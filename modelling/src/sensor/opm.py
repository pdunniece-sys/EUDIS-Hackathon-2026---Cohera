"""A QTFM-like scalar optically pumped magnetometer: the quantum sensor.

    from sensor import ScalarOPM, FieldModel

    opm = ScalarOPM()
    readings = opm.emulate(trajectory, field_model, seed=0)
    anomaly = opm.to_anomaly(readings, trajectory, field_model)

Every error term is switchable and every value is a named parameter, so a run
can isolate one effect. With all of them off the emulator returns the map value
exactly, which is what `tests/test_sensor.py` checks first.

Where the numbers come from
---------------------------
Three are measured, from the real flight in notebooks/dynamic.ipynb:

    platform + environment noise   1.35 nT RMS above 0.1 Hz, with mains
                                   notched out. **This is not sensor noise.**
                                   It is everything else: the airframe, the
                                   air, the ground. Labelling it sensor noise
                                   would flatter the environment and blame the
                                   instrument.
    mains interference             about 5 nT RMS, 93% of all noise power in
                                   that flight. 60 Hz in Ontario, 50 Hz in
                                   Ireland - hence a parameter.
    broadband floor                about 4 pT/rtHz observed at 100-125 Hz.

One is from the datasheet: 3 pT/rtHz instrument noise. The observed 4 pT/rtHz
floor sits just above it, which is the expected relationship and a decent check
that neither figure is wildly wrong.

Two are datasheet-stated but unmeasured by us, and they are the weak points:
heading error and the dead zone. We have no rotation test, and the March
flight never put the sensor in an awkward attitude - its field-to-axis angle
only spanned about 50 to 126 degrees, nowhere near a dead zone. So the shapes
below are plausible defaults, not fitted curves. Both are parameters for that
reason.

The dead zone is the one that bites
-----------------------------------
A scalar OPM works by driving an atomic resonance with light. When the field
lines up with the optical axis there is no transverse component to precess
about, the resonance disappears, and the sensor stops reading - it does not
read badly, it reads nothing. The emulator returns NaN there, never a number,
because a plausible-looking value in a dead zone would be the single most
dangerous output this module could produce.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field

import numpy as np
from scipy import signal

from ins import rotations
from sensor.timing import decimate
from sensor.types import Readings

# QuSpin QTFM, from the datasheet.
INSTRUMENT_NOISE_PT_PER_RTHZ = 3.0
DEAD_ZONE_DEG = 7.0
HEADING_ERROR_NT = 3.0          # datasheet: "<3 nT uncompensated"

# Measured in notebooks/dynamic.ipynb on the March 2026 flight.
PLATFORM_NOISE_NT_RMS = 1.35
MAINS_NT_RMS = 5.0
# Slope of the platform noise amplitude spectrum, fitted to that flight.
# Steeply red: two thirds of the power is below 0.3 Hz. See _coloured_noise.
PLATFORM_NOISE_SLOPE = -1.5


def default_heading_error(theta_rad: np.ndarray, amplitude_nt: float) -> np.ndarray:
    """Heading error against the field-to-sensor-axis angle.

    A placeholder shape: zero when the field is along or across the sensor
    axis, largest in between, bounded by `amplitude_nt`.

    **Replace this.** QuSpin publish a Heading Error Compensation document with
    the real response for the QTFM, and a measured curve from our own rotation
    test would be better still. The magnitude here is right (the datasheet says
    under 3 nT uncompensated); the shape is a guess, and it is a parameter so
    that swapping it in is a one-line change rather than an edit to this file.
    """
    return amplitude_nt * np.sin(2.0 * theta_rad)


@dataclass
class ScalarOPM:
    """A QTFM-like scalar magnetometer.

    Switches, all defaulting to on, so that turning one off is a deliberate
    experiment rather than the normal state.

        axis_body           (3,) sensor optical axis in body FRD. Default is
                            forward, which is a mounting choice, not a finding.
        rate_hz             native output rate, 250 Hz
        output_rate_hz      what the navigation filter wants, 10 Hz
        latency_s           the sensor's own reporting delay
        mains_hz            50 in Ireland, 60 in North America
    """

    axis_body: np.ndarray = dc_field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    rate_hz: float = 250.0
    output_rate_hz: float = 10.0
    latency_s: float = 0.0

    instrument_noise_pt_per_rthz: float = INSTRUMENT_NOISE_PT_PER_RTHZ
    platform_noise_nt_rms: float = PLATFORM_NOISE_NT_RMS
    platform_noise_highpass_hz: float = 0.1
    platform_noise_slope: float = PLATFORM_NOISE_SLOPE
    mains_hz: float = 50.0
    mains_nt_rms: float = MAINS_NT_RMS
    heading_error_nt: float = HEADING_ERROR_NT
    heading_error_fn: object = default_heading_error
    dead_zone_deg: float = DEAD_ZONE_DEG

    def __post_init__(self):
        self.axis_body = np.asarray(self.axis_body, dtype=float).reshape(3)
        norm = np.linalg.norm(self.axis_body)
        if norm == 0:
            raise ValueError("axis_body cannot be the zero vector")
        self.axis_body = self.axis_body / norm

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def theta(self, trajectory, field) -> np.ndarray:
        """Angle between the field and the sensor's optical axis, radians.

        This single number drives both the heading error and the dead zone, so
        it is exposed: plotting a flight's theta tells you immediately whether
        a dead-zone crossing was ever close.
        """
        if trajectory.att_quat is None:
            raise ValueError(
                "a scalar OPM's errors depend on how the sensor sits in the "
                "field, so the trajectory must carry attitude"
            )
        axis_ned = rotations.rotate(trajectory.att_quat,
                                    np.tile(self.axis_body, (len(trajectory), 1)))
        cos_theta = np.einsum("ij,ij->i", axis_ned, field.b_hat)
        return np.arccos(np.clip(cos_theta, -1.0, 1.0))

    def in_dead_zone(self, theta_rad: np.ndarray) -> np.ndarray:
        """True where the field is too close to the sensor axis to read.

        Both ends: along the axis and directly against it are equally blind.
        """
        limit = np.radians(self.dead_zone_deg)
        return (theta_rad < limit) | (theta_rad > np.pi - limit)

    # ------------------------------------------------------------------
    # The emulator
    # ------------------------------------------------------------------

    def emulate(self, trajectory, field_model, seed: int | None = 0,
                decimate_output: bool = True) -> Readings:
        """What this sensor would report flying `trajectory`.

        `field_model` may be a FieldModel or an already-computed FieldSamples,
        so a demo can build the field once and run several sensors against it.
        """
        field = (field_model if hasattr(field_model, "b_ned")
                 else field_model.at(trajectory))
        rng = np.random.default_rng(seed)

        n = len(field)
        dt = 1.0 / self.rate_hz
        reading = field.total_nt.copy()

        theta = self.theta(trajectory, field)
        dead = self.in_dead_zone(theta)

        # --- heading error -------------------------------------------------
        if self.heading_error_nt:
            reading = reading + self.heading_error_fn(theta, self.heading_error_nt)

        # --- noise that grows towards the dead zone ------------------------
        # The resonance weakens as the field swings towards the optical axis,
        # so the sensor gets noisier before it gives up entirely. Scaling by
        # 1/sin(theta), floored at the dead-zone edge, is the simplest form
        # with that behaviour. The shape is a guess; the existence of the
        # effect is not.
        floor = np.sin(np.radians(self.dead_zone_deg))
        noise_scale = 1.0 / np.maximum(np.sin(theta), floor)

        # --- instrument noise, from the datasheet --------------------------
        if self.instrument_noise_pt_per_rthz:
            # A spectral density in pT/rtHz becomes a per-sample standard
            # deviation of density / sqrt(dt), then pT -> nT.
            sigma = self.instrument_noise_pt_per_rthz / np.sqrt(dt) / 1000.0
            reading = reading + rng.normal(0.0, sigma, n) * noise_scale

        # --- platform and environment noise, measured ----------------------
        if self.platform_noise_nt_rms:
            reading = reading + self._coloured_noise(rng, n, dt)

        # --- mains ---------------------------------------------------------
        if self.mains_nt_rms and self.mains_hz:
            # Amplitude is sqrt(2) times RMS for a sine. The phase is random
            # per run but fixed within it, because mains does not re-randomise
            # mid-flight.
            phase = rng.uniform(0, 2 * np.pi)
            reading = reading + (self.mains_nt_rms * np.sqrt(2)
                                 * np.sin(2 * np.pi * self.mains_hz * field.t + phase))

        valid = ~dead
        reading = np.where(valid, reading, np.nan)

        if not decimate_output:
            return Readings(field.t, reading, valid, name="QTFM-like OPM",
                            rate_hz=self.rate_hz, latency_s=self.latency_s)

        out = decimate(field.t, reading, valid, self.rate_hz, self.output_rate_hz,
                       latency_s=self.latency_s)
        return Readings(out.t, out.values, out.valid, name="QTFM-like OPM",
                        rate_hz=self.output_rate_hz, latency_s=out.total_delay_s)

    def _coloured_noise(self, rng, n: int, dt: float) -> np.ndarray:
        """Platform and environment noise: red, not white.

        The shape matters as much as the size. Measured on the March flight
        (the band breakdown is in notebooks/dynamic.ipynb):

            0.1 - 0.3 Hz    2516 pT/rtHz    65% of the variance
            0.3 - 1.0 Hz     321
            1.0 - 3.0 Hz      81
            3   - 100 Hz    8-11            nearly flat
            100 - 125 Hz       3.4

        So almost all of it is below 1 Hz and it falls off steeply - close to
        f^-1.5 - until it meets a near-white floor. Modelling it as white noise
        scaled to 1.35 nT RMS would put 120 pT/rtHz at 100 Hz, thirty times the
        measured floor, and would swamp the instrument's own 3 pT/rtHz.

        Synthesised in the frequency domain because the shape is the
        specification: give every bin above the corner an amplitude of
        f^slope, randomise the phases, transform back, then scale the result so
        the band RMS matches what was measured.
        """
        freqs = np.fft.rfftfreq(n, dt)
        shape = np.zeros_like(freqs)
        above = freqs >= self.platform_noise_highpass_hz
        if not above.any():
            return np.zeros(n)
        shape[above] = freqs[above] ** self.platform_noise_slope

        phases = rng.normal(0.0, 1.0, len(freqs)) + 1j * rng.normal(0.0, 1.0, len(freqs))
        noise = np.fft.irfft(phases * shape, n=n)

        # Scale through the same high-pass the measurement used, not against
        # the raw RMS. The measured 1.35 nT was the standard deviation *after*
        # a second-order high-pass at 0.1 Hz, and since two thirds of this
        # noise sits within an octave of that corner, the filter removes a
        # quarter of it. Normalising the raw series instead would leave the
        # emulator a reliable 25% quiet.
        nyquist = 0.5 / dt
        corner = self.platform_noise_highpass_hz / nyquist
        if corner < 1.0:
            measured = signal.filtfilt(*signal.butter(2, corner, "high"), noise)
        else:
            measured = noise

        rms = np.sqrt(np.mean(measured**2))
        return noise * (self.platform_noise_nt_rms / rms) if rms > 0 else noise

    # ------------------------------------------------------------------
    # What the navigation filter actually compares with the map
    # ------------------------------------------------------------------

    def to_anomaly(self, readings: Readings, trajectory, field_model) -> np.ndarray:
        """Reading minus the core field minus the diurnal model.

        The result is the quantity to compare against the Tellus map: a crustal
        anomaly in nT. Everything that is not crust has been subtracted using a
        *model*, not truth, so the residual carries the model's errors too -
        which is the honest version, because that is what a real system has.

        Note what this cannot remove. The diurnal model here is a sine wave
        standing in for a reference observatory. A real deployment subtracts
        measured MagIE data; until then the diurnal residual is a systematic
        error sitting directly on the signal.
        """
        return _subtract_background(readings, trajectory, field_model)


def _subtract_background(readings: Readings, trajectory, field_model) -> np.ndarray:
    """Shared by both sensor types: strip the core field and the diurnal model.

    The readings are on their own clock - later than the trajectory, by the
    sensor's latency and filter delay - so the background is evaluated at the
    position the aircraft held when each reading was *taken*.
    """
    if not hasattr(field_model, "core_field_ned"):
        raise TypeError(
            "to_anomaly needs the FieldModel itself, not a FieldSamples: it has "
            "to evaluate the background at the readings' own timestamps"
        )

    taken_at = readings.t - readings.latency_s
    where = trajectory.at(np.clip(taken_at, trajectory.t[0], trajectory.t[-1]))

    core = np.linalg.norm(
        field_model.core_field_ned(where.lat, where.lon, where.alt_m), axis=1)
    diurnal = np.asarray(field_model.diurnal(where.t))
    return readings.field_nt - core - diurnal
