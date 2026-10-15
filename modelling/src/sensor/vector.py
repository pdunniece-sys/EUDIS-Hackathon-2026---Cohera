"""An RM3100-like vector magnetometer: the cheap sensor, for comparison.

    from sensor import VectorMagnetometer

    rm3100 = VectorMagnetometer()
    readings = rm3100.emulate(trajectory, field_model, seed=0)

This exists to answer "why quantum". A three-axis magnetometer costs a few
pounds, draws almost nothing and is already on the aircraft. If its magnitude
were good enough to navigate on, there would be no case for an OPM.

Why a vector sensor's magnitude is worse than a scalar sensor's reading
----------------------------------------------------------------------
Not because the parts are worse - because of what has to be true for the
magnitude to be right.

A scalar OPM measures `|B|` directly, through an atomic resonance. Its reading
does not depend on which way it is pointing, except through the small heading
error and the dead zone.

A vector magnetometer measures three components and `|B|` is computed from
them. That means every per-axis error - bias, scale factor, misalignment -
feeds into the magnitude *weighted by how the sensor happens to be oriented*.
Turn the aircraft and the weighting changes, so the magnitude error changes
with heading even though the field has not.

That is the crux. A 0.5% scale error on one axis is 250 nT on a 50,000 nT
field, and it swings with heading as the aircraft flies its survey lines. The
crustal anomaly we are trying to read is tens of nT. The error is not just
larger than the signal, it is *correlated with the flight pattern*, which is
the worst possible shape for it to have: it looks like geology.

Calibration helps a great deal, and a well-calibrated vector magnetometer is a
genuinely useful instrument. It does not close a gap this size, and the
calibration has to hold across temperature and time.

Where the numbers come from
---------------------------
PNI RM3100 datasheet, at cycle count 200: **15 nT noise**, 13 nT sensitivity.
Bias, scale factor and misalignment are not specified per-part - they are what
calibration is for - so the defaults here are typical uncalibrated values and
are marked as assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field

import numpy as np

from ins import rotations
from sensor.opm import _subtract_background
from sensor.timing import decimate
from sensor.types import Readings

# PNI RM3100 datasheet, cycle count 200.
RM3100_NOISE_NT = 15.0

# ASSUMED, not datasheet. Typical of an uncalibrated three-axis part.
BIAS_NT = 50.0
SCALE_ERROR = 0.005          # 0.5%
MISALIGNMENT_DEG = 0.5


@dataclass
class VectorMagnetometer:
    """An RM3100-like three-axis magnetometer, reporting |B|.

        bias_nt           (3,) per-axis offset, or a scalar sigma to draw from
        scale_error       per-axis scale factor error, as a fraction
        misalignment_deg  how far the axes are from truly orthogonal and from
                          the body frame
        noise_nt          per-axis white noise, 15 nT from the datasheet
        axes_body         (3, 3) sensor axes in body FRD. Identity by default.
    """

    bias_nt: float = BIAS_NT
    scale_error: float = SCALE_ERROR
    misalignment_deg: float = MISALIGNMENT_DEG
    noise_nt: float = RM3100_NOISE_NT
    rate_hz: float = 250.0
    output_rate_hz: float = 10.0
    latency_s: float = 0.0
    axes_body: np.ndarray = dc_field(default_factory=lambda: np.eye(3))

    def realise(self, rng):
        """Draw this unit's fixed errors: bias, scale and misalignment.

        Drawn once per sensor, not per sample - they are properties of the
        part, not noise. Returned so a surprising result can be explained.
        """
        bias = (np.asarray(self.bias_nt, dtype=float).reshape(3)
                if np.ndim(self.bias_nt) else rng.normal(0.0, self.bias_nt, 3))
        scale = rng.normal(0.0, self.scale_error, 3) if self.scale_error else np.zeros(3)

        misalign = np.eye(3)
        if self.misalignment_deg:
            # Small rotations about each axis: the sensor's axes are not quite
            # the body's, and not quite perpendicular to each other.
            angles = np.radians(rng.normal(0.0, self.misalignment_deg, 3))
            misalign = rotations.to_matrix(rotations.exp_map(angles))[0]
        return bias, scale, misalign

    def emulate(self, trajectory, field_model, seed: int | None = 0,
                decimate_output: bool = True) -> Readings:
        """What this sensor would report, as a magnitude.

        The field is rotated into the sensor's own axes, corrupted there - which
        is where the errors physically live - and only then collapsed to a
        magnitude. Doing it in that order is what makes the error depend on
        heading, which is the entire point of the comparison.
        """
        field = (field_model if hasattr(field_model, "b_ned")
                 else field_model.at(trajectory))
        if trajectory.att_quat is None:
            raise ValueError(
                "a vector magnetometer's errors depend on its orientation, so "
                "the trajectory must carry attitude"
            )

        rng = np.random.default_rng(seed)
        bias, scale, misalign = self.realise(rng)
        n = len(field)

        # True field in the sensor's frame.
        b_body = rotations.rotate_inverse(trajectory.att_quat, field.b_ned)
        b_sensor = b_body @ self.axes_body.T @ misalign.T

        # Per-axis errors, applied where they actually occur.
        measured = b_sensor * (1.0 + scale) + bias
        if self.noise_nt:
            measured = measured + rng.normal(0.0, self.noise_nt, (n, 3))

        reading = np.linalg.norm(measured, axis=1)
        valid = np.ones(n, dtype=bool)      # no dead zone: it reads everywhere

        if not decimate_output:
            return Readings(field.t, reading, valid, name="RM3100-like vector",
                            rate_hz=self.rate_hz, latency_s=self.latency_s)

        out = decimate(field.t, reading, valid, self.rate_hz, self.output_rate_hz,
                       latency_s=self.latency_s)
        return Readings(out.t, out.values, out.valid, name="RM3100-like vector",
                        rate_hz=self.output_rate_hz, latency_s=out.total_delay_s)

    def to_anomaly(self, readings: Readings, trajectory, field_model) -> np.ndarray:
        """Reading minus core field minus diurnal, same as the OPM."""
        return _subtract_background(readings, trajectory, field_model)
