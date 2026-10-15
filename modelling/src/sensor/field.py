"""The magnetic field that is actually there, at the aircraft, sample by sample.

    from sensor import FieldModel
    from tellus import MagneticGrid

    model = FieldModel(grid=MagneticGrid("a7_tmi"), date=date(2026, 10, 15))
    field = model.at(trajectory)
    field.total_nt            # what a perfect scalar sensor would read

Four things add up to the field at a flying drone:

    core        ~49,000 nT in Ireland. The Earth's dynamo, from IGRF. Smooth,
                predictable, and almost all of the magnitude.
    anomaly     tens to hundreds of nT. The crust, from the Tellus map. This
                is the only part that says anything about position, and it is
                about 0.5% of the total.
    diurnal     tens of nT over a day. The ionosphere, driven by the Sun.
                Changes with time, not position, which is what makes it a
                nuisance rather than a signal.
    platform    the drone's own field. Zero by default - see interference.py.

The signal we navigate on is the second one. Everything here exists to get the
other three out of the way honestly rather than assuming them away.

Why the field is assembled as a vector
--------------------------------------
A scalar magnetometer reads `|B|`, not any one component. So the only way to
work out what a sensor reads is to add the parts as vectors and then take the
magnitude at the end. That automatically gets two things right that are easy
to fumble:

  - a crustal anomaly adds along the core field direction, so a +100 nT
    anomaly raises the total by about 100 nT
  - an interference vector `b` only shows up in `|B|` through its component
    along the field, `b . B_hat`. Interference perpendicular to the field is
    nearly invisible to a scalar sensor and fully visible to a vector one.

Both fall out of `|sum of vectors|` without being special-cased.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field as dc_field

import numpy as np

from sensor.interference import NoInterference
from sensor.types import FieldSamples

# IGRF is slow to evaluate and smooth in space, so it is computed on a coarse
# grid of trajectory samples and interpolated between them. Over a survey
# flight the core field changes by a few nT across the whole block, so this
# costs far less than a nanotesla.
IGRF_EVERY_N_SAMPLES = 500


@dataclass
class DiurnalModel:
    """The daily swing in the field, driven by ionospheric currents.

        amplitude_nt   peak-to-peak size. Quiet days are tens of nT at Irish
                       latitudes; a magnetic storm is hundreds.
        period_hours   24 by default.
        phase_hours    when the peak falls, in hours after local midnight.
                       Default 12, so the extreme is around local noon, which
                       is when the driving currents are strongest.

    This is a sine wave, and a real diurnal variation is not. It is here so
    that the emulator has a time-varying term to remove, which is the shape of
    the real problem: the correction has to come from a reference station,
    because nothing about a single moving sensor can separate a field that
    changes with time from one that changes with position.

    The real version reads MagIE observatory data (Birr and Dunsink, the Irish
    magnetic observatories). Until that is wired in, keep the amplitude
    modest and do not treat the residual as calibrated.
    """

    amplitude_nt: float = 20.0
    period_hours: float = 24.0
    phase_hours: float = 12.0

    def __call__(self, t_s: np.ndarray, start: dt.datetime | None = None) -> np.ndarray:
        hours = np.asarray(t_s, dtype=float) / 3600.0
        if start is not None:
            hours = hours + start.hour + start.minute / 60.0
        radians = 2 * np.pi * (hours - self.phase_hours) / self.period_hours
        return 0.5 * self.amplitude_nt * np.sin(radians)


class NoDiurnal:
    """A perfectly steady ionosphere. Useful for isolating other effects."""

    def __call__(self, t_s, start=None) -> np.ndarray:
        return np.zeros(len(np.atleast_1d(t_s)))

    def __repr__(self) -> str:
        return "<NoDiurnal>"


@dataclass
class FieldModel:
    """Assembles the true field along a trajectory.

        grid           a MagneticGrid for the crustal anomaly, or None to
                       leave the anomaly at zero
        date           the flight date, for IGRF. The core field drifts by
                       tens of nT a year, so this is not cosmetic.
        diurnal        a DiurnalModel, or NoDiurnal()
        interference   a callable from interference.py
        continue_to_altitude  lift the map from the 60 m survey height to the
                       flight altitude before sampling. On by default,
                       because not doing it is a silent error of the same
                       order as the signal.
    """

    grid: object = None
    date: dt.date = dc_field(default_factory=lambda: dt.date(2026, 10, 15))
    diurnal: object = dc_field(default_factory=DiurnalModel)
    interference: object = dc_field(default_factory=NoInterference)
    continue_to_altitude: bool = True
    _continued_cache: dict = dc_field(default_factory=dict, repr=False)

    def core_field_ned(self, lat, lon, alt_m) -> np.ndarray:
        """IGRF core field as an (n, 3) NED vector in nT.

        Two conversions happen here and both are easy to get wrong, so they are
        in one place: ppigrf wants height in **kilometres**, and it returns
        **east, north, up** while everything in this project is north, east,
        **down**.
        """
        import ppigrf

        lat = np.atleast_1d(np.asarray(lat, dtype=float))
        lon = np.atleast_1d(np.asarray(lon, dtype=float))
        alt_m = np.atleast_1d(np.asarray(alt_m, dtype=float))

        when = dt.datetime(self.date.year, self.date.month, self.date.day)
        east, north, up = ppigrf.igrf(lon, lat, alt_m / 1000.0, when)
        east = np.asarray(east).reshape(-1)
        north = np.asarray(north).reshape(-1)
        up = np.asarray(up).reshape(-1)
        return np.column_stack([north, east, -up])      # NED: down = -up

    def _core_along(self, trajectory) -> np.ndarray:
        """Core field at every trajectory sample, interpolated from a coarse set."""
        n = len(trajectory)
        step = max(1, n // IGRF_EVERY_N_SAMPLES)
        picks = np.unique(np.concatenate([np.arange(0, n, step), [n - 1]]))

        coarse = self.core_field_ned(trajectory.lat[picks], trajectory.lon[picks],
                                     trajectory.alt_m[picks])
        if len(picks) == n:
            return coarse
        return np.column_stack([
            np.interp(np.arange(n), picks, coarse[:, i]) for i in range(3)
        ])

    def _anomaly_along(self, trajectory) -> np.ndarray:
        """Crustal anomaly at each sample, from the map at flight altitude."""
        n = len(trajectory)
        if self.grid is None:
            return np.zeros(n)

        grid = self.grid
        if self.continue_to_altitude:
            # One continuation for the whole flight, keyed on the altitude
            # rounded to a metre - re-running the FFT per sample would be
            # absurd, and a metre of altitude changes the map by very little.
            altitude = float(np.round(np.nanmean(trajectory.alt_m)))
            if altitude not in self._continued_cache:
                from tellus.continuation import continue_to_altitude

                self._continued_cache[altitude] = continue_to_altitude(grid, altitude)
            grid = self._continued_cache[altitude]

        return grid.sample_track(trajectory.lat, trajectory.lon)

    def at(self, trajectory) -> FieldSamples:
        """The full field along a trajectory.

        The anomaly is added along the core field direction, because a total
        field anomaly map records a change in magnitude, not a vector.
        """
        core_ned = self._core_along(trajectory)
        core_nt = np.linalg.norm(core_ned, axis=1)
        core_hat = core_ned / core_nt[:, None]

        anomaly = self._anomaly_along(trajectory)
        diurnal = np.asarray(self.diurnal(trajectory.t))

        # Interference is produced in the body frame and rotated into NED.
        try:
            body = self.interference(trajectory, b_ned=core_ned)
        except TypeError:
            body = self.interference(trajectory)
        body = np.asarray(body, dtype=float).reshape(-1, 3)

        if np.any(body) and trajectory.att_quat is None:
            raise ValueError(
                "interference is in the body frame, so rotating it into the "
                "navigation frame needs the trajectory's attitude"
            )
        if np.any(body):
            from ins import rotations
            interference_ned = rotations.rotate(trajectory.att_quat, body)
        else:
            interference_ned = np.zeros_like(core_ned)

        # The anomaly and the diurnal both act along the ambient field.
        scalar_part = np.nan_to_num(anomaly, nan=0.0) + diurnal
        b_ned = core_ned + core_hat * scalar_part[:, None] + interference_ned

        return FieldSamples(trajectory.t, b_ned, core_nt, anomaly, diurnal,
                            interference_ned)
