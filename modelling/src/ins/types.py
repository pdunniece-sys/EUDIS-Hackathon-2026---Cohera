"""The data types every other part of the INS talks in.

Three of them, and nothing else crosses a module boundary:

    Trajectory  - where the vehicle was (or where the INS thinks it was)
    ImuSamples  - what the accelerometers and gyros measured
    InsState    - the integrator's working state, for one vehicle or for many

Frame conventions, fixed everywhere in this package
---------------------------------------------------
Body frame is **FRD**: x forward, y right, z down. Same as ArduPilot, so a
logged IMU needs no axis shuffling.

Navigation frame is **NED**: x north, y east, z down. Down is positive, so an
altitude of 60 m is a down-coordinate of -60.

Attitude is a quaternion rotating **body to NED**, stored scalar-first as
(w, x, y, z). Applying it to a body-frame vector gives that vector in NED.

Accelerometers measure **specific force**, not acceleration. Sitting level and
still, the z accelerometer reads about -9.81 m/s^2, not +9.81 and not 0. This
catches people out; it is the same sign convention ArduPilot logs use.

One thing this deliberately does NOT do
---------------------------------------
No field is ever filled in with a plausible-looking default. A Trajectory with
no velocity has `vel_ned = None`, not an array of zeros, and anything that
needs velocity says so rather than quietly integrating a lie. Gaps inside an
array that should hold a reading are NaN for the same reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# WGS84, used by every lat/lon conversion in this package.
WGS84_A = 6_378_137.0           # semi-major axis, m
WGS84_F = 1.0 / 298.257_223_563  # flattening
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)  # first eccentricity squared


@dataclass
class Trajectory:
    """Where the vehicle was, sample by sample.

    Used for three different things, which is why so much is optional:
    the simulated truth, a GNSS track loaded from a log, and the INS estimate.

        t         seconds from the start of this trajectory
        lat, lon  degrees, WGS84
        alt_m     metres above the WGS84 ellipsoid (not sea level)
        vel_ned   (n, 3) m/s in the navigation frame, or None
        att_quat  (n, 4) body->NED quaternion, scalar first, or None
    """

    t: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    alt_m: np.ndarray
    vel_ned: np.ndarray | None = None
    att_quat: np.ndarray | None = None
    name: str = ""
    source: Path | None = None

    def __post_init__(self):
        self.t = np.asarray(self.t, dtype=float)
        self.lat = np.asarray(self.lat, dtype=float)
        self.lon = np.asarray(self.lon, dtype=float)
        self.alt_m = np.asarray(self.alt_m, dtype=float)
        n = len(self.t)
        for field_name in ("lat", "lon", "alt_m"):
            got = len(getattr(self, field_name))
            if got != n:
                raise ValueError(f"{field_name} has {got} samples, t has {n}")
        if self.vel_ned is not None:
            self.vel_ned = np.asarray(self.vel_ned, dtype=float).reshape(-1, 3)
        if self.att_quat is not None:
            self.att_quat = np.asarray(self.att_quat, dtype=float).reshape(-1, 4)

    def __len__(self) -> int:
        return len(self.t)

    @property
    def duration_s(self) -> float:
        return float(self.t[-1] - self.t[0])

    def __repr__(self) -> str:
        label = f" {self.name!r}" if self.name else ""
        return f"<Trajectory{label} {len(self)} samples, {self.duration_s:.1f} s>"

    def at(self, times_s) -> "Trajectory":
        """Interpolate this trajectory onto other timestamps.

        Linear in latitude, longitude and altitude, which is fine at these
        sample rates. Quaternions are interpolated component-wise and
        renormalised - crude, but adequate between samples milliseconds apart.
        """
        times_s = np.asarray(times_s, dtype=float)
        pick = lambda a: np.interp(times_s, self.t, a)
        vel = att = None
        if self.vel_ned is not None:
            vel = np.column_stack([np.interp(times_s, self.t, self.vel_ned[:, i])
                                   for i in range(3)])
        if self.att_quat is not None:
            att = np.column_stack([np.interp(times_s, self.t, self.att_quat[:, i])
                                   for i in range(4)])
            att = att / np.linalg.norm(att, axis=1, keepdims=True)
        return Trajectory(times_s, pick(self.lat), pick(self.lon), pick(self.alt_m),
                          vel, att, name=self.name, source=self.source)

    def horizontal_error_m(self, truth: "Trajectory") -> np.ndarray:
        """Distance on the ground between this trajectory and a reference one.

        The reference is interpolated onto our timestamps first. Returns NaN
        wherever our timestamps fall outside the reference's span, rather than
        silently extrapolating it flat and reporting a flattering error.
        """
        ref = truth.at(self.t)
        outside = (self.t < truth.t[0]) | (self.t > truth.t[-1])

        # Local flat approximation about the reference track. Good to well
        # under a metre over the few kilometres a survey flight covers.
        lat0 = float(np.nanmean(ref.lat))
        m_per_deg_lat, m_per_deg_lon = metres_per_degree(lat0)
        north = (self.lat - ref.lat) * m_per_deg_lat
        east = (self.lon - ref.lon) * m_per_deg_lon

        error = np.hypot(north, east)
        error[outside] = np.nan
        return error


@dataclass
class ImuSamples:
    """What the IMU measured, on its own clock.

        t        seconds, strictly increasing, NOT assumed evenly spaced
        accel_b  (n, 3) specific force in body FRD, m/s^2
        gyro_b   (n, 3) angular rate in body FRD, rad/s

    Irregular spacing is normal in real logs - ArduPilot drops samples and
    the QuSpin logger has its own jitter - so `dt` is computed per step from
    the timestamps rather than assumed.
    """

    t: np.ndarray
    accel_b: np.ndarray
    gyro_b: np.ndarray
    name: str = ""
    source: Path | None = None

    def __post_init__(self):
        self.t = np.asarray(self.t, dtype=float)
        self.accel_b = np.asarray(self.accel_b, dtype=float).reshape(-1, 3)
        self.gyro_b = np.asarray(self.gyro_b, dtype=float).reshape(-1, 3)
        if not (len(self.t) == len(self.accel_b) == len(self.gyro_b)):
            raise ValueError(
                f"length mismatch: t={len(self.t)}, accel={len(self.accel_b)}, "
                f"gyro={len(self.gyro_b)}"
            )
        if len(self.t) > 1 and np.any(np.diff(self.t) <= 0):
            bad = int(np.argmin(np.diff(self.t)))
            raise ValueError(
                f"timestamps must strictly increase; they do not at index {bad} "
                f"(t={self.t[bad]:.6f} -> {self.t[bad + 1]:.6f})"
            )

    def __len__(self) -> int:
        return len(self.t)

    @property
    def dt(self) -> np.ndarray:
        """Interval before each sample. The first entry repeats the second,
        because there is no interval before the first sample."""
        if len(self.t) < 2:
            return np.array([np.nan])
        d = np.diff(self.t)
        return np.concatenate([[d[0]], d])

    @property
    def rate_hz(self) -> float:
        return (len(self.t) - 1) / (self.t[-1] - self.t[0])

    def window(self, start_s: float, end_s: float) -> "ImuSamples":
        """The samples between two times, inclusive."""
        keep = (self.t >= start_s) & (self.t <= end_s)
        if not keep.any():
            raise ValueError(f"no IMU samples between {start_s} and {end_s} s")
        return ImuSamples(self.t[keep], self.accel_b[keep], self.gyro_b[keep],
                          name=self.name, source=self.source)

    def __repr__(self) -> str:
        label = f" {self.name!r}" if self.name else ""
        return f"<ImuSamples{label} {len(self)} samples, {self.rate_hz:.1f} Hz>"


@dataclass
class InsState:
    """The integrator's working state, for one vehicle or for a thousand.

        pos_ned  (N, 3) metres from the navigation-frame origin
        vel_ned  (N, 3) m/s
        quat     (N, 4) body->NED, scalar first

    Always 2-D, even for a single vehicle, so that `propagate` has exactly one
    code path. A particle filter holds one of these with N = particle count and
    steps the whole cloud in a single call.
    """

    pos_ned: np.ndarray
    vel_ned: np.ndarray
    quat: np.ndarray

    def __post_init__(self):
        self.pos_ned = np.atleast_2d(np.asarray(self.pos_ned, dtype=float))
        self.vel_ned = np.atleast_2d(np.asarray(self.vel_ned, dtype=float))
        self.quat = np.atleast_2d(np.asarray(self.quat, dtype=float))
        if not (self.pos_ned.shape[0] == self.vel_ned.shape[0] == self.quat.shape[0]):
            raise ValueError(
                f"N differs between fields: pos={self.pos_ned.shape}, "
                f"vel={self.vel_ned.shape}, quat={self.quat.shape}"
            )

    @classmethod
    def at_rest(cls, n: int = 1) -> "InsState":
        """N vehicles at the origin, stationary, level and facing north."""
        return cls(np.zeros((n, 3)), np.zeros((n, 3)),
                   np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)))

    @property
    def n(self) -> int:
        return self.pos_ned.shape[0]

    def copy(self) -> "InsState":
        return InsState(self.pos_ned.copy(), self.vel_ned.copy(), self.quat.copy())

    def __repr__(self) -> str:
        return f"<InsState N={self.n}>"


def metres_per_degree(lat_deg: float) -> tuple[float, float]:
    """How many metres one degree of latitude and of longitude span, at a latitude.

    Uses the WGS84 meridional and transverse radii of curvature, so this is
    correct to the shape of the ellipsoid rather than assuming a sphere.
    """
    lat = np.radians(lat_deg)
    s = np.sin(lat)
    # Meridional radius: how far north you travel per radian of latitude.
    r_meridian = WGS84_A * (1 - WGS84_E2) / (1 - WGS84_E2 * s**2) ** 1.5
    # Transverse radius: the parallel's radius, shrinking with cos(lat).
    r_transverse = WGS84_A / np.sqrt(1 - WGS84_E2 * s**2)
    return (np.radians(1.0) * r_meridian,
            np.radians(1.0) * r_transverse * np.cos(lat))
