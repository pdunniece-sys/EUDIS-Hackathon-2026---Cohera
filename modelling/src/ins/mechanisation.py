"""Strapdown inertial navigation: turn accelerometer and gyro readings into a position.

    from ins import Mechanisation, NavFrame, InsState

    frame = NavFrame(lat0=52.45, lon0=-6.40, alt0=60.0)
    mech = Mechanisation(frame)
    state = mech.initialise(lat=52.45, lon=-6.40, alt_m=60.0,
                            vel_ned=[8, 0, 0], att_quat=[1, 0, 0, 0])

    for accel_b, gyro_b, dt in samples:
        state = mech.propagate(state, accel_b, gyro_b, dt)

Everything is vectorised over the leading axis of the state, so stepping one
vehicle and stepping ten thousand particles are the same call.

The integration scheme, and why it is this one
----------------------------------------------
Semi-implicit Euler: velocity updates first, then position uses the *new*
velocity.

    q_new   = q * exp(gyro * dt)
    a_ned   = R(q) @ accel_b + gravity        (attitude at the step start)
    vel_new = vel + a_ned * dt
    pos_new = pos + vel_new * dt

Chosen because it inverts exactly. `inverse_propagate` in trajectory.py runs
these four lines backwards to recover the IMU readings that would produce a
known path, and because the inverse is exact rather than approximate, feeding
those readings back in reproduces the path to numerical precision. That turns
"does the integrator work" into a test with a real pass mark instead of a
hand-waved tolerance.

Higher-order integration would be more accurate per step at a given rate. It
is not worth it here: at 100-400 Hz the integration error is thousands of
times smaller than the sensor errors we are trying to measure.

What is modelled, and what is left out
--------------------------------------
Modelled: latitude- and altitude-dependent normal gravity, and optionally the
Coriolis term from Earth's rotation.

Left out: the transport rate (the navigation frame tipping as the vehicle
moves over the curved Earth), and any flexing of the local frame. Over the
10-30 km a survey flight covers these are far below the sensor errors. The
navigation frame is treated as flat and fixed, anchored at the start point.

One thing this deliberately does NOT do
---------------------------------------
It does not quietly stabilise the horizontal channels. A free-running INS
position diverges, and that divergence is the entire measurement - damping it
with a pseudo-measurement would produce a flattering number that means
nothing. Only the vertical channel is aided, because a free INS altitude
diverges so violently that it corrupts the horizontal channels through the
gravity projection, and because every real drone has a barometer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ins import rotations
from ins.types import InsState, Trajectory, metres_per_degree

# Earth's rotation rate, rad/s (WGS84 / IERS).
EARTH_RATE = 7.292_115e-5


def normal_gravity(lat_deg, alt_m=0.0):
    """Gravity magnitude in m/s^2 at a latitude and height.

    Somigliana's formula on the WGS84 ellipsoid, plus a first-order free-air
    correction for height. Gravity is about 0.5% stronger at the poles than at
    the equator and weakens by 3 parts per million per metre of altitude -
    small, but a constant 9.81 would be a systematic error sitting directly on
    the vertical channel.
    """
    lat = np.radians(np.asarray(lat_deg, dtype=float))
    s2 = np.sin(lat) ** 2
    g0 = 9.780_326_771_5 * (1 + 0.005_302_44 * s2 - 0.000_005_82 * s2**2)
    return g0 - 3.086e-6 * np.asarray(alt_m, dtype=float)


@dataclass
class NavFrame:
    """A local NED frame pinned to one point on the Earth.

    Positions inside the INS are plain metres north, east and down from this
    anchor, which keeps the integration in well-conditioned numbers. Latitude
    and longitude are only reconstructed on the way out, so that
    `MagneticGrid.sample_track` can be handed the result directly.

    The conversion uses the WGS84 radii of curvature at the anchor latitude and
    holds them fixed. Over 10 km that costs well under a metre; it is not
    suitable for a flight that crosses a continent.
    """

    lat0: float
    lon0: float
    alt0: float = 0.0

    def __post_init__(self):
        self._m_per_deg_lat, self._m_per_deg_lon = metres_per_degree(self.lat0)
        if self._m_per_deg_lon <= 0:
            raise ValueError(f"anchor latitude {self.lat0} is at or past a pole")

    def to_latlon(self, pos_ned):
        """(N, 3) metres north/east/down -> latitude, longitude, altitude."""
        pos_ned = np.atleast_2d(np.asarray(pos_ned, dtype=float))
        lat = self.lat0 + pos_ned[:, 0] / self._m_per_deg_lat
        lon = self.lon0 + pos_ned[:, 1] / self._m_per_deg_lon
        alt = self.alt0 - pos_ned[:, 2]      # down is positive, altitude is not
        return lat, lon, alt

    def from_latlon(self, lat, lon, alt_m):
        """Latitude, longitude, altitude -> (N, 3) metres north/east/down."""
        lat = np.atleast_1d(np.asarray(lat, dtype=float))
        lon = np.atleast_1d(np.asarray(lon, dtype=float))
        alt_m = np.atleast_1d(np.asarray(alt_m, dtype=float))
        return np.column_stack([
            (lat - self.lat0) * self._m_per_deg_lat,
            (lon - self.lon0) * self._m_per_deg_lon,
            self.alt0 - alt_m,
        ])


@dataclass
class BaroAiding:
    """Pulls the INS vertical channel towards a barometric altitude.

    A two-gain complementary filter, critically damped, set by one time
    constant. Altitude error feeds back on both the down-position and the
    down-velocity, which is enough to stop the vertical channel running away
    without touching the horizontal.

    `tau_s` is how quickly the INS altitude is pulled onto the barometer.
    Two seconds is slow enough that barometer noise does not get injected into
    the velocity, fast enough that accelerometer bias cannot build.
    """

    tau_s: float = 2.0

    @property
    def gains(self) -> tuple[float, float]:
        wn = 1.0 / self.tau_s
        return 2.0 * wn, wn**2        # on position, on velocity


def propagate(state: InsState, accel_b, gyro_b, dt, *,
              gravity_ned, earth_rate_ned=None,
              baro_alt_m=None, baro_gains=None, alt0=0.0) -> InsState:
    """One strapdown step. Pure: the input state is never modified.

        state         (N, ...) current position, velocity and attitude
        accel_b       (3,) or (N, 3) specific force, body FRD, m/s^2
        gyro_b        (3,) or (N, 3) angular rate, body FRD, rad/s
        dt            seconds, a scalar or (N,)
        gravity_ned   (3,) or (N, 3) gravity vector, NED, so z is positive
        earth_rate_ned  (3,) Earth rotation in NED, or None to ignore Coriolis
        baro_alt_m    scalar or (N,) aiding altitude, or None for a free
                      vertical channel
        baro_gains    (k_pos, k_vel) from BaroAiding.gains, required if
                      baro_alt_m is given
        alt0          the navigation frame's anchor altitude, needed to compare
                      the down-coordinate against an absolute baro altitude

    Readings are broadcast against the state, so one IMU sample can drive a
    whole particle cloud, or each particle can have its own perturbed reading.
    """
    n = state.n
    accel_b = np.broadcast_to(np.atleast_2d(np.asarray(accel_b, float)), (n, 3))
    gyro_b = np.broadcast_to(np.atleast_2d(np.asarray(gyro_b, float)), (n, 3))
    gravity_ned = np.broadcast_to(np.atleast_2d(np.asarray(gravity_ned, float)), (n, 3))
    dt_col = np.reshape(np.broadcast_to(np.asarray(dt, float), (n,)), (n, 1))

    # 1. Attitude. The increment is in the body frame, so it multiplies on the
    #    right of the current attitude.
    quat = rotations.normalise(
        rotations.multiply(state.quat, rotations.exp_map(gyro_b * dt_col))
    )

    # 2. Specific force into the navigation frame, using the attitude at the
    #    start of the step. Adding gravity turns specific force into true
    #    acceleration: level and still, the accelerometer reads (0, 0, -g),
    #    gravity is (0, 0, +g), and the sum is correctly zero.
    accel_ned = rotations.rotate(state.quat, accel_b) + gravity_ned

    # 3. Coriolis, if asked for. Earth's rotation deflects anything moving over
    #    the surface. Off by default: over 10-30 minutes with MEMS sensors the
    #    gyro bias swamps it, and leaving it out keeps the model easy to reason
    #    about. The synthetic IMU is built through the same flag, so truth and
    #    integration always agree about whether it exists.
    if earth_rate_ned is not None:
        omega = np.broadcast_to(np.atleast_2d(np.asarray(earth_rate_ned, float)), (n, 3))
        accel_ned = accel_ned - 2.0 * np.cross(omega, state.vel_ned)

    # 4. Velocity, then position from the new velocity.
    vel = state.vel_ned + accel_ned * dt_col
    pos = state.pos_ned + vel * dt_col

    # 5. Vertical aiding. Only the down channel is touched.
    if baro_alt_m is not None:
        if baro_gains is None:
            raise ValueError("baro_alt_m given without baro_gains")
        k_pos, k_vel = baro_gains
        baro = np.reshape(np.broadcast_to(np.asarray(baro_alt_m, float), (n,)), (n,))
        # Altitude the INS currently believes, in the same datum as the baro.
        alt_ins = alt0 - pos[:, 2]
        err = alt_ins - baro                      # positive: INS thinks it is too high
        dt_flat = dt_col[:, 0]
        pos[:, 2] += k_pos * err * dt_flat        # down grows, so altitude falls
        vel[:, 2] += k_vel * err * dt_flat

    return InsState(pos, vel, quat)


@dataclass
class Mechanisation:
    """A configured strapdown integrator: the navigation frame plus the options.

    Holding the configuration in one object means the forward integration and
    the synthetic-IMU generation cannot disagree about gravity, Earth rotation
    or the frame anchor - they are handed the same instance.
    """

    frame: NavFrame
    earth_rotation: bool = False
    baro: BaroAiding | None = field(default_factory=BaroAiding)

    @property
    def earth_rate_ned(self):
        """Earth's rotation vector in the local NED frame, or None if disabled.

        It points at the celestial pole: north-ish and upward, hence the
        negative down-component.
        """
        if not self.earth_rotation:
            return None
        lat = np.radians(self.frame.lat0)
        return EARTH_RATE * np.array([np.cos(lat), 0.0, -np.sin(lat)])

    def gravity_ned(self, pos_ned=None):
        """Gravity in the navigation frame, (3,). Down is positive."""
        alt = self.frame.alt0
        if pos_ned is not None:
            alt = self.frame.alt0 - float(np.mean(np.atleast_2d(pos_ned)[:, 2]))
        return np.array([0.0, 0.0, float(normal_gravity(self.frame.lat0, alt))])

    def initialise(self, lat, lon, alt_m, vel_ned, att_quat, n: int = 1) -> InsState:
        """Build a starting state, optionally replicated across N particles."""
        pos = self.frame.from_latlon(lat, lon, alt_m)
        state = InsState(np.tile(pos, (n, 1)),
                         np.tile(np.asarray(vel_ned, float).reshape(1, 3), (n, 1)),
                         np.tile(rotations.normalise(
                             np.asarray(att_quat, float).reshape(1, 4)), (n, 1)))
        return state

    def propagate(self, state, accel_b, gyro_b, dt, baro_alt_m=None) -> InsState:
        """One step, with this object's gravity, Earth-rotation and aiding settings."""
        return propagate(
            state, accel_b, gyro_b, dt,
            gravity_ned=self.gravity_ned(state.pos_ned),
            earth_rate_ned=self.earth_rate_ned,
            baro_alt_m=baro_alt_m if self.baro is not None else None,
            baro_gains=self.baro.gains if self.baro is not None else None,
            alt0=self.frame.alt0,
        )

    def integrate(self, imu, state, baro_alt_m=None) -> Trajectory:
        """Run a whole ImuSamples through, collecting the path.

        `baro_alt_m` is one altitude per IMU sample, or None. The returned
        trajectory carries latitude, longitude, altitude, velocity and attitude
        so it can be compared against truth or handed straight to the magnetic
        grid sampler.
        """
        n = len(imu)
        if state.n != 1:
            raise ValueError(f"integrate() tracks a single vehicle, got N={state.n}")

        pos = np.empty((n, 3))
        vel = np.empty((n, 3))
        quat = np.empty((n, 4))
        dt = imu.dt

        for i in range(n):
            # The first sample has no interval before it, so nothing is
            # integrated across it - it just records the initial state.
            if i > 0:
                aid = None if baro_alt_m is None else baro_alt_m[i]
                state = self.propagate(state, imu.accel_b[i - 1], imu.gyro_b[i - 1],
                                       dt[i], baro_alt_m=aid)
            pos[i] = state.pos_ned[0]
            vel[i] = state.vel_ned[0]
            quat[i] = state.quat[0]

        lat, lon, alt = self.frame.to_latlon(pos)
        return Trajectory(imu.t, lat, lon, alt, vel, quat, name="INS estimate")
