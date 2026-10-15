"""Build a flight path, then work out what a perfect IMU would have measured on it.

    from ins import lawnmower, synthetic_flight

    waypoints = lawnmower((52.45, -6.40), length_m=1200, width_m=600, spacing_m=120)
    truth, imu = synthetic_flight(waypoints, alt_m=60.0, speed_mps=8.0)

`truth` is where the drone really went. `imu` is what a flawless accelerometer
and gyro would have read along that path. Feed `imu` straight back into the
mechanisation and the path comes back out to within a few centimetres over
twenty minutes; add errors from errors.py first and the difference is the
drift we are trying to measure.

How the path is made smooth
---------------------------
Waypoints give a polyline with sharp corners. A drone cannot fly a sharp
corner, and more importantly an infinite acceleration at the corner would make
the derived IMU readings meaningless.

So the polyline is walked at constant speed to get evenly spaced samples, then
the position series is smoothed with a Gaussian kernel whose width is set by
how long the turn should take. That rounds every corner by the same amount,
and because it acts on position, velocity and acceleration come out smooth for
free. A real multirotor slows slightly in a turn; so does this one, as a
consequence of the smoothing rather than by being told to.

How the attitude is worked out
------------------------------
A multirotor steers by tilting. Its rotors only push along the body's own
up-axis, so to accelerate sideways it must lean into the acceleration. That
makes the attitude a consequence of the path, not a free choice:

  - the body up-axis must point along the total specific force, which is
    gravity plus whatever acceleration the path demands
  - the nose points along the track, which fixes the remaining freedom

Roll and pitch therefore fall out of lateral and longitudinal acceleration
automatically, and yaw follows the direction of travel. There is no separate
attitude model to keep in step with the position model.

One thing this deliberately does NOT do
---------------------------------------
It does not pretend to be a flight dynamics model. There is no rotor, no mass,
no wind and no controller. It produces a kinematically consistent path with
plausible attitudes, which is all the INS needs - the INS only ever sees
specific force and angular rate.
"""

from __future__ import annotations

import numpy as np

from ins import rotations
from ins.mechanisation import Mechanisation, NavFrame
from ins.types import ImuSamples, Trajectory, metres_per_degree


def straight_line(start, end):
    """Two waypoints: `[(lat, lon), (lat, lon)]`. Here for symmetry with lawnmower."""
    return [tuple(start), tuple(end)]


def lawnmower(origin, length_m: float, width_m: float, spacing_m: float,
              heading_deg: float = 0.0):
    """Waypoints for a survey pattern: long parallel lines joined at the ends.

        origin       (lat, lon) of the first corner
        length_m     how long each line is
        width_m      how far the pattern extends sideways
        spacing_m    gap between adjacent lines
        heading_deg  direction of the long lines, clockwise from north

    Returns a list of (lat, lon). This is the pattern the real QuSpin flights
    flew, and the one a magnetic survey would use, because it covers ground
    evenly and crosses its own track often enough to check repeatability.
    """
    if spacing_m <= 0 or length_m <= 0 or width_m <= 0:
        raise ValueError("length, width and spacing must all be positive")

    lat0, lon0 = origin
    m_per_deg_lat, m_per_deg_lon = metres_per_degree(lat0)

    # Along-line and across-line unit vectors, in metres north/east.
    heading = np.radians(heading_deg)
    along = np.array([np.cos(heading), np.sin(heading)])
    across = np.array([-np.sin(heading), np.cos(heading)])

    n_lines = max(2, int(round(width_m / spacing_m)) + 1)
    points = []
    for i in range(n_lines):
        offset = across * (i * spacing_m)
        # Alternate direction so the drone flies a continuous path rather than
        # teleporting back to the same end of every line.
        ends = [0.0, length_m] if i % 2 == 0 else [length_m, 0.0]
        for distance in ends:
            north, east = offset + along * distance
            points.append((lat0 + north / m_per_deg_lat,
                           lon0 + east / m_per_deg_lon))
    return points


def trajectory_from_waypoints(waypoints, alt_m: float, speed_mps: float,
                              rate_hz: float = 100.0, turn_time_s: float = 4.0,
                              climb_profile=None, name="truth") -> Trajectory:
    """Turn waypoints into a smooth, flyable trajectory sampled at `rate_hz`.

        turn_time_s    roughly how long a corner takes. Sets the smoothing
                       width, so it also sets the turn radius: at 8 m/s a
                       4 s turn is a radius of about 30 m, which is what a
                       survey multirotor actually flies.
        climb_profile  optional callable t -> altitude in metres, for a flight
                       that does not hold one height. Default is level.

    The velocity stored on the result uses the same finite-difference
    convention as the integrator, so initialising an INS from this trajectory
    introduces no step-one mismatch.
    """
    waypoints = [(float(a), float(b)) for a, b in waypoints]
    if len(waypoints) < 2:
        raise ValueError("need at least two waypoints")
    if speed_mps <= 0 or rate_hz <= 0:
        raise ValueError("speed and rate must be positive")

    lat0, lon0 = waypoints[0]
    frame = NavFrame(lat0, lon0, alt_m)
    corners = np.array([frame.from_latlon(lat, lon, alt_m)[0, :2]
                        for lat, lon in waypoints])

    # 1. Walk the polyline at constant speed. Cumulative distance along the
    #    corners becomes the lookup for evenly spaced samples.
    leg_lengths = np.linalg.norm(np.diff(corners, axis=0), axis=1)
    if np.any(leg_lengths == 0):
        raise ValueError("two consecutive waypoints are identical")
    distance_at_corner = np.concatenate([[0.0], np.cumsum(leg_lengths)])
    total = distance_at_corner[-1]

    dt = 1.0 / rate_hz
    n = int(np.floor(total / (speed_mps * dt))) + 1
    if n < 10:
        raise ValueError(f"path is only {total:.0f} m - too short to sample")
    travelled = np.arange(n) * speed_mps * dt

    north = np.interp(travelled, distance_at_corner, corners[:, 0])
    east = np.interp(travelled, distance_at_corner, corners[:, 1])

    # 2. Round the corners. A Gaussian kernel of this width turns each sharp
    #    vertex into a smooth arc taking roughly turn_time_s to fly.
    sigma_samples = max(1.0, turn_time_s * rate_hz / 6.0)
    north = _gaussian_smooth(north, sigma_samples)
    east = _gaussian_smooth(east, sigma_samples)

    t = np.arange(n) * dt
    if climb_profile is None:
        alt = np.full(n, alt_m)
    else:
        alt = np.asarray([float(climb_profile(ti)) for ti in t])
        alt = _gaussian_smooth(alt, sigma_samples)

    down = frame.alt0 - alt
    pos_ned = np.column_stack([north, east, down])

    # 3. Velocity, matching the integrator's convention exactly:
    #    vel[k] is the velocity used to step from k-1 to k.
    vel_ned = np.empty_like(pos_ned)
    vel_ned[1:] = np.diff(pos_ned, axis=0) / dt
    vel_ned[0] = vel_ned[1]

    # 4. Attitude from the acceleration the path demands.
    accel_ned = np.empty_like(vel_ned)
    accel_ned[:-1] = np.diff(vel_ned, axis=0) / dt
    accel_ned[-1] = accel_ned[-2]
    gravity = Mechanisation(frame).gravity_ned()
    att_quat = _attitude_for_path(accel_ned, vel_ned, gravity)

    lat, lon, alt_out = frame.to_latlon(pos_ned)
    traj = Trajectory(t, lat, lon, alt_out, vel_ned, att_quat, name=name)
    traj.frame = frame          # kept so the IMU derivation reuses the same anchor
    return traj


def _gaussian_smooth(x: np.ndarray, sigma_samples: float) -> np.ndarray:
    """Smooth a 1-D series without distorting either end.

    The kernel is truncated at three sigma. The ends need padding, and the
    choice of padding matters more than it looks: a plain mirror reflects the
    path back on itself, which makes the smoothed drone decelerate into the
    first sample and accelerate out of the last one. The flight then starts
    from a near-standstill, the initial heading is undefined, and the derived
    gyro reading at sample zero is enormous.

    Odd reflection - mirroring through the end *point* rather than across it -
    continues the first and last legs straight on at constant speed. Since
    those legs are straight anyway, the padding is exact and the endpoints come
    out untouched.
    """
    half = int(np.ceil(3 * sigma_samples))
    offsets = np.arange(-half, half + 1)
    kernel = np.exp(-0.5 * (offsets / sigma_samples) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(x, half, mode="reflect", reflect_type="odd")
    return np.convolve(padded, kernel, mode="same")[half:half + len(x)]


def _attitude_for_path(accel_ned, vel_ned, gravity_ned) -> np.ndarray:
    """The attitude a multirotor must hold to fly this acceleration.

    Specific force is what the rotors have to produce: the acceleration the
    path needs, minus gravity. The body's up-axis lines up with it, and the
    nose points along the track. Together those fix the attitude completely.
    """
    # Specific force in NED. Hovering, this is (0, 0, -g): straight up.
    specific_force = accel_ned - gravity_ned
    magnitude = np.linalg.norm(specific_force, axis=1, keepdims=True)
    if np.any(magnitude < 1e-6):
        raise ValueError("path demands free-fall; no attitude can fly it")

    # Body z (down) is opposite the specific force.
    z_body = -specific_force / magnitude

    # Nose along the track. Where the drone is barely moving the track
    # direction is meaningless, so hold the last usable heading instead of
    # letting the nose spin on numerical noise.
    speed = np.linalg.norm(vel_ned[:, :2], axis=1)
    heading = np.arctan2(vel_ned[:, 1], vel_ned[:, 0])
    usable = speed > 0.1
    if not usable.any():
        heading = np.zeros_like(heading)
    else:
        heading = np.interp(np.arange(len(heading)),
                            np.flatnonzero(usable), heading[usable])
    nose = np.column_stack([np.cos(heading), np.sin(heading), np.zeros_like(heading)])

    # Right axis is perpendicular to both; then forward is made exactly
    # perpendicular to the other two rather than assumed to be.
    y_body = np.cross(z_body, nose)
    y_norm = np.linalg.norm(y_body, axis=1, keepdims=True)
    y_body = y_body / np.where(y_norm < 1e-9, 1.0, y_norm)
    x_body = np.cross(y_body, z_body)

    # Columns of a body->NED matrix are the body axes expressed in NED.
    matrix = np.stack([x_body, y_body, z_body], axis=-1)
    return rotations.from_matrix(matrix)


def inverse_propagate(truth: Trajectory, mech: Mechanisation) -> ImuSamples:
    """What a perfect IMU would have measured along a known trajectory.

    This runs the mechanisation's four lines backwards. Because that inversion
    is exact rather than approximate, integrating the result reproduces the
    input path to numerical precision - which is what makes the perfect-IMU
    test in tests/test_ins.py a real check rather than a tolerance fudge.

    The trajectory must carry velocity and attitude; a position-only track
    (a GNSS log, say) does not contain enough to derive angular rates.
    """
    if truth.vel_ned is None or truth.att_quat is None:
        raise ValueError(
            "inverse_propagate needs velocity and attitude. A position-only "
            "trajectory cannot determine what the gyros read."
        )

    n = len(truth)
    if n < 3:
        raise ValueError(f"need at least 3 samples, got {n}")

    dt = np.diff(truth.t)
    pos_ned = mech.frame.from_latlon(truth.lat, truth.lon, truth.alt_m)
    vel = truth.vel_ned
    quat = truth.att_quat

    # Acceleration over each step, from the velocity either side of it.
    accel_ned = np.empty((n, 3))
    accel_ned[:-1] = np.diff(vel, axis=0) / dt[:, None]
    # Nothing is integrated across the final sample, so its reading is never
    # used. Repeat the previous one rather than leave a NaN that would
    # propagate if someone did use it.
    accel_ned[-1] = accel_ned[-2]

    # Undo gravity and, if it is switched on, Coriolis - using the same
    # configuration object the forward pass will use, so the two cannot
    # disagree about whether Earth rotates.
    specific_force_ned = accel_ned - mech.gravity_ned(pos_ned)
    earth_rate = mech.earth_rate_ned
    if earth_rate is not None:
        specific_force_ned = specific_force_ned + 2.0 * np.cross(
            np.broadcast_to(earth_rate, (n, 3)), vel)

    accel_b = rotations.rotate_inverse(quat, specific_force_ned)

    # Angular rate from the rotation between consecutive attitudes. The
    # increment is expressed in the body frame, hence conjugate-on-the-left.
    gyro_b = np.empty((n, 3))
    increment = rotations.multiply(rotations.conjugate(quat[:-1]), quat[1:])
    gyro_b[:-1] = rotations.log_map(increment) / dt[:, None]
    gyro_b[-1] = gyro_b[-2]

    return ImuSamples(truth.t, accel_b, gyro_b, name="perfect IMU")


def synthetic_flight(waypoints, alt_m: float, speed_mps: float,
                     rate_hz: float = 100.0, turn_time_s: float = 4.0,
                     mech: Mechanisation | None = None, **kwargs):
    """Build a path and its perfect IMU in one call.

    Returns `(truth, imu)`. Pass a `Mechanisation` to control gravity, Earth
    rotation and the frame anchor; otherwise one is made with the defaults and
    attached to the truth trajectory as `truth.mech` so the runner can reuse
    exactly the same settings.
    """
    truth = trajectory_from_waypoints(waypoints, alt_m, speed_mps,
                                      rate_hz=rate_hz, turn_time_s=turn_time_s,
                                      **kwargs)
    if mech is None:
        mech = Mechanisation(truth.frame)
    imu = inverse_propagate(truth, mech)
    truth.mech = mech
    return truth, imu


def simulated_baro(truth: Trajectory, seed: int | None = None,
                   noise_m: float = 0.2, drift_m_per_min: float = 0.5) -> np.ndarray:
    """A barometer reading for each sample of a trajectory.

    Truth plus white noise plus a slow drift. The drift is what a real
    barometer does as the weather moves - it is why baro altitude is trusted
    for short-term changes and not as an absolute height.

    Default 0.2 m of noise and 0.5 m per minute of wander are typical of a
    small MEMS barometer indoors-to-outdoors; both are arguments because both
    depend heavily on the day.
    """
    rng = np.random.default_rng(seed)
    minutes = (truth.t - truth.t[0]) / 60.0
    # One smooth wander for the whole flight, not a fresh random number each
    # sample - a barometer drifts, it does not jump.
    drift = drift_m_per_min * minutes * rng.normal(0.0, 1.0)
    return truth.alt_m + drift + rng.normal(0.0, noise_m, len(truth))
