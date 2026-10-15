"""Read IMU and position data out of real log files.

    from ins.loaders import load_ardupilot_bin

    imu, truth, baro = load_ardupilot_bin("flight.BIN")
    run = run_ins(imu, truth, t_cut=300.0)

Two formats so far. ArduPilot DataFlash is the one that matters, because it is
what the flight controller already records on every flight. The QuSpin dynamic
CSV is here because we have one and it carries an IMU, but see the warning on
that function - its axes are not established.

Unit conversions, all of them, once
-----------------------------------
ArduPilot DataFlash logs are already SI by the time pymavlink has applied the
format multipliers:

    IMU.AccX/Y/Z    m/s^2       no conversion
    IMU.GyrX/Y/Z    rad/s       no conversion
    GPS.Lat/Lng     degrees     no conversion (stored as 1e7 ints, scaled by the reader)
    GPS.Alt         metres      above mean sea level, NOT the ellipsoid
    ATT.Roll/Pitch/Yaw  degrees -> radians here
    BARO.Alt        metres      relative to where the vehicle was armed

The QuSpin export is not SI and needs real conversion:

    Accelerometer*  milli-g     -> m/s^2, multiply by 9.80665 / 1000
    Gyro*           deg/s       -> rad/s, multiply by pi / 180
    Height          metres      above the WGS84 ellipsoid

One thing these deliberately do NOT do
--------------------------------------
They do not guess at a missing stream. If a log has no ATT messages there is no
way to know which way the vehicle was pointing, so loading raises instead of
substituting level flight - an INS started from a wrong attitude produces a
confident, wrong answer, which is worse than no answer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ins import rotations
from ins.types import ImuSamples, Trajectory

# Body FRD is what this package uses everywhere. ArduPilot already logs in it.
IDENTITY_FRAME = np.eye(3)

G0 = 9.806_65
DEG = np.pi / 180.0


def load_ardupilot_bin(path, imu_instance: int = 0, require_att: bool = True):
    """Read an ArduPilot DataFlash .BIN log.

    Returns `(imu, truth, baro_alt_m)`:

        imu     ImuSamples on the IMU's own clock, body FRD, SI units
        truth   Trajectory from GPS position and ATT attitude, with velocity
                differentiated from position
        baro_alt_m  a callable-free array aligned to the IMU timestamps, or
                None if the log has no BARO messages

    `imu_instance` picks which IMU: a Pixhawk has two or three, and they do not
    agree. Instance 0 is the one the EKF normally uses.

    All streams are put on a single clock - seconds from the first IMU sample -
    taken from each message's TimeUS field.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")

    try:
        from pymavlink import DFReader
    except ImportError as exc:
        raise ImportError(
            "Reading ArduPilot logs needs pymavlink.\n"
            "Run:  pip install -e \"./modelling[dev]\"   (from the repo root)"
        ) from exc

    reader = (DFReader.DFReader_binary(str(path)) if path.suffix.lower() == ".bin"
              else DFReader.DFReader_text(str(path)))

    wanted = {"IMU", "GPS", "ATT", "BARO"}
    collected: dict[str, list] = {name: [] for name in wanted}
    while True:
        message = reader.recv_match(type=list(wanted))
        if message is None:
            break
        collected[message.get_type()].append(message)

    missing = [name for name in ("IMU", "GPS") if not collected[name]]
    if missing:
        raise ValueError(
            f"{path.name} has no {' or '.join(missing)} messages, so there is "
            f"nothing to integrate. Found: "
            f"{ {k: len(v) for k, v in collected.items()} }"
        )
    if require_att and not collected["ATT"]:
        raise ValueError(
            f"{path.name} has no ATT messages. Without a logged attitude there "
            f"is no way to know which way the vehicle faced at the GNSS cut, "
            f"and an INS started from the wrong attitude is confidently wrong. "
            f"Pass require_att=False only if you intend to supply an attitude "
            f"some other way."
        )

    # --- IMU, and the clock everything else is referred to -----------------
    imu_rows = collected["IMU"]
    # Newer logs carry the instance in field I; older ones use separate
    # message names and have no I field, in which case there is only one.
    if hasattr(imu_rows[0], "I"):
        picked = [m for m in imu_rows if int(m.I) == imu_instance]
        if not picked:
            available = sorted({int(m.I) for m in imu_rows})
            raise ValueError(
                f"no IMU instance {imu_instance} in {path.name}; "
                f"the log has instances {available}"
            )
        imu_rows = picked

    t_us = np.array([m.TimeUS for m in imu_rows], dtype=float)
    order = np.argsort(t_us)
    t_us = t_us[order]
    t0 = t_us[0]

    accel = np.array([[m.AccX, m.AccY, m.AccZ] for m in imu_rows], dtype=float)[order]
    gyro = np.array([[m.GyrX, m.GyrY, m.GyrZ] for m in imu_rows], dtype=float)[order]

    # Duplicate timestamps happen when a log is stitched or a message repeats.
    # ImuSamples insists on a strictly increasing clock, so drop them here
    # where we can say what happened.
    keep = np.concatenate([[True], np.diff(t_us) > 0])
    if not keep.all():
        dropped = int((~keep).sum())
        print(f"  note: dropped {dropped} IMU samples with repeated timestamps")

    imu_t = (t_us[keep] - t0) / 1e6
    imu = ImuSamples(imu_t, accel[keep], gyro[keep],
                     name=f"{path.name} IMU{imu_instance}", source=path)

    # --- Truth: GPS position, ATT attitude ---------------------------------
    gps = collected["GPS"]
    gps_t = (np.array([m.TimeUS for m in gps], dtype=float) - t0) / 1e6
    gps_order = np.argsort(gps_t)
    gps_t = gps_t[gps_order]
    lat = np.array([m.Lat for m in gps], dtype=float)[gps_order]
    lon = np.array([m.Lng for m in gps], dtype=float)[gps_order]
    alt = np.array([m.Alt for m in gps], dtype=float)[gps_order]

    # ArduPilot logs GPS rows before a fix with latitude and longitude at
    # exactly zero. Those are not positions, so they become NaN rather than
    # dragging the track to the Gulf of Guinea.
    no_fix = (lat == 0.0) & (lon == 0.0)
    lat[no_fix] = np.nan
    lon[no_fix] = np.nan
    alt[no_fix] = np.nan
    good = ~no_fix
    if not good.any():
        raise ValueError(f"{path.name} never got a GPS fix")
    gps_t, lat, lon, alt = gps_t[good], lat[good], lon[good], alt[good]

    att_quat = None
    if collected["ATT"]:
        att = collected["ATT"]
        att_t = (np.array([m.TimeUS for m in att], dtype=float) - t0) / 1e6
        att_order = np.argsort(att_t)
        att_t = att_t[att_order]
        roll = np.array([m.Roll for m in att], dtype=float)[att_order] * DEG
        pitch = np.array([m.Pitch for m in att], dtype=float)[att_order] * DEG
        yaw = np.array([m.Yaw for m in att], dtype=float)[att_order] * DEG
        # onto the GPS clock, which is what the truth trajectory runs on
        att_quat = rotations.from_euler(
            np.interp(gps_t, att_t, roll),
            np.interp(gps_t, att_t, pitch),
            # Yaw wraps at 360, so interpolate it as a direction rather than a
            # number - otherwise a crossing produces a spurious half-turn.
            np.arctan2(np.interp(gps_t, att_t, np.sin(yaw)),
                       np.interp(gps_t, att_t, np.cos(yaw))),
        )

    vel_ned = _velocity_from_positions(gps_t, lat, lon, alt)
    truth = Trajectory(gps_t, lat, lon, alt, vel_ned, att_quat,
                       name=f"{path.name} GPS", source=path)

    # --- Barometer, onto the IMU clock -------------------------------------
    baro_alt = None
    if collected["BARO"]:
        baro = collected["BARO"]
        baro_t = (np.array([m.TimeUS for m in baro], dtype=float) - t0) / 1e6
        baro_order = np.argsort(baro_t)
        # BARO.Alt is relative to the arming point, while GPS.Alt is above mean
        # sea level. Shifting the baro onto the GPS datum at the first fix puts
        # them in the same frame, which is what the vertical aiding needs.
        raw = np.array([m.Alt for m in baro], dtype=float)[baro_order]
        baro_t = baro_t[baro_order]
        offset = float(alt[0] - np.interp(gps_t[0], baro_t, raw))
        baro_alt = np.interp(imu.t, baro_t, raw) + offset

    return imu, truth, baro_alt


def load_quspin_dynamic_csv(path, body_from_sensor=None, imu_rate_hz=None):
    """Read the IMU out of a QuSpin dynamic CSV export.

    Returns `(imu, track)`. The track carries position only - no attitude - so
    it cannot initialise an INS on its own; it is there to say where the
    aircraft was.

    **The sensor axes are not established.** `body_from_sensor` is a 3x3 matrix
    taking the CSV's accelerometer and gyro axes into body FRD, and it defaults
    to the identity, which is a guess, not a finding. The QuSpin IMU sits in
    the magnetometer housing at whatever orientation it was mounted, and the
    export does not record it. Before trusting any INS result from this file,
    work the mapping out from a known manoeuvre - level flight should put
    about -9.81 m/s^2 on the body z accelerometer and near zero on x and y -
    and pass the matrix in explicitly.

    `imu_rate_hz` resamples onto a uniform clock. The native rate is 250 Hz,
    and the raw timestamps are usable as they are, so the default is None
    (leave them alone).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")

    import pandas as pd

    frame = IDENTITY_FRAME if body_from_sensor is None else np.asarray(
        body_from_sensor, dtype=float).reshape(3, 3)

    needed = ["Logger_Time (us) Original", "AccelerometerX", "AccelerometerY",
              "AccelerometerZ", "GyroX", "GyroY", "GyroZ",
              "Latitude", "Longitude", "Height"]
    df = pd.read_csv(path)
    absent = [c for c in needed if c not in df.columns]
    if absent:
        raise ValueError(
            f"{path.name} is missing columns {absent}. This loader expects the "
            f"'flat_interpolation' dynamic export; see notebooks/dynamic.ipynb "
            f"for what that file looks like."
        )

    t = df["Logger_Time (us) Original"].to_numpy(dtype=float) / 1e6
    t = t - t[0]

    # milli-g -> m/s^2. 1000 mg is one standard gravity.
    accel_sensor = df[["AccelerometerX", "AccelerometerY",
                       "AccelerometerZ"]].to_numpy(dtype=float) * (G0 / 1000.0)
    # deg/s -> rad/s
    gyro_sensor = df[["GyroX", "GyroY", "GyroZ"]].to_numpy(dtype=float) * DEG

    # Into body FRD, by whatever mapping the caller supplied.
    accel_b = accel_sensor @ frame.T
    gyro_b = gyro_sensor @ frame.T

    keep = np.concatenate([[True], np.diff(t) > 0])
    imu = ImuSamples(t[keep], accel_b[keep], gyro_b[keep],
                     name=f"{path.name} IMU", source=path)

    if imu_rate_hz is not None:
        uniform = np.arange(imu.t[0], imu.t[-1], 1.0 / imu_rate_hz)
        imu = ImuSamples(
            uniform,
            np.column_stack([np.interp(uniform, imu.t, imu.accel_b[:, i]) for i in range(3)]),
            np.column_stack([np.interp(uniform, imu.t, imu.gyro_b[:, i]) for i in range(3)]),
            name=imu.name + f" @{imu_rate_hz:g}Hz", source=path)

    lat = df["Latitude"].to_numpy(dtype=float)[keep]
    lon = df["Longitude"].to_numpy(dtype=float)[keep]
    alt = df["Height"].to_numpy(dtype=float)[keep]
    vel = _velocity_from_positions(t[keep], lat, lon, alt)
    track = Trajectory(t[keep], lat, lon, alt, vel, None,
                       name=f"{path.name} track", source=path)

    return imu, track


def _velocity_from_positions(t, lat, lon, alt) -> np.ndarray:
    """Differentiate a position track into an NED velocity.

    Uses the same convention as the mechanisation - `vel[k]` is the velocity
    that carries the vehicle from sample k-1 to k - so a trajectory built here
    can initialise an INS without a step-one mismatch.

    A GPS track differentiated like this is noisy. It is good enough to
    initialise from and to score against; it is not a velocity reference.
    """
    from ins.types import metres_per_degree

    m_per_deg_lat, m_per_deg_lon = metres_per_degree(float(np.nanmean(lat)))
    north = lat * m_per_deg_lat
    east = lon * m_per_deg_lon
    down = -alt

    dt = np.diff(t)
    vel = np.empty((len(t), 3))
    vel[1:] = np.column_stack([np.diff(north) / dt, np.diff(east) / dt,
                               np.diff(down) / dt])
    vel[0] = vel[1] if len(t) > 1 else 0.0
    return vel
