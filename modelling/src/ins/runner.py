"""Cut GNSS at a moment in time and see where the INS thinks it is afterwards.

    from ins import run_ins

    run = run_ins(imu, truth, t_cut=0.0, grade="consumer_mems")
    run.summary()
    run.error_at(600)      # metres off, ten minutes after the cut

This is the question the whole module exists to answer. Everything before
`t_cut` is assumed to be a normally navigating aircraft with GNSS; at `t_cut`
the INS is initialised from the truth and from then on it integrates its own
IMU and nothing else.

Why the initial error matters as much as the sensor grade
--------------------------------------------------------
An INS does not start from nothing - it starts from whatever the GNSS-aided
filter believed at the moment it lost its fix, including that filter's own
errors. A velocity error of 0.1 m/s is unremarkable at the point of a GNSS
cut, and on its own it puts the position 60 m out after ten minutes without
any sensor error at all. An initial attitude error does worse, because it
tilts gravity into the horizontal channels exactly as a gyro bias does.

So `initial_error` is not a detail. Running with the sensor grade alone
answers "how good is the IMU"; running with both answers the operational
question, which is "how far can we go after the fix drops out".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ins import rotations
from ins.errors import ErrorModel, ErrorRealisation, grade as lookup_grade
from ins.mechanisation import Mechanisation, NavFrame
from ins.trajectory import simulated_baro
from ins.types import ImuSamples, Trajectory


@dataclass
class InitialError:
    """What the navigation filter had wrong at the moment GNSS was lost.

    Deterministic offsets, not standard deviations - so a run is reproducible
    and a single effect can be isolated. All default to zero.

        pos_m      (3,) north, east, down error in metres
        vel_mps    (3,) north, east, down error in m/s
        att_deg    (3,) roll, pitch, yaw error in degrees, applied as a
                   misalignment of the navigation frame
    """

    pos_m: np.ndarray = field(default_factory=lambda: np.zeros(3))
    vel_mps: np.ndarray = field(default_factory=lambda: np.zeros(3))
    att_deg: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self):
        self.pos_m = np.asarray(self.pos_m, dtype=float).reshape(3)
        self.vel_mps = np.asarray(self.vel_mps, dtype=float).reshape(3)
        self.att_deg = np.asarray(self.att_deg, dtype=float).reshape(3)


@dataclass
class InsRun:
    """One free-running INS integration and how badly it went."""

    estimate: Trajectory
    truth: Trajectory
    error_m: np.ndarray            # horizontal error, one per estimate sample
    t_cut: float
    grade_name: str
    realisation: ErrorRealisation | None = None

    @property
    def elapsed_s(self) -> np.ndarray:
        """Seconds since the GNSS cut."""
        return self.estimate.t - self.t_cut

    def error_at(self, seconds_after_cut: float) -> float:
        """Horizontal error a given time after the cut, interpolated.

        NaN if the run is not that long, rather than silently returning the
        error at the end and understating the drift.
        """
        elapsed = self.elapsed_s
        if seconds_after_cut > elapsed[-1]:
            return float("nan")
        return float(np.interp(seconds_after_cut, elapsed, self.error_m))

    def summary(self) -> None:
        """Print the drift at the usual checkpoints."""
        print(f"{self.grade_name}, {self.elapsed_s[-1] / 60:.1f} min after the cut")
        for minutes in (1, 5, 10, 20):
            err = self.error_at(minutes * 60)
            shown = "   -   " if np.isnan(err) else f"{err:8.1f}"
            print(f"   {minutes:2d} min  {shown} m")

    def __repr__(self) -> str:
        return (f"<InsRun {self.grade_name!r} {self.elapsed_s[-1] / 60:.1f} min, "
                f"final error {self.error_m[-1]:.0f} m>")


def run_ins(imu: ImuSamples, truth: Trajectory, t_cut: float = 0.0,
            initial_error: InitialError | None = None,
            grade: str | ErrorModel | None = None,
            seed: int | None = 0,
            mech: Mechanisation | None = None,
            baro_aiding: bool = True,
            baro_seed: int | None = 0) -> InsRun:
    """Integrate the IMU forward from `t_cut` with no further GNSS.

        imu            the IMU samples, clean. Errors are added here, so the
                       same clean set can be reused for every grade.
        truth          where the vehicle really went, used to initialise at
                       `t_cut` and to score the result afterwards
        t_cut          when GNSS is lost, in the same seconds as imu.t
        initial_error  what the filter had wrong at the cut, or None for a
                       perfect start
        grade          a name from GRADES, an ErrorModel, or None for a
                       flawless IMU
        seed           seeds the sensor errors; the same seed gives the same
                       run, every time
        baro_aiding    hold the vertical channel to a simulated barometer.
                       On by default - see mechanisation.py for why the
                       vertical is aided and the horizontal is not.

    Returns an InsRun carrying the estimated trajectory and the horizontal
    error against truth.
    """
    if not (imu.t[0] <= t_cut <= imu.t[-1]):
        raise ValueError(
            f"t_cut={t_cut} is outside the IMU data, which spans "
            f"{imu.t[0]:.1f} to {imu.t[-1]:.1f} s"
        )

    # The mechanisation must be the one the IMU was derived with, or gravity
    # and Earth rotation will not match and the "perfect" case will not close.
    if mech is None:
        mech = getattr(truth, "mech", None)
    if mech is None:
        frame = NavFrame(float(truth.lat[0]), float(truth.lon[0]), float(truth.alt_m[0]))
        mech = Mechanisation(frame)
    if not baro_aiding:
        mech = Mechanisation(mech.frame, earth_rotation=mech.earth_rotation, baro=None)

    # 1. Only the IMU from the cut onwards is used.
    flight = imu.window(t_cut, imu.t[-1])

    # 2. Corrupt it, unless we are testing the integrator itself.
    model = None
    if grade is not None:
        model = lookup_grade(grade) if isinstance(grade, str) else grade
    realisation = None
    if model is not None:
        flight, realisation = model.corrupt(flight, seed=seed)

    # 3. Start from the truth at the cut, optionally wrong.
    start = truth.at([t_cut])
    if start.vel_ned is None or start.att_quat is None:
        raise ValueError(
            "truth must carry velocity and attitude to initialise an INS. "
            "A position-only track does not say which way the vehicle faced."
        )
    pos_ned = mech.frame.from_latlon(start.lat, start.lon, start.alt_m)[0]
    vel_ned = start.vel_ned[0]
    quat = start.att_quat[0]

    if initial_error is not None:
        pos_ned = pos_ned + initial_error.pos_m
        vel_ned = vel_ned + initial_error.vel_mps
        # A navigation-frame misalignment, so it multiplies on the left.
        misalignment = rotations.exp_map(np.radians(initial_error.att_deg))
        quat = rotations.normalise(rotations.multiply(misalignment, quat))[0]

    lat, lon, alt = mech.frame.to_latlon(pos_ned)
    state = mech.initialise(lat[0], lon[0], alt[0], vel_ned, quat)

    # 4. Barometer, if the vertical channel is being aided.
    baro = None
    if mech.baro is not None:
        baro = simulated_baro(truth.at(flight.t), seed=baro_seed)

    # 5. Integrate, then score.
    estimate = mech.integrate(flight, state, baro_alt_m=baro)
    error = estimate.horizontal_error_m(truth)

    name = model.name if model is not None else "perfect"
    return InsRun(estimate, truth, error, t_cut, name, realisation)
