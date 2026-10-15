"""Inertial navigation: how far the drone drifts once GNSS is gone.

    from ins import synthetic_flight, run_ins, GRADES

    truth, imu = synthetic_flight(lawnmower((52.45, -6.40), 1200, 600, 120),
                                  alt_m=60.0, speed_mps=8.0)
    result = run_ins(imu, truth, t_cut=0.0, grade="consumer_mems")
    result.error_m[-1]        # how far off it is at the end

See mechanisation.py for the frame conventions - they are fixed throughout and
getting one wrong is the classic way to waste a day on this.
"""

from ins.errors import GRADES, ErrorModel, ErrorRealisation, grade
from ins.loaders import load_ardupilot_bin, load_quspin_dynamic_csv
from ins.mechanisation import BaroAiding, Mechanisation, NavFrame, normal_gravity, propagate
from ins.runner import InitialError, InsRun, run_ins
from ins.trajectory import (
    inverse_propagate,
    lawnmower,
    straight_line,
    synthetic_flight,
    trajectory_from_waypoints,
)
from ins.types import ImuSamples, InsState, Trajectory

__all__ = [
    "GRADES", "ErrorModel", "ErrorRealisation", "grade",
    "Mechanisation", "NavFrame", "BaroAiding", "propagate", "normal_gravity",
    "Trajectory", "ImuSamples", "InsState",
    "synthetic_flight", "trajectory_from_waypoints", "inverse_propagate",
    "lawnmower", "straight_line",
    "run_ins", "InsRun", "InitialError",
    "load_ardupilot_bin", "load_quspin_dynamic_csv",
]
