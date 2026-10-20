"""What is wrong with a real IMU, and how to put those faults into a perfect one.

    from ins import GRADES

    model = GRADES["consumer_mems"]
    dirty, realised = model.corrupt(perfect_imu, seed=0)

Four error sources per axis, for both the accelerometers and the gyros:

  turn-on bias       a constant offset, different every time the thing is
                     switched on. Drawn once per run.
  bias random walk   that offset wandering slowly during the flight, which is
                     what "bias instability" means.
  white noise        sample-to-sample noise. Quoted as a spectral density, so
                     the per-sample size depends on the sample rate.
  scale factor       readings proportional to the truth but a fraction out.
                     Drawn once per run, optional.

Why the bias dominates
----------------------
White noise integrates away: averaged over a long flight it mostly cancels,
leaving an error that grows as the square root of time. A bias does not
cancel. An accelerometer bias `b` integrates straight into a position error of
`0.5 * b * t^2`, and a gyro bias tilts the whole navigation frame so that
gravity leaks into the horizontal channels, giving an error that grows as
`t^3`. Those two terms are the entire story of why an unaided INS drifts, and
the tests in tests/test_ins.py check both exponents directly.

A note on the numbers below
---------------------------
Where a figure is on a manufacturer's datasheet it is cited. Where it is not -
and bias instability usually is not - it is marked as an assumption. Treat
every preset as a starting point to tune against a real log, not as a
specification. They are deliberately not tuned to flatter the result.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ins.types import ImuSamples

# Unit conversions, written once, used below with the source units in comments.
G0 = 9.806_65                 # standard gravity, m/s^2
DEG = np.pi / 180.0           # degrees -> radians
DEG_PER_HOUR = DEG / 3600.0   # deg/hr -> rad/s
# Angle random walk: deg/sqrt(hr) -> rad/s/sqrt(Hz). An ARW of N deg/sqrt(hr)
# is a gyro white-noise density of N * (pi/180) / 60 rad/s/sqrt(Hz), because
# one sqrt(hour) is 60 sqrt(seconds).
DEG_PER_SQRT_HOUR = DEG / 60.0


@dataclass
class ErrorRealisation:
    """The specific errors drawn for one run.

    Kept and returned so a result can be explained after the fact: if a run
    drifts unusually far, this says whether it got an unlucky bias draw.
    Construct one by hand to force an exact error, which is what the tests do.
    """

    accel_bias: np.ndarray            # m/s^2, (3,)
    gyro_bias: np.ndarray             # rad/s, (3,)
    accel_scale: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gyro_scale: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self):
        self.accel_bias = np.asarray(self.accel_bias, dtype=float).reshape(3)
        self.gyro_bias = np.asarray(self.gyro_bias, dtype=float).reshape(3)
        self.accel_scale = np.asarray(self.accel_scale, dtype=float).reshape(3)
        self.gyro_scale = np.asarray(self.gyro_scale, dtype=float).reshape(3)

    def __repr__(self) -> str:
        return (f"<ErrorRealisation accel_bias={np.round(self.accel_bias, 4)} m/s^2, "
                f"gyro_bias={np.round(self.gyro_bias / DEG, 4)} deg/s>")


@dataclass
class ErrorModel:
    """An IMU grade: the size of each error source, per axis.

    All fields are in SI and are standard deviations, not maxima. Any of them
    can be zero, which switches that error source off - useful for isolating
    one effect at a time.
    """

    accel_noise_density: float = 0.0      # m/s^2 / sqrt(Hz)
    gyro_noise_density: float = 0.0       # rad/s / sqrt(Hz)
    accel_turnon_bias: float = 0.0        # m/s^2
    gyro_turnon_bias: float = 0.0         # rad/s
    accel_bias_rw: float = 0.0            # m/s^2 / sqrt(s)
    gyro_bias_rw: float = 0.0             # rad/s / sqrt(s)
    accel_scale: float = 0.0              # fraction, e.g. 0.001 = 0.1%
    gyro_scale: float = 0.0               # fraction
    name: str = ""
    note: str = ""

    @classmethod
    def perfect(cls) -> "ErrorModel":
        """No errors at all. The baseline the integrator is tested against."""
        return cls(name="perfect")

    @classmethod
    def constant_bias(cls, accel=0.0, gyro=0.0) -> "ErrorModel":
        """A noiseless IMU with one exact, repeatable bias on every axis.

        Not a real sensor - a diagnostic. The analytic drift laws
        (0.5*b*t^2 and the gyro t^3) only hold when nothing else is going on,
        so the tests that check those exponents use this.
        """
        model = cls(name="constant_bias")
        model._fixed = (float(accel), float(gyro))
        return model

    def realise(self, rng: np.random.Generator) -> ErrorRealisation:
        """Draw the once-per-run errors: turn-on bias and scale factor."""
        fixed = getattr(self, "_fixed", None)
        if fixed is not None:
            accel, gyro = fixed
            return ErrorRealisation(np.full(3, accel), np.full(3, gyro))
        return ErrorRealisation(
            accel_bias=rng.normal(0.0, self.accel_turnon_bias, 3),
            gyro_bias=rng.normal(0.0, self.gyro_turnon_bias, 3),
            accel_scale=rng.normal(0.0, self.accel_scale, 3),
            gyro_scale=rng.normal(0.0, self.gyro_scale, 3),
        )

    def corrupt(self, imu: ImuSamples, seed: int | None = None,
                realisation: ErrorRealisation | None = None):
        """Add this grade's errors to a clean set of IMU samples.

        Returns `(corrupted, realisation)`. The same seed always gives the
        same output, so a drift figure can be reproduced exactly.

        Pass `realisation` to reuse errors drawn earlier - that is how the same
        physical IMU can be run against two different trajectories.
        """
        rng = np.random.default_rng(seed)
        if realisation is None:
            realisation = self.realise(rng)

        n = len(imu)
        dt = imu.dt
        dt_col = dt.reshape(-1, 1)

        # White noise. A spectral density is per root-hertz, so the standard
        # deviation of one sample is density / sqrt(dt) - sampling faster gives
        # noisier individual samples, and the same noise after averaging.
        accel_white = rng.normal(0.0, 1.0, (n, 3)) * self.accel_noise_density / np.sqrt(dt_col)
        gyro_white = rng.normal(0.0, 1.0, (n, 3)) * self.gyro_noise_density / np.sqrt(dt_col)

        # Bias random walk: each step adds a small independent nudge, and the
        # nudges accumulate. Scaling by sqrt(dt) makes the total wander depend
        # on elapsed time rather than on how often we sampled.
        accel_walk = np.cumsum(
            rng.normal(0.0, 1.0, (n, 3)) * self.accel_bias_rw * np.sqrt(dt_col), axis=0)
        gyro_walk = np.cumsum(
            rng.normal(0.0, 1.0, (n, 3)) * self.gyro_bias_rw * np.sqrt(dt_col), axis=0)

        accel = (imu.accel_b * (1.0 + realisation.accel_scale)
                 + realisation.accel_bias + accel_walk + accel_white)
        gyro = (imu.gyro_b * (1.0 + realisation.gyro_scale)
                + realisation.gyro_bias + gyro_walk + gyro_white)

        label = f"{imu.name} + {self.name}" if imu.name else self.name
        return ImuSamples(imu.t, accel, gyro, name=label, source=imu.source), realisation

    def __repr__(self) -> str:
        return (f"<ErrorModel {self.name!r} "
                f"accel_bias={self.accel_turnon_bias * 1e3 / G0:.1f} mg, "
                f"gyro_bias={self.gyro_turnon_bias / DEG_PER_HOUR:.2f} deg/hr>")


# ----------------------------------------------------------------------
# Presets
#
# Three grades, spanning the range that matters for this project: what is
# already on the drone, what could be afforded, and what the problem would
# look like with no sensor limit at all.
#
# Every value is a standard deviation. Read the source note on each before
# quoting any of these numbers anywhere.
# ----------------------------------------------------------------------

GRADES: dict[str, ErrorModel] = {

    # TDK InvenSense ICM-42688-P, the IMU on a Pixhawk 6C and most current
    # flight controllers.
    #
    # SOURCED from the TDK datasheet (DS-000347) and product page:
    #   gyro noise  2.8 mdps/sqrt(Hz)
    #   accel noise 70 ug/sqrt(Hz) on Z, 65 on X and Y - 70 used for all three
    #
    # ASSUMED, not on any datasheet. Bias instability is not specified for this
    # part, and the quoted offset tolerance (tens of mg over temperature) is a
    # worst case before calibration, not the residual after a flight
    # controller has calibrated it. These are the residuals we would expect
    # post-calibration, and they are the first thing to tune against a real
    # log:
    #   turn-on bias       0.02 m/s^2 (~2 mg) accel, 0.05 deg/s gyro
    #   bias random walk   1e-4 m/s^2/sqrt(s), 3e-4 deg/s/sqrt(s)
    #   scale factor       0.1%
    "consumer_mems": ErrorModel(
        accel_noise_density=70e-6 * G0,            # 70 ug/sqrt(Hz) -> m/s^2/sqrt(Hz)
        gyro_noise_density=0.0028 * DEG,           # 2.8 mdps/sqrt(Hz) -> rad/s/sqrt(Hz)
        accel_turnon_bias=0.02,                    # m/s^2, ASSUMED
        gyro_turnon_bias=0.05 * DEG,               # 0.05 deg/s, ASSUMED
        accel_bias_rw=1e-4,                        # m/s^2/sqrt(s), ASSUMED
        gyro_bias_rw=3e-4 * DEG,                   # deg/s/sqrt(s), ASSUMED
        accel_scale=0.001,                         # 0.1%, ASSUMED
        gyro_scale=0.001,
        name="consumer_mems",
        note="ICM-42688-P class. Noise from the TDK datasheet; biases assumed.",
    ),

    # Our own X500's Pixhawk 6C, IMU0, MEASURED from a 37 min static log
    # (hardware/flight_logs/2026-10-15_static_imu_test.bin.zip, 15 Oct 2026,
    # indoors, no GPS). Analysis in notebooks/imu_static.ipynb. Allan figures
    # are the RMS over the three axes of the steady part (after the board
    # heater reached 45 degC, 4 min in).
    #
    # MEASURED, Allan deviation:
    #   gyro noise   0.0055 / 0.0141 / 0.0040 deg/s/rtHz (x/y/z) - 2-5x the
    #                datasheet's 2.8 mdps/rtHz, worst on y
    #   accel noise  97 / 90 / 256 ug/rtHz - worst on z
    #   gyro bias RW 0.00024 / 0.00062 / 0.00016 deg/s/rts
    #   accel bias RW 7 (y) and 23 (z) ug/rts; x never turned up in 33 min
    # MEASURED, residual gyro bias after the EKF has learnt it: raw gyro mean
    #   minus the EKF's XKF1 estimate, under 0.008 deg/s on every axis over
    #   three windows. That includes the Earth's rotation (up to 0.004 deg/s
    #   here), so it is an upper bound, and it is ONE power-up, not a spread.
    #   The raw bias it learnt was 0.26 / -0.72 / 0.22 deg/s - gyros are
    #   calibrated at boot (24.5 degC) and the heater then takes them to 45,
    #   with INS_TCAL off. A GNSS cut before the EKF has converged sees that.
    # EKF's own estimate: accel bias 0.02 m/s^2 on z, matching the 0.024 m/s^2
    #   shortfall of |accel| against local gravity.
    # ASSUMED, not measurable statically: scale factors, kept from above.
    "x500_measured": ErrorModel(
        accel_noise_density=166e-6 * G0,           # MEASURED, RMS of 97/90/256 ug/rtHz
        gyro_noise_density=0.0090 * DEG,           # MEASURED, RMS over axes
        accel_turnon_bias=0.02,                    # MEASURED (EKF, z axis only)
        gyro_turnon_bias=0.005 * DEG,              # MEASURED residual after EKF, upper bound
        accel_bias_rw=1.65e-4,                     # MEASURED, y/z only, m/s^2/sqrt(s)
        gyro_bias_rw=0.0004 * DEG,                 # MEASURED, RMS over axes
        accel_scale=0.001,                         # ASSUMED
        gyro_scale=0.001,                          # ASSUMED
        name="x500_measured",
        note="Our Pixhawk 6C IMU0, static log 15 Oct 2026. Post-EKF residual bias.",
    ),

    # Fibre-optic gyro class, the sort of unit fitted to a crewed aircraft or a
    # serious UAV. Not a specific product - these are the order-of-magnitude
    # bands for "tactical grade" as given in the standard reference tables
    # (e.g. Groves, Principles of GNSS, Inertial and Multisensor Integrated
    # Navigation Systems, 2nd ed., ch. 4). Any real unit should be looked up.
    "tactical": ErrorModel(
        accel_noise_density=20e-6 * G0,            # ~20 ug/sqrt(Hz)
        gyro_noise_density=0.05 * DEG_PER_SQRT_HOUR,   # 0.05 deg/sqrt(hr) ARW
        accel_turnon_bias=1e-3 * G0,               # ~1 mg
        gyro_turnon_bias=1.0 * DEG_PER_HOUR,       # ~1 deg/hr
        accel_bias_rw=1e-5,
        gyro_bias_rw=0.1 * DEG_PER_HOUR / np.sqrt(3600.0),
        accel_scale=1e-4,
        gyro_scale=1e-4,
        name="tactical",
        note="Class-typical bands, not a specific part. Verify before quoting.",
    ),

    # Navigation grade: ring-laser or high-end FOG, the class used on airliners
    # and submarines. Included to show the shape of the problem when the IMU is
    # effectively not the limit - which is the interesting comparison, because
    # it says how much a better IMU would actually buy. Same source caveat as
    # tactical.
    "navigation": ErrorModel(
        accel_noise_density=5e-6 * G0,             # ~5 ug/sqrt(Hz)
        gyro_noise_density=0.002 * DEG_PER_SQRT_HOUR,  # 0.002 deg/sqrt(hr) ARW
        accel_turnon_bias=25e-6 * G0,              # ~25 ug
        gyro_turnon_bias=0.01 * DEG_PER_HOUR,      # ~0.01 deg/hr
        accel_bias_rw=1e-6,
        gyro_bias_rw=0.001 * DEG_PER_HOUR / np.sqrt(3600.0),
        accel_scale=1e-5,
        gyro_scale=1e-5,
        name="navigation",
        note="Class-typical bands, not a specific part. Verify before quoting.",
    ),
}


def grade(name: str) -> ErrorModel:
    """Look up a preset by name, with a useful message if it is not there."""
    if name not in GRADES:
        raise KeyError(f"unknown IMU grade {name!r}. Available: {sorted(GRADES)}")
    return GRADES[name]
