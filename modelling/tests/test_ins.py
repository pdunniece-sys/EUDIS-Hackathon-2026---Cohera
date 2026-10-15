"""Checks that the INS is right, not just that it runs.

The first three are the ones that matter. A strapdown INS has a handful of
classic ways to be subtly wrong - a transposed rotation matrix, a sign error on
gravity, quaternion multiplication the wrong way round - and all of them still
produce plausible-looking output. What they cannot do is reproduce the analytic
drift laws, because those depend on the errors propagating through the
mechanisation in exactly the right way:

    perfect IMU          ->  the original path comes back
    accelerometer bias b ->  position error of 0.5 * b * t^2
    gyro bias w          ->  position error of g * w * t^3 / 6

The third one is the sharpest test in the file. It only comes out as t^3 if the
gyro error tilts the navigation frame and gravity then leaks into the
horizontal channels with the right sign and magnitude - which exercises the
attitude integration, the rotation convention and the gravity model all at
once.
"""

import numpy as np
import pytest

from ins import (
    GRADES,
    ImuSamples,
    InsState,
    Mechanisation,
    NavFrame,
    Trajectory,
    inverse_propagate,
    lawnmower,
    propagate,
    run_ins,
    synthetic_flight,
)
from ins.errors import ErrorModel, ErrorRealisation
from ins.loaders import G0, load_quspin_dynamic_csv
from ins.mechanisation import normal_gravity

LAT, LON, ALT = 52.45, -6.40, 60.0      # on the Tellus A7 block


def hovering(duration_s=600.0, rate_hz=100.0):
    """A perfectly stationary vehicle: level, still, facing north.

    The analytic drift laws assume nothing else is happening, so the bias tests
    use this rather than a real flight path.
    """
    n = int(duration_s * rate_hz) + 1
    t = np.arange(n) / rate_hz
    g = normal_gravity(LAT, ALT)
    truth = Trajectory(
        t, np.full(n, LAT), np.full(n, LON), np.full(n, ALT),
        np.zeros((n, 3)), np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)), name="hover",
    )
    truth.mech = Mechanisation(NavFrame(LAT, LON, ALT))
    # At rest the accelerometer measures specific force straight up: (0, 0, -g).
    imu = ImuSamples(t, np.tile([0.0, 0.0, -g], (n, 1)), np.zeros((n, 3)),
                     name="hover")
    return truth, imu


def with_exact_bias(imu, accel=(0, 0, 0), gyro=(0, 0, 0)):
    """Add one precise, repeatable bias and no noise at all."""
    model = ErrorModel(name="exact bias")
    dirty, _ = model.corrupt(imu, realisation=ErrorRealisation(accel, gyro))
    return dirty


# ----------------------------------------------------------------------
# 1. A perfect IMU reproduces the path it was derived from
# ----------------------------------------------------------------------

def test_perfect_imu_reproduces_a_curved_path():
    """Integrating a flawless IMU must give back the trajectory it came from.

    The path is a real survey pattern with turns, climbs in bank angle and
    yaw rates of tens of degrees per second, so this is not a straight line
    that would hide a frame error.

    The pass mark is 1 m after 10 minutes. It actually closes to well under a
    millimetre, because the mechanisation's inverse is exact rather than
    approximate - so if this test ever reports tens of centimetres, something
    has broken, even though it would still pass.
    """
    waypoints = lawnmower((LAT, LON), length_m=1500, width_m=700, spacing_m=140)
    truth, imu = synthetic_flight(waypoints, alt_m=ALT, speed_mps=8.0, rate_hz=100.0)
    assert truth.duration_s > 600, "need at least 10 minutes of flight to test"

    run = run_ins(imu, truth, t_cut=0.0, grade=None)

    assert run.error_at(600) < 1.0
    assert np.nanmax(run.error_m) < 1.0
    # And the much tighter bound that says the inverse is still exact.
    assert np.nanmax(run.error_m) < 0.01


def test_the_synthetic_flight_is_physically_sensible():
    """Guard the trajectory builder itself.

    An earlier version padded the smoothing by plain reflection, which made the
    drone start from a standstill and produced a 4400 deg/s gyro reading on the
    first sample. Everything downstream still ran, and the INS still closed,
    because the IMU was consistent with the (nonsense) path. These bounds are
    what would have caught it.
    """
    truth, imu = synthetic_flight(lawnmower((LAT, LON), 1200, 600, 120),
                                  alt_m=ALT, speed_mps=8.0)
    speed = np.linalg.norm(truth.vel_ned, axis=1)

    assert speed.min() > 3.0, "the drone should never nearly stop"
    assert speed.max() < 8.5, "the drone should never exceed the commanded speed"
    assert np.degrees(np.abs(imu.gyro_b)).max() < 180.0, "implausible rotation rate"
    # Level-ish flight: the mean z accelerometer should be about -g.
    assert abs(imu.accel_b[:, 2].mean() + normal_gravity(LAT, ALT)) < 0.1


# ----------------------------------------------------------------------
# 2 and 3. The analytic drift laws
# ----------------------------------------------------------------------

def test_accel_bias_drifts_as_t_squared():
    """A constant accelerometer bias integrates twice into 0.5 * b * t^2."""
    truth, imu = hovering(duration_s=600.0)
    bias = 0.01                                   # m/s^2, north axis only
    run = run_ins(with_exact_bias(imu, accel=(bias, 0, 0)), truth,
                  t_cut=0.0, grade=None)

    for seconds in (60, 120, 300, 600):
        predicted = 0.5 * bias * seconds**2
        assert run.error_at(seconds) == pytest.approx(predicted, rel=0.10)


def test_gyro_bias_drifts_as_t_cubed():
    """A gyro bias tilts the frame, gravity leaks in, and error grows as t^3.

    Checked two ways: against the closed form g*w*t^3/6, and by fitting the
    exponent directly, which is the part that cannot be faked by a
    coincidentally-scaled error.
    """
    truth, imu = hovering(duration_s=600.0)
    rate = np.radians(0.01)                       # rad/s about the east axis
    run = run_ins(with_exact_bias(imu, gyro=(0, rate, 0)), truth,
                  t_cut=0.0, grade=None)
    g = normal_gravity(LAT, ALT)

    for seconds in (60, 120, 300, 600):
        predicted = g * rate * seconds**3 / 6.0
        assert run.error_at(seconds) == pytest.approx(predicted, rel=0.10)

    times = np.array([100.0, 200.0, 400.0])
    errors = np.array([run.error_at(s) for s in times])
    exponent = np.polyfit(np.log(times), np.log(errors), 1)[0]
    assert exponent == pytest.approx(3.0, abs=0.05)


def test_a_flawless_imu_does_not_drift_at_all():
    """The hover case with no errors must stay put, to sub-millimetre."""
    truth, imu = hovering(duration_s=600.0)
    run = run_ins(imu, truth, t_cut=0.0, grade=None)
    assert np.nanmax(run.error_m) < 1e-3


# ----------------------------------------------------------------------
# 4. Vectorisation
# ----------------------------------------------------------------------

def test_vectorised_propagate_matches_the_single_case():
    """One thousand identical particles must step exactly like one vehicle.

    This is what makes the module reusable by a particle filter: the same
    propagate() call steps the whole cloud, and it has to agree bit for bit
    with the single-vehicle path or the filter and the reference INS will
    quietly diverge.
    """
    mech = Mechanisation(NavFrame(LAT, LON, ALT))
    rng = np.random.default_rng(7)
    accel = np.array([0.3, -0.2, -9.7])
    gyro = np.radians([2.0, -1.5, 8.0])

    one = mech.initialise(LAT, LON, ALT, [8.0, 0.5, 0.0], [1.0, 0.0, 0.0, 0.0])
    many = mech.initialise(LAT, LON, ALT, [8.0, 0.5, 0.0], [1.0, 0.0, 0.0, 0.0],
                           n=1000)
    assert many.n == 1000

    for _ in range(200):
        one = mech.propagate(one, accel, gyro, 0.01)
        many = mech.propagate(many, accel, gyro, 0.01)

    assert np.allclose(np.broadcast_to(one.pos_ned, (1000, 3)), many.pos_ned, atol=0)
    assert np.allclose(np.broadcast_to(one.vel_ned, (1000, 3)), many.vel_ned, atol=0)
    assert np.allclose(np.broadcast_to(one.quat, (1000, 4)), many.quat, atol=0)


def test_particles_can_each_have_their_own_reading():
    """A filter needs to push a different IMU sample into every particle."""
    mech = Mechanisation(NavFrame(LAT, LON, ALT), baro=None)
    state = InsState.at_rest(n=4)
    accel = np.tile([0.0, 0.0, -9.81], (4, 1))
    accel[:, 0] = [0.0, 1.0, 2.0, 3.0]            # a different bias per particle

    stepped = mech.propagate(state, accel, np.zeros((4, 3)), 1.0)
    # North velocity should order itself the same way as the north accelerations.
    assert np.all(np.diff(stepped.vel_ned[:, 0]) > 0)
    assert stepped.vel_ned[0, 0] == pytest.approx(0.0, abs=1e-9)


# ----------------------------------------------------------------------
# 5. Unit conversions in the loaders
# ----------------------------------------------------------------------

def test_quspin_csv_unit_conversions(tmp_path):
    """milli-g to m/s^2 and deg/s to rad/s, on a file we construct by hand."""
    csv = tmp_path / "fake_dynamic.csv"
    csv.write_text(
        "Logger_Time (us) Original,AccelerometerX,AccelerometerY,AccelerometerZ,"
        "GyroX,GyroY,GyroZ,Latitude,Longitude,Height\n"
        "0,1000,0,-1000,180,0,-90,44.0,-79.0,200.0\n"
        "4000,0,500,0,0,90,0,44.00001,-79.00001,200.5\n"
        "8000,0,0,0,0,0,0,44.00002,-79.00002,201.0\n"
    )
    imu, track = load_quspin_dynamic_csv(csv)

    # 1000 mg is exactly one standard gravity.
    assert imu.accel_b[0, 0] == pytest.approx(G0)
    assert imu.accel_b[0, 2] == pytest.approx(-G0)
    assert imu.accel_b[1, 1] == pytest.approx(G0 / 2)
    # 180 deg/s is pi rad/s.
    assert imu.gyro_b[0, 0] == pytest.approx(np.pi)
    assert imu.gyro_b[0, 2] == pytest.approx(-np.pi / 2)
    assert imu.gyro_b[1, 1] == pytest.approx(np.pi / 2)

    # The clock starts at zero and is in seconds, from microseconds.
    assert imu.t[0] == 0.0
    assert imu.t[1] == pytest.approx(0.004)
    assert imu.rate_hz == pytest.approx(250.0)

    # Position comes through untouched, and carries no attitude.
    assert track.lat[0] == pytest.approx(44.0)
    assert track.alt_m[2] == pytest.approx(201.0)
    assert track.att_quat is None


def test_quspin_axis_mapping_is_applied(tmp_path):
    """The frame matrix is honoured, because the sensor axes are not known."""
    csv = tmp_path / "fake.csv"
    csv.write_text(
        "Logger_Time (us) Original,AccelerometerX,AccelerometerY,AccelerometerZ,"
        "GyroX,GyroY,GyroZ,Latitude,Longitude,Height\n"
        "0,1000,0,0,0,0,0,44.0,-79.0,200.0\n"
        "4000,1000,0,0,0,0,0,44.0,-79.0,200.0\n"
        "8000,1000,0,0,0,0,0,44.0,-79.0,200.0\n"
    )
    # Swap x and z: sensor x should land on body z.
    swap = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    imu, _ = load_quspin_dynamic_csv(csv, body_from_sensor=swap)
    assert imu.accel_b[0, 2] == pytest.approx(G0)
    assert imu.accel_b[0, 0] == pytest.approx(0.0)


def test_quspin_loader_rejects_the_wrong_file(tmp_path):
    """A clear message beats a KeyError three frames down."""
    csv = tmp_path / "wrong.csv"
    csv.write_text("time,field\n0,53000\n")
    with pytest.raises(ValueError, match="missing columns"):
        load_quspin_dynamic_csv(csv)


def test_ardupilot_loader_needs_pymavlink_or_a_file():
    """Either way the error names the problem rather than throwing an ImportError
    from somewhere inside the reader."""
    from ins.loaders import load_ardupilot_bin
    with pytest.raises(FileNotFoundError):
        load_ardupilot_bin("no_such_log.BIN")


# ----------------------------------------------------------------------
# 6. Reproducibility
# ----------------------------------------------------------------------

def test_the_same_seed_gives_the_same_flight():
    """A drift figure has to be reproducible to be quotable."""
    truth, imu = synthetic_flight(lawnmower((LAT, LON), 1000, 500, 125),
                                  alt_m=ALT, speed_mps=8.0)
    first = run_ins(imu, truth, 0.0, grade="consumer_mems", seed=42)
    again = run_ins(imu, truth, 0.0, grade="consumer_mems", seed=42)
    different = run_ins(imu, truth, 0.0, grade="consumer_mems", seed=43)

    assert np.array_equal(first.error_m, again.error_m)
    assert not np.array_equal(first.error_m, different.error_m)


def test_grades_rank_in_the_expected_order():
    """A better IMU must drift less. Trivially true, and worth pinning down."""
    truth, imu = synthetic_flight(lawnmower((LAT, LON), 1000, 500, 125),
                                  alt_m=ALT, speed_mps=8.0)
    errors = {name: run_ins(imu, truth, 0.0, grade=name, seed=3).error_at(300)
              for name in ("consumer_mems", "tactical", "navigation")}
    assert errors["consumer_mems"] > errors["tactical"] > errors["navigation"]


# ----------------------------------------------------------------------
# Odds and ends that have bitten before
# ----------------------------------------------------------------------

def test_initial_velocity_error_alone_drifts_linearly():
    """A velocity error at the cut grows as v*t, with no sensor error needed.

    Worth its own test because it is the term people forget: losing GNSS with
    0.1 m/s of velocity error costs 60 m in ten minutes from a flawless IMU.
    """
    from ins import InitialError
    truth, imu = hovering(duration_s=600.0)
    run = run_ins(imu, truth, 0.0,
                  initial_error=InitialError(vel_mps=[0.1, 0.0, 0.0]), grade=None)
    for seconds in (60, 300, 600):
        assert run.error_at(seconds) == pytest.approx(0.1 * seconds, rel=0.02)


def test_error_is_nan_past_the_end_of_truth():
    """Scoring beyond the reference must not silently extrapolate it flat."""
    truth, imu = hovering(duration_s=60.0)
    run = run_ins(imu, truth, 0.0, grade=None)
    assert np.isnan(run.error_at(600))


def test_imu_rejects_a_non_increasing_clock():
    """Out-of-order timestamps are a real failure mode in stitched logs."""
    with pytest.raises(ValueError, match="strictly increase"):
        ImuSamples([0.0, 0.1, 0.05], np.zeros((3, 3)), np.zeros((3, 3)))


def test_inverse_propagate_refuses_a_position_only_track():
    """A GNSS track has no attitude, so it cannot yield gyro readings."""
    mech = Mechanisation(NavFrame(LAT, LON, ALT))
    track = Trajectory([0, 1, 2], [LAT] * 3, [LON] * 3, [ALT] * 3)
    with pytest.raises(ValueError, match="velocity and attitude"):
        inverse_propagate(track, mech)


def test_earth_rotation_flag_is_consistent_end_to_end():
    """With Coriolis switched on, the synthetic IMU and the integrator must
    still agree - the flag has to reach both or the 'perfect' case breaks.

    Barometer aiding is off here on purpose. With Coriolis on, baro noise
    wobbles the vertical velocity, and 2*omega x v turns any vertical velocity
    into a small horizontal acceleration. That coupling is real physics, not an
    error, but it is about a centimetre over ten minutes and it would otherwise
    be the thing this test measures instead of the flag plumbing.
    """
    waypoints = lawnmower((LAT, LON), 800, 400, 100)
    mech = Mechanisation(NavFrame(LAT, LON, ALT), earth_rotation=True)
    truth, imu = synthetic_flight(waypoints, alt_m=ALT, speed_mps=8.0, mech=mech)
    run = run_ins(imu, truth, 0.0, grade=None, mech=mech, baro_aiding=False)
    assert np.nanmax(run.error_m) < 0.01


def test_coriolis_actually_changes_the_answer():
    """Guard against the Earth-rotation flag being silently ignored.

    Driving the integrator with an IMU derived *without* Coriolis while
    integrating *with* it must produce a visible disagreement. If this test
    ever passes trivially, the flag has stopped reaching the mechanisation and
    the test above would still pass for the wrong reason.
    """
    waypoints = lawnmower((LAT, LON), 800, 400, 100)
    without = Mechanisation(NavFrame(LAT, LON, ALT), earth_rotation=False)
    with_rotation = Mechanisation(NavFrame(LAT, LON, ALT), earth_rotation=True)

    truth, imu = synthetic_flight(waypoints, alt_m=ALT, speed_mps=8.0, mech=without)
    mismatched = run_ins(imu, truth, 0.0, grade=None, mech=with_rotation,
                         baro_aiding=False)
    assert np.nanmax(mismatched.error_m) > 1.0


def test_baro_aiding_holds_the_vertical_channel():
    """With an accelerometer bias on the down axis, aiding should bound the
    altitude error while leaving the horizontal free to diverge."""
    truth, imu = hovering(duration_s=600.0)
    dirty = with_exact_bias(imu, accel=(0, 0, 0.05))

    aided = run_ins(dirty, truth, 0.0, grade=None, baro_aiding=True)
    free = run_ins(dirty, truth, 0.0, grade=None, baro_aiding=False)

    aided_alt_error = np.abs(aided.estimate.alt_m - ALT).max()
    free_alt_error = np.abs(free.estimate.alt_m - ALT).max()
    assert aided_alt_error < 5.0
    assert free_alt_error > 100.0
