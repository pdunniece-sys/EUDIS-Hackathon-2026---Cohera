"""Checks on the magnetometer emulator and on upward continuation.

The first test is the important one. With every error switched off, the
emulated anomaly must equal the map value along the track - not approximately,
to within a nanotesla. If that holds, then everything the emulator adds
afterwards is a named, switchable error term rather than an unexplained
offset, and any difference between the OPM and the vector sensor is a
difference in their error models rather than an accident of the plumbing.

Tests needing the Tellus grid are skipped if it has not been downloaded, the
same way tests/test_sample.py handles it.
"""

import datetime as dt

import numpy as np
import pytest
from scipy import signal

from ins import lawnmower, synthetic_flight
from sensor import (
    DiurnalModel,
    FieldModel,
    NoDiurnal,
    NoInterference,
    ParametricInterference,
    ScalarOPM,
    VectorMagnetometer,
)
from tellus import MagneticGrid, products
from tellus.continuation import continue_to_altitude, upward_continue

LAT, LON, ALT = 52.45, -6.40, 60.0
FLIGHT_DATE = dt.date(2026, 10, 15)

grid_available = pytest.mark.skipif(
    not products.get("a7_tmi").tif_path.exists(),
    reason="run scripts/fetch_tellus.py first",
)


@pytest.fixture(scope="module")
def flight():
    """A short survey flight, shared by every test that needs one."""
    truth, _ = synthetic_flight(lawnmower((LAT, LON), 600, 300, 100),
                                alt_m=ALT, speed_mps=8.0, rate_hz=250.0)
    return truth


@pytest.fixture(scope="module")
def grid():
    return MagneticGrid("a7_tmi").subset((LON - 0.05, LAT - 0.05, LON + 0.05, LAT + 0.05))


def silent_opm(**kwargs):
    """An OPM with every error term switched off."""
    defaults = dict(instrument_noise_pt_per_rthz=0.0, platform_noise_nt_rms=0.0,
                    mains_nt_rms=0.0, heading_error_nt=0.0, dead_zone_deg=0.0)
    defaults.update(kwargs)
    return ScalarOPM(**defaults)


# ----------------------------------------------------------------------
# 1. With no errors, the emulator returns the map
# ----------------------------------------------------------------------

@grid_available
def test_perfect_opm_returns_the_map_value(flight, grid):
    """Errors off, diurnal off: the emulated anomaly IS the map, to under 1 nT.

    This is the test that says the field assembly, the IGRF subtraction and the
    sensor plumbing all agree. Everything else in this file measures a
    deliberate departure from it.
    """
    field = FieldModel(grid=grid, date=FLIGHT_DATE, diurnal=NoDiurnal(),
                       interference=NoInterference(), continue_to_altitude=False)
    readings = silent_opm().emulate(flight, field, seed=0, decimate_output=False)
    anomaly = silent_opm().to_anomaly(readings, flight, field)

    truth_anomaly = grid.sample_track(flight.lat, flight.lon)
    good = np.isfinite(truth_anomaly) & np.isfinite(anomaly)
    assert good.sum() > 100, "the track should mostly be on the grid"
    assert np.abs(anomaly[good] - truth_anomaly[good]).max() < 1.0


@grid_available
def test_the_diurnal_model_is_removed_by_to_anomaly(flight, grid):
    """A diurnal swing must not survive into the anomaly, since the same model
    is used to add it and to take it away."""
    field = FieldModel(grid=grid, date=FLIGHT_DATE,
                       diurnal=DiurnalModel(amplitude_nt=50.0),
                       continue_to_altitude=False)
    readings = silent_opm().emulate(flight, field, seed=0, decimate_output=False)
    anomaly = silent_opm().to_anomaly(readings, flight, field)

    truth_anomaly = grid.sample_track(flight.lat, flight.lon)
    good = np.isfinite(truth_anomaly) & np.isfinite(anomaly)
    assert np.abs(anomaly[good] - truth_anomaly[good]).max() < 1.0


# ----------------------------------------------------------------------
# 2. The noise matches what was measured in dynamic.ipynb
# ----------------------------------------------------------------------

def test_noise_matches_the_measured_flight(flight):
    """1.35 nT RMS above 0.1 Hz with mains notched, ~3-4 pT/rtHz floor.

    These are the figures from notebooks/dynamic.ipynb. The emulator exists to
    reproduce that flight's noise, so this test is the link between the two.
    """
    field = FieldModel(grid=None, date=FLIGHT_DATE, diurnal=NoDiurnal())
    opm = ScalarOPM(mains_hz=60.0)          # the measured flight was in Ontario
    readings = opm.emulate(flight, field, seed=3, decimate_output=False)

    fs = opm.rate_hz
    x = readings.field_nt - np.nanmean(readings.field_nt)

    # Notch the mains out, exactly as the notebook does.
    notched = x.copy()
    for f0 in (60.0, 120.0):
        notched = signal.filtfilt(*signal.iirnotch(f0, 30, fs), notched)
    high = signal.filtfilt(*signal.butter(2, 0.1 / (fs / 2), "high"), notched)

    rms = np.sqrt(np.mean(high**2))
    assert rms == pytest.approx(1.35, rel=0.20)

    # And the broadband floor, where nothing real or mains-driven lives.
    f, p = signal.welch(x, fs=fs, nperseg=4096)
    band = (f >= 100) & (f <= 125)
    floor_pt = np.sqrt(np.median(p[band])) * 1000.0
    assert 2.0 < floor_pt < 6.0, f"floor {floor_pt:.1f} pT/rtHz outside 2-6"


def test_mains_frequency_is_a_parameter(flight):
    """50 Hz in Ireland, 60 Hz in Ontario. A hardcoded 60 would pass silently
    on Irish data while leaving the interference in."""
    field = FieldModel(grid=None, date=FLIGHT_DATE, diurnal=NoDiurnal())

    peaks = {}
    for mains_hz in (50.0, 60.0):
        opm = ScalarOPM(mains_hz=mains_hz, platform_noise_nt_rms=0.0,
                        instrument_noise_pt_per_rthz=0.0, heading_error_nt=0.0)
        readings = opm.emulate(flight, field, seed=1, decimate_output=False)
        x = readings.field_nt - np.nanmean(readings.field_nt)
        f, p = signal.welch(x, fs=opm.rate_hz, nperseg=4096)
        peaks[mains_hz] = f[np.argmax(p)]

    assert peaks[50.0] == pytest.approx(50.0, abs=1.0)
    assert peaks[60.0] == pytest.approx(60.0, abs=1.0)


# ----------------------------------------------------------------------
# 3. The dead zone
# ----------------------------------------------------------------------

def test_dead_zone_is_nan_exactly_within_seven_degrees(flight):
    """NaN when the field is within 7 degrees of the sensor axis, either way,
    and a number everywhere else. Never a plausible-looking value inside."""
    field_model = FieldModel(grid=None, date=FLIGHT_DATE, diurnal=NoDiurnal())
    samples = field_model.at(flight)

    # Point the sensor along the field so the flight actually crosses a dead
    # zone - mounted forward, a level flight in Ireland never would.
    opm = ScalarOPM(axis_body=[0.0, 0.0, 1.0], dead_zone_deg=7.0)
    theta = opm.theta(flight, samples)
    readings = opm.emulate(flight, samples, seed=0, decimate_output=False)

    inside = (theta < np.radians(7.0)) | (theta > np.pi - np.radians(7.0))
    assert np.array_equal(np.isnan(readings.field_nt), inside)
    assert np.array_equal(readings.valid, ~inside)


def test_dead_zone_can_be_switched_off(flight):
    """Zero degrees means no dead zone at all, for isolating other effects."""
    field_model = FieldModel(grid=None, date=FLIGHT_DATE, diurnal=NoDiurnal())
    opm = ScalarOPM(axis_body=[0.0, 0.0, 1.0], dead_zone_deg=0.0)
    readings = opm.emulate(flight, field_model, seed=0, decimate_output=False)
    assert readings.valid.all()


def test_noise_rises_approaching_the_dead_zone(flight):
    """The resonance weakens before it disappears, so readings get noisier
    near the axis rather than failing without warning."""
    field_model = FieldModel(grid=None, date=FLIGHT_DATE, diurnal=NoDiurnal())
    samples = field_model.at(flight)
    opm = ScalarOPM(axis_body=[0.0, 0.0, 1.0], platform_noise_nt_rms=0.0,
                    mains_nt_rms=0.0, heading_error_nt=0.0)
    theta = opm.theta(flight, samples)
    readings = opm.emulate(flight, samples, seed=0, decimate_output=False)

    residual = readings.field_nt - samples.total_nt
    near = np.isfinite(residual) & (theta < np.radians(25))
    far = np.isfinite(residual) & (theta > np.radians(60))
    if near.sum() > 50 and far.sum() > 50:
        assert np.std(residual[near]) > np.std(residual[far])


# ----------------------------------------------------------------------
# 4. The comparison that justifies the quantum sensor
# ----------------------------------------------------------------------

def test_vector_magnitude_error_varies_with_heading_and_the_opm_does_not(flight):
    """The point of the whole module.

    A scale error on a vector magnetometer feeds into |B| weighted by the
    sensor's orientation, so the magnitude error swings as the aircraft turns.
    A scalar OPM reads |B| directly and does not - beyond its small heading
    error term, which is why that term is switched off here.
    """
    field_model = FieldModel(grid=None, date=FLIGHT_DATE, diurnal=NoDiurnal())
    samples = field_model.at(flight)
    heading = np.degrees(np.arctan2(flight.vel_ned[:, 1], flight.vel_ned[:, 0])) % 360

    vector = VectorMagnetometer(scale_error=0.005, bias_nt=0.0,
                                misalignment_deg=0.0, noise_nt=0.0)
    opm = silent_opm()

    vec_err = (vector.emulate(flight, samples, seed=2, decimate_output=False).field_nt
               - samples.total_nt)
    opm_err = (opm.emulate(flight, samples, seed=2, decimate_output=False).field_nt
               - samples.total_nt)

    # Spread of the mean error across heading octants.
    def spread_across_heading(err):
        bins = np.digitize(heading, np.arange(0, 360, 45))
        means = [err[bins == b].mean() for b in np.unique(bins)
                 if (bins == b).sum() > 50]
        return np.ptp(means) if len(means) > 2 else 0.0

    vector_spread = spread_across_heading(vec_err)
    opm_spread = spread_across_heading(opm_err)

    assert vector_spread > 20.0, "a 0.5% scale error should swing with heading"
    assert opm_spread < 1e-6, "a perfect scalar sensor has no heading dependence"
    assert vector_spread > 1000 * max(opm_spread, 1e-9)


def test_opm_heading_error_is_bounded_by_its_parameter(flight):
    """The heading error term must stay inside the datasheet's 3 nT."""
    field_model = FieldModel(grid=None, date=FLIGHT_DATE, diurnal=NoDiurnal())
    samples = field_model.at(flight)
    opm = silent_opm(heading_error_nt=3.0)
    err = (opm.emulate(flight, samples, seed=0, decimate_output=False).field_nt
           - samples.total_nt)
    assert np.nanmax(np.abs(err)) <= 3.0 + 1e-9


# ----------------------------------------------------------------------
# 5. Upward continuation
# ----------------------------------------------------------------------

@grid_available
def test_a_flat_grid_is_unchanged_by_continuation(grid):
    """Continuing a constant field changes nothing: it has no short
    wavelengths to attenuate."""
    flat = upward_continue(_grid_of(grid, np.full(grid.shape, 42.0)), 40.0)
    interior = flat.values[30:-30, 30:-30]
    assert np.allclose(interior, 42.0, atol=0.5)


@grid_available
def test_continuation_by_zero_is_the_identity(grid):
    same = upward_continue(grid, 0.0)
    both_good = np.isfinite(grid.values) & np.isfinite(same.values)
    assert np.allclose(grid.values[both_good], same.values[both_good])


@grid_available
def test_a_spike_spreads_out_and_its_peak_drops(grid):
    """The physical behaviour: height blurs the field and lowers the extremes."""
    values = np.zeros(grid.shape)
    centre = (grid.shape[0] // 2, grid.shape[1] // 2)
    values[centre] = 1000.0
    spike = _grid_of(grid, values)

    lifted = upward_continue(spike, 100.0)

    assert lifted.values[centre] < 1000.0, "the peak must drop"
    neighbours = lifted.values[centre[0] - 3:centre[0] + 4,
                               centre[1] - 3:centre[1] + 4]
    assert np.count_nonzero(np.abs(neighbours) > 1e-6) > 1, "it must spread"
    # Upward continuation conserves the integral of the field.
    assert lifted.values.sum() == pytest.approx(1000.0, rel=0.10)


@grid_available
def test_continuation_attenuates_short_wavelengths_more(grid):
    """The defining property: height is a low-pass filter on the map."""
    lifted = upward_continue(grid, 100.0)
    good = np.isfinite(grid.values) & np.isfinite(lifted.values)
    assert np.nanstd(lifted.values[good]) < np.nanstd(grid.values[good])


@grid_available
def test_downward_continuation_is_refused(grid):
    """Not supported, deliberately - see the module docstring."""
    with pytest.raises(ValueError, match="negative"):
        upward_continue(grid, -20.0)
    with pytest.raises(ValueError, match="BELOW"):
        continue_to_altitude(grid, flight_altitude_m=30.0)


@grid_available
def test_continue_to_altitude_subtracts_the_survey_height(grid):
    """Flying at 100 m over a 60 m survey is a 40 m continuation, not 100."""
    direct = upward_continue(grid, 40.0)
    via_altitude = continue_to_altitude(grid, flight_altitude_m=100.0)
    good = np.isfinite(direct.values) & np.isfinite(via_altitude.values)
    assert np.allclose(direct.values[good], via_altitude.values[good])


# ----------------------------------------------------------------------
# Timing, interference and reproducibility
# ----------------------------------------------------------------------

def test_decimation_reports_its_delay(flight):
    """A 250 Hz sensor on a 10 Hz filter arrives late, and says by how much."""
    field_model = FieldModel(grid=None, date=FLIGHT_DATE, diurnal=NoDiurnal())
    opm = ScalarOPM(output_rate_hz=10.0, latency_s=0.02)
    readings = opm.emulate(flight, field_model, seed=0)

    assert readings.rate_hz == pytest.approx(10.0)
    assert readings.latency_s > 0.02, "filter delay must be added to the latency"
    assert readings.latency_s < 0.5
    assert len(readings) < len(flight) / 20


@grid_available
def test_decimation_does_not_add_a_start_up_transient(flight, grid):
    """The decimated reading must still match the map, to about a nanotesla.

    This is a regression test. The anti-alias filter used to start from rest,
    so its output ramped up from zero towards a 49,000 nT field across the
    whole length of the filter. That put a 19,656 nT spike on the first
    readings and 175 nT of RMS error on the flight - while every undecimated
    test still passed, because the undecimated path never touches the filter.
    """
    field_model = FieldModel(grid=grid, date=FLIGHT_DATE, diurnal=NoDiurnal(),
                             continue_to_altitude=False)
    field = field_model.at(flight)
    opm = silent_opm()
    readings = opm.emulate(flight, field, seed=0, decimate_output=True)
    anomaly = opm.to_anomaly(readings, flight, field_model)

    taken_at = np.clip(readings.t - readings.latency_s, flight.t[0], flight.t[-1])
    expected = np.interp(taken_at, flight.t, field.anomaly_nt)
    error = anomaly - expected
    good = np.isfinite(error)

    assert np.sqrt(np.mean(error[good] ** 2)) < 1.0
    assert np.abs(error[good]).max() < 5.0


def test_interference_is_off_unless_asked_for(flight):
    """The default aircraft is magnetically silent, which is a fiction we
    choose deliberately rather than a measurement."""
    field_model = FieldModel(grid=None, date=FLIGHT_DATE, diurnal=NoDiurnal())
    samples = field_model.at(flight)
    assert not np.any(samples.interference_ned)


def test_permanent_interference_shows_up_and_swings_with_heading(flight):
    """A fixed body-frame field projects onto the Earth's field differently as
    the aircraft turns - which is exactly why it is hard to tell from geology."""
    field_model = FieldModel(
        grid=None, date=FLIGHT_DATE, diurnal=NoDiurnal(),
        interference=ParametricInterference(permanent_nt=[100.0, 0.0, 0.0]))
    samples = field_model.at(flight)

    assert np.any(samples.interference_ned)
    clean = FieldModel(grid=None, date=FLIGHT_DATE, diurnal=NoDiurnal()).at(flight)
    difference = samples.total_nt - clean.total_nt
    # It is not a constant offset: it depends on which way the drone points.
    assert np.ptp(difference) > 10.0


def test_the_same_seed_gives_the_same_readings(flight):
    field_model = FieldModel(grid=None, date=FLIGHT_DATE, diurnal=NoDiurnal())
    opm = ScalarOPM()
    first = opm.emulate(flight, field_model, seed=11)
    again = opm.emulate(flight, field_model, seed=11)
    other = opm.emulate(flight, field_model, seed=12)

    assert np.array_equal(np.nan_to_num(first.field_nt), np.nan_to_num(again.field_nt))
    assert not np.array_equal(np.nan_to_num(first.field_nt), np.nan_to_num(other.field_nt))


def test_igrf_gives_a_sensible_irish_field():
    """About 49,000 nT, pointing north and steeply down. If this ever fails,
    the east/north/up to north/east/down conversion has been broken."""
    model = FieldModel(grid=None, date=FLIGHT_DATE)
    ned = model.core_field_ned([LAT], [LON], [ALT])[0]

    assert np.linalg.norm(ned) == pytest.approx(49_000, rel=0.05)
    assert ned[0] > 0, "the field should point north"
    assert ned[2] > 0, "and steeply DOWN in the northern hemisphere"
    inclination = np.degrees(np.arctan2(ned[2], np.hypot(ned[0], ned[1])))
    assert 65 < inclination < 75, f"Irish inclination should be about 69, got {inclination:.1f}"


def test_scalar_sensor_barely_sees_interference_across_the_field(flight):
    """A scalar sensor only feels the component of interference along B.
    Perpendicular interference is nearly invisible to it - and fully visible
    to a vector sensor, which is a real difference between the two."""
    model = FieldModel(grid=None, date=FLIGHT_DATE, diurnal=NoDiurnal())
    clean = model.at(flight)
    b_hat = clean.b_hat[0]

    # A 200 nT vector perpendicular to the field, held in the navigation frame.
    across = np.cross(b_hat, [0.0, 0.0, 1.0])
    across = 200.0 * across / np.linalg.norm(across)
    shifted = clean.b_ned + across

    change = np.abs(np.linalg.norm(shifted, axis=1) - clean.total_nt)
    assert change.max() < 2.0, "perpendicular interference should barely move |B|"


def _grid_of(template, values):
    """A grid with this geometry but different values, for continuation tests."""
    from tellus.continuation import _copy_with
    return _copy_with(template, values, "test grid")
