"""Checks that the QuSpin log parser reads the file correctly.

    pytest

Skipped automatically if the log is not in data/QuSpin/.
"""

import numpy as np
import pytest

from quspin import read_log
from quspin.parse import DEFAULT_LOG


@pytest.fixture(scope="module")
def log():
    if not DEFAULT_LOG.exists():
        pytest.skip(f"No QuSpin log at {DEFAULT_LOG}")
    return read_log()


def test_sample_rate_is_250_hz(log):
    """The counter wraps 0-249 once a second, so the rate must come out at 250."""
    assert log.rate_hz == pytest.approx(250, abs=1)


def test_timestamps_increase(log):
    """Including the four that were repaired - a step backwards would break interpolation."""
    assert np.all(np.diff(log.time_s) > 0)
    assert log.time_s[0] == 0


def test_field_is_a_plausible_total_field(log):
    """Total field, not an anomaly, so it should sit near Earth's background."""
    assert 20_000 < log.field_nt.mean() < 70_000
    assert log.field_nt.std() < 500  # a single short flight should not vary hugely


def test_aux_channels_run_at_a_quarter_of_the_sample_rate(log):
    """Four channels share every fourth line, so each lands at about 62.5 Hz."""
    for channel in (log.accel, log.gyro, log.mag_vector, log.temperature):
        rate = len(channel.time_s) / log.duration_s
        assert rate == pytest.approx(62.5, abs=1), channel.name
        assert np.all(np.diff(channel.time_s) > 0)

    assert log.accel.values.shape[1] == 3
    assert log.temperature.values.ndim == 1


def test_positions_decode_to_somewhere_real(log):
    """A wrong scale factor on lat/lon would put the flight in the sea, or nowhere."""
    assert len(log.fixes) > 0
    assert -90 <= log.fixes.lat.min() <= log.fixes.lat.max() <= 90
    assert -180 <= log.fixes.lon.min() <= log.fixes.lon.max() <= 180
    assert -500 < log.fixes.alt_m.min() < 9000

    # The whole flight should be one small area, not scattered over the globe.
    east, north = log.fixes.local_xy()
    assert np.hypot(np.ptp(east), np.ptp(north)) < 50_000


def test_position_interpolation_covers_every_sample(log):
    lat, lon, alt = log.position_at(log.time_s)
    assert lat.shape == log.field_nt.shape
    assert np.isfinite(lat).all()


def test_airborne_removes_the_ground_time(log):
    flight = log.airborne()
    assert flight.duration_s < log.duration_s
    assert flight.fixes.alt_m.min() > log.fixes.alt_m.min()
    # Startup transients happen on the ground, so the flight should be calmer.
    assert flight.field_nt.std() < log.field_nt.std()


def test_smoothing_removes_noise_but_keeps_the_signal(log):
    flight = log.airborne()
    smooth = flight.smoothed_field(cutoff_hz=1.0)

    assert smooth.shape == flight.field_nt.shape
    assert smooth.std() < flight.field_nt.std()      # noise is gone
    assert smooth.std() > 1.0                        # but the geology is not
    assert abs(smooth.mean() - flight.field_nt.mean()) < 1.0  # no offset introduced
