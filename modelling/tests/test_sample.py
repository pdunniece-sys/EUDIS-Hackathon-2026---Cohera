"""Checks that the sampling behaves the way a navigation prototype needs it to.

    pytest

These need the A7 grid on disk. If it is not there, run:

    python scripts/fetch_tellus.py
"""

import numpy as np
import pytest
from pyproj import Transformer

from tellus import MagneticGrid, products
from tellus.sample import straight_track


@pytest.fixture(scope="module")
def grid():
    if not products.get("a7_tmi").tif_path.exists():
        pytest.skip("Grid not downloaded - run: python scripts/fetch_tellus.py")
    return MagneticGrid("a7_tmi")


@pytest.fixture(scope="module")
def to_wgs84(grid):
    return Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True)


def cell_centre_lonlat(grid, to_wgs84, row, col):
    """The WGS84 position of the centre of one cell."""
    x, y = grid.transform @ (col + 0.5, row + 0.5)
    lon, lat = to_wgs84.transform(x, y)
    return lat, lon


def a_valid_cell(grid):
    """Row/col of some cell well inside the surveyed area."""
    rows, cols = grid.shape
    inner = grid.values[rows // 4 : 3 * rows // 4, cols // 4 : 3 * cols // 4]
    r, c = np.argwhere(np.isfinite(inner))[len(np.argwhere(np.isfinite(inner))) // 2]
    return r + rows // 4, c + cols // 4


def test_sampling_a_cell_centre_returns_that_cells_value(grid, to_wgs84):
    """The headline guarantee: ask for a cell centre, get that cell back."""
    row, col = a_valid_cell(grid)
    expected = grid.values[row, col]

    lat, lon = cell_centre_lonlat(grid, to_wgs84, row, col)
    assert grid.sample(lat, lon) == pytest.approx(expected, abs=1e-4)


def test_midway_between_two_cells_is_their_average(grid):
    """Confirms the interpolation is bilinear rather than nearest-neighbour.

    Nearest-neighbour would snap to one of the two cells; bilinear lands
    exactly halfway between them.
    """
    row, col = a_valid_cell(grid)
    left, right = grid.values[row, col], grid.values[row, col + 1]
    assert np.isfinite([left, right]).all()

    # Half a cell east of the first cell's centre, in the grid's own CRS.
    x, y = grid.transform @ (col + 1.0, row + 0.5)
    got = grid._sample_native(np.array([x]), np.array([y]))[0]

    assert got == pytest.approx((left + right) / 2, abs=1e-6)
    assert got != pytest.approx(left, abs=1e-6)  # i.e. it did not just snap


def test_outside_the_grid_is_nan_not_zero(grid):
    """A silent zero here would read as a real magnetic measurement."""
    for lat, lon in [(0.0, 0.0), (53.35, -6.26), (52.5, -20.0)]:  # Gulf of Guinea, Dublin, Atlantic
        assert np.isnan(grid.sample(lat, lon))


def test_unsurveyed_cells_are_nan(grid, to_wgs84):
    """Inside the grid's rectangle but outside the flown area is also NaN."""
    row, col = np.argwhere(~np.isfinite(grid.values))[0]
    lat, lon = cell_centre_lonlat(grid, to_wgs84, row, col)
    assert np.isnan(grid.sample(lat, lon))


def test_sample_track_matches_sampling_one_at_a_time(grid):
    """The vectorised path must not drift from the single-point path."""
    lats, lons = straight_track(*products.A7_DEMO_TRACK, n=25)
    one_by_one = [grid.sample(lat, lon) for lat, lon in zip(lats, lons)]
    np.testing.assert_allclose(grid.sample_track(lats, lons), one_by_one)


def test_track_variation_is_physically_sensible(grid):
    """The demo track should look like real geology, not a flat line or nonsense."""
    lats, lons = straight_track(*products.A7_DEMO_TRACK, n=400)
    values = grid.sample_track(lats, lons)

    assert np.isfinite(values).all(), "the demo track should be fully inside coverage"

    peak_to_peak = values.max() - values.min()
    assert 10 < peak_to_peak < 5000, f"{peak_to_peak:.0f} nT is not a credible swing"
    assert values.std() > 5, "the profile is essentially flat - suspect a bad read"

    # Airborne anomaly grids are levelled to sit near zero overall.
    assert abs(np.median(values)) < 1000


def test_subset_gives_the_same_answers_as_the_full_grid(grid):
    """A cut-out is a speed optimisation, so it must not change any value."""
    lats, lons = straight_track(*products.A7_DEMO_TRACK, n=200)
    bbox = (lons.min(), lats.min(), lons.max(), lats.max())

    small = grid.subset(bbox)
    assert small.values.size < grid.values.size

    np.testing.assert_allclose(
        small.sample_track(lats, lons), grid.sample_track(lats, lons), atol=1e-9
    )


def test_grid_is_stored_north_up(grid):
    """Row 0 must be the northern edge. Getting this backwards flips every map."""
    assert grid.transform.e < 0
    _, bottom, _, top = grid.bounds
    assert top > bottom
