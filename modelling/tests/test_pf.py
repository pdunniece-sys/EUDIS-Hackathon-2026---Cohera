"""Checks on the particle filter.

The first two tests use a small synthetic grid and no Tellus data: they pin
down the two mechanisms that most often go silently wrong in a particle
filter - resampling that does not preserve the distribution, and particles
off the map being scored as if the map said something about them.
"""

import numpy as np
import pytest
from affine import Affine
from pyproj import Transformer
from rasterio.crs import CRS

from navfilter import ParticleFilter, PFConfig, grid_from_array, systematic_resample

LAT, LON = 52.45, -6.40


@pytest.fixture(scope="module")
def small_grid():
    """A 2 km square of 50 m cells in Irish Transverse Mercator, centred on
    (LAT, LON), holding a smooth ramp so every cell has a finite value."""
    crs = CRS.from_epsg(2157)
    x0, y0 = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(LON, LAT)
    n, cell = 40, 50.0
    transform = Affine(cell, 0, x0 - n * cell / 2, 0, -cell, y0 + n * cell / 2)
    rr, cc = np.mgrid[0:n, 0:n]
    return grid_from_array(10.0 * rr + 3.0 * cc, transform, crs)


# ----------------------------------------------------------------------
# 1. Resampling
# ----------------------------------------------------------------------

def test_systematic_resample_preserves_the_distribution():
    rng = np.random.default_rng(0)
    n = 500
    w = np.full(n, 0.5 / (n - 1))
    w[7] = 0.5                         # one particle holds half the mass

    idx = systematic_resample(w, rng)

    assert idx.shape == (n,)
    assert idx.min() >= 0 and idx.max() < n
    # Systematic resampling copies a particle floor(Nw) or ceil(Nw) times.
    assert np.sum(idx == 7) in (n // 2, n // 2 + 1)
    # The light particles share the rest, each at most once.
    others = np.bincount(idx[idx != 7], minlength=n)
    assert others.max() <= 1


def test_systematic_resample_matches_weights_on_average():
    rng = np.random.default_rng(1)
    w = rng.dirichlet(np.ones(50))
    counts = sum(np.bincount(systematic_resample(w, rng), minlength=50)
                 for _ in range(200))
    np.testing.assert_allclose(counts / (200 * 50), w, atol=0.003)


# ----------------------------------------------------------------------
# 2. Off the map
# ----------------------------------------------------------------------

def test_particles_off_the_map_get_zero_weight(small_grid):
    pf = ParticleFilter(small_grid, PFConfig(n_particles=4, sigma_total_nt=5.0))
    lat = np.array([LAT, LAT, LAT + 1.0, LAT])        # third is 110 km away
    lon = np.array([LON, LON + 0.001, LON, LON + 1.0])  # fourth is 68 km away
    z = small_grid.sample(LAT, LON)
    logw = np.full(4, -np.log(4))

    new, used = pf.weight(logw, lat, lon, z)
    w = np.exp(new)

    assert used
    assert w[2] == 0.0 and w[3] == 0.0
    assert w[:2].sum() == pytest.approx(1.0)
    assert w[0] > w[1]                 # the exact match beats the near miss


def test_all_off_map_update_leaves_weights_unchanged(small_grid):
    pf = ParticleFilter(small_grid, PFConfig(n_particles=3, sigma_total_nt=5.0))
    logw = np.log(np.array([0.2, 0.3, 0.5]))
    lat = np.full(3, LAT + 1.0)
    lon = np.full(3, LON)

    new, used = pf.weight(logw, lat, lon, 42.0)

    assert not used
    np.testing.assert_array_equal(new, logw)
