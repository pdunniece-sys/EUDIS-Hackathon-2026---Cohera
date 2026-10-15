"""Checks on the particle filter.

The first two groups use a small synthetic grid and no Tellus data: they pin
down the two mechanisms that most often go silently wrong in a particle
filter - resampling that does not preserve the distribution, and particles
off the map being scored as if the map said something about them.

The last two fly five minutes over A7 and skip if the grid is not downloaded.
One checks the filter does no harm when nothing is wrong; the other is the
headline claim in miniature.
"""

import datetime as dt

import numpy as np
import pytest
from affine import Affine
from pyproj import Transformer
from rasterio.crs import CRS

from ins import lawnmower, run_ins, synthetic_flight
from navfilter import (
    ParticleFilter,
    PFConfig,
    degrade_map,
    grid_from_array,
    systematic_resample,
)
from sensor import FieldModel, ScalarOPM
from tellus import MagneticGrid, products
from tellus.continuation import continue_to_altitude

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


# ----------------------------------------------------------------------
# 3-4. Whole filter on a short flight over Tellus A7
# ----------------------------------------------------------------------

grid_available = pytest.mark.skipif(
    not products.get("a7_tmi").tif_path.exists(),
    reason="run scripts/fetch_tellus.py first",
)


@pytest.fixture(scope="module")
def survey():
    """A five-minute lawnmower over A7, the field along it, and a FieldModel."""
    truth, imu = synthetic_flight(lawnmower((LAT, LON), 800, 450, 150),
                                  alt_m=60.0, speed_mps=8.0, rate_hz=250.0)
    grid = MagneticGrid("a7_tmi").subset((LON - 0.05, LAT - 0.05, LON + 0.05, LAT + 0.05))
    model = FieldModel(grid=grid, date=dt.date(2026, 10, 15))
    return truth, imu, grid, model, model.at(truth)


def silent_opm():
    return ScalarOPM(rate_hz=250.0, instrument_noise_pt_per_rthz=0.0,
                     platform_noise_nt_rms=0.0, mains_nt_rms=0.0,
                     heading_error_nt=0.0, dead_zone_deg=0.0)


def filter_flight(survey, opm, grade, filter_map, **config):
    truth, imu, _, model, field = survey
    ins_run = run_ins(imu, truth, t_cut=0.0, grade=grade, seed=0)
    readings = opm.emulate(truth, field, seed=0)
    anomaly = opm.to_anomaly(readings, ins_run.estimate, model)
    core = lambda la, lo, al: np.linalg.norm(model.core_field_ned(la, lo, al), axis=1)
    pf = ParticleFilter(filter_map, PFConfig(n_particles=500, grade=grade, seed=0, **config))
    return ins_run, pf.run(ins_run.estimate, readings.t - readings.latency_s, anomaly,
                           truth=truth, core_fn=core)


@grid_available
def test_perfect_imu_exact_map_no_noise_stays_close(survey):
    """With nothing wrong anywhere, the filter must not invent an error.

    The INS is exact, so all the filter can do is hold the cloud on the truth;
    a filter that wanders here is being pushed by its own machinery.
    """
    _, _, grid, _, _ = survey
    _, run = filter_flight(survey, silent_opm(), None, continue_to_altitude(grid, 60.0))
    assert np.nanmax(run.error_m) < 30.0


@grid_available
def test_consumer_imu_filter_beats_ins_fivefold(survey):
    """The headline claim, small: the map holds a consumer INS that has drifted
    kilometres to a fraction of that, with a realistic sensor on a degraded map."""
    _, _, grid, _, _ = survey
    filter_map = degrade_map(continue_to_altitude(grid, 60.0), seed=1)
    ins_run, run = filter_flight(survey, ScalarOPM(rate_hz=250.0), "consumer_mems",
                                 filter_map)
    assert run.error_m[-1] < 0.2 * run.ins_error_m[-1]
    assert ins_run.error_m[-1] > 1000.0      # the INS really was lost


# ----------------------------------------------------------------------
# Self-map gridding
# ----------------------------------------------------------------------

def test_grid_lines_reproduces_a_plane_between_lines():
    """Linear gridding of lines across a linear field must be exact between
    them - so any residual on real data is the field or the sensor, not the
    gridding."""
    from navfilter import selfmap

    lat, lon = np.meshgrid(LAT + np.arange(5) * 50 / 111_000,     # 5 lines, 50 m apart
                           LON + np.linspace(0, 500 / 68_000, 200), indexing="ij")
    lat, lon = lat.ravel(), lon.ravel()
    plane = lambda la, lo: 3000.0 * (la - LAT) + 1000.0 * (lo - LON)
    n = len(lat)
    log = selfmap.SurveyLog(np.arange(n) * 0.1, np.zeros(n), lat, lon, np.full(n, 60.0),
                            plane(lat, lon), np.repeat(np.arange(5), 200),
                            np.ones(n, dtype=int), model=None)

    grid = selfmap.grid_lines(log, log.line % 2 == 0, cell_m=10)
    mid_lat = lat[log.line == 1]
    mid_lon = lon[log.line == 1]
    got = grid.sample_track(mid_lat, mid_lon)
    ok = np.isfinite(got)
    assert ok.mean() > 0.9
    np.testing.assert_allclose(got[ok], plane(mid_lat, mid_lon)[ok], atol=0.01)
