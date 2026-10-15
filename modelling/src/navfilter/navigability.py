"""Where in Ireland magnetic navigation works, and how well, before you fly.

    from navfilter.navigability import information_map

    nav = information_map(MagneticGrid("merge_tmi"))     # a few seconds, all Ireland
    nav.bound_m()          # best-case position error everywhere, metres
    nav.classes()          # good / marginal / flat / too rough for the map
    nav.at(54.08, -9.02)   # one point

The idea in one paragraph
-------------------------
A scalar reading says where you are only through how the map *changes*: a
flat map gives the same number everywhere. Each reading constrains position
along the gradient, by sigma / |grad B|. Over a short track the constraints
from many readings add up as an information matrix

    J = (n / sigma^2) * mean over the track of  grad B grad B^T

and the Cramer-Rao bound says no unbiased estimator can do better than
J^-1. The *weakest direction* is what matters, so the reported bound is

    bound = sigma_eff / sqrt(n * lambda_min(mean grad B grad B^T))

A pure ramp - big gradient, one direction - pins you across it and not at
all along it. |grad B| alone would call that excellent; lambda_min says it is
not.

Information the filter can actually use
---------------------------------------
Checked against particle-filter runs at seven places, the first version of
this (gradients of the full 50 m grid, a fixed sigma) failed both ways:

  * a rough field (207 nT spread) scored a 7 m bound, and the filter lost
    lock at 1.1 km - its 200 m map was 50 nT wrong there, five times the
    error the filter allows for;
  * a nearly flat field (1.9 nT spread) scored 148 m from fine texture a
    200 m map does not contain, and the filter wandered to 1.8 km.

So both halves now come from the map the filter has. Gradients are taken on
the grid smoothed to `map_cell_m` (200 m, the Tellus line spacing and what
degrade_map uses), and what that smoothing removes is map error, added to
sigma in quadrature. Where that map error alone exceeds the filter's sigma,
the cell is classed "too rough": the filter as built is overconfident there,
whatever the bound says.

What the number is, and is not
------------------------------
A *lower bound* on error for a filter that has already locked on. It ignores
INS drift, the correlation of map errors along track, and confusion between
distant look-alike places. Use it to rank places and to plan routes, not to
quote a guaranteed accuracy.

It is computed at the survey's own height (60 m for Tellus) unless the grid
passed in has been upward-continued first. Flying higher smooths the field
and makes every bound worse.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from affine import Affine
from scipy import ndimage

from tellus.sample import MagneticGrid

# Defaults matching the particle filter as configured: 0.5 Hz updates at
# 8 m/s is one reading every 16 m, and the effective sigma including the x2
# correlation inflation is ~9.8 nT (navfilter/particle.py).
SIGMA_NT = 9.8
SAMPLE_SPACING_M = 16.0
TRACK_M = 1000.0
MAP_CELL_M = 200.0

# Classes returned by NavigabilityMap.classes(), in plotting order.
CLASSES = {0: "no survey", 1: "good", 2: "marginal", 3: "flat", 4: "too rough for the map"}


@dataclass
class NavigabilityMap:
    """Track-averaged gradient structure on a coarse grid.

        t_nn, t_ne, t_ee  mean of grad B grad B^T over the window, (nT/m)^2,
                          from the map at `map_cell_m` resolution
        map_err2          mean square of what that map misses, nT^2
        coverage          fraction of the window the survey covered
        transform, crs    the coarse grid's geometry, same CRS as the map
    """

    t_nn: np.ndarray
    t_ne: np.ndarray
    t_ee: np.ndarray
    map_err2: np.ndarray
    coverage: np.ndarray
    transform: Affine
    crs: object
    window_m: float
    map_cell_m: float
    _to_grid: object = None

    def eigen(self):
        """(lambda_min, lambda_max) of the structure tensor, per cell."""
        mean = 0.5 * (self.t_nn + self.t_ee)
        diff = np.sqrt((0.5 * (self.t_nn - self.t_ee)) ** 2 + self.t_ne**2)
        return mean - diff, mean + diff

    def map_error_nt(self):
        return np.sqrt(self.map_err2)

    def bound_m(self, sigma_nt: float = SIGMA_NT, track_m: float = TRACK_M,
                spacing_m: float = SAMPLE_SPACING_M, min_coverage: float = 0.8):
        """Best-case position error, worst direction, metres. NaN off-survey."""
        lam_min, _ = self.eigen()
        n = track_m / spacing_m
        sigma = np.sqrt(sigma_nt**2 + self.map_err2)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = sigma / np.sqrt(n * np.maximum(lam_min, 0.0))
        out[~(self.coverage >= min_coverage)] = np.nan
        return out

    def classes(self, sigma_nt: float = SIGMA_NT, good_m: float = 50.0,
                flat_m: float = 250.0, **kwargs) -> np.ndarray:
        """Integer class per cell, see CLASSES.

        "too rough" wins over everything: if the map's own error exceeds the
        filter's sigma, the filter will trust a map that is wrong by more than
        it allows for, and a good bound there is not to be believed.
        """
        bound = self.bound_m(sigma_nt=sigma_nt, **kwargs)
        out = np.zeros(bound.shape, dtype=np.int8)
        out[bound < good_m] = 1
        out[(bound >= good_m) & (bound < flat_m)] = 2
        out[bound >= flat_m] = 3
        out[np.isfinite(bound) & (self.map_error_nt() > sigma_nt)] = 4
        return out

    def anisotropy(self):
        """sqrt(lambda_min / lambda_max): 1 for texture in every direction,
        0 for a pure ramp you could slide along."""
        lo, hi = self.eigen()
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.sqrt(np.clip(lo, 0, None) / hi)

    def _cells(self, lat, lon):
        x, y = self._to_grid.transform(np.atleast_1d(lon), np.atleast_1d(lat))
        col, row = ~self.transform @ (np.asarray(x), np.asarray(y))
        row, col = np.floor(row).astype(int), np.floor(col).astype(int)
        ok = ((row >= 0) & (row < self.t_nn.shape[0])
              & (col >= 0) & (col < self.t_nn.shape[1]))
        return row, col, ok

    def at(self, lat, lon, layer=None, **kwargs):
        """A layer at WGS84 points (nearest coarse cell). Default: the bound."""
        values = self.bound_m(**kwargs) if layer is None else layer
        row, col, ok = self._cells(lat, lon)
        out = np.full(len(row), np.nan)
        out[ok] = values[row[ok], col[ok]]
        return out

    @property
    def extent(self):
        rows, cols = self.t_nn.shape
        left, top = self.transform @ (0, 0)
        right, bottom = self.transform @ (cols, rows)
        return left, right, bottom, top


def _smooth_nan(a: np.ndarray, sigma_cells: float) -> np.ndarray:
    """Gaussian smoothing that ignores NaN instead of spreading it."""
    good = np.isfinite(a)
    num = ndimage.gaussian_filter(np.where(good, a, 0.0), sigma_cells, mode="nearest")
    den = ndimage.gaussian_filter(good.astype(a.dtype), sigma_cells, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[~good] = np.nan
    return out


def information_map(grid: MagneticGrid, cell_m: float = 250.0,
                    window_m: float = TRACK_M, map_cell_m: float = MAP_CELL_M,
                    strip_rows: int = 400) -> NavigabilityMap:
    """Gradient structure of `grid` as a `map_cell_m` map would hold it,
    averaged over `window_m`, on `cell_m` cells.

    Works through the grid in strips of rows, with enough overlap for the
    smoothing, so the national grid (57 M cells) never needs more than a few
    hundred MB at once.
    """
    res = float(np.mean(grid.resolution))
    f = max(1, int(round(cell_m / res)))
    rows, cols = grid.shape
    out_r, out_c = int(np.ceil(rows / f)), int(np.ceil(cols / f))
    sums = np.zeros((5, out_r, out_c))       # t_nn, t_ne, t_ee, map_err2, count
    total = np.zeros((out_r, out_c))

    # A block average over map_cell_m has the same variance as a Gaussian of
    # sigma = width / sqrt(12); close enough to stand in for degrade_map.
    map_sigma = map_cell_m / np.sqrt(12.0) / res
    margin = int(np.ceil(4 * map_sigma)) + 1
    step = strip_rows - (strip_rows % f)

    for r0 in range(0, rows, step):
        r1 = min(rows, r0 + step)
        lo, hi = max(0, r0 - margin), min(rows, r1 + margin)
        raw = grid.values[lo:hi].astype(np.float32)
        smooth = _smooth_nan(raw, map_sigma)
        # Rows run north to south, so d/dnorth is minus d/drow.
        d_row, d_col = np.gradient(smooth, res)
        keep = slice(r0 - lo, r0 - lo + (r1 - r0))
        g_n, g_e = -d_row[keep], d_col[keep]
        resid2 = (raw[keep] - smooth[keep]) ** 2
        layers = (g_n * g_n, g_n * g_e, g_e * g_e, resid2)
        good = np.isfinite(layers[0]) & np.isfinite(layers[2]) & np.isfinite(resid2)

        n_rows = r1 - r0
        pad_r, pad_c = (-n_rows) % f, (-cols) % f

        def pool(a):
            """Sum f x f blocks; the ragged edge is zero-padded."""
            a = np.pad(a, ((0, pad_r), (0, pad_c)))
            return a.reshape(a.shape[0] // f, f, a.shape[1] // f, f).sum(axis=(1, 3))

        k0 = r0 // f
        k1 = k0 + (n_rows + pad_r) // f
        for i, layer in enumerate(layers):
            sums[i, k0:k1] += pool(np.where(good, layer, 0.0))
        sums[4, k0:k1] += pool(good.astype(np.float32))              # cells with data
        total[k0:k1] += pool(np.ones(good.shape, dtype=np.float32))  # all cells

    # Window average by normalised convolution, so survey edges and gaps are
    # averaged over the cells that exist rather than diluted with zeros.
    sigma_cells = window_m / cell_m / 2.0
    window = lambda a: ndimage.gaussian_filter(a, sigma_cells, mode="constant")
    count = window(sums[4])
    with np.errstate(invalid="ignore", divide="ignore"):
        t_nn, t_ne, t_ee, map_err2 = (window(sums[i]) / count for i in range(4))
        coverage = count / window(total)

    transform = grid.transform @ Affine.scale(f, f)
    return NavigabilityMap(t_nn, t_ne, t_ee, map_err2, coverage, transform, grid.crs,
                           window_m, map_cell_m, _to_grid=grid._to_grid)
