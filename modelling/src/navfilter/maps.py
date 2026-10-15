"""The map the filter is allowed to use, which is deliberately not the truth.

    from navfilter import degrade_map

    filter_map = degrade_map(truth_map, cell_m=200, error_nt=3.0, corr_m=500, seed=1)

Why the filter's map must be worse than the one that made the readings
----------------------------------------------------------------------
The emulated readings are sampled from the full-resolution Tellus grid. If the
filter then compares them against that same grid, the only mismatch left is
sensor noise, and any filter - even a bad one - locks on and looks perfect.
This is the "inverse crime" of simulation: generating the data and inverting it
with the same model.

A real aircraft never has that luxury. The published map was gridded from
survey lines a few hundred metres apart, so it is smooth where the true field
is not, and it carries levelling and gridding errors that are correlated over
hundreds of metres. `degrade_map` imitates both:

    1. block-average to ~`cell_m` cells and interpolate back, which throws away
       the short wavelengths a gridded survey never really resolved;
    2. add a smooth, Gaussian-correlated error field of `error_nt` RMS with a
       correlation length of `corr_m`, standing in for levelling and
       gridding error.

Both numbers are ASSUMED, not measured against Tellus. They are chosen to be
plausible and on the pessimistic side; a residual study against an
independent survey line would replace them.

What this deliberately does NOT do
----------------------------------
It does not change the grid's geometry. The degraded map sits on exactly the
same cells as the original, so a position error is always a navigation error
and never a registration error between two maps.
"""

from __future__ import annotations

import numpy as np
from affine import Affine
from pyproj import Transformer
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator

from tellus.continuation import _copy_with
from tellus.sample import WGS84, MagneticGrid


def degrade_map(grid: MagneticGrid, cell_m: float = 200.0, error_nt: float = 3.0,
                corr_m: float = 500.0, seed: int | None = 0) -> MagneticGrid:
    """A coarser, slightly wrong copy of `grid`, for the filter to navigate on.

        cell_m    block size to average over, metres. Rounded to a whole number
                  of source cells.
        error_nt  RMS of the added map error, nT
        corr_m    correlation length of that error (Gaussian kernel sigma), m
        seed      the error field is random; the same seed gives the same map

    NaN cells stay NaN, and a block containing any coverage averages only the
    cells it has, so the survey edge does not grow or shrink by much.
    """
    values = grid.values
    res = float(np.mean(grid.resolution))
    factor = max(1, int(round(cell_m / res)))

    smoothed = _block_resample(values, factor) if factor > 1 else values.copy()

    if error_nt > 0:
        rng = np.random.default_rng(seed)
        noise = ndimage.gaussian_filter(rng.normal(size=values.shape),
                                        sigma=corr_m / res, mode="reflect")
        noise *= error_nt / noise.std()
        smoothed = smoothed + noise

    smoothed[~np.isfinite(values)] = np.nan
    name = f"{grid.name} degraded ({factor * res:.0f} m, {error_nt:g} nT)"
    return _copy_with(grid, smoothed, name)


def _block_resample(values: np.ndarray, factor: int) -> np.ndarray:
    """Block-average by `factor`, then bilinear back onto the original cells."""
    rows, cols = values.shape
    pr, pc = (-rows) % factor, (-cols) % factor
    padded = np.pad(values, ((0, pr), (0, pc)), constant_values=np.nan)
    blocks = padded.reshape(padded.shape[0] // factor, factor,
                            padded.shape[1] // factor, factor)
    with np.errstate(invalid="ignore"):
        counts = np.isfinite(blocks).sum(axis=(1, 3))
        coarse = np.nansum(blocks, axis=(1, 3)) / np.where(counts, counts, 1)
    coarse[counts == 0] = np.nan

    # Coarse cell centres, in units of fine cells, so the interpolation lands
    # each fine cell where it actually sits inside its block.
    cr = (np.arange(coarse.shape[0]) + 0.5) * factor - 0.5
    cc = (np.arange(coarse.shape[1]) + 0.5) * factor - 0.5
    # A missing coarse cell would poison its whole neighbourhood through the
    # interpolation; fill with the mean for the interpolation only, the caller
    # restores NaN from the original mask afterwards.
    filled = np.where(np.isfinite(coarse), coarse, np.nanmean(coarse))
    interp = RegularGridInterpolator((cr, cc), filled, bounds_error=False,
                                     fill_value=None)
    rr, ccf = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    return interp(np.column_stack([rr.ravel(), ccf.ravel()])).reshape(rows, cols)


def grid_from_array(values: np.ndarray, transform: Affine, crs,
                    name: str = "synthetic", units: str = "nT") -> MagneticGrid:
    """A MagneticGrid built from an array, for tests and synthetic maps.

    `transform` maps (col, row) to the top-left corner of each cell in `crs`,
    the same convention rasterio uses, so sampling behaves exactly as it does
    on a real Tellus grid.
    """
    out = MagneticGrid.__new__(MagneticGrid)
    out.values = np.asarray(values, dtype=float)
    out.transform = transform
    out.crs = crs
    out.units = units
    out.name = name
    out.path = None
    out._to_grid = Transformer.from_crs(WGS84, crs, always_xy=True)
    out._to_wgs84 = Transformer.from_crs(crs, WGS84, always_xy=True)
    return out
