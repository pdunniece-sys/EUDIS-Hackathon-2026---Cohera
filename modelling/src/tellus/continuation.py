"""Move a magnetic anomaly grid to a different altitude.

    from tellus import MagneticGrid
    from tellus.continuation import upward_continue

    grid = MagneticGrid("a7_tmi").subset(bbox)
    higher = upward_continue(grid, 40.0)       # Tellus flew at 60 m, so this is 100 m

Tellus was flown at 60 m above ground. A drone at 100 m does not see the same
field: altitude acts as a low-pass filter on the anomaly, because the
contribution of a feature falls off faster the smaller that feature is. Sharp,
shallow structure fades with height while broad structure barely changes.

The maths, in one line
----------------------
In the Fourier domain each wavenumber `k` is simply multiplied by `exp(-|k| dz)`.
Short wavelengths (large `|k|`) are attenuated hard, long wavelengths are left
nearly alone. That is the whole operation.

Why downward continuation is refused
------------------------------------
Going down means multiplying by `exp(+|k| dz)`, which amplifies the shortest
wavelengths most - and the shortest wavelengths in a real grid are mostly
gridding artefacts and noise. A few tens of metres of downward continuation
turns a usable map into noise with a plausible-looking texture, which is worse
than no map. So this module refuses below survey height rather than offering a
tunable instability.

If you genuinely need the field below 60 m, the answer is to fly a new survey,
not to continue this one downwards.
"""

from __future__ import annotations

import numpy as np

from tellus.sample import MagneticGrid

# Tellus was flown at this height above ground. Everything in the grid is the
# field as it was at 60 m, so continuing to a drone altitude means continuing
# by (drone height - 60).
TELLUS_SURVEY_HEIGHT_M = 60.0


def upward_continue(grid: MagneticGrid, dz_m: float, taper_cells: int = 16) -> MagneticGrid:
    """Continue a grid upward by `dz_m` metres. Returns a new grid.

        dz_m         how far UP to move, in metres. Must be >= 0.
        taper_cells  width of the cosine taper applied around the edges before
                     the transform.

    Two practical details the textbook version leaves out, both of which matter
    here:

    **NaNs.** A real survey grid has gaps - off the edge of the block, and over
    ground the survey did not cover. An FFT cannot digest a NaN, so gaps are
    filled with the grid mean for the transform and put back as NaN afterwards.
    Values near a large gap are therefore less trustworthy than values in the
    middle of good coverage.

    **Edges.** An FFT treats the grid as repeating forever, so a value of +400
    on the left edge sits right next to -200 on the right edge, and that
    artificial cliff rings across the whole result. The grid is mirrored and
    tapered to zero at the boundary first, which removes the discontinuity.
    """
    if dz_m < 0:
        raise ValueError(
            f"dz_m={dz_m} is negative, which is downward continuation. That "
            f"amplifies short wavelengths exponentially and turns gridding "
            f"noise into fake anomalies. Refused deliberately - see the module "
            f"docstring."
        )
    if dz_m == 0:
        return _copy_with(grid, grid.values.copy(), grid.name)

    values = grid.values
    gaps = ~np.isfinite(values)
    if gaps.all():
        raise ValueError("grid is entirely NaN, nothing to continue")

    # Work on the anomaly about its own mean, so the taper pulls towards a
    # sensible baseline rather than towards an arbitrary zero.
    mean = float(np.nanmean(values))
    filled = np.where(gaps, mean, values) - mean

    padded, (row0, col0) = _mirror_pad(filled, taper_cells)
    padded = padded * _edge_taper(padded.shape, taper_cells)

    dy, dx = grid.resolution[1], grid.resolution[0]
    rows, cols = padded.shape

    # Wavenumbers in radians per metre. The 2*pi matters - leaving it out is
    # the classic way to get an answer that looks right but attenuates by the
    # wrong amount.
    ky = 2 * np.pi * np.fft.fftfreq(rows, d=dy)
    kx = 2 * np.pi * np.fft.fftfreq(cols, d=dx)
    k = np.hypot(*np.meshgrid(ky, kx, indexing="ij"))

    spectrum = np.fft.fft2(padded)
    continued = np.real(np.fft.ifft2(spectrum * np.exp(-k * dz_m)))

    out = continued[row0:row0 + values.shape[0], col0:col0 + values.shape[1]] + mean
    out[gaps] = np.nan

    name = f"{grid.name} +{dz_m:g} m"
    return _copy_with(grid, out, name)


def continue_to_altitude(grid: MagneticGrid, flight_altitude_m: float,
                         survey_height_m: float = TELLUS_SURVEY_HEIGHT_M,
                         **kwargs) -> MagneticGrid:
    """Continue a survey grid to the altitude a drone is actually flying.

    Convenience wrapper that does the subtraction for you, because getting the
    direction of that subtraction wrong is easy and silent.
    """
    dz = flight_altitude_m - survey_height_m
    if dz < 0:
        raise ValueError(
            f"flying at {flight_altitude_m} m is BELOW the {survey_height_m} m "
            f"survey height, which needs downward continuation. Refused - see "
            f"tellus/continuation.py. Fly at or above survey height, or accept "
            f"the 60 m map as an approximation and say so."
        )
    return upward_continue(grid, dz, **kwargs)


def _mirror_pad(values: np.ndarray, taper_cells: int):
    """Pad by mirroring, so the grid meets its own reflection at every edge.

    Returns the padded array and where the original sits inside it. Padding by
    half the grid size is the usual recommendation; this caps it so a large
    grid does not quadruple in memory for no benefit.
    """
    rows, cols = values.shape
    pad_r = min(max(taper_cells * 2, rows // 4), rows)
    pad_c = min(max(taper_cells * 2, cols // 4), cols)
    padded = np.pad(values, ((pad_r, pad_r), (pad_c, pad_c)), mode="reflect")
    return padded, (pad_r, pad_c)


def _edge_taper(shape, taper_cells: int) -> np.ndarray:
    """A cosine window that falls to zero in the outermost `taper_cells`.

    Flat across the middle, so the interior - the part anyone will sample - is
    untouched.
    """
    def ramp(n):
        w = np.ones(n)
        t = min(taper_cells, n // 2)
        if t > 0:
            edge = 0.5 * (1 - np.cos(np.pi * np.arange(t) / t))
            w[:t] = edge
            w[-t:] = edge[::-1]
        return w

    return np.outer(ramp(shape[0]), ramp(shape[1]))


def _copy_with(grid: MagneticGrid, values: np.ndarray, name: str) -> MagneticGrid:
    """A new MagneticGrid sharing this one's geometry but holding new values."""
    out = MagneticGrid.__new__(MagneticGrid)
    out.values = values
    out.transform = grid.transform
    out.crs = grid.crs
    out.units = grid.units
    out.name = name
    out.path = grid.path
    out._to_grid = grid._to_grid
    out._to_wgs84 = grid._to_wgs84
    return out
