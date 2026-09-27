"""Read magnetic anomaly values out of a Tellus grid at given latitudes/longitudes.

    from tellus import MagneticGrid

    grid = MagneticGrid("a7_tmi")           # loads the GeoTIFF into memory once
    grid.sample(52.8, -6.3)                 # one point  -> nT
    grid.sample_track(lats, lons)           # a flight path -> array of nT

The grid is held in memory as a plain numpy array, so sampling thousands of
points along a trajectory costs almost nothing after the first load.

Two things this deliberately does NOT do
----------------------------------------
It never returns 0.0 to mean "no data". Anywhere off the edge of the survey,
or over a cell the survey did not cover, you get NaN. A silent zero would look
like a real magnetic reading and quietly corrupt a navigation fix.

It never falls back to nearest-neighbour. Values come from bilinear
interpolation of the four surrounding cells, so a track crossing the grid
returns a smooth profile rather than a 50 m staircase.
"""

from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from pyproj import Transformer

from tellus import products

WGS84 = "EPSG:4326"


class MagneticGrid:
    """A magnetic anomaly grid, loaded once and sampled many times."""

    def __init__(self, source, units: str | None = None, name: str | None = None):
        """Load a grid.

        source: a product key from products.py ("a7_tmi"), or a path to a
                GeoTIFF that scripts/fetch_tellus.py produced.
        """
        if isinstance(source, str) and source in products.PRODUCTS:
            product = products.get(source)
            path = product.tif_path
            units = units or product.units
            name = name or product.description
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} does not exist yet.\n"
                    f"Run:  python scripts/fetch_tellus.py --product {source}"
                )
        else:
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"{path} does not exist.")

        with rasterio.open(path) as src:
            self.values = src.read(1).astype("float64")
            self.transform = src.transform
            self.crs = src.crs
            nodata = src.nodata

        # Everything that is not a real reading becomes NaN, once, here.
        if nodata is not None and np.isfinite(nodata):
            self.values[self.values == nodata] = np.nan
        self.values[np.abs(self.values) > 1e6] = np.nan

        self.units = units or "nT"
        self.name = name or path.name
        self.path = path
        self._to_grid = Transformer.from_crs(WGS84, self.crs, always_xy=True)
        self._to_wgs84 = Transformer.from_crs(self.crs, WGS84, always_xy=True)

    # ------------------------------------------------------------------
    # Describing the grid
    # ------------------------------------------------------------------

    @property
    def shape(self) -> tuple[int, int]:
        """(rows, columns)."""
        return self.values.shape

    @property
    def resolution(self) -> tuple[float, float]:
        """Cell size in metres, (x, y). Always positive."""
        return abs(self.transform.a), abs(self.transform.e)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Outer edges in the grid's own CRS: (left, bottom, right, top)."""
        rows, cols = self.shape
        left, top = self.transform @ (0, 0)
        right, bottom = self.transform @ (cols, rows)
        return left, bottom, right, top

    @property
    def bounds_wgs84(self) -> tuple[float, float, float, float]:
        """Outer edges as (min_lon, min_lat, max_lon, max_lat).

        The grid is a rectangle in ITM but a slightly skewed shape in lat/lon,
        so this is the bounding box of all four corners, not just two of them.
        """
        left, bottom, right, top = self.bounds
        xs = [left, right, left, right]
        ys = [bottom, bottom, top, top]
        lons, lats = self._to_wgs84.transform(xs, ys)
        return min(lons), min(lats), max(lons), max(lats)

    def __repr__(self) -> str:
        rows, cols = self.shape
        return f"<MagneticGrid {self.name!r} {cols}x{rows} cells, {self.units}>"

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, lat: float, lon: float) -> float:
        """Anomaly value at one WGS84 point. NaN if there is no coverage."""
        return float(self.sample_track([lat], [lon])[0])

    def sample_track(self, lats, lons) -> np.ndarray:
        """Anomaly values along a trajectory. NaN wherever there is no coverage.

        Vectorised - pass the whole track at once rather than looping.
        """
        lats = np.asarray(lats, dtype="float64")
        lons = np.asarray(lons, dtype="float64")
        if lats.shape != lons.shape:
            raise ValueError(f"lats and lons differ in shape: {lats.shape} vs {lons.shape}")

        x, y = self._to_grid.transform(lons, lats)
        return self._sample_native(np.asarray(x), np.asarray(y))

    def _sample_native(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Bilinear sample at coordinates already in the grid's own CRS."""
        rows, cols = self.shape

        # Fractional pixel position. ~transform gives continuous (col, row)
        # measured from the grid's top-left *corner*, so subtracting half a
        # cell puts whole numbers exactly on cell centres - which is what
        # bilinear interpolation needs to anchor to.
        col_f, row_f = ~self.transform @ (x, y)
        col_f = np.asarray(col_f) - 0.5
        row_f = np.asarray(row_f) - 0.5

        # Outside the ring of cell centres there is nothing to interpolate
        # between, so those points are NaN rather than extrapolated.
        inside = (row_f >= 0) & (row_f <= rows - 1) & (col_f >= 0) & (col_f <= cols - 1)

        out = np.full(row_f.shape, np.nan, dtype="float64")
        if not inside.any():
            return out

        rr = row_f[inside]
        cc = col_f[inside]

        # Top-left cell of each 2x2 block, clamped so a point sitting exactly
        # on the last row or column still has a neighbour to pair with.
        r0 = np.clip(np.floor(rr).astype(int), 0, rows - 2)
        c0 = np.clip(np.floor(cc).astype(int), 0, cols - 2)
        dr = rr - r0
        dc = cc - c0

        v00 = self.values[r0, c0]
        v01 = self.values[r0, c0 + 1]
        v10 = self.values[r0 + 1, c0]
        v11 = self.values[r0 + 1, c0 + 1]

        top = v00 * (1 - dc) + v01 * dc
        bottom = v10 * (1 - dc) + v11 * dc
        # NaN in any of the four corners propagates through this arithmetic on
        # its own, which is what we want: no guessing at the survey edge.
        out[inside] = top * (1 - dr) + bottom * dr
        return out

    # ------------------------------------------------------------------
    # Cutting out a smaller area
    # ------------------------------------------------------------------

    def subset(self, bbox, margin_m: float = 1000.0) -> "MagneticGrid":
        """Cut out a smaller grid covering a lat/lon box.

        bbox:     (min_lon, min_lat, max_lon, max_lat)
        margin_m: extra metres kept around the edge, so points near the
                  boundary still have neighbours for interpolation.

        Returns a new MagneticGrid holding only that area. Block A7 in full is
        about 30 MB in memory; a 20 km box cut out of it is under 1 MB, which
        matters if you are holding several regions at once or passing the array
        to something else.

        This is about memory, not speed. Sampling time is dominated by the
        WGS84 -> grid reprojection rather than by the size of the array, so a
        subset samples at roughly the same rate as the full grid.
        """
        min_lon, min_lat, max_lon, max_lat = bbox
        corner_lons = [min_lon, max_lon, min_lon, max_lon]
        corner_lats = [min_lat, min_lat, max_lat, max_lat]
        xs, ys = self._to_grid.transform(corner_lons, corner_lats)

        left = min(xs) - margin_m
        right = max(xs) + margin_m
        bottom = min(ys) - margin_m
        top = max(ys) + margin_m

        # Convert to whole rows and columns, clipped to what we actually have.
        rows, cols = self.shape
        c_lo, r_lo = ~self.transform @ (left, top)
        c_hi, r_hi = ~self.transform @ (right, bottom)
        c_lo = int(np.clip(np.floor(c_lo), 0, cols - 1))
        r_lo = int(np.clip(np.floor(r_lo), 0, rows - 1))
        c_hi = int(np.clip(np.ceil(c_hi), c_lo + 1, cols))
        r_hi = int(np.clip(np.ceil(r_hi), r_lo + 1, rows))

        clipped = MagneticGrid.__new__(MagneticGrid)
        clipped.values = self.values[r_lo:r_hi, c_lo:c_hi].copy()
        # Same cell size, origin shifted to the top-left of the cut-out.
        clipped.transform = self.transform @ Affine.translation(c_lo, r_lo)
        clipped.crs = self.crs
        clipped.units = self.units
        clipped.name = f"{self.name} (subset)"
        clipped.path = self.path
        clipped._to_grid = self._to_grid
        clipped._to_wgs84 = self._to_wgs84
        return clipped


def straight_track(start, end, n: int = 300):
    """Points evenly spaced along a straight line between two lat/lon pairs.

        lats, lons = straight_track((52.5, -6.5), (52.4, -6.3), n=300)

    Straight in latitude/longitude, which over a 20 km hop is indistinguishable
    from a straight line on the ground. Good enough for a simulated trajectory;
    it is not a great-circle route and is not meant for long distances.
    """
    (lat0, lon0), (lat1, lon1) = start, end
    return np.linspace(lat0, lat1, n), np.linspace(lon0, lon1, n)
