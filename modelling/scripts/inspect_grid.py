#!/usr/bin/env python3
"""Describe a downloaded grid and draw a quick look at it.

    python scripts/inspect_grid.py                # default block (A7)
    python scripts/inspect_grid.py -p a9_tmi

Prints the grid's geometry and statistics, then writes two images to outputs/:

    <product>_map.png        the grid itself, drawn two ways
    <product>_histogram.png  the spread of values

The map is drawn twice on purpose. The left panel is the honest one: a
diverging blue/red scale with white pinned to zero, so you can see at a glance
which way an anomaly points. The right panel copies the rainbow scale
Geological Survey Ireland use in their own published images, which is there for
one job - comparing our picture against theirs to confirm the grid is the right
way up. That colour scale is read straight out of the downloaded zip, so it is
genuinely theirs and not an approximation.
"""

import argparse
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import matplotlib
import numpy as np
import rasterio

matplotlib.use("Agg")  # write files, never try to open a window
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tellus import MagneticGrid, products  # noqa: E402

INK = "#3d3d3a"
MUTED = "#8a8a82"


def describe(product) -> MagneticGrid:
    """Print everything worth knowing about the grid before trusting it."""
    grid = MagneticGrid(product.key)

    # A few properties only exist on the file, not on the loaded array.
    with rasterio.open(product.tif_path) as src:
        dtype, nodata = src.dtypes[0], src.nodata

    rows, cols = grid.shape
    left, bottom, right, top = grid.bounds
    min_lon, min_lat, max_lon, max_lat = grid.bounds_wgs84
    res_x, res_y = grid.resolution

    valid = grid.values[np.isfinite(grid.values)]
    covered = 100 * valid.size / grid.values.size

    print(f"{product.description}\n")
    print(f"  file          {product.tif_path.relative_to(products.ROOT)}")
    print(f"  CRS           {grid.crs}  ({grid.crs.to_string()})")
    print(f"  dimensions    {cols} columns x {rows} rows")
    print(f"  cell size     {res_x:g} m x {res_y:g} m")
    print(f"  dtype         {dtype}")
    print(f"  nodata        {nodata}")
    print()
    print(f"  bounds (ITM)  east  {left:>10,.0f} to {right:>10,.0f} m")
    print(f"                north {bottom:>10,.0f} to {top:>10,.0f} m")
    print(f"  bounds (WGS84) lon  {min_lon:>10.4f} to {max_lon:>10.4f}")
    print(f"                 lat  {min_lat:>10.4f} to {max_lat:>10.4f}")
    print()
    print(f"  valid cells   {valid.size:,} of {grid.values.size:,}  ({covered:.1f}% covered)")
    print(f"  min           {valid.min():>10.2f} {product.units}")
    print(f"  max           {valid.max():>10.2f} {product.units}")
    print(f"  mean          {valid.mean():>10.2f} {product.units}")
    print(f"  std           {valid.std():>10.2f} {product.units}")
    return grid


def geosoft_colormap(zip_path: Path):
    """Read GSI's own 'Clra 32' colour scale out of the downloaded zip.

    They ship it as a QGIS style file, which stores the scale as a list of
    positions between 0 and 1 with an RGB colour at each. Returns None if the
    zip is not there, in which case we simply skip that panel.
    """
    if not zip_path.exists():
        return None
    try:
        with zipfile.ZipFile(zip_path) as archive:
            name = next(n for n in archive.namelist() if n.endswith("Geosoft_Clra_32_QGIS.xml"))
            root = ET.fromstring(archive.read(name))
    except (StopIteration, KeyError, ET.ParseError):
        return None

    props = {p.get("k"): p.get("v") for p in root.iter("prop")}

    def rgb(value):
        r, g, b = (int(c) / 255 for c in value.split(",")[:3])
        return r, g, b

    stops = [(0.0, rgb(props["color1"]))]
    for stop in props["stops"].split(":"):
        position, colour = stop.split(";")
        stops.append((float(position), rgb(colour)))
    stops.append((1.0, rgb(props["color2"])))
    return LinearSegmentedColormap.from_list("geosoft_clra32", stops)


def plot_map(grid, product, path: Path) -> None:
    valid = grid.values[np.isfinite(grid.values)]

    # A handful of extreme cells would wash out everything else, so the colour
    # scale is clipped to the middle 98% of values and kept symmetric about
    # zero. Values beyond the limits still render, just saturated.
    limit = float(np.percentile(np.abs(valid), 99))
    left, bottom, right, top = grid.bounds
    extent = (left / 1000, right / 1000, bottom / 1000, top / 1000)  # km, for readable ticks

    geosoft = geosoft_colormap(product.zip_path)
    panels = [("Diverging, white at zero", "RdBu_r", TwoSlopeNorm(0, -limit, limit))]
    if geosoft is not None:
        panels.append(("GSI's own scale, for comparison", geosoft, None))

    fig, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 8.5))
    axes = np.atleast_1d(axes)

    for ax, (subtitle, cmap, norm) in zip(axes, panels):
        image = ax.imshow(
            grid.values,
            extent=extent,
            origin="upper",   # row 0 is the north edge - see the check in fetch_tellus.py
            cmap=cmap,
            norm=norm,
            vmin=None if norm else -limit,
            vmax=None if norm else limit,
            interpolation="nearest",
        )
        ax.set_title(subtitle, fontsize=10, color=MUTED, pad=8)
        ax.set_xlabel("ITM easting (km)", fontsize=9, color=MUTED)
        ax.set_ylabel("ITM northing (km)", fontsize=9, color=MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#dcdcd6")
        bar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03, extend="both")
        bar.set_label(product.units, fontsize=9, color=MUTED)
        bar.ax.tick_params(colors=MUTED, labelsize=8)

    fig.suptitle(product.description, fontsize=13, color=INK, y=0.97)
    fig.text(0.5, 0.015,
             "North is up. Compare the right panel with GSI's published map to confirm it.",
             ha="center", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)
    print(f"  wrote {path.relative_to(products.ROOT)}")


def plot_histogram(grid, product, path: Path) -> None:
    valid = grid.values[np.isfinite(grid.values)]
    limit = float(np.percentile(np.abs(valid), 99.5))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(valid, bins=250, range=(-limit, limit), color="#4a7fb5", linewidth=0)
    ax.axvline(0, color=MUTED, linewidth=1)

    ax.set_xlabel(f"anomaly ({product.units})", fontsize=10, color=MUTED)
    ax.set_ylabel("number of cells", fontsize=10, color=MUTED)
    ax.set_title(f"Spread of values - {product.description}", fontsize=11, color=INK, pad=10)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color="#ececE6", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#dcdcd6")

    fig.text(0.99, 0.02, f"clipped to +/-{limit:.0f} {product.units}",
             ha="right", fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)
    print(f"  wrote {path.relative_to(products.ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-p", "--product", default=products.DEFAULT_PRODUCT)
    args = parser.parse_args()

    product = products.get(args.product)
    grid = describe(product)

    products.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print()
    plot_map(grid, product, products.OUTPUT_DIR / f"{product.key}_map.png")
    plot_histogram(grid, product, products.OUTPUT_DIR / f"{product.key}_histogram.png")


if __name__ == "__main__":
    main()
