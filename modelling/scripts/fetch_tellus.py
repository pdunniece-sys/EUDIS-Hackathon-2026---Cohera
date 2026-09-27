#!/usr/bin/env python3
"""Download a Tellus magnetic grid and convert it into a GeoTIFF.

    python scripts/fetch_tellus.py                 # default block (A7)
    python scripts/fetch_tellus.py --list          # show what else is available
    python scripts/fetch_tellus.py -p a9_tmi       # a different block

Three steps. Each one is skipped if its output is already sitting there, so
re-running after an interruption picks up where it left off.

    1. download   the zip from Geological Survey Ireland   -> data/raw/
    2. extract    the .gxf grid out of the zip             -> data/raw/
    3. convert    the .gxf into a float32 GeoTIFF          -> data/processed/

Why step 3 exists
-----------------
A .gxf file is a plain-text grid: a small header, then the cell values written
out as text, row by row. That is portable and human-readable, but it is slow
to parse every time you open it, and the format has nowhere to record which
coordinate system the numbers belong to. Converting once to a GeoTIFF fixes
both - the values become binary float32, and the CRS travels with the file.
"""

import argparse
import sys
import zipfile
from pathlib import Path

import numpy as np
import rasterio
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tellus import products  # noqa: E402

CHUNK = 1 << 20  # 1 MB


def human(n_bytes: float) -> str:
    return f"{n_bytes / 1e6:.1f} MB"


def download(url: str, dest: Path) -> None:
    """Stream a URL to disk, resuming a partial file rather than restarting."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        head = requests.head(url, timeout=30, allow_redirects=True)
        head.raise_for_status()
    except requests.RequestException as exc:
        raise SystemExit(
            f"Could not reach the download server.\n"
            f"  url:   {url}\n"
            f"  error: {exc}\n\n"
            f"Geological Survey Ireland's server does go down occasionally. Check\n"
            f"https://www.gsi.ie/data-and-maps/geophysics/ in a browser; if that\n"
            f"loads but this does not, the file may have been moved or renamed."
        )

    total = int(head.headers.get("Content-Length", 0))
    have = dest.stat().st_size if dest.exists() else 0

    if total and have == total:
        print(f"  already downloaded ({human(total)})")
        return
    if have > total > 0:
        print("  local file is larger than the server's - starting again")
        dest.unlink()
        have = 0

    headers = {"Range": f"bytes={have}-"} if have else {}
    if have:
        print(f"  resuming from {human(have)}")

    try:
        with requests.get(url, headers=headers, stream=True, timeout=60) as response:
            response.raise_for_status()
            # 206 means the server honoured our Range request and we can append.
            # A plain 200 means it ignored it and is sending the whole file.
            if have and response.status_code != 206:
                print("  server will not resume - downloading from the start")
                have = 0
            mode = "ab" if have else "wb"

            with open(dest, mode) as handle:
                written = have
                for chunk in response.iter_content(CHUNK):
                    handle.write(chunk)
                    written += len(chunk)
                    if total:
                        pct = 100 * written / total
                        print(
                            f"\r  {human(written)} / {human(total)}  ({pct:4.1f}%)",
                            end="",
                            flush=True,
                        )
            print()
    except requests.RequestException as exc:
        raise SystemExit(
            f"\nDownload failed after {human(dest.stat().st_size if dest.exists() else 0)}.\n"
            f"  error: {exc}\n\n"
            f"The partial file was kept - run this command again to resume."
        )

    final = dest.stat().st_size
    if total and final != total:
        raise SystemExit(
            f"Download ended early: got {human(final)}, expected {human(total)}.\n"
            f"Run this command again to resume."
        )


def extract(zip_path: Path, member: str, dest: Path) -> None:
    """Pull one .gxf out of the zip without unpacking the rest of it."""
    if dest.exists():
        print(f"  already extracted ({human(dest.stat().st_size)})")
        return

    try:
        with zipfile.ZipFile(zip_path) as archive:
            try:
                info = archive.getinfo(member)
            except KeyError:
                names = "\n".join(f"    {n}" for n in archive.namelist() if n.endswith(".gxf"))
                raise SystemExit(
                    f"{member} is not in {zip_path.name}.\n"
                    f"The .gxf files it does contain:\n{names}\n\n"
                    f"Update the gxf_member for this product in src/tellus/products.py."
                )
            print(f"  extracting {human(info.file_size)} of text...")
            with archive.open(info) as src, open(dest, "wb") as out:
                while block := src.read(CHUNK):
                    out.write(block)
    except zipfile.BadZipFile:
        raise SystemExit(
            f"{zip_path} is not a valid zip - the download was probably truncated.\n"
            f"Delete it and run this command again."
        )


def convert(gxf_path: Path, tif_path: Path, crs: str) -> None:
    """Read the GXF grid and write it out as a GeoTIFF with a CRS attached."""
    if tif_path.exists():
        print(f"  already converted ({human(tif_path.stat().st_size)})")
        return

    tif_path.parent.mkdir(parents=True, exist_ok=True)
    print("  reading the GXF (this is the slow part - it is text)...")

    with rasterio.open(gxf_path) as src:
        values = src.read(1).astype("float32")
        profile = src.profile
        gxf_nodata = src.nodata

    # GXF marks empty cells with a sentinel value, normally -1e32. Turn those
    # into NaN so that nothing downstream can mistake them for a real reading
    # of zero nT. Anything beyond the magnetic range is treated as a sentinel
    # too, in case a grid uses a different one.
    blank = ~np.isfinite(values) | (np.abs(values) > 1e6)
    if gxf_nodata is not None:
        blank |= values == np.float32(gxf_nodata)
    values[blank] = np.nan
    print(f"  {blank.sum():,} of {values.size:,} cells are outside the survey area")

    profile.update(
        driver="GTiff",
        dtype="float32",
        count=1,
        crs=crs,
        nodata=np.nan,
        compress="deflate",
        predictor=3,   # floating-point predictor - much better ratios on smooth grids
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )
    with rasterio.open(tif_path, "w", **profile) as dst:
        dst.write(values, 1)


def verify(tif_path: Path, units: str) -> None:
    """Open the result and confirm it is a usable, north-up, georeferenced raster."""
    with rasterio.open(tif_path) as src:
        values = src.read(1, masked=True)
        finite = values.compressed()
        if finite.size == 0:
            raise SystemExit(f"{tif_path.name} opened but contains no valid data.")

        # In a north-up raster the y pixel size is negative: row 0 is the
        # northernmost row. If this is ever positive the grid is upside down.
        if src.transform.e >= 0:
            raise SystemExit(
                f"{tif_path.name} is stored south-up (transform.e = {src.transform.e}).\n"
                f"Every map and every sample drawn from it would be flipped."
            )

        print(f"  {src.width} x {src.height} cells at {abs(src.transform.a):g} m")
        print(f"  CRS {src.crs}")
        print(f"  {finite.min():.1f} to {finite.max():.1f} {units}, "
              f"mean {finite.mean():.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-p", "--product", default=products.DEFAULT_PRODUCT)
    parser.add_argument("--list", action="store_true", help="list available products and exit")
    args = parser.parse_args()

    if args.list:
        for key, product in products.PRODUCTS.items():
            print(f"{key:12s} {product.description} [{product.units}]")
        return

    product = products.get(args.product)
    print(f"{product.description}\n")

    print(f"1. download  {product.zip_path.name}")
    download(product.zip_url, product.zip_path)

    print(f"\n2. extract   {product.gxf_path.name}")
    extract(product.zip_path, product.gxf_member, product.gxf_path)

    print(f"\n3. convert   {product.tif_path.name}")
    convert(product.gxf_path, product.tif_path, product.crs)

    print("\n4. verify")
    verify(product.tif_path, product.units)

    print(f"\nReady: {product.tif_path.relative_to(products.ROOT)}")


if __name__ == "__main__":
    main()
