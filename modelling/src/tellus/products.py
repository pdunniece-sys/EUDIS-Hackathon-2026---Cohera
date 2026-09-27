"""Where the Tellus magnetic grids live, and what is inside each download.

Every entry here was checked against the live server rather than guessed.
Adding another survey block means adding one Product below - nothing else
in the codebase needs to change.
"""

from dataclasses import dataclass
from pathlib import Path

# Project layout. Everything resolves from here so scripts work from any cwd.
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"          # zips exactly as downloaded
PROCESSED_DIR = ROOT / "data" / "processed"  # GeoTIFFs we converted
OUTPUT_DIR = ROOT / "outputs"            # plots

# Irish Transverse Mercator.
#
# IMPORTANT, and the one place we are not simply reading the file: the .gxf
# files contain no projection information at all - no #PRJ block, nothing.
# The coordinate system has to be supplied by us. EPSG:2157 is correct because:
#
#   1. GSI's own map service for this exact product reports wkid 2157
#      (gsi.geodata.gov.ie/server/rest/services/Geophysics/...Magnetic_Intensity...)
#   2. the data.gov.ie dataset is titled "... Ireland (ROI/NI) ITM ..."
#   3. the grid origins (e.g. 653050 E, 593800 N) only make sense in ITM.
#      Irish Grid (EPSG:29903) eastings would be under 400 000.
#
# If a future product turns out to be on Irish Grid, override it per-product.
IRISH_TRANSVERSE_MERCATOR = "EPSG:2157"


@dataclass(frozen=True)
class Product:
    """One grid we can download, and enough detail to turn it into a GeoTIFF."""

    key: str            # what you type on the command line
    description: str
    units: str          # nT, nT/m or radians
    zip_url: str
    gxf_member: str     # path of the .gxf *inside* the zip
    crs: str = IRISH_TRANSVERSE_MERCATOR

    @property
    def zip_path(self) -> Path:
        return RAW_DIR / self.zip_url.rsplit("/", 1)[-1]

    @property
    def gxf_path(self) -> Path:
        return RAW_DIR / Path(self.gxf_member).name

    @property
    def tif_path(self) -> Path:
        return PROCESSED_DIR / f"{self.key}.tif"


_A7 = "https://gsi.geodata.gov.ie/downloads/Geophysics/Data/GSI_Tellus_A7_MAG_GRIDS_2019B.zip"
_A9 = "https://gsi.geodata.gov.ie/downloads/Geophysics/Data/GSI_Tellus_A9_MAG_GRIDS_2022.zip"
_MERGE = "https://gsi.geodata.gov.ie/downloads/Geophysics/Data/GSI_Tellus_MAG_MERGE_GRIDS_2022.zip"

PRODUCTS = {
    p.key: p
    for p in [
        # Block A7 - south-east Ireland (Wicklow/Wexford/Carlow), 93 MB zip.
        # The default: strong magnetic variation over the Leinster Granite and
        # the Avoca volcanic belt, without the size of the national merge.
        Product(
            key="a7_tmi",
            description="Block A7 total magnetic intensity anomaly (microlevelled)",
            units="nT",
            zip_url=_A7,
            gxf_member="A7_MAG_GRIDS_2019B/GXF/MAG_A7_MIC_2019B.gxf",
        ),
        Product(
            key="a7_1vd",
            description="Block A7 first vertical derivative (upward continued 150 m, pole reduced)",
            units="nT/m",
            zip_url=_A7,
            gxf_member="A7_MAG_GRIDS_2019B/GXF/MAG_A7_2019B_UP150_pole30_1VD.gxf",
        ),
        Product(
            key="a7_tdr",
            description="Block A7 tilt derivative (upward continued 150 m, pole reduced)",
            units="radians",
            zip_url=_A7,
            gxf_member="A7_MAG_GRIDS_2019B/GXF/MAG_A7_2019B_UP150_pole30_TDR.gxf",
        ),
        # Block A9 - Cork. Smallest useful block (43 MB) if you want a quick test.
        Product(
            key="a9_tmi",
            description="Block A9 total magnetic intensity anomaly (microlevelled)",
            units="nT",
            zip_url=_A9,
            gxf_member="MAG_A9_GRIDS_2022/GXF/A9_MIC_MAG_2022.gxf",
        ),
        # All of Ireland. 875 MB zip, 451 MB of GXF text inside - expect the
        # download and the conversion to take a while.
        Product(
            key="merge_tmi",
            description="National merged total magnetic intensity anomaly, all Ireland",
            units="nT",
            zip_url=_MERGE,
            gxf_member="MAG_MERGE_2022/GXF/TELLUS_MAG_MERGE_2022_TMI.gxf",
        ),
    ]
}

DEFAULT_PRODUCT = "a7_tmi"


def get(key: str) -> Product:
    """Look up a product, with a helpful message if the key is wrong."""
    try:
        return PRODUCTS[key]
    except KeyError:
        available = "\n".join(f"  {k:12s} {p.description}" for k, p in PRODUCTS.items())
        raise SystemExit(f"Unknown product {key!r}. Available:\n{available}")


# A 17 km line across the strongest feature in block A7: it starts over the
# magnetic low and crosses into the high over the Wexford volcanic belt, so a
# profile along it swings by roughly 350 nT. Both the test and the notebook use
# it as a known-interesting flight path. Endpoints are (latitude, longitude).
A7_DEMO_TRACK = ((52.502, -6.498), (52.383, -6.325))
