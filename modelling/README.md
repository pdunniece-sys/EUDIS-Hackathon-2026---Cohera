# Modelling — simulation and data tooling

The simulation and modelling part of the magnetic anomaly navigation project.
Everything here is self-contained; see the [repo README](../README.md) for how it
fits with the rest of the work.

Two independent halves, one for each kind of data:

- **Tellus** — the Irish airborne magnetic anomaly *map*, sampled along a path.
- **QuSpin** — a raw magnetometer *flight log*, parsed into usable arrays.

They share nothing but this folder. Each has its own module, scripts and notebook.

All commands below are run from this directory.

---

# Tellus magnetic anomaly sampling

Sample the Irish Tellus airborne magnetic anomaly field along a flight path,
for a magnetic-anomaly navigation prototype.

```python
from tellus import MagneticGrid

grid = MagneticGrid("a7_tmi")
grid.sample(52.45, -6.40)          # -> 74.53  (nT)
grid.sample_track(lats, lons)      # -> array of nT along a trajectory
```

## Getting started

```bash
# from the repo root, once
python3 -m venv .venv
source .venv/bin/activate
pip install -e "./modelling[dev]"

# then from this directory
python scripts/fetch_tellus.py     # downloads ~93 MB, takes a few minutes
python scripts/inspect_grid.py     # prints the grid's details, writes plots
pytest                             # 16 checks
jupyter lab notebooks/explore.ipynb
```

## Layout

```
scripts/fetch_tellus.py     download a grid and convert it to a GeoTIFF
scripts/inspect_grid.py     print the grid's details and draw a quick look
src/tellus/products.py      which grids exist and where they live
src/tellus/sample.py        the MagneticGrid class - all the sampling
tests/test_sample.py        checks that sampling behaves correctly
notebooks/explore.ipynb     load, sample, plot, run the checks

scripts/inspect_quspin.py   summarise a flight log and draw an overview
src/quspin/parse.py         the log parser - format notes are in its docstring
tests/test_quspin.py        checks that parsing is correct
notebooks/quspin.ipynb      raw text to usable arrays, step by step

data/                       downloads, grids and logs (not in git)
outputs/                    generated plots (not in git)
```

Nothing under `data/` or `outputs/` is tracked. The Tellus grids come back with
`scripts/fetch_tellus.py`; the QuSpin logs have to be copied into
`data/QuSpin/` by hand.

## The one thing to know about the data

Each Tellus download contains the same grid three times, and only one of them
is usable:

| Format | What it is |
|---|---|
| `.gxf` | **The actual numbers**, in nanotesla. Plain text. This is what we read. |
| `.grd` | The same numbers in Geosoft's private binary format. GDAL cannot open it. |
| `.tif` | **A picture** of the grid. RGB pixels, values already discarded. |

The `.tif` is the familiar format and the tempting one, but it is a rendered
image with hill-shading baked in and a non-linear colour scale, so the values
cannot be recovered from it. GSI's own readme confirms this: *".tiff files are
a georeferenced coloured raster"*, while *".gxf are ASCII grid files and are an
interrogatable raster file"*.

Their ArcGIS map service has the same limitation — querying a point returns
`{"RGB.Red": 159, ...}`, not nanotesla. There is no WCS and no ImageServer, so
downloading the file is the only route to values.

`scripts/fetch_tellus.py` therefore downloads the zip, extracts the `.gxf`, and
converts it once into a float32 GeoTIFF with a coordinate system attached.
Everything else reads that GeoTIFF.

## Coordinate system — an assumption, not a reading

The `.gxf` files contain **no projection information at all**. There is no
`#PRJ` block to read, so the CRS has to be supplied. We assign
**EPSG:2157 (IRENET95 / Irish Transverse Mercator)**, on three pieces of
evidence:

1. GSI's own map service for this exact product reports `wkid: 2157`.
2. The data.gov.ie dataset is titled "… Ireland (ROI/NI) **ITM** …".
3. The grid origins (653 050 E, 593 800 N for block A7) only make sense in ITM.
   Irish Grid (EPSG:29903) eastings would be below 400 000.

This is set in one place, `IRISH_TRANSVERSE_MERCATOR` in `src/tellus/products.py`.

## What was verified

- **Orientation.** The classic failure here is a north/south flip. The grid was
  rendered for a fixed bounding box and compared against GSI's own rendering of
  the same ground from their map service. The stepped survey boundary and every
  anomaly line up. `test_grid_is_stored_north_up` and a check in
  `fetch_tellus.py` both assert row 0 is the northern edge.
- **Values.** Block A7 reads −326.5 to 779.9 nT, matching the `#ZMINIMUM` and
  `#ZMAXIMUM` in the source file exactly. The mean is 0.2 nT, which is what an
  anomaly field should be.
- **Interpolation.** Sampling a cell centre returns that cell's value;
  sampling halfway between two cells returns their average, confirming bilinear
  rather than nearest-neighbour.
- **No silent zeros.** Off the grid, and over cells the survey did not cover,
  sampling returns `NaN`.

## Available grids

`python scripts/fetch_tellus.py --list`

| Key | Area | Zip |
|---|---|---|
| `a7_tmi` | Wicklow / Wexford / Carlow — the default | 93 MB |
| `a7_1vd`, `a7_tdr` | Derivatives of the same block | same zip |
| `a9_tmi` | Cork — smaller, if you want a faster test | 43 MB |
| `merge_tmi` | All of Ireland | 875 MB |

All are 50 m cells. Add another by adding one `Product` to `src/tellus/products.py`.

## Licence and attribution

The data is published by Geological Survey Ireland under
**Creative Commons Attribution 4.0 International (CC BY 4.0)**. Anything built
on it must carry:

> Contains Irish Public Sector Data (Geological Survey Ireland) licensed under a
> Creative Commons Attribution 4.0 International (CC BY 4.0) licence.

The national merged grid (`merge_tmi`) also covers Northern Ireland with data
from the Geological Survey of Northern Ireland, which is licensed under the
**UK Open Government Licence v3.0** and needs its own attribution.

Source: <https://www.gsi.ie/data-and-maps/geophysics/>

## Survey details

Flown 60 m above ground, along lines 200 m apart, sampled roughly every 6 m.
Gridded by GSI at 50 m using minimum curvature. The values are the total
magnetic intensity *anomaly* — diurnally corrected and with the International
Geomagnetic Reference Field removed — so they sit around zero rather than
around Ireland's ~49 000 nT background field.


---

# QuSpin flight logs

A QuSpin QTFM scalar magnetometer flown on a quadcopter, logged raw off the
serial port. Completely separate from the Tellus half.

```python
from quspin import read_log

log = read_log()              # data/QuSpin/*.txt
log.summary()
flight = log.airborne()       # drop the ground time at each end
flight.smoothed_field()       # with the 60 Hz interference filtered out
```

```bash
python scripts/inspect_quspin.py --airborne
jupyter lab notebooks/quspin.ipynb
```

## What the raw file looks like

No header, no columns. One line per reading, fields marked by punctuation:

```
223637639!53098.490_@066>222476s114a17.09b-47.85c-1025.88
    |         |        |     |    |   |
    |         |        |     |    |   +-- one auxiliary reading
    |         |        |     |    +------ signal level
    |         |        |     +----------- a second clock, milliseconds
    |         |        +----------------- sample counter, 0-249
    |         +-------------------------- TOTAL FIELD in nT
    +------------------------------------ timestamp in microseconds
```

The counter wraps once a second, which is how you know the rate is 250 Hz.
Only one auxiliary reading fits per line, so four channels take turns and each
arrives at 62.5 Hz: accelerometer (mg), gyroscope (deg/s), a 3-axis
magnetometer (uT) and sensor temperature (C). Mixed in once a second are
`GNSSFIX` lines carrying latitude and longitude as integers scaled by 10^7.

`src/quspin/parse.py` documents the format in full, along with exactly what the
parser changes and what it leaves untouched.

## Three things to know before using the numbers

1. **It is total field, not an anomaly.** About 53,000 nT, because Earth's
   background is still in it. The Tellus grid has that removed. The two are not
   directly comparable without subtracting a reference field.
2. **The 3-axis magnetometer disagrees with the scalar reading** — about 37 uT
   against 53 uT. It is an uncalibrated auxiliary sensor, useful for attitude,
   not as a measurement of the field.
3. **GNSS altitude is above the WGS84 ellipsoid**, not sea level. In southern
   Ontario the two differ by roughly 36 m.

## The 60 Hz problem

The raw trace carries about 5 nT of noise, essentially all of it at exactly
60 Hz — North American mains, picked up from power lines near the survey. It
does not correlate with heading or rotation rate, so it is environmental rather
than the aircraft.

Geology cannot change faster than the drone flies over it, so everything real
sits below about 0.1 Hz. `smoothed_field()` low-passes at 1 Hz, which removes
the interference and leaves the signal: 4.85 nT of noise out, 7.28 nT of
geology kept.

## What the example log contains

`T2M0-00N6_Data2.txt`, 8 June 2026, King Township, Ontario (44.08 N, 79.81 W).
21.6 minutes, of which 15.2 airborne at 42 m above launch. A lawnmower survey
of a 700 x 330 m plot, 6.9 km of track at 6.6 m/s.

Repeatability, measured by binning into 25 m cells and comparing passes over
the same ground:

| | |
|---|---|
| spread within a cell (measurement error) | **0.7 nT** median |
| spread between cells (real geology) | **7.6 nT** |
| signal to error | **12 : 1** |

That is a good sensor result. The limitation is the area, not the instrument:
700 x 330 m with 33 nT of total variation is too small a patch to fix a
position against a map.
