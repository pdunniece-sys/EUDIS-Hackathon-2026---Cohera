# Modelling — simulation and data tooling

The simulation and modelling part of the magnetic anomaly navigation project.
Everything here is self-contained; see the [repo README](../README.md) for how it
fits with the rest of the work.

Three independent parts:

- **Tellus** — the Irish airborne magnetic anomaly *map*, sampled along a path.
- **QuSpin** — a raw magnetometer *flight log*, parsed into usable arrays.
- **INS** — how far the drone drifts on inertial sensors alone once GNSS stops.
- **Sensor** — what a magnetometer on the drone would actually read.

Each has its own module, scripts and tests, and they share only the data types
they have to.

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
notebooks/dynamic.ipynb     the processed CSV export, and what it can tell us

scripts/run_ins_demo.py     drift after a GNSS cut, per IMU grade
src/ins/mechanisation.py    the strapdown integrator - frame conventions here
src/ins/trajectory.py       synthetic flight paths and their perfect IMU
src/ins/errors.py           IMU error model and the three grade presets
src/ins/loaders.py          ArduPilot .BIN and QuSpin CSV readers
src/ins/types.py            Trajectory, ImuSamples, InsState
src/ins/rotations.py        quaternion helpers
tests/test_ins.py           checks the analytic drift laws, not just that it runs

scripts/run_opm_demo.py     emulated OPM vs vector magnetometer over A7
src/sensor/field.py         IGRF + anomaly + diurnal + interference -> the true field
src/sensor/opm.py           the QTFM-like scalar OPM and its error model
src/sensor/vector.py        the RM3100-like vector magnetometer, for comparison
src/sensor/interference.py  the drone's own field. Zero unless switched on
src/sensor/timing.py        latency, anti-alias filtering and decimation
src/tellus/continuation.py  move a grid to flight altitude (upward only)
tests/test_sensor.py        checks the emulator against the measured flight

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

---

# INS drift — how long can we coast without GNSS?

Magnetic matching gives a position fix, but not continuously and not instantly.
Between fixes the drone is on its inertial sensors alone. This module answers
the question that sets the whole design: **if GNSS is cut, how long before the
inertial solution is too far out to be useful?**

```python
from ins import lawnmower, synthetic_flight, run_ins

truth, imu = synthetic_flight(lawnmower((52.45, -6.40), 1200, 900, 150),
                              alt_m=60.0, speed_mps=8.0)
run = run_ins(imu, truth, t_cut=0.0, grade="consumer_mems")
run.summary()
```

```bash
python scripts/run_ins_demo.py                   # 20 min synthetic flight
python scripts/run_ins_demo.py --log flight.BIN  # a real ArduPilot log
```

## The answer

A 22 minute survey flight over the A7 block, GNSS cut at the start:

| grade | 1 min | 5 min | 10 min | 20 min |
|---|---|---|---|---|
| navigation | 0.3 m | 4.1 m | 20.8 m | 134 m |
| tactical | 12.7 m | 106 m | 362 m | 1.2 km |
| consumer MEMS | 461 m | 48.5 km | 186 km | 754 km |

The consumer MEMS row is the one on the drone today. It is past a 50 m Tellus
cell in **under ten seconds**, and the numbers after that are not a measurement
of anything useful — they are the t³ law running away.

Read the table this way: inertial navigation is not an alternative to magnetic
matching here, it is the thing that has to be corrected, and it needs
correcting on a timescale of seconds, not minutes. That constrains the fix
rate, which constrains everything else.

## Why the error grows the way it does

Two terms dominate, and both are checked against their closed forms in
`tests/test_ins.py`:

| source | position error |
|---|---|
| accelerometer bias `b` | `0.5 · b · t²` |
| gyro bias `ω` | `g · ω · t³ / 6` |

A gyro bias tilts the navigation frame, gravity leaks into the horizontal
channels, and that leak integrates twice — hence t³, and hence the gyro bias
setting every column past about a minute. The 10 and 20 minute figures scale
almost linearly with the gyro bias assumed in `src/ins/errors.py`, so treat
them as proportional to an assumption, not as measurements.

## Where the numbers come from, and which are assumed

The `consumer_mems` preset is the ICM-42688-P class, the IMU on a Pixhawk 6C.

**Sourced** from the TDK datasheet: gyro noise 2.8 mdps/√Hz, accelerometer
noise 70 µg/√Hz.

**Assumed**, because no datasheet specifies it: turn-on bias, bias
instability and scale factor. Bias instability is not published for this part,
and the quoted offset tolerance is a pre-calibration worst case rather than
what survives a flight controller's calibration. Every assumed figure is
marked as such in `errors.py`. The `tactical` and `navigation` presets are
class-typical bands from the standard reference tables, not specific products.

This matters because the headline drift is driven almost entirely by an
assumed number. Tuning it against a real ArduPilot log is the single highest
value thing that could be done to this module.

## What is simulated and what is real

`synthetic_flight` builds a lawnmower path, leans the aircraft into its own
accelerations the way a multirotor must, and then runs the integrator
*backwards* to find the IMU readings that would produce exactly that path.
Feed those back in and the original trajectory returns to under a millimetre
over twenty minutes — so any drift seen afterwards is the error model, never
the integrator.

Real data comes in through `src/ins/loaders.py`. ArduPilot `.BIN` is the format
that matters, since the flight controller already logs it. The QuSpin dynamic
CSV loads too, with one caveat worth repeating: **its IMU axes are not
established**. Loading the example file gives a mean specific force of
9.853 m/s², which confirms the unit conversion, but it sits about 12° off
vertical — so the sensor is mounted tilted and the identity axis mapping is a
guess. `body_from_sensor` is an explicit parameter for that reason.

## What this does not model

No transport rate, no vibration, no coning or sculling, no temperature. Earth
rotation is implemented but off by default — over 10 to 30 minutes with MEMS
sensors the gyro bias is far larger, and the synthetic IMU is generated through
the same flag so truth and integration always agree.

The vertical channel is aided from a barometer, on by default. The horizontal
is deliberately left free-running, because that divergence is the entire
measurement.

---

# Magnetometer emulation — what the quantum sensor would read

The INS module says how fast the drone gets lost. This one says what the
magnetometer has to work with in order to find it again. Given a flight path,
it produces what a sensor on that aircraft would actually report, errors and
all.

```python
from sensor import FieldModel, ScalarOPM
from tellus import MagneticGrid

field = FieldModel(grid=MagneticGrid("a7_tmi"))
opm = ScalarOPM()
readings = opm.emulate(truth, field, seed=0)
anomaly = opm.to_anomaly(readings, truth, field)   # what the filter compares with the map
```

```bash
python scripts/run_opm_demo.py                  # 20 min over A7
python scripts/run_opm_demo.py --mains-hz 60    # as if flown in Ontario
python scripts/run_opm_demo.py --interference   # with the drone's own field on
```

## The answer

A 22 minute survey over A7 at 60 m, both sensors reading the same true field:

| sensor | RMS error vs the map | swing with heading |
|---|---|---|
| QTFM-like scalar OPM | **2.7 nT** | 5.0 nT |
| RM3100-like vector | **107 nT** | 60 nT |

The anomaly along that track spans 28 to 100 nT. So the vector sensor's error
is larger than the entire signal, and the OPM's is a few percent of it.

But the second column is the real argument. A large error could be calibrated
out. An error that **moves with heading** cannot be told apart from geology on
a lawnmower survey, because the heading is what the survey pattern varies. That
is the case for a quantum sensor, and it is a consequence of how the two
instruments work rather than of how much they cost:

- a scalar OPM measures `|B|` directly, through an atomic resonance, so its
  reading barely depends on which way it points
- a vector magnetometer measures three components and computes `|B|`, so every
  per-axis bias, scale and misalignment error enters weighted by orientation

A 0.5% scale error on one axis is 250 nT of a 50,000 nT field, and it swings as
the aircraft turns.

## Where the numbers come from

Three are **measured**, from the March flight in `notebooks/dynamic.ipynb`:

| | |
|---|---|
| platform + environment noise | 1.35 nT RMS above 0.1 Hz |
| mains interference | ~5 nT RMS, 93% of all noise power |
| broadband floor | ~3.4 pT/√Hz at 100-125 Hz |

The first is labelled platform-and-environment, **not** sensor noise. It is the
airframe, the air and the ground; calling it sensor noise would blame the
instrument for its surroundings.

Its *shape* matters as much as its size, and getting that wrong is easy. Two
thirds of the power sits between 0.1 and 0.3 Hz and it falls off as roughly
`f^-1.5` to a near-white floor. Modelling it as white noise at the same RMS
would put 120 pT/√Hz at 100 Hz — thirty times the measured floor, and enough to
bury the instrument's own contribution. The emulator synthesises it in the
frequency domain for that reason, and `tests/test_sensor.py` checks the result
band by band against the real flight.

Two are from the QuSpin datasheet: 3 pT/√Hz instrument noise, and heading error
under 3 nT uncompensated. The RM3100 figure of 15 nT per-axis noise is from the
PNI datasheet at cycle count 200.

Two are **assumed and are the weak points**: the shape of the heading error
curve, and the dead zone. We have no rotation test, and the March flight never
put the sensor in an awkward attitude — its field-to-axis angle only spanned
about 50 to 126 degrees. Both are parameters so a measured curve can replace
them in one line. QuSpin publish a Heading Error Compensation document that
should go in first.

## The dead zone

A scalar OPM drives an atomic resonance with light. When the field lines up
with the optical axis there is nothing transverse to precess about, the
resonance vanishes, and the sensor stops reading — it does not read badly, it
reads nothing. Inside ±7° of the axis the emulator returns NaN, never a number,
and the noise rises as the dead zone is approached rather than failing without
warning.

This is a mounting constraint, not just a modelling detail. In Ireland the
field dips about 69° from horizontal, so where the sensor axis points decides
whether a normal survey attitude ever crosses a dead zone.

## Timing

The sensor runs at 250 Hz and a navigation filter wants about 10 Hz, so there
is an anti-alias filter and a decimation, and both cost delay. The emulator
reports it: **376 ms** at the defaults, which at 8 m/s is three metres of
along-track position.

That is small against a 50 m map cell and large against the sensor's own 0.7 nT
repeatability, so it matters for comparing readings with each other and not
much for the fix itself. The filter is a causal linear-phase FIR rather than a
zero-phase one, because a real-time filter cannot use future samples and a
simulator that quietly does is flattering itself.

## Upward continuation

Tellus was flown at 60 m. A drone at 100 m sees a smoother field, because
altitude is a low-pass filter on the anomaly. `tellus/continuation.py` does the
standard FFT version — multiply the spectrum by `exp(-|k| dz)` — handling the
grid's NaN gaps and tapering the edges so the transform does not ring.

**Downward continuation is refused.** It means multiplying by `exp(+|k| dz)`,
which amplifies the shortest wavelengths hardest, and the shortest wavelengths
in a real grid are mostly gridding artefacts. A few tens of metres of it turns
a usable map into noise with a convincing texture. If the field below 60 m is
needed, the answer is a new survey.

## What is still missing

The diurnal variation is a sine wave standing in for a reference observatory.
The real version subtracts measured MagIE data from Birr and Dunsink; until
that is wired in, the diurnal residual is a systematic error sitting directly on
the signal, and the emulator can only remove exactly the model it added.

Platform interference defaults to **zero**, which is a fiction we choose
deliberately. We have never measured our own drone's magnetic signature, and a
plausible-looking default would be an invented number quietly inflating every
result. Switching it on is one argument; fitting it needs a Tolles-Lawson
calibration flight we have not flown.
