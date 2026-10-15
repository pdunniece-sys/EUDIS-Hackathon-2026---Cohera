"""Self-map test: navigate a real QTFM survey on a map made from its own lines.

    from navfilter import selfmap

    log = selfmap.load_survey(CSV)                       # 10 Hz, lines labelled
    smap = selfmap.grid_lines(log, log.line % 2 == 0)    # map from every other line
    ins = selfmap.drifted_ins(log, window, "tactical")   # real path + simulated drift

Why this test exists
--------------------
Everything else in this package navigates on Tellus with emulated readings.
This is the one place where the magnetometer, its noise, the aircraft's own
field and the heading error are all real: a QuSpin QTFM flown on a drone in
Ontario on 14 March 2026. There is no Tellus map there, so the map is built
from the flight itself - half the survey lines are gridded into a map and the
filter navigates along the other half, which it has never seen.

Two things to know when reading the result
------------------------------------------
On a lawnmower, alternate lines are flown in opposite directions, so "map from
the even lines, test on the odd ones" also means "map flown westbound, test
flown eastbound". Any heading-dependent error - the drone's own field, OPM
heading error - appears as a constant offset between map and test. That is a
real effect the filter has to live with, and it is measured, not removed.

The INS is not real. The log's IMU axes are not established (see
ins/loaders.py), so the INS track is the real GPS path plus the position error
of a simulated INS flown on a matching synthetic lawnmower. The magnetic side
is entirely real; the drift is a model.

What this deliberately does NOT do
----------------------------------
Use any reading from a map line as a measurement. Those readings built the
map, and matching them against it would be the inverse crime again.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
from affine import Affine
from pyproj import Transformer
from rasterio.crs import CRS
from scipy.interpolate import griddata
from scipy.ndimage import uniform_filter1d

from ins import lawnmower, run_ins, synthetic_flight
from ins.types import Trajectory, metres_per_degree
from navfilter.maps import grid_from_array
from sensor import FieldModel

QTFM_RATE_HZ = 250.0
# 25 samples at 250 Hz is 0.1 s: a 10 Hz output, and a whole number of mains
# cycles at both 50 Hz (5) and 60 Hz (6), so a plain block mean nulls mains.
BLOCK = 25


@dataclass
class SurveyLog:
    """A QTFM flight at 10 Hz, with survey lines labelled.

        t           seconds from the start of the log
        field_nt    total field, block-averaged
        lat, lon, alt_m   GPS position (alt is ellipsoidal, as logged)
        anomaly_nt  field minus IGRF at the GPS position
        line        survey line index, numbered south to north; -1 off-line
        direction   +1 eastbound, -1 westbound, 0 off-line
    """

    t: np.ndarray
    field_nt: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    alt_m: np.ndarray
    anomaly_nt: np.ndarray
    line: np.ndarray
    direction: np.ndarray
    model: FieldModel

    def trajectory(self, keep=slice(None)) -> Trajectory:
        return Trajectory(self.t[keep], self.lat[keep], self.lon[keep], self.alt_m[keep],
                          name="GPS")

    def core_fn(self):
        return lambda lat, lon, alt: np.linalg.norm(
            self.model.core_field_ned(lat, lon, alt), axis=1)

    def window(self, first_line: int, last_line: int) -> slice:
        """Samples from the start of one line to the end of another."""
        idx = np.flatnonzero((self.line >= first_line) & (self.line <= last_line))
        return slice(idx[0], idx[-1] + 1)


def load_survey(path, date: dt.date = dt.date(2026, 3, 14),
                min_line_m: float = 400.0) -> SurveyLog:
    """Read a QuSpin dynamic CSV and find its survey lines.

    A line is a stretch of straight flight at least `min_line_m` long; lines
    are numbered by their mean position across track, south to north. This
    assumes roughly east-west lines, which is how the March flight was flown.
    """
    import pandas as pd

    cols = ["Logger_Time (us) Original", "Mag Data[nT]", "Latitude", "Longitude", "Height"]
    df = pd.read_csv(path, usecols=cols)
    n = len(df) // BLOCK * BLOCK
    block = lambda a: np.asarray(a, dtype=float)[:n].reshape(-1, BLOCK).mean(axis=1)

    t = block((df[cols[0]] - df[cols[0]].iloc[0]) / 1e6)
    field = block(df[cols[1]])
    lat, lon, alt = block(df.Latitude), block(df.Longitude), block(df.Height)

    model = FieldModel(date=date)
    core = np.linalg.norm(model.core_field_ned(lat, lon, alt), axis=1)

    # Straight east-west flight: smoothed velocity mostly east or west.
    m_lat, m_lon = metres_per_degree(float(lat.mean()))
    north, east = (lat - lat.mean()) * m_lat, (lon - lon.mean()) * m_lon
    vn = np.gradient(uniform_filter1d(north, 20), t)
    ve = np.gradient(uniform_filter1d(east, 20), t)
    speed = np.hypot(vn, ve)
    straight = (np.abs(vn) < 0.25 * speed) & (speed > 2.0)

    edges = np.flatnonzero(np.diff(np.r_[0, straight.astype(int), 0]))
    segments = [(a, b) for a, b in zip(edges[::2], edges[1::2])
                if abs(east[b - 1] - east[a]) >= min_line_m]
    segments.sort(key=lambda s: north[s[0]:s[1]].mean())

    line = np.full(len(t), -1)
    direction = np.zeros(len(t), dtype=int)
    for k, (a, b) in enumerate(segments):
        line[a:b] = k
        direction[a:b] = 1 if ve[a:b].mean() > 0 else -1

    return SurveyLog(t, field, lat, lon, alt, field - core, line, direction, model)


def _utm_crs(lon: float, lat: float) -> CRS:
    zone = int((lon + 180) // 6) + 1
    return CRS.from_epsg((32600 if lat >= 0 else 32700) + zone)


def grid_lines(log: SurveyLog, use: np.ndarray, cell_m: float = 10.0):
    """Grid the readings selected by `use` into a MagneticGrid.

    Linear interpolation between the lines, in UTM metres, NaN outside the
    lines' convex hull - so the map stops where the survey stops. 10 m cells,
    not 50: the lines are ~25 m apart, so a 50 m cell would average across
    two of them.
    """
    crs = _utm_crs(float(log.lon.mean()), float(log.lat.mean()))
    to_utm = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = (np.asarray(v) for v in to_utm.transform(log.lon[use], log.lat[use]))

    left, top = x.min() - 2 * cell_m, y.max() + 2 * cell_m
    cols = int(np.ceil((x.max() + 2 * cell_m - left) / cell_m))
    rows = int(np.ceil((top - (y.min() - 2 * cell_m)) / cell_m))
    gx = left + (np.arange(cols) + 0.5) * cell_m
    gy = top - (np.arange(rows) + 0.5) * cell_m
    values = griddata((x, y), log.anomaly_nt[use], tuple(np.meshgrid(gx, gy)),
                      method="linear")
    return grid_from_array(values, Affine(cell_m, 0, left, 0, -cell_m, top), crs,
                           name="self-map")


def drifted_ins(log: SurveyLog, window: slice, grade: str, seed: int = 0,
                speed_mps: float = 8.0) -> Trajectory:
    """The real GPS path over `window`, plus a simulated INS's position error.

    The error comes from flying a synthetic lawnmower shaped like this survey
    (same line length, spacing, heading and speed) through ins.run_ins with
    the given grade, cut at its start. Its north/east error against time is
    then added to the real path. Beyond the synthetic flight's end the error
    is held at its last value rather than invented.
    """
    truth = log.trajectory(window)
    lines = np.unique(log.line[window][log.line[window] >= 0])
    m_lat, m_lon = metres_per_degree(float(truth.lat[0]))

    # Survey shape from the data: line length and spacing.
    first = log.line == lines[0]
    length = abs(np.ptp(log.lon[first])) * m_lon
    across = [log.lat[log.line == k].mean() * m_lat for k in lines]
    spacing = float(np.mean(np.diff(across))) if len(lines) > 1 else 25.0
    heading = 270.0 if log.direction[first].mean() < 0 else 90.0

    synth, imu = synthetic_flight(
        lawnmower((truth.lat[0], truth.lon[0]), length, spacing * (len(lines) - 1),
                  spacing, heading_deg=heading),
        alt_m=float(truth.alt_m.mean()), speed_mps=speed_mps, rate_hz=100.0)
    est = run_ins(imu, synth, t_cut=0.0, grade=grade, seed=seed).estimate
    ref = synth.at(est.t)
    err_n = (est.lat - ref.lat) * m_lat
    err_e = (est.lon - ref.lon) * m_lon

    elapsed = truth.t - truth.t[0]
    return Trajectory(truth.t,
                      truth.lat + np.interp(elapsed, est.t - est.t[0], err_n) / m_lat,
                      truth.lon + np.interp(elapsed, est.t - est.t[0], err_e) / m_lon,
                      truth.alt_m, name=f"{grade} INS (simulated drift)")
