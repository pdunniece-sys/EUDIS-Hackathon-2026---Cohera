"""The particle filter in three calls, for notebooks and scripts.

    from navfilter import scenario

    scenario.show_location(53.10, -7.90)                 # where is that, is it mapped?
    sim = scenario.simulate(53.10, -7.90, minutes=10)    # fly it, run INS and filter
    scenario.plot(sim.runs, sim.filter_map, sim.truth)

    run = scenario.filter_log(ins_track, t_s, field_nt)  # your own data instead

`simulate` is everything run_pf_demo.py does, for any point in Ireland the
national Tellus grid covers. `filter_log` is the same filter fed with a track
and magnetometer readings you supply, so the only thing that changes between
simulation and a real flight is where those two inputs come from.

What `filter_log` deliberately does NOT do
------------------------------------------
It does not correct diurnal variation. The simulator's diurnal model is a
made-up sine wave, and subtracting it from real data would add error rather
than remove it. Until MagIE observatory data is wired in, expect ~10-20 nT of
slow drift on the measured anomaly on a real day. See the Known gaps in the
project notes.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from ins import lawnmower, run_ins, synthetic_flight
from ins.types import Trajectory
from navfilter.maps import degrade_map
from navfilter.particle import ParticleFilter, PFConfig, PFRun
from sensor import FieldModel, NoDiurnal, Readings, ScalarOPM
from tellus import MagneticGrid
from tellus.continuation import continue_to_altitude

GRADE_COLOURS = {"consumer_mems": "#9c3a2e", "tactical": "#2f5f8f",
                 "navigation": "#2f6f4f", "perfect": "#6b5b95", "log": "#2f5f8f"}
INK, MUTED = "#3d3d3a", "#8a8a82"


@lru_cache(maxsize=2)
def load_map(product: str = "merge_tmi") -> MagneticGrid:
    """The whole grid, loaded once per session. The national grid is ~460 MB
    in memory, so it is cached rather than reloaded on every call."""
    return MagneticGrid(product)


def _cut_out(lats, lons, product: str, margin_deg: float = 0.05) -> MagneticGrid:
    """The part of the map a flight needs, plus a margin for drifted particles."""
    return load_map(product).subset((np.min(lons) - margin_deg, np.min(lats) - margin_deg,
                                     np.max(lons) + margin_deg, np.max(lats) + margin_deg))


def _core_fn(model: FieldModel):
    return lambda lat, lon, alt: np.linalg.norm(model.core_field_ned(lat, lon, alt), axis=1)


# ----------------------------------------------------------------------
# Picking a place
# ----------------------------------------------------------------------

def show_location(lat: float, lon: float, product: str = "merge_tmi", box_km: float = 3.0):
    """Two panels: where the point is on the whole map, and the field around it.

    Prints the anomaly at the point, or says plainly that it is off the survey.
    """
    import matplotlib.pyplot as plt

    grid = load_map(product)
    value = grid.sample(lat, lon)
    print(f"{lat:.4f}, {lon:.4f}: " + (f"{value:.1f} nT" if np.isfinite(value)
                                       else "NOT on the survey - pick another point"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    step = max(1, max(grid.shape) // 800)            # decimated for drawing only
    left, bottom, right, top = grid.bounds
    lo, hi = np.nanpercentile(grid.values[::step, ::step], [2, 98])
    ax1.imshow(grid.values[::step, ::step], extent=(left, right, bottom, top),
               cmap="RdBu_r", vmin=lo, vmax=hi)
    x, y = grid._to_grid.transform(lon, lat)
    ax1.plot(x, y, "k*", ms=14, mec="white")
    ax1.set_title(grid.name, loc="left", fontsize=9, color=INK)
    ax1.set_axis_off()

    d = box_km / 2 / 111.0
    local = grid.subset((lon - 1.6 * d, lat - d, lon + 1.6 * d, lat + d), margin_m=0)
    # The number that matters for navigation: how much the field varies here.
    # Under ~5 nT the filter has almost nothing to match; tens of nT is good.
    print(f"   field varies by {np.nanstd(local.values):.1f} nT (std) within "
          f"{box_km:g} km")
    l2, b2, r2, t2 = local.bounds
    im = ax2.imshow(local.values, extent=(l2, r2, b2, t2), cmap="RdBu_r")
    ax2.plot(x, y, "k*", ms=14, mec="white")
    fig.colorbar(im, ax=ax2, label="anomaly (nT)")
    ax2.set_title(f"{box_km:g} km around the point - more contrast here means a "
                  "better fix", loc="left", fontsize=9, color=INK)
    ax2.set_axis_off()
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# Simulated flight
# ----------------------------------------------------------------------

@dataclass
class Simulation:
    truth: Trajectory
    filter_map: MagneticGrid
    runs: dict          # grade -> PFRun

    def summary(self):
        for run in self.runs.values():
            run.summary()
            print()


def simulate(lat: float, lon: float, minutes: float = 10.0,
             grades=("consumer_mems", "tactical"), particles: int = 2000,
             seed: int = 0, cold_start: bool = False, alt_m: float = 60.0,
             speed_mps: float = 8.0, spacing_m: float = 150.0,
             product: str = "merge_tmi", **config) -> Simulation:
    """Fly a lawnmower from (lat, lon), cut GNSS at the start, filter it.

    The lawnmower's south-west corner is (lat, lon) and it runs about
    `minutes` long. Extra keyword arguments go to PFConfig, e.g.
    `update_hz=1.0` or `sigma_total_nt=8`.
    """
    path_m = minutes * 60.0 * speed_mps
    line_m = np.sqrt(path_m * spacing_m)
    n_lines = max(2, int(np.ceil(path_m / (line_m + spacing_m))))
    waypoints = lawnmower((lat, lon), length_m=line_m,
                          width_m=(n_lines - 1) * spacing_m, spacing_m=spacing_m)
    # 250 Hz so the OPM's mains line (50 Hz) sits well inside Nyquist.
    truth, imu = synthetic_flight(waypoints, alt_m=alt_m, speed_mps=speed_mps,
                                  rate_hz=250.0)

    grid = _cut_out(truth.lat, truth.lon, product)
    if np.isnan(grid.sample_track(truth.lat, truth.lon)).any():
        raise ValueError(
            f"part of a {minutes:g} min flight from {lat}, {lon} is off the "
            f"survey. Check with show_location(), or move or shorten the flight.")

    model = FieldModel(grid=grid, date=dt.date(2026, 10, 15))
    field = model.at(truth)
    # The filter's map is deliberately worse than the truth - see maps.py.
    filter_map = degrade_map(continue_to_altitude(grid, alt_m), seed=seed + 1)

    opm = ScalarOPM(rate_hz=250.0)
    readings = opm.emulate(truth, field, seed=seed)
    taken_at = readings.t - readings.latency_s

    runs = {}
    for grade in grades:
        g = None if grade == "perfect" else grade
        ins_run = run_ins(imu, truth, t_cut=0.0, grade=g, seed=seed)
        anomaly = opm.to_anomaly(readings, ins_run.estimate, model)
        cfg = PFConfig(n_particles=particles, grade=g, seed=seed,
                       sigma_pos0_m=500.0 if cold_start else 30.0, **config)
        runs[grade] = ParticleFilter(filter_map, cfg).run(
            ins_run.estimate, taken_at, anomaly, truth=truth, name=grade,
            core_fn=_core_fn(model))
    return Simulation(truth, filter_map, runs)


# ----------------------------------------------------------------------
# Your own data
# ----------------------------------------------------------------------

def filter_log(ins: Trajectory, t_s, field_nt, grade: str | None = "consumer_mems",
               truth: Trajectory | None = None, date: dt.date | None = None,
               alt_m: float | None = None, product: str = "merge_tmi",
               name: str = "log", **config) -> PFRun:
    """Run the filter on a track and magnetometer readings you supply.

        ins       where the aircraft *thinks* it is with GNSS off: a Trajectory
                  of t, lat, lon, alt_m. Its first sample is the GNSS cut.
        t_s       when each magnetometer reading was taken, on the same clock
        field_nt  total field in nT, as the OPM reports it (~49,000 here).
                  NaN for a dropped reading.
        grade     which IMU the track came from, for the filter's process
                  noise; None means a near-perfect INS
        truth     the real path, if you have one (e.g. GNSS logged but not
                  used), purely to score the result
        alt_m     flight altitude for the map; defaults to the track's mean

    Extra keyword arguments go to PFConfig.
    """
    t_s = np.asarray(t_s, dtype=float)
    field_nt = np.asarray(field_nt, dtype=float)
    alt = float(np.nanmean(ins.alt_m)) if alt_m is None else alt_m

    grid = _cut_out(ins.lat, ins.lon, product, margin_deg=0.1)
    filter_map = continue_to_altitude(grid, max(alt, 60.0))
    model = FieldModel(grid=None, date=date or dt.date.today(), diurnal=NoDiurnal())

    readings = Readings(t_s, field_nt, np.isfinite(field_nt), name=name)
    anomaly = ScalarOPM().to_anomaly(readings, ins, model)

    cfg = PFConfig(grade=grade, **config)
    return ParticleFilter(filter_map, cfg).run(ins, t_s, anomaly, truth=truth,
                                               name=name, core_fn=_core_fn(model))


# ----------------------------------------------------------------------
# One picture
# ----------------------------------------------------------------------

def plot(runs: dict, grid: MagneticGrid, truth: Trajectory | None = None,
         title: str = "", path=None):
    """Error against time (if there is a truth) and the tracks on the map."""
    import matplotlib.pyplot as plt

    scored = truth is not None
    fig, axes = plt.subplots(1, 2 if scored else 1, figsize=(14 if scored else 8, 6),
                             gridspec_kw=dict(width_ratios=[1.15, 1]) if scored else None)
    ax, mp = (axes if scored else (None, axes))

    if scored:
        for name, run in runs.items():
            c = GRADE_COLOURS.get(name, INK)
            ax.plot(run.elapsed_s / 60, run.ins_error_m, "--", lw=1.2, color=c,
                    label=f"{name}, INS only")
            ax.plot(run.elapsed_s / 60, run.error_m, lw=1.6, color=c,
                    label=f"{name}, INS + MagNav")
            ax.fill_between(run.elapsed_s / 60, 0.1, run.sigma2_m, color=c,
                            alpha=0.10, lw=0)
        ax.set_yscale("log")
        ax.set_ylim(1, None)
        ax.set_xlabel("minutes since GNSS cut")
        ax.set_ylabel("horizontal error (m)")
        ax.set_title("position error; shading is the filter's own 2σ",
                     loc="left", color=INK)
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=0.25, which="both")

    # Frame the map on the truth if there is one, otherwise on the estimate.
    first = next(iter(runs.values()))
    frame = truth if scored else first.estimate
    fx, fy = grid._to_grid.transform(frame.lon, frame.lat)
    pad = 800.0
    left, bottom, right, top = grid.bounds
    lo, hi = np.nanpercentile(grid.values, [2, 98])
    im = mp.imshow(grid.values, extent=(left, right, bottom, top), cmap="RdBu_r",
                   vmin=lo, vmax=hi)
    fig.colorbar(im, ax=mp, shrink=0.8, label="anomaly (nT)")
    if scored:
        mp.plot(fx, fy, color=INK, lw=1.8, label="truth")
    for name, run in runs.items():
        c = GRADE_COLOURS.get(name, INK)
        if run.ins_track is not None:
            x, y = grid._to_grid.transform(run.ins_track.lon, run.ins_track.lat)
            mp.plot(x, y, "--", color=c, lw=1.0, label=f"{name}, INS only")
        x, y = grid._to_grid.transform(run.estimate.lon, run.estimate.lat)
        mp.plot(x, y, color=c, lw=1.3, label=f"{name}, INS + MagNav")
    mp.set_xlim(min(fx) - pad, max(fx) + pad)
    mp.set_ylim(min(fy) - pad, max(fy) + pad)
    mp.set_aspect("equal")
    mp.set_xlabel("ITM easting (m)")
    mp.set_ylabel("ITM northing (m)")
    mp.set_title("tracks over the anomaly map", loc="left", color=INK)
    mp.legend(frameon=False, fontsize=7, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.12))

    if title:
        fig.suptitle(title, color=MUTED, fontsize=10, x=0.01, ha="left")
    fig.text(0.01, 0.01, "Magnetic data: Geological Survey Ireland, Tellus, CC BY 4.0; "
             "Northern Ireland under OGL v3.0", color=MUTED, fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    if path is not None:
        fig.savefig(path, dpi=140)
    return fig
