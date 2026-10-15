#!/usr/bin/env python3
"""The headline: how far an INS drifts without GNSS, with and without the map.

    python scripts/run_pf_demo.py                       # 20 min, both grades
    python scripts/run_pf_demo.py --minutes 5
    python scripts/run_pf_demo.py --cold-start          # 500 m initial spread
    python scripts/run_pf_demo.py --grades tactical --particles 5000

Flies the same lawnmower over Tellus A7 as run_opm_demo.py, cuts GNSS at the
start, and for each IMU grade runs:

    1. the INS on its own - the baseline, and the exact run the filter corrects
    2. the QTFM-like OPM along the true path
    3. the particle filter, on a *degraded* copy of the map (navfilter/maps.py
       says why the filter must not get the map the readings came from)

Writes outputs/pf_headline.png and prints the error table.

What the left panel claims, and what it does not
------------------------------------------------
The INS-only lines scale with assumed IMU biases (ins/errors.py), so their
height is an assumption. The shaded band is the filter's own 2-sigma claim;
where the solid line stays inside it the filter is honest about itself. The
gap between dashed and solid is the result.
"""

import argparse
import datetime as dt
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ins import inverse_propagate, run_ins  # noqa: E402
from navfilter import ParticleFilter, PFConfig, degrade_map  # noqa: E402
from run_opm_demo import (  # noqa: E402
    ALTITUDE_M,
    INK,
    MUTED,
    SENSOR_RATE_HZ,
    START_LAT,
    START_LON,
    TRUTH_COLOUR,
    build_flight,
)
from sensor import FieldModel, ScalarOPM  # noqa: E402
from tellus import MagneticGrid  # noqa: E402
from tellus.continuation import continue_to_altitude  # noqa: E402
from tellus.products import ROOT  # noqa: E402

OUTPUT_DIR = ROOT / "outputs"
GRADE_COLOURS = {"consumer_mems": "#9c3a2e", "tactical": "#2f5f8f",
                 "navigation": "#2f6f4f"}
FLIGHT_DATE = dt.date(2026, 10, 15)


def plot(runs, truth, grid, path: Path, subtitle: str):
    fig, (ax, mp) = plt.subplots(1, 2, figsize=(14, 6),
                                 gridspec_kw=dict(width_ratios=[1.15, 1]))

    # 1. Error against time since the cut.
    for name, (ins_run, pf_run) in runs.items():
        colour = GRADE_COLOURS.get(name, INK)
        ax.plot(ins_run.elapsed_s / 60, ins_run.error_m, "--", lw=1.2,
                color=colour, label=f"{name}, INS only")
        ax.plot(pf_run.elapsed_s / 60, pf_run.error_m, lw=1.6, color=colour,
                label=f"{name}, INS + MagNav")
        ax.fill_between(pf_run.elapsed_s / 60, 0.1, pf_run.sigma2_m,
                        color=colour, alpha=0.10, lw=0)
    ax.set_yscale("log")
    ax.set_ylim(1, None)
    ax.set_xlabel("minutes since GNSS cut")
    ax.set_ylabel("horizontal error (m)")
    ax.set_title("position error; shading is the filter's own 2σ",
                 loc="left", color=INK)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25, which="both")

    # 2. The tracks, on the map, in the grid's own metres.
    tx, ty = grid._to_grid.transform(truth.lon, truth.lat)
    pad = 800.0
    xlim = (min(tx) - pad, max(tx) + pad)
    ylim = (min(ty) - pad, max(ty) + pad)
    left, bottom, right, top = grid.bounds
    vals = grid.values
    lo, hi = np.nanpercentile(vals, [2, 98])
    im = mp.imshow(vals, extent=(left, right, bottom, top), cmap="RdBu_r",
                   vmin=lo, vmax=hi, origin="upper")
    fig.colorbar(im, ax=mp, shrink=0.8, label="anomaly (nT)")
    mp.plot(tx, ty, color=TRUTH_COLOUR, lw=1.8, label="truth")
    for name, (ins_run, pf_run) in runs.items():
        colour = GRADE_COLOURS.get(name, INK)
        e = ins_run.estimate
        x, y = grid._to_grid.transform(e.lon, e.lat)
        mp.plot(x, y, "--", color=colour, lw=1.0, label=f"{name}, INS only")
        x, y = grid._to_grid.transform(pf_run.estimate.lon, pf_run.estimate.lat)
        mp.plot(x, y, color=colour, lw=1.3, label=f"{name}, INS + MagNav")
    mp.set_xlim(xlim)
    mp.set_ylim(ylim)
    mp.set_aspect("equal")
    mp.set_xlabel("ITM easting (m)")
    mp.set_ylabel("ITM northing (m)")
    mp.set_title("tracks over the anomaly map", loc="left", color=INK)
    mp.legend(frameon=False, fontsize=7, loc="upper right")

    fig.suptitle(f"Magnetic anomaly navigation over Tellus A7 — {subtitle}",
                 color=MUTED, fontsize=10, x=0.01, ha="left")
    fig.text(0.01, 0.01, "Magnetic data: Geological Survey Ireland, Tellus, CC BY 4.0",
             color=MUTED, fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--minutes", type=float, default=20.0)
    parser.add_argument("--particles", type=int, default=2000)
    parser.add_argument("--sigma-nt", type=float, default=None,
                        help="total measurement sigma; default is the PFConfig budget")
    parser.add_argument("--cold-start", action="store_true",
                        help="500 m initial position spread instead of 30 m")
    parser.add_argument("--update-hz", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grades", nargs="+", default=["consumer_mems", "tactical"])
    parser.add_argument("--product", default="a7_tmi")
    args = parser.parse_args()
    started = time.time()

    truth = build_flight(args.minutes)
    imu = inverse_propagate(truth, truth.mech)

    grid = MagneticGrid(args.product).subset(
        (START_LON - 0.1, START_LAT - 0.1, START_LON + 0.1, START_LAT + 0.1))
    model = FieldModel(grid=grid, date=FLIGHT_DATE)
    field = model.at(truth)

    # The filter's map: the same grid at flight altitude, then made worse.
    filter_map = degrade_map(continue_to_altitude(grid, ALTITUDE_M), seed=args.seed + 1)
    print(f"filter map: {filter_map}")

    opm = ScalarOPM(rate_hz=SENSOR_RATE_HZ)
    readings = opm.emulate(truth, field, seed=args.seed)
    taken_at = readings.t - readings.latency_s
    print(f"readings: {readings}")

    def core_fn(lat, lon, alt):
        return np.linalg.norm(model.core_field_ned(lat, lon, alt), axis=1)

    runs = {}
    for grade in args.grades:
        ins_run = run_ins(imu, truth, t_cut=0.0, grade=grade, seed=args.seed)
        # The background comes off at the INS position: the aircraft has no other.
        anomaly = opm.to_anomaly(readings, ins_run.estimate, model)
        config = PFConfig(n_particles=args.particles, grade=grade,
                          sigma_pos0_m=500.0 if args.cold_start else 30.0,
                          sigma_total_nt=args.sigma_nt, update_hz=args.update_hz,
                          seed=args.seed)
        pf_run = ParticleFilter(filter_map, config).run(
            ins_run.estimate, taken_at, anomaly, truth=truth, name=grade,
            core_fn=core_fn)
        runs[grade] = (ins_run, pf_run)
        print()
        pf_run.summary()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot(runs, truth, filter_map, OUTPUT_DIR / "pf_headline.png",
         f"{truth.duration_s / 60:.0f} min lawnmower at {ALTITUDE_M:.0f} m, "
         f"{args.particles} particles"
         + (", cold start" if args.cold_start else ""))
    print(f"total {time.time() - started:.0f} s")


if __name__ == "__main__":
    main()
