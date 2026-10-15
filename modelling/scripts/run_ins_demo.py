#!/usr/bin/env python3
"""How far the drone drifts after GNSS is cut, for three grades of IMU.

    python scripts/run_ins_demo.py                      # 20 min synthetic flight
    python scripts/run_ins_demo.py --minutes 10         # a shorter one
    python scripts/run_ins_demo.py --log flight.BIN     # a real ArduPilot log

Flies a synthetic survey pattern over the Tellus A7 block at 60 m and 8 m/s,
cuts GNSS at the start, and integrates each IMU grade forward on its own.
Prints the horizontal error at 1, 5, 10 and 20 minutes, and writes one plot to
outputs/.

Reading the result
------------------
The y axis is logarithmic and spans several decades, because that is the shape
of the problem: unaided inertial error grows as the cube of time once the gyro
bias dominates, so a chart with a linear axis shows one grade and three flat
lines.

The numbers are large. That is the point - it is the argument for magnetic
matching rather than against inertial navigation. The useful question is not
"how far can the INS go alone" but "how long can it coast between magnetic
fixes", which is the first minute of this chart, not the twentieth.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ins import GRADES, lawnmower, run_ins, synthetic_flight  # noqa: E402
from ins.runner import InitialError  # noqa: E402
from tellus.products import ROOT  # noqa: E402

OUTPUT_DIR = ROOT / "outputs"
INK = "#3d3d3a"
MUTED = "#8a8a82"

# Over the Tellus A7 block (Wicklow / Wexford), so the same flight can later be
# handed to MagneticGrid.sample_track without moving it.
START_LAT, START_LON = 52.45, -6.40
ALTITUDE_M = 60.0          # the height Tellus itself was flown at
SPEED_MPS = 8.0            # a survey multirotor's cruise

# One colour per grade, best to worst, so the plot reads in order.
COLOURS = {"navigation": "#2f6f4f", "tactical": "#b07d2b", "consumer_mems": "#9c3a2e"}
CHECKPOINTS_MIN = (1, 5, 10, 20)


def synthetic_survey(minutes: float):
    """A lawnmower pattern long enough to fly for the requested time.

    The pattern is sized from the duration rather than fixed, so --minutes
    gives a real flight of that length instead of looping a short one.
    """
    path_length_m = minutes * 60.0 * SPEED_MPS
    spacing_m = 150.0

    # Size a square-ish block whose total flown distance comes to path_length_m.
    # The cross-legs between lines count towards that distance too, which is
    # why n_lines divides by (length + spacing) and not just length.
    line_length_m = np.sqrt(path_length_m * spacing_m)
    # Round the line count up, so a 20 minute request gives at least 20
    # minutes of flight and the 20 minute column is not blank.
    n_lines = max(2, int(np.ceil(path_length_m / (line_length_m + spacing_m))))
    width_m = (n_lines - 1) * spacing_m

    waypoints = lawnmower((START_LAT, START_LON), length_m=line_length_m,
                          width_m=width_m, spacing_m=spacing_m)
    truth, imu = synthetic_flight(waypoints, alt_m=ALTITUDE_M,
                                  speed_mps=SPEED_MPS, rate_hz=100.0)
    print(f"synthetic flight: {truth.duration_s / 60:.1f} min, "
          f"{len(waypoints)} waypoints, "
          f"{line_length_m:.0f} x {width_m:.0f} m block at {ALTITUDE_M:.0f} m")
    return truth, imu


def real_flight(log_path: Path):
    """Load a real ArduPilot log and use its GPS track as truth."""
    from ins import load_ardupilot_bin

    imu, truth, baro = load_ardupilot_bin(log_path)
    print(f"{log_path.name}: {len(imu)} IMU samples at {imu.rate_hz:.0f} Hz, "
          f"{truth.duration_s / 60:.1f} min of GPS")
    if baro is None:
        print("  note: no BARO messages, so the vertical channel runs free")
    return truth, imu


def print_table(runs: dict) -> None:
    """The error table, worst grade last so the eye lands on the best case."""
    width = max(len(name) for name in runs)
    header = "  ".join(f"{m:>5d} min" for m in CHECKPOINTS_MIN)
    print(f"\nhorizontal error after the GNSS cut\n")
    print(f"  {'grade':<{width}}  {header}")
    print(f"  {'-' * width}  {'-' * len(header)}")
    for name, run in runs.items():
        cells = []
        for minutes in CHECKPOINTS_MIN:
            err = run.error_at(minutes * 60)
            cells.append("      -  " if np.isnan(err) else f"{_metres(err):>9s}")
        print(f"  {name:<{width}}  {'  '.join(cells)}")
    print()


def _metres(value: float) -> str:
    """Metres under a kilometre, kilometres above it. Keeps the table readable
    when the same column spans centimetres to hundreds of kilometres."""
    if value < 1000:
        return f"{value:.1f} m"
    return f"{value / 1000:.1f} km"


def plot_drift(runs: dict, path: Path, subtitle: str) -> None:
    """Error against time, one line per grade, log y."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for name, run in runs.items():
        minutes = run.elapsed_s / 60.0
        ax.plot(minutes, run.error_m, lw=1.6, label=name.replace("_", " "),
                color=COLOURS.get(name, INK))

    # The scale that actually matters for magnetic matching: a 50 m Tellus
    # cell. Once the INS is further out than this, the map cannot tell the
    # filter which cell it is in without searching a wide area.
    ax.axhline(50, color=MUTED, ls="--", lw=1.0)
    ax.text(0.02, 52, "50 m - one Tellus cell", color=MUTED, fontsize=9,
            va="bottom", transform=ax.get_yaxis_transform())

    ax.set_yscale("log")
    # The error is exactly zero at the cut, which on a log axis would hand ten
    # empty decades to a single point and squash everything that matters into
    # the top inch. Start the axis at decimetres - below that, nothing about
    # this problem is interesting.
    top = max(np.nanmax(run.error_m) for run in runs.values())
    ax.set_ylim(0.1, top * 3)
    ax.set_xlabel("minutes since GNSS was lost")
    ax.set_ylabel("horizontal error (m)")
    ax.set_title("Free-running INS drift", loc="left", color=INK)
    ax.set_title(subtitle, loc="right", color=MUTED, fontsize=9)
    ax.grid(alpha=0.25, which="both")
    ax.legend(loc="upper left", frameon=False)

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", type=Path, default=None,
                        help="an ArduPilot .BIN log to use instead of a synthetic flight")
    parser.add_argument("--minutes", type=float, default=20.0,
                        help="length of the synthetic flight (default 20)")
    parser.add_argument("--seed", type=int, default=1,
                        help="seeds the sensor errors; same seed, same result")
    parser.add_argument("--vel-error", type=float, default=0.0,
                        help="initial velocity error in m/s at the cut, north axis "
                             "(try 0.1 - it costs 60 m in ten minutes on its own)")
    args = parser.parse_args()

    if args.log is not None:
        truth, imu = real_flight(args.log)
        subtitle = args.log.name
    else:
        truth, imu = synthetic_survey(args.minutes)
        subtitle = f"synthetic, {truth.duration_s / 60:.0f} min at {SPEED_MPS:.0f} m/s"

    initial = None
    if args.vel_error:
        initial = InitialError(vel_mps=[args.vel_error, 0.0, 0.0])
        print(f"starting with {args.vel_error} m/s of velocity error")

    # Best grade first so the printed table and the legend agree.
    runs = {}
    for name in ("navigation", "tactical", "consumer_mems"):
        runs[name] = run_ins(imu, truth, t_cut=float(imu.t[0]),
                             initial_error=initial, grade=name, seed=args.seed)

    print_table(runs)

    for name, run in runs.items():
        # In deg/hr, because a navigation-grade bias rounds to zero in deg/s.
        bias_deg_per_hr = np.degrees(run.realisation.gyro_bias) * 3600.0
        print(f"  {name:<14} drew a gyro bias of "
              f"{np.array2string(bias_deg_per_hr, precision=2, suppress_small=False)} deg/hr")
    print("\n  The gyro bias is what sets the 10-20 minute numbers: error from a")
    print("  gyro bias grows as t^3, so those columns scale almost linearly with")
    print("  the figures assumed in src/ins/errors.py. Re-run with a different")
    print("  --seed to see the spread.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = args.log.stem if args.log else "synthetic"
    plot_drift(runs, OUTPUT_DIR / f"ins_drift_{stem}.png", subtitle)


if __name__ == "__main__":
    main()
