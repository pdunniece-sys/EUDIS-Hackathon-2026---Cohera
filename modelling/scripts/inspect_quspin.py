#!/usr/bin/env python3
"""Describe a QuSpin log and draw an overview of the flight.

    python scripts/inspect_quspin.py                 # whole log
    python scripts/inspect_quspin.py --airborne      # flight only
    python scripts/inspect_quspin.py path/to/log.txt

Prints what is in the file, then writes one overview image to outputs/ with
four panels: the flight path coloured by field, the field against time, the
altitude, and how hard the platform was being thrown about.

It also prints a repeatability figure, which is the number that decides
whether the data is good enough to navigate with. See the comment on
repeatability() below.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from quspin import read_log  # noqa: E402
from quspin.parse import ROOT  # noqa: E402

OUTPUT_DIR = ROOT / "outputs"
INK = "#3d3d3a"
MUTED = "#8a8a82"


def repeatability(log, cell_m: float = 25.0) -> None:
    """How closely the sensor agrees with itself over the same ground.

    A survey flies a lawnmower pattern, so the aircraft passes near the same
    spot more than once. Bin the readings into square cells and the spread
    *within* a cell is measurement error, while the spread *between* cells is
    real geology. The ratio of the two is what tells you whether there is a
    usable signal here.
    """
    east, north = log.fixes.local_xy()
    field = log.field_at_fixes()

    keys = np.column_stack([np.floor(east / cell_m), np.floor(north / cell_m)])
    cells: dict[tuple[float, float], list[float]] = {}
    for key, value in zip(map(tuple, keys), field):
        cells.setdefault(key, []).append(value)

    revisited = [v for v in cells.values() if len(v) > 1]
    if not revisited:
        print(f"\nno {cell_m:.0f} m cell was visited twice - cannot judge repeatability")
        return

    within = np.array([np.std(v) for v in revisited])
    between = np.std([np.mean(v) for v in cells.values()])

    print(f"\nrepeatability, in {cell_m:.0f} m cells")
    print(f"  cells visited more than once   {len(revisited)} of {len(cells)}")
    print(f"  spread within a cell (error)   {np.median(within):.1f} nT median,"
          f" {np.percentile(within, 90):.1f} nT at the 90th percentile")
    print(f"  spread between cells (signal)  {between:.1f} nT")
    print(f"  signal to error                {between / max(np.median(within), 1e-9):.0f} : 1")


def overview(log, path: Path) -> None:
    east, north = log.fixes.local_xy()
    field_at_fix = log.field_at_fixes()
    minutes = log.time_s / 60

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Where it flew, coloured by what it measured.
    ax = axes[0, 0]
    ax.plot(east, north, color="#ccccc4", linewidth=0.5, zorder=0)
    dots = ax.scatter(east, north, c=field_at_fix, cmap="RdYlBu_r", s=10,
                      vmin=np.percentile(field_at_fix, 2),
                      vmax=np.percentile(field_at_fix, 98))
    ax.plot(east[0], north[0], "o", color="black", markersize=7)
    ax.set_aspect("equal")
    ax.set_title("Flight path, coloured by total field (dot = start)",
                 fontsize=10, color=MUTED)
    ax.set_xlabel("metres east of centre", fontsize=9, color=MUTED)
    ax.set_ylabel("metres north of centre", fontsize=9, color=MUTED)
    fig.colorbar(dots, ax=ax, label="nT", fraction=0.046)

    # 2. The raw measurement.
    ax = axes[0, 1]
    ax.plot(minutes, log.field_nt, linewidth=0.3, color="#4a7fb5")
    ax.set_title(f"Total field, {log.rate_hz:.0f} Hz", fontsize=10, color=MUTED)
    ax.set_xlabel("minutes", fontsize=9, color=MUTED)
    ax.set_ylabel("nT", fontsize=9, color=MUTED)

    # 3. Altitude, which is how you tell flight from ground time.
    ax = axes[1, 0]
    ax.plot(log.fixes.time_s / 60, log.fixes.alt_m, linewidth=1.4, color="#b5734a")
    ax.set_title("GNSS altitude above the ellipsoid", fontsize=10, color=MUTED)
    ax.set_xlabel("minutes", fontsize=9, color=MUTED)
    ax.set_ylabel("metres", fontsize=9, color=MUTED)

    # 4. How much the platform was moving - manoeuvres show up here.
    ax = axes[1, 1]
    gyro_rate = np.linalg.norm(log.gyro.values, axis=1)
    ax.plot(log.gyro.time_s / 60, gyro_rate, linewidth=0.4, color="#7a6ba8")
    ax.set_title("Rotation rate (all three gyro axes combined)",
                 fontsize=10, color=MUTED)
    ax.set_xlabel("minutes", fontsize=9, color=MUTED)
    ax.set_ylabel("degrees per second", fontsize=9, color=MUTED)

    for ax in axes.flat:
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.grid(color="#f0f0ea", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("bottom", "left"):
            ax.spines[side].set_color("#dcdcd6")

    name = log.source.name if log.source else "QuSpin log"
    fig.suptitle(name, fontsize=13, color=INK, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)
    print(f"\nwrote {path.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logfile", nargs="?", default=None)
    parser.add_argument("--airborne", action="store_true",
                        help="cut the ground time off each end before reporting")
    args = parser.parse_args()

    log = read_log(args.logfile) if args.logfile else read_log()

    if args.airborne:
        log = log.airborne()
        print("(trimmed to the airborne section)\n")

    log.summary()
    repeatability(log)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_airborne" if args.airborne else ""
    stem = log.source.stem if log.source else "quspin"
    overview(log, OUTPUT_DIR / f"{stem}{suffix}.png")


if __name__ == "__main__":
    main()
