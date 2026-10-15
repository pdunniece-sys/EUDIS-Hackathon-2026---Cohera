#!/usr/bin/env python3
"""What a magnetometer would read flying a survey, and how wrong it would be.

    python scripts/run_opm_demo.py                  # 20 min flight over A7
    python scripts/run_opm_demo.py --minutes 10
    python scripts/run_opm_demo.py --mains-hz 60    # as if flown in Ontario
    python scripts/run_opm_demo.py --interference   # switch the drone's own field on

Flies a synthetic lawnmower over Tellus block A7 at 60 m, works out the true
field along it, and emulates two sensors reading it: a QTFM-like scalar OPM
and an RM3100-like vector magnetometer. Prints the RMS error of each against
the map and writes one four-panel plot to outputs/.

The comparison this is for
--------------------------
Both sensors are trying to report the same thing - the crustal anomaly, tens
of nT, which is the only part of the field that says where you are. The OPM
measures the field's magnitude directly. The vector sensor measures three
components and computes the magnitude, so its per-axis errors enter weighted by
how the aircraft happens to be pointing.

The bottom-right panel is the whole argument. If the vector sensor's error were
merely large, calibration would fix it. What the panel shows is that the error
*moves with heading*, which on a lawnmower pattern means it moves with the
survey lines - so it looks exactly like geology.
"""

import argparse
import datetime as dt
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ins import lawnmower, synthetic_flight  # noqa: E402
from sensor import (  # noqa: E402
    FieldModel,
    ParametricInterference,
    ScalarOPM,
    VectorMagnetometer,
)
from tellus import MagneticGrid  # noqa: E402
from tellus.products import ROOT  # noqa: E402

OUTPUT_DIR = ROOT / "outputs"
INK = "#3d3d3a"
MUTED = "#8a8a82"
OPM_COLOUR = "#2f6f4f"
VECTOR_COLOUR = "#9c3a2e"
TRUTH_COLOUR = "#3d3d3a"

START_LAT, START_LON = 52.45, -6.40
ALTITUDE_M = 60.0
SPEED_MPS = 8.0
SENSOR_RATE_HZ = 250.0


def build_flight(minutes: float):
    """A lawnmower sized to take about `minutes` to fly."""
    spacing_m = 150.0
    path_m = minutes * 60.0 * SPEED_MPS
    line_m = np.sqrt(path_m * spacing_m)
    n_lines = max(2, int(np.ceil(path_m / (line_m + spacing_m))))

    waypoints = lawnmower((START_LAT, START_LON), length_m=line_m,
                          width_m=(n_lines - 1) * spacing_m, spacing_m=spacing_m)
    truth, _ = synthetic_flight(waypoints, alt_m=ALTITUDE_M, speed_mps=SPEED_MPS,
                                rate_hz=SENSOR_RATE_HZ)
    print(f"flight: {truth.duration_s / 60:.1f} min, {line_m:.0f} m lines, "
          f"{n_lines} of them, at {ALTITUDE_M:.0f} m")
    return truth


def score(sensor, readings, truth, model, field):
    """Error of a sensor's reported anomaly against the map it should match.

    The readings are late - sensor latency plus filter delay - so the truth is
    looked up at the moment each reading was *taken*, not the moment it was
    reported. Skipping that step would show a spurious error that is really
    just the delay.
    """
    anomaly = sensor.to_anomaly(readings, truth, model)
    taken_at = np.clip(readings.t - readings.latency_s, truth.t[0], truth.t[-1])

    true_anomaly = np.interp(taken_at, truth.t, field.anomaly_nt)
    heading = np.interp(taken_at, truth.t,
                        np.degrees(np.arctan2(truth.vel_ned[:, 1],
                                              truth.vel_ned[:, 0])) % 360)

    error = anomaly - true_anomaly
    good = np.isfinite(error)
    rms = float(np.sqrt(np.mean(error[good] ** 2))) if good.any() else float("nan")
    return dict(t=taken_at, anomaly=anomaly, true_anomaly=true_anomaly,
                error=error, heading=heading, rms=rms, readings=readings)


def heading_spread(result, n_bins: int = 12) -> float:
    """How much the mean error moves as the aircraft turns.

    The number that separates the two sensors. A scalar OPM should be near
    zero here; a vector sensor should not.
    """
    edges = np.linspace(0, 360, n_bins + 1)
    bins = np.digitize(result["heading"], edges)
    means = [np.nanmean(result["error"][bins == b]) for b in range(1, n_bins + 1)
             if np.isfinite(result["error"][bins == b]).sum() > 20]
    return float(np.ptp(means)) if len(means) > 2 else float("nan")


def plot(results, path: Path, subtitle: str):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    opm, vector = results["OPM"], results["vector"]

    # 1. The anomaly each sensor reports, against the map.
    ax = axes[0][0]
    ax.plot(opm["t"] / 60, opm["true_anomaly"], lw=1.4, color=TRUTH_COLOUR,
            label="true anomaly (map)")
    ax.plot(opm["t"] / 60, opm["anomaly"], lw=0.9, color=OPM_COLOUR, label="OPM")
    ax.set_ylabel("anomaly (nT)")
    ax.set_xlabel("minutes")
    ax.set_title("the OPM tracks the map", loc="left", color=INK)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)

    # 2. The vector sensor on the same axes, which needs a different scale.
    ax = axes[0][1]
    ax.plot(vector["t"] / 60, vector["true_anomaly"], lw=1.4, color=TRUTH_COLOUR,
            label="true anomaly (map)")
    ax.plot(vector["t"] / 60, vector["anomaly"], lw=0.9, color=VECTOR_COLOUR,
            label="RM3100-like")
    ax.set_ylabel("anomaly (nT)")
    ax.set_xlabel("minutes")
    ax.set_title("the vector sensor does not", loc="left", color=INK)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)

    # 3. Error against time.
    ax = axes[1][0]
    ax.plot(opm["t"] / 60, opm["error"], lw=0.8, color=OPM_COLOUR,
            label=f"OPM, {opm['rms']:.1f} nT RMS")
    ax.plot(vector["t"] / 60, vector["error"], lw=0.8, color=VECTOR_COLOUR,
            label=f"vector, {vector['rms']:.0f} nT RMS")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_ylabel("error (nT)")
    ax.set_xlabel("minutes")
    ax.set_yscale("symlog", linthresh=10)
    ax.set_title("error against time", loc="left", color=INK)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)

    # 4. Error against heading - the point of the whole comparison.
    ax = axes[1][1]
    for name, result, colour in (("OPM", opm, OPM_COLOUR),
                                 ("vector", vector, VECTOR_COLOUR)):
        edges = np.linspace(0, 360, 25)
        centres = 0.5 * (edges[:-1] + edges[1:])
        bins = np.digitize(result["heading"], edges)
        means = np.array([np.nanmean(result["error"][bins == b])
                          if np.isfinite(result["error"][bins == b]).sum() > 5
                          else np.nan for b in range(1, len(edges))])
        ax.plot(centres, means, "o-", ms=3, lw=1.3, color=colour,
                label=f"{name}, swings {heading_spread(result):.1f} nT")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("heading (deg)")
    ax.set_ylabel("mean error (nT)")
    ax.set_yscale("symlog", linthresh=10)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_title("error against heading", loc="left", color=INK)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)

    fig.suptitle(f"Magnetometer emulation over Tellus A7 — {subtitle}",
                 color=MUTED, fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--minutes", type=float, default=20.0)
    parser.add_argument("--mains-hz", type=float, default=50.0,
                        help="50 in Ireland, 60 in North America")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--interference", action="store_true",
                        help="switch on a plausible drone field (100 nT permanent)")
    parser.add_argument("--product", default="a7_tmi")
    args = parser.parse_args()

    truth = build_flight(args.minutes)

    grid = MagneticGrid(args.product).subset(
        (START_LON - 0.1, START_LAT - 0.1, START_LON + 0.1, START_LAT + 0.1))
    print(f"map: {grid}")

    interference = (ParametricInterference(permanent_nt=[100.0, -40.0, 60.0])
                    if args.interference else None)
    model = FieldModel(grid=grid, date=dt.date(2026, 10, 15),
                       **({"interference": interference} if interference else {}))

    # Build the field once; both sensors read the same reality.
    field = model.at(truth)
    print(f"field: {field}")
    print(f"       core {field.core_nt.mean():.0f} nT, "
          f"anomaly {np.nanmin(field.anomaly_nt):.0f} to {np.nanmax(field.anomaly_nt):.0f} nT, "
          f"diurnal +-{np.abs(field.diurnal_nt).max():.0f} nT")

    opm = ScalarOPM(mains_hz=args.mains_hz, rate_hz=SENSOR_RATE_HZ)
    vector = VectorMagnetometer(rate_hz=SENSOR_RATE_HZ)

    results = {
        "OPM": score(opm, opm.emulate(truth, field, seed=args.seed), truth, model, field),
        "vector": score(vector, vector.emulate(truth, field, seed=args.seed),
                        truth, model, field),
    }

    theta = np.degrees(opm.theta(truth, field))
    print(f"\nsensor geometry: field-to-axis angle spans {theta.min():.0f} to "
          f"{theta.max():.0f} deg (dead zone below {opm.dead_zone_deg:.0f})")
    print(f"readings: {results['OPM']['readings']}")

    print("\nRMS error against the map")
    print(f"  {'sensor':<16} {'RMS':>12}  {'swing with heading':>20}")
    print(f"  {'-' * 16} {'-' * 12}  {'-' * 20}")
    for name, result in results.items():
        print(f"  {name:<16} {result['rms']:>9.1f} nT  "
              f"{heading_spread(result):>17.1f} nT")

    ratio = results["vector"]["rms"] / results["OPM"]["rms"]
    print(f"\n  the vector sensor is {ratio:.0f}x worse, and its error moves with")
    print(f"  heading, which on a lawnmower pattern is indistinguishable from geology.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_interference" if args.interference else ""
    plot(results, OUTPUT_DIR / f"opm_emulation{suffix}.png",
         f"{truth.duration_s / 60:.0f} min, {args.mains_hz:.0f} Hz mains"
         + (", drone field on" if args.interference else ""))


if __name__ == "__main__":
    main()
