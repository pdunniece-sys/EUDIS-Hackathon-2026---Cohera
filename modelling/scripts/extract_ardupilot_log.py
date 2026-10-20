#!/usr/bin/env python3
"""Turn an ArduPilot DataFlash .bin log into one CSV per message type.

    python scripts/extract_ardupilot_log.py ../hardware/flight_logs/2026-10-15_static_imu_test.bin
    python scripts/extract_ardupilot_log.py LOG.bin --all          # every message type
    python scripts/extract_ardupilot_log.py LOG.bin --types IMU XKF1

A .bin is ArduPilot's own binary format - nothing on a Mac opens it, which is
why it looks broken. pymavlink reads it. The CSVs land in
data/ardupilot/<log name>/, one per message type (IMU.csv, XKF1.csv, ...),
plus PARM.csv for the parameters the aircraft was flying with.

Every CSV gets a `time_s` column: seconds from the first message in the log,
so all of them share one clock. Values are in the units ArduPilot logs, which
for the ones that matter here are SI already (IMU accel m/s^2, gyro rad/s,
temperature degC), except EKF gyro biases in XKF1 GX/GY/GZ, which are deg/s.

Rows with several instances (IMU, BARO, GPS, XKF*) keep the instance column
(`I` or `C`) rather than being split, so filter on it.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tellus.products import ROOT  # noqa: E402

# What a static IMU test or a flight analysis needs. --all takes everything.
DEFAULT_TYPES = ["IMU", "XKF1", "XKF2", "XKF4", "HEAT", "BARO", "ATT", "GPS",
                 "MAG", "VIBE", "BAT", "MODE", "MSG", "PARM"]


def extract(path: Path, types=None, out_dir: Path | None = None) -> Path:
    from pymavlink import DFReader

    reader = DFReader.DFReader_binary(str(path))
    rows: dict[str, list] = {}
    t0 = None
    while True:
        msg = reader.recv_match(type=types) if types else reader.recv_match()
        if msg is None:
            break
        kind = msg.get_type()
        if kind in ("FMT", "FMTU", "MULT", "UNIT"):
            continue
        d = msg.to_dict()
        d.pop("mavpackettype", None)
        if "TimeUS" in d:
            t0 = d["TimeUS"] if t0 is None else t0
            d["time_s"] = (d["TimeUS"] - t0) / 1e6
        rows.setdefault(kind, []).append(d)

    out_dir = out_dir or ROOT / "data" / "ardupilot" / path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    for kind, data in sorted(rows.items()):
        frame = pd.DataFrame(data)
        if "time_s" in frame:
            frame = frame[["time_s"] + [c for c in frame.columns if c != "time_s"]]
        frame.to_csv(out_dir / f"{kind}.csv", index=False)
        print(f"  {kind:6s} {len(frame):9d} rows")
    print(f"wrote {out_dir.relative_to(ROOT) if out_dir.is_relative_to(ROOT) else out_dir}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", type=Path)
    parser.add_argument("--types", nargs="+", default=DEFAULT_TYPES)
    parser.add_argument("--all", action="store_true", help="every message type")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    extract(args.log, None if args.all else args.types, args.out)


if __name__ == "__main__":
    main()
