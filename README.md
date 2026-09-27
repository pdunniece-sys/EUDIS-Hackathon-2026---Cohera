# Cohera — magnetic anomaly navigation

Navigating without GNSS: a magnetometer reads the local magnetic field, and the
reading is matched against a published anomaly map to work out position.

EUDIS Defence Hackathon 2026, Challenge 1.

## Layout

| | |
|---|---|
| [modelling/](modelling/) | Map sampling, flight-log parsing, analysis. See its [README](modelling/README.md). |
| `hardware/` | Drawings, CAD, engineering documents. Placeholder. |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e "./modelling[dev]"
```

`pytest` and the scripts run from inside `modelling/`.

## Data

Not in git — 240 MB, and the logs aren't public.

- Tellus grids: `python scripts/fetch_tellus.py` from `modelling/`.
- OPM logs: ask repo owner
