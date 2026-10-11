<p align="center">
  <img src="Branding/Lockup,%20ket@2x.png" alt="Cohera — Quantum Navigation" width="100%">
</p>

# Cohera — quantum navigation

Navigation that keeps working when GNSS is jammed, spoofed or simply absent.

The Earth's crust has a magnetic fingerprint: small, stable variations in field
strength from place to place that cannot be jammed or faked from a distance.
Cohera flies a quantum magnetometer, reads the local field, and matches that
reading against a published magnetic anomaly map to work out where it is.

Built for the EUDIS Defence Hackathon 2026, Challenge 1.

## How it works

1. **Sense** — an optically pumped magnetometer (QuSpin QTFM) measures total
   field at 250 Hz from a drone.
2. **Clean** — interference is filtered out and the Earth's background field
   removed, leaving the anomaly.
3. **Match** — the anomaly trace along the flight path is compared with the
   Tellus airborne survey of Ireland (Geological Survey Ireland, 50 m cells) to
   fix position.

## What's here

| | |
|---|---|
| [modelling/](modelling/) | Map sampling, flight-log parsing and analysis. Start with its [README](modelling/README.md). |
| [hardware/](hardware/) | Drawings, CAD and engineering documents. Placeholder for now. |
| [Branding/](Branding/) | Logo and lockup. |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e "./modelling[dev]"
```

`pytest` and the scripts run from inside `modelling/`.

## Data

Not in git — 240 MB, and the flight logs aren't public.

- **Tellus grids:** `python scripts/fetch_tellus.py` from `modelling/`.
- **Magnetometer logs:** ask the repo owner.

Contains Irish Public Sector Data (Geological Survey Ireland) licensed under a
Creative Commons Attribution 4.0 International (CC BY 4.0) licence.
