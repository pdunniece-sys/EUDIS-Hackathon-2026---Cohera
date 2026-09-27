# Magnetic anomaly navigation — Dublin Hackathon

Navigating by the Earth's magnetic field instead of GNSS: a magnetometer on an
aircraft reads the local field, and the reading is matched against a published
magnetic anomaly map to work out where the aircraft is.

This repo holds the whole team's work. It is split by discipline, one folder
each, and each folder has its own README explaining what is in it.

## Layout

| Folder | What lives there | State |
|---|---|---|
| [modelling/](modelling/) | Simulation and modelling. Magnetic map sampling, magnetometer flight-log parsing, the supporting analysis. | Working |
| [hardware/](hardware/) | Working drawings, CAD, engineering documents, BOM. | Placeholder |

`modelling/` is a small part of the project and is the only part with code in it
so far. Start with [modelling/README.md](modelling/README.md) — it documents the
data, the pitfalls in it, and what has been verified.

`Main.py` at the top level predates this structure and is empty. It is kept only
so nobody wonders where it went; delete it whenever you like.

## Getting set up

Python work is all under `modelling/`, and one virtual environment at the repo
root serves it:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e "./modelling[dev]"
```

Then, from `modelling/`:

```bash
python scripts/fetch_tellus.py     # downloads the magnetic grid, ~93 MB
pytest                             # 16 checks
```

Run `pytest` from inside `modelling/` rather than the repo root — that is where
the test configuration lives.

## Data is not in this repo

**Nothing under `modelling/data/` is tracked by git**, and neither is
`modelling/outputs/`. The survey grids and flight logs come to about 240 MB,
which is past what GitHub will accept in ordinary git objects, so a fresh clone
arrives without them. Two different situations:

- **The Tellus magnetic grids** are public and reproducible. Run
  `python scripts/fetch_tellus.py` from `modelling/` and you have them.
- **The QuSpin flight logs** are not public. They have to be copied in by hand,
  into `modelling/data/QuSpin/`. Ask Peter for them.

If you add data of your own, put it under the relevant folder and say in that
folder's README where it came from and how to get it again. Do not commit it.

## Keeping the repo trackable

A few conventions, so this does not turn into a pile:

- **One folder per discipline**, each with a README that stands on its own. Add a
  row to the table above when you add a folder.
- **Binaries and data stay out of git.** Drawings and engineering documents are
  the exception — those are the deliverable, so commit them, but export a PDF
  alongside any native CAD file so the rest of us can open it.
- **Say where numbers came from.** If a value in a document or a slide came out
  of code in here, name the script or notebook that produced it. Half the value
  of `modelling/` is that its claims are traceable to the data.
- **Notebooks are committed with their outputs intact**, so they can be read
  without running them. Re-run before committing if you changed the code.

## Attribution

The Tellus magnetic data is Geological Survey Ireland, under CC BY 4.0, and
anything published from it must carry their attribution notice. The exact
wording, and the separate notice needed for the Northern Ireland coverage, is in
[modelling/README.md](modelling/README.md#licence-and-attribution).
