"""What the emulator hands back, and what it needs to get there.

Two types cross module boundaries:

    FieldSamples  the true magnetic field at the aircraft, sample by sample
    Readings      what a sensor reported, which is not the same thing

Keeping them apart is the point. `FieldSamples` is physics - what is actually
there. `Readings` is instrumentation - what a device claimed, after its noise,
its blind spots and its filter delay. The whole module exists to turn the first
into the second, and a navigation filter only ever sees the second.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class FieldSamples:
    """The true field at the aircraft, broken into the parts it is made of.

        t              seconds, matching the trajectory
        b_ned          (n, 3) total field vector in NED, nT
        core_nt        (n,) magnitude of the IGRF core field alone
        anomaly_nt     (n,) crustal anomaly from the map, at flight altitude
        diurnal_nt     (n,) the daily variation model
        interference_ned  (n, 3) the platform's own field, in NED, nT

    The parts are kept because the interesting questions are all about which
    part dominates. `b_ned` is their sum, and `total_nt` is what a flawless
    scalar magnetometer would read.
    """

    t: np.ndarray
    b_ned: np.ndarray
    core_nt: np.ndarray
    anomaly_nt: np.ndarray
    diurnal_nt: np.ndarray
    interference_ned: np.ndarray

    def __post_init__(self):
        self.t = np.asarray(self.t, dtype=float)
        self.b_ned = np.asarray(self.b_ned, dtype=float).reshape(-1, 3)

    def __len__(self) -> int:
        return len(self.t)

    @property
    def total_nt(self) -> np.ndarray:
        """Magnitude of the true field: what a perfect scalar sensor would read."""
        return np.linalg.norm(self.b_ned, axis=1)

    @property
    def b_hat(self) -> np.ndarray:
        """(n, 3) unit vector along the field."""
        return self.b_ned / np.linalg.norm(self.b_ned, axis=1, keepdims=True)

    def __repr__(self) -> str:
        return (f"<FieldSamples {len(self)} samples, "
                f"total {self.total_nt.mean():.0f} nT, "
                f"anomaly {np.nanmin(self.anomaly_nt):.0f} to "
                f"{np.nanmax(self.anomaly_nt):.0f} nT>")


@dataclass
class Readings:
    """What a sensor reported.

        t          seconds. May differ from the trajectory's clock: a sensor
                   has a latency and may run at its own rate.
        field_nt   total field as reported, including every error
        valid      False where the reading is unusable - in a dead zone, or
                   contaminated by one through the anti-alias filter

    `field_nt` is NaN wherever `valid` is False. Both are kept because they
    answer different questions: `valid` says why, NaN makes it impossible to
    use the number by accident.
    """

    t: np.ndarray
    field_nt: np.ndarray
    valid: np.ndarray
    name: str = ""
    rate_hz: float = float("nan")
    latency_s: float = 0.0
    source: Path | None = None

    def __post_init__(self):
        self.t = np.asarray(self.t, dtype=float)
        self.field_nt = np.asarray(self.field_nt, dtype=float)
        self.valid = np.asarray(self.valid, dtype=bool)
        if not (len(self.t) == len(self.field_nt) == len(self.valid)):
            raise ValueError(
                f"length mismatch: t={len(self.t)}, field={len(self.field_nt)}, "
                f"valid={len(self.valid)}"
            )
        # A reading that is not valid must not carry a usable-looking number.
        self.field_nt = np.where(self.valid, self.field_nt, np.nan)

    def __len__(self) -> int:
        return len(self.t)

    @property
    def fraction_valid(self) -> float:
        return float(self.valid.mean())

    def __repr__(self) -> str:
        return (f"<Readings {self.name!r} {len(self)} samples @ {self.rate_hz:.1f} Hz, "
                f"{self.fraction_valid * 100:.1f}% valid, "
                f"{self.latency_s * 1e3:.0f} ms behind>")
