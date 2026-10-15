"""The drone's own magnetic field, as seen by a sensor bolted to it.

    from sensor import ParametricInterference

    interference = ParametricInterference(permanent_nt=[20, -5, 40])

Every interference model is a callable taking a trajectory and returning an
`(n, 3)` field in **body frame**, nanotesla. Body frame because that is where
the aircraft's own field lives: it turns with the aircraft, which is exactly
what makes it hard to tell apart from real geology, and exactly what makes it
show up as a heading-dependent error.

The default is zero
-------------------
`NoInterference` is the default everywhere, so interference is something you
switch on deliberately rather than something quietly inflating your error bars.
That matters because we have no measurement of our own drone's signature yet -
a plausible-looking default would be a number we invented appearing in results.

The three terms, and why aeromagnetics splits them this way
-----------------------------------------------------------
This is the standard decomposition behind Tolles-Lawson compensation, the
method every aeromagnetic survey uses:

  permanent   the airframe's own magnetisation. Fixed in the body frame, so in
              the navigation frame it swings around as the aircraft turns.
  induced     the Earth's field magnetising soft iron in the airframe. Depends
              on the aircraft's attitude relative to the field, so it is a
              matrix acting on the field expressed in body coordinates.
  current     motor and battery currents. Roughly proportional to throttle,
              and the term that is worst on a multirotor because the power
              wiring runs the length of the airframe.

Fitting these to a real aircraft needs a dedicated calibration flight - a box
of pitch, roll and yaw manoeuvres at altitude over magnetically quiet ground.
We have not flown one. Until we do, `ArrayInterference` is the hook for
dropping in measured values.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ins import rotations


class NoInterference:
    """A magnetically silent aircraft. The default, and a fiction.

    Real drones are not magnetically silent. This exists so that interference
    is always an explicit choice.
    """

    def __call__(self, trajectory) -> np.ndarray:
        return np.zeros((len(trajectory), 3))

    def __repr__(self) -> str:
        return "<NoInterference>"


@dataclass
class ParametricInterference:
    """Permanent, induced and current-driven interference, in body frame.

        permanent_nt   (3,) fixed magnetisation of the airframe, nT
        induced        (3, 3) soft-iron matrix. The induced field is
                       `induced @ B_body`, so entries are dimensionless and
                       of order 1e-3 for a small airframe.
        current_nt_per_amp  (3,) field per amp of motor current
        current_a      a callable t -> amps, or a constant. Default 20 A,
                       a plausible hover draw for a survey multirotor.

    None of these defaults are measured. They are zero unless you set them.
    """

    permanent_nt: np.ndarray = field(default_factory=lambda: np.zeros(3))
    induced: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    current_nt_per_amp: np.ndarray = field(default_factory=lambda: np.zeros(3))
    current_a: object = 20.0

    def __post_init__(self):
        self.permanent_nt = np.asarray(self.permanent_nt, dtype=float).reshape(3)
        self.induced = np.asarray(self.induced, dtype=float).reshape(3, 3)
        self.current_nt_per_amp = np.asarray(
            self.current_nt_per_amp, dtype=float).reshape(3)

    def __call__(self, trajectory, b_ned=None) -> np.ndarray:
        """Body-frame interference, (n, 3) nT.

        `b_ned` is the ambient field, needed only for the induced term. Without
        it the induced term is skipped, because inventing an ambient field to
        magnetise the airframe with would be worse than leaving the term out.
        """
        n = len(trajectory)
        out = np.tile(self.permanent_nt, (n, 1))

        if b_ned is not None and np.any(self.induced):
            if trajectory.att_quat is None:
                raise ValueError(
                    "the induced term needs attitude - it depends on how the "
                    "airframe sits in the Earth's field"
                )
            b_body = rotations.rotate_inverse(trajectory.att_quat, b_ned)
            out = out + b_body @ self.induced.T

        if np.any(self.current_nt_per_amp):
            amps = (self.current_a(trajectory.t) if callable(self.current_a)
                    else np.full(n, float(self.current_a)))
            out = out + np.asarray(amps).reshape(-1, 1) * self.current_nt_per_amp

        return out

    def __repr__(self) -> str:
        return (f"<ParametricInterference permanent={self.permanent_nt} nT, "
                f"induced={'set' if np.any(self.induced) else 'off'}, "
                f"current={'set' if np.any(self.current_nt_per_amp) else 'off'}>")


@dataclass
class ArrayInterference:
    """Interference measured on our own aircraft, interpolated onto the flight.

        t          seconds
        body_nt    (n, 3) body-frame field, nT

    This is where a Tolles-Lawson fit from a real calibration flight goes once
    we have flown one. Until then it has no data to hold.
    """

    t: np.ndarray
    body_nt: np.ndarray

    def __post_init__(self):
        self.t = np.asarray(self.t, dtype=float)
        self.body_nt = np.asarray(self.body_nt, dtype=float).reshape(-1, 3)
        if len(self.t) != len(self.body_nt):
            raise ValueError(
                f"t has {len(self.t)} samples, body_nt has {len(self.body_nt)}")

    def __call__(self, trajectory, b_ned=None) -> np.ndarray:
        return np.column_stack([
            np.interp(trajectory.t, self.t, self.body_nt[:, i]) for i in range(3)
        ])

    def __repr__(self) -> str:
        return f"<ArrayInterference {len(self.t)} samples>"
