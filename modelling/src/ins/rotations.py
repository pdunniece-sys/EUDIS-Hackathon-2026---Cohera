"""Quaternion and rotation helpers, all vectorised over the first axis.

Quaternions are (..., 4) arrays, **scalar first**: (w, x, y, z). Every one of
them in this package rotates body FRD to navigation NED, so:

    rotate(q, v_body) -> v_ned
    rotate_inverse(q, v_ned) -> v_body

Pass one quaternion as (4,) or a thousand as (1000, 4); the shape comes back
the way it went in.

Why hand-rolled rather than scipy.spatial.transform
---------------------------------------------------
Two reasons. `exp_map` and `log_map` here are exact inverses of each other, and
the mechanisation relies on that to reconstruct a perfect IMU from a known
trajectory - a round trip through a library's own small-angle handling is not
guaranteed to close. And scipy stores quaternions scalar-*last*, which is a
silent sign-and-order trap sitting in the middle of a navigation filter.
"""

from __future__ import annotations

import numpy as np

# Below this rotation angle (radians) the exponential and logarithmic maps use
# their Taylor expansions instead of dividing by a vanishing sine.
_SMALL_ANGLE = 1e-9


def normalise(q: np.ndarray) -> np.ndarray:
    """Scale quaternions back to unit length.

    Integrating attitude lets the norm drift, so this is called after every
    step. A zero quaternion cannot be normalised and raises rather than
    returning something arbitrary.
    """
    q = np.asarray(q, dtype=float)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm == 0):
        raise ValueError("cannot normalise a zero quaternion")
    return q / norm


def conjugate(q: np.ndarray) -> np.ndarray:
    """The inverse rotation, for unit quaternions."""
    q = np.asarray(q, dtype=float)
    return np.concatenate([q[..., :1], -q[..., 1:]], axis=-1)


def multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compose two rotations: the result applies `b` first, then `a`.

    So a body-frame increment is applied as `multiply(q, dq)` - the increment
    goes on the right, because it is expressed in the frame `q` starts from.
    """
    a = np.atleast_2d(np.asarray(a, dtype=float))
    b = np.atleast_2d(np.asarray(b, dtype=float))
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    out = np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)
    return out


def exp_map(rotvec: np.ndarray) -> np.ndarray:
    """Rotation vector (axis * angle, radians) -> quaternion.

    This is how a gyro reading becomes an attitude increment: multiply the
    body rates by the timestep and pass the result in.
    """
    rotvec = np.atleast_2d(np.asarray(rotvec, dtype=float))
    angle = np.linalg.norm(rotvec, axis=-1, keepdims=True)

    # sin(a/2)/a, written so it stays finite as the angle goes to zero.
    half = 0.5 * angle
    scale = np.where(angle < _SMALL_ANGLE, 0.5 - angle**2 / 48.0,
                     np.sin(half) / np.where(angle == 0, 1.0, angle))
    return np.concatenate([np.cos(half), scale * rotvec], axis=-1)


def log_map(q: np.ndarray) -> np.ndarray:
    """Quaternion -> rotation vector. The exact inverse of `exp_map`.

    Used to recover what a gyro must have measured between two known
    attitudes, which is how the synthetic IMU is built.
    """
    q = np.atleast_2d(normalise(q))
    # A quaternion and its negation are the same rotation; pick the one with
    # w >= 0 so the recovered angle is the short way round, never the long way.
    q = np.where(q[..., :1] < 0, -q, q)
    w = np.clip(q[..., :1], -1.0, 1.0)
    vec = q[..., 1:]
    sin_half = np.linalg.norm(vec, axis=-1, keepdims=True)

    angle = 2.0 * np.arctan2(sin_half, w)
    scale = np.where(sin_half < _SMALL_ANGLE, 2.0 + angle**2 / 12.0,
                     angle / np.where(sin_half == 0, 1.0, sin_half))
    return scale * vec


def to_matrix(q: np.ndarray) -> np.ndarray:
    """Quaternion -> (N, 3, 3) rotation matrix taking body vectors to NED."""
    q = np.atleast_2d(normalise(q))
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], axis=-2)


def rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Body frame -> NED."""
    m = to_matrix(q)
    v = np.atleast_2d(np.asarray(v, dtype=float))
    return np.einsum("nij,nj->ni", m, np.broadcast_to(v, (m.shape[0], 3)))


def rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """NED -> body frame."""
    m = to_matrix(q)
    v = np.atleast_2d(np.asarray(v, dtype=float))
    return np.einsum("nji,nj->ni", m, np.broadcast_to(v, (m.shape[0], 3)))


def from_euler(roll, pitch, yaw) -> np.ndarray:
    """Roll, pitch, yaw in **radians** -> quaternion.

    The usual aerospace sequence: yaw about down, then pitch about right, then
    roll about forward. Matches what ArduPilot's ATT message reports (in
    degrees, so convert before calling).
    """
    roll = np.atleast_1d(np.asarray(roll, dtype=float))
    pitch = np.atleast_1d(np.asarray(pitch, dtype=float))
    yaw = np.atleast_1d(np.asarray(yaw, dtype=float))
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.stack([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], axis=-1)


def to_euler(q: np.ndarray) -> np.ndarray:
    """Quaternion -> (N, 3) of roll, pitch, yaw in radians.

    Pitch is clamped at +-90 degrees, where roll and yaw stop being separable.
    Straight and level flight is nowhere near that, but a clamp beats a NaN
    out of arcsin if something upstream goes wrong.
    """
    q = np.atleast_2d(normalise(q))
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.stack([roll, pitch, yaw], axis=-1)


def from_matrix(m: np.ndarray) -> np.ndarray:
    """(N, 3, 3) body->NED rotation matrix -> quaternion.

    Used to build an attitude from axes worked out geometrically, which is how
    the synthetic trajectory turns "thrust points this way, nose points that
    way" into a quaternion.

    There are four algebraically equivalent ways to extract a quaternion from a
    matrix, and each one divides by a quantity that can vanish. All four are
    computed and the per-row winner is the one with the largest divisor, which
    is what keeps this stable at every attitude.
    """
    m = np.asarray(m, dtype=float).reshape(-1, 3, 3)
    m00, m01, m02 = m[:, 0, 0], m[:, 0, 1], m[:, 0, 2]
    m10, m11, m12 = m[:, 1, 0], m[:, 1, 1], m[:, 1, 2]
    m20, m21, m22 = m[:, 2, 0], m[:, 2, 1], m[:, 2, 2]

    # The four divisors, one per branch.
    divisors = np.stack([1 + m00 + m11 + m22, 1 + m00 - m11 - m22,
                         1 - m00 + m11 - m22, 1 - m00 - m11 + m22], axis=-1)
    best = np.argmax(divisors, axis=-1)
    s = np.sqrt(np.maximum(np.take_along_axis(divisors, best[:, None], 1)[:, 0], 1e-12))
    half_over_s = 0.5 / s

    candidates = np.stack([
        np.stack([0.5 * s, (m21 - m12) * half_over_s,
                  (m02 - m20) * half_over_s, (m10 - m01) * half_over_s], -1),
        np.stack([(m21 - m12) * half_over_s, 0.5 * s,
                  (m01 + m10) * half_over_s, (m02 + m20) * half_over_s], -1),
        np.stack([(m02 - m20) * half_over_s, (m01 + m10) * half_over_s,
                  0.5 * s, (m12 + m21) * half_over_s], -1),
        np.stack([(m10 - m01) * half_over_s, (m02 + m20) * half_over_s,
                  (m12 + m21) * half_over_s, 0.5 * s], -1),
    ], axis=1)

    q = np.take_along_axis(candidates, best[:, None, None], 1)[:, 0, :]
    return normalise(q)
