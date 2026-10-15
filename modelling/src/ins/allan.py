"""Allan deviation: read an IMU's error model off a log of it sitting still.

    from ins.allan import allan_deviation, fit_allan

    taus, adev = allan_deviation(gyro_x, rate_hz=400)
    fit = fit_allan(taus, adev)      # noise density, bias instability, random walk

The Allan deviation averages the signal over windows of length tau and asks
how much consecutive averages differ. Each error source has its own slope on
a log-log plot, which is why one static log separates them:

    slope -1/2   white noise.      sigma(1 s) is the noise density, per rtHz
    flat bottom  bias instability. minimum / 0.664 (the standard factor)
    slope +1/2   bias random walk. sigma = K sqrt(tau / 3)

What the fit maps onto in ErrorModel: noise density -> *_noise_density,
random walk K -> *_bias_rw. Bias instability has no field of its own in
ErrorModel; it is reported because it is the number datasheets quote.

What this deliberately does NOT do
----------------------------------
Measure turn-on bias or scale factor. Turn-on bias is a different number at
each power-up, so one log gives one draw of it, not its spread; scale factor
needs the sensor turned through known rates. Neither can come from Allan
variance, and both are left to the caller to source honestly.

It also assumes the input is stationary. A log that is still warming up has a
temperature-driven drift that looks like random walk and inflates K; cut the
warm-up off first.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def allan_deviation(x, rate_hz: float, n_taus: int = 60, min_clusters: int = 9):
    """Overlapping Allan deviation of an evenly sampled signal.

    Returns `(taus, adev)` in seconds and the signal's own units. Taus are
    log-spaced up to the length that still leaves `min_clusters` independent
    averages, past which the estimate is too noisy to read.
    """
    x = np.asarray(x, dtype=float)
    max_m = max(1, len(x) // min_clusters)
    ms = np.unique(np.logspace(0, np.log10(max_m), n_taus).astype(int))
    c = np.cumsum(np.r_[0.0, x])
    adev = np.empty(len(ms))
    for i, m in enumerate(ms):
        means = (c[m:] - c[:-m]) / m
        diff = means[m:] - means[:-m]
        adev[i] = np.sqrt(0.5 * np.mean(diff**2))
    return ms / rate_hz, adev


@dataclass
class AllanFit:
    noise_density: float      # units / sqrt(Hz)
    bias_instability: float   # units
    instability_tau_s: float
    bias_rw: float            # units / sqrt(s); NaN if the curve never turns up


def fit_allan(taus, adev) -> AllanFit:
    """Read the three standard terms off an Allan deviation curve.

    Noise density is sigma at tau = 1 s, which assumes the curve is still on
    its -1/2 slope there - true for MEMS parts, whose bias instability bottom
    is tens of seconds out. The random walk is fitted only to the part of the
    curve after the minimum; if there are fewer than three points there the
    log was too short to see it, and NaN says so rather than guessing.
    """
    taus, adev = np.asarray(taus), np.asarray(adev)
    i = int(np.argmin(adev))
    after = taus > taus[i]
    k = (float(np.median(adev[after] / np.sqrt(taus[after] / 3.0)))
         if after.sum() >= 3 else float("nan"))
    return AllanFit(noise_density=float(np.interp(1.0, taus, adev)),
                    bias_instability=float(adev[i] / 0.664),
                    instability_tau_s=float(taus[i]),
                    bias_rw=k)
