"""Getting a 250 Hz sensor onto a 10 Hz filter, and knowing how late it is.

Three separate delays, which get conflated and should not be:

    latency     the sensor's own. The reading labelled t was taken slightly
                before t, and the device may not say by how much.
    filter      the anti-alias low-pass has a group delay. A linear-phase FIR
                of N taps delays everything by exactly (N-1)/2 samples, which
                is why this module uses one: the delay is a number we can
                state rather than something that varies with frequency.
    decimation  dropping from 250 Hz to 10 Hz quantises the timestamp to the
                output interval, 100 ms.

Added up these are typically 50-150 ms, which at 8 m/s is 0.4 to 1.2 metres of
along-track position. That is small against a 50 m map cell and large against
the sensor's own 0.7 nT repeatability, so it matters for comparing readings
with each other and not much for the position fix. `decimate` returns the
total so the number is stated rather than assumed.

Why not a zero-phase filter
---------------------------
`filtfilt` would remove the delay entirely by running the filter backwards as
well as forwards. It is the right tool for analysing a recorded file and the
wrong one here, because it uses future samples. A real-time navigation filter
cannot do that, and a simulator that quietly does is flattering itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass
class DecimationResult:
    """Decimated samples, and an honest account of how late they are."""

    t: np.ndarray
    values: np.ndarray
    valid: np.ndarray
    filter_delay_s: float
    total_delay_s: float
    n_taps: int

    def __repr__(self) -> str:
        return (f"<DecimationResult {len(self.t)} samples, "
                f"{self.total_delay_s * 1e3:.0f} ms total delay>")


def decimate(t, values, valid, rate_in_hz: float, rate_out_hz: float,
             latency_s: float = 0.0, transition_width: float = 0.4,
             stopband_db: float = 40.0):
    """Anti-alias filter and downsample, reporting the delay.

        rate_out_hz       the filter's rate, 10 Hz by default
        latency_s         the sensor's own latency, added to the total
        transition_width  cutoff as a fraction of the output Nyquist. 0.4 of
                          5 Hz is a 2 Hz cutoff, which is far above anything
                          geological (below ~0.1 Hz) and well below the output
                          Nyquist, so nothing real is lost and nothing folds
                          back.
        stopband_db       how hard anything above the cutoff is rejected.
                          40 dB is a factor of 100 and costs about 0.37 s of
                          delay; 60 dB costs 0.7 s. This is the knob that
                          trades delay against aliasing.

    Returns a `DecimationResult`. The timestamps carry the full delay, so a
    reading sits at the moment it actually describes plus how long it took to
    arrive - which is what a navigation filter needs in order to associate it
    with the right position.
    """
    t = np.asarray(t, dtype=float)
    values = np.asarray(values, dtype=float)
    valid = np.asarray(valid, dtype=bool)

    if rate_out_hz > rate_in_hz:
        raise ValueError(
            f"output rate {rate_out_hz} Hz is above the input rate {rate_in_hz} Hz; "
            f"this decimates, it does not interpolate"
        )

    output_nyquist = rate_out_hz / 2.0
    cutoff_hz = transition_width * output_nyquist

    # Size the filter from what it actually has to achieve: `stopband_db` of
    # rejection, with the transition spread across the whole gap between the
    # cutoff and the output Nyquist. Kaiser's formula gives the tap count that
    # does that and no more.
    #
    # This is worth doing properly rather than by rule of thumb. An earlier
    # version asked for an unnecessarily sharp transition and got 501 taps -
    # a full second of group delay, which at 8 m/s is eight metres of
    # along-track position error. The honest design is a quarter of that.
    #
    # Even so the delay is not negligible, and it is the price of decimating
    # 25:1 in a single stage. A production system would cascade two stages
    # (250 -> 50 -> 10) and pay far less. We report the number instead.
    n_taps, beta = signal.kaiserord(stopband_db, (output_nyquist - cutoff_hz)
                                    / (rate_in_hz / 2))
    n_taps |= 1                      # odd, so the group delay is a whole sample
    n_taps = min(n_taps, max(11, len(values) // 4 | 1))
    taps = signal.firwin(n_taps, cutoff_hz, window=("kaiser", beta), fs=rate_in_hz)

    # A NaN would poison n_taps samples of output, so dead-zone gaps are
    # bridged for the filter and then marked invalid afterwards. The filter
    # sees a plausible number; the caller is told not to trust it.
    bridged = values.copy()
    if (~valid).any():
        if valid.any():
            bridged[~valid] = np.interp(t[~valid], t[valid], values[valid])
        else:
            bridged[:] = 0.0

    # Start the filter in steady state rather than from rest. A total field is
    # about 49,000 nT, so a filter starting from zero ramps up across its whole
    # length and the first output samples are tens of thousands of nanotesla
    # wrong. Seeding the state with the first sample removes that transient
    # instead of leaving it to be trimmed off afterwards and half-missed.
    zi = signal.lfilter_zi(taps, 1.0) * bridged[0]
    filtered, _ = signal.lfilter(taps, 1.0, bridged, zi=zi)

    filter_delay_samples = (n_taps - 1) // 2
    filter_delay_s = filter_delay_samples / rate_in_hz

    # Any invalid sample inside the filter window contaminates the output.
    contaminated = ~valid
    if contaminated.any():
        window = np.ones(n_taps, dtype=float)
        contaminated = np.convolve(contaminated.astype(float), window,
                                   mode="same") > 0

    step = int(round(rate_in_hz / rate_out_hz))
    picks = np.arange(0, len(values), step)
    # Drop the filter's start-up, where the output is still filling with taps.
    picks = picks[picks >= filter_delay_samples]

    total_delay_s = filter_delay_s + latency_s
    return DecimationResult(
        t=t[picks] + total_delay_s,
        values=filtered[picks],
        valid=~contaminated[picks],
        filter_delay_s=filter_delay_s,
        total_delay_s=total_delay_s,
        n_taps=n_taps,
    )
