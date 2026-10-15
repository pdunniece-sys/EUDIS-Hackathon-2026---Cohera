"""Particle filter v1: correct a free-running INS by matching the magnetic map.

    from navfilter import ParticleFilter, PFConfig

    pf = ParticleFilter(filter_map, PFConfig(grade="consumer_mems"))
    run = pf.run(ins_run.estimate, taken_at, anomaly_nt, truth=truth)
    run.summary()

An error-state filter, not a cloud of INSes
-------------------------------------------
The INS is integrated once - the same corrupted run used as the INS-only
baseline, so the comparison is like for like. Each particle is a hypothesis
about how wrong that one INS is, horizontally:

    x_i = [dN, dE, dvN, dvE]      metres and m/s
    corrected position = INS position + (dN, dE)

Running a full mechanisation per particle would cost N times as much and buy
nothing here: the vertical is barometer-aided and attitude does not enter the
map likelihood. The price is that the cloud's motion model is a crude one -
constant velocity error plus a random walk - so `q_vel` has to be large enough
to cover how the real INS error actually grows.

One cycle, at `update_hz`
-------------------------
    predict   dp += dv dt;  dv += N(0, q_vel sqrt(dt))
    weight    log w += -0.5 ((z - map(INS + dp)) / sigma)^2, -inf off the map
    estimate  weighted mean and 2x2 covariance of (dN, dE)
    resample  systematic, when ESS < N/2, then roughen

Why 2 Hz when the OPM gives 10
------------------------------
Consecutive 10 Hz readings are not independent: the platform noise is red
(65% of its variance below 0.3 Hz) and the map error is correlated over
hundreds of metres, which the aircraft crosses in tens of seconds. Treating
every reading as independent multiplies the same error into the weights five
times and the cloud collapses onto a confident wrong answer. Every reading
that is used is still an exact sample, not an average.

No peeking
----------
`run` sees the INS estimate, the measured anomaly and its own map. `truth`
is accepted only so the result can be scored after the loop has finished;
nothing inside the loop touches it.

What this deliberately does NOT do
----------------------------------
- Feed corrections back into the INS. It is open loop, so INS errors keep
  growing underneath and the particles have to keep chasing them.
- Estimate IMU biases. A constant velocity error is the only systematic term
  the state carries.
- Model the vertical. The barometer has that.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ins.errors import GRADES
from ins.types import Trajectory, metres_per_degree

G0 = 9.80665

# Velocity-error random walk per grade, m/s/sqrt(s), before the x3 inflation.
# Accelerometer bias random walk plus gravity times the gyro bias random walk
# (a tilt error leaks g into the horizontal). This is an engineering
# approximation of how fast the INS velocity error wanders, not a derivation,
# and both inputs are themselves ASSUMED in ins/errors.py.
Q_VEL_INFLATION = 3.0
Q_VEL_FLOOR = 1e-3   # m/s/sqrt(s); keeps a perfect-IMU cloud from freezing


def default_q_vel(grade: str | None) -> float:
    """Process noise for the velocity error, derived from an IMU grade."""
    if grade is None:
        return Q_VEL_FLOOR
    model = GRADES[grade]
    q = model.accel_bias_rw + G0 * model.gyro_bias_rw
    return max(Q_VEL_INFLATION * q, Q_VEL_FLOOR)


@dataclass
class PFConfig:
    """Everything that sets the filter's behaviour, in one place.

        n_particles      cloud size
        sigma_pos0_m     position spread at the GNSS cut. 30 m is a warm
                         handover from a GNSS-aided filter; 500 m is a cold
                         start
        sigma_vel0_mps   velocity spread at the cut
        grade            IMU grade name, used only for the default q_vel
        q_vel            velocity random walk, m/s/sqrt(s). None derives it
                         from `grade`
        sigma_*_nt       measurement error budget, combined in quadrature:
                         sensor (OPM after decimation), map (gridding and
                         levelling, see maps.py), interference (platform,
                         unmeasured). The defaults total about 5 nT and are
                         ASSUMED
        sigma_total_nt   overrides the budget with one number
        update_hz        measurement updates per second (see module docstring)
        snapshot_every   keep the cloud every this many updates, for replay
    """

    n_particles: int = 2000
    sigma_pos0_m: float = 30.0
    sigma_vel0_mps: float = 0.2
    grade: str | None = None
    q_vel: float | None = None
    sigma_sensor_nt: float = 2.0
    sigma_map_nt: float = 4.0
    sigma_interf_nt: float = 2.0
    sigma_total_nt: float | None = None
    update_hz: float = 2.0
    resample_frac: float = 0.5
    roughen: float = 0.2
    roughen_floor_m: float = 1.0
    roughen_floor_mps: float = 0.01
    snapshot_every: int = 10
    seed: int | None = 0

    @property
    def sigma_nt(self) -> float:
        if self.sigma_total_nt is not None:
            return float(self.sigma_total_nt)
        return float(np.sqrt(self.sigma_sensor_nt**2 + self.sigma_map_nt**2
                             + self.sigma_interf_nt**2))

    @property
    def q(self) -> float:
        return self.q_vel if self.q_vel is not None else default_q_vel(self.grade)


@dataclass
class PFRun:
    """One filter run, scored against truth if truth was given.

        t            update times (the first entry is the GNSS cut)
        estimate     INS track plus the mean offset, at `t`
        cov          (n, 2, 2) north/east position covariance, m^2
        ess          effective sample size after each weighting
        error_m      horizontal error of `estimate` against truth
        ins_error_m  the INS-only error at the same times
        snapshots    list of dicts: t, particles (N, 4), weights, ins_lat, ins_lon
    """

    t: np.ndarray
    estimate: Trajectory
    cov: np.ndarray
    ess: np.ndarray
    error_m: np.ndarray
    ins_error_m: np.ndarray
    snapshots: list = field(default_factory=list)
    t_cut: float = 0.0
    name: str = ""
    n_skipped: int = 0

    @property
    def elapsed_s(self) -> np.ndarray:
        return self.t - self.t_cut

    @property
    def sigma2_m(self) -> np.ndarray:
        """2-sigma horizontal radius from the covariance (sqrt of the trace)."""
        return 2.0 * np.sqrt(np.trace(self.cov, axis1=1, axis2=2))

    def _at(self, series: np.ndarray, seconds: float) -> float:
        if seconds > self.elapsed_s[-1]:
            return float("nan")
        return float(np.interp(seconds, self.elapsed_s, series))

    def summary(self) -> None:
        print(f"{self.name}, {self.elapsed_s[-1] / 60:.1f} min after the cut"
              + (f", {self.n_skipped} updates skipped off-map" if self.n_skipped else ""))
        print(f"   {'':6} {'INS only':>10} {'INS+MagNav':>11} {'PF 2sigma':>10}")
        for minutes in (1, 5, 10, 20):
            s = minutes * 60
            cells = [self._at(self.ins_error_m, s), self._at(self.error_m, s),
                     self._at(self.sigma2_m, s)]
            shown = ["      -   " if np.isnan(c) else f"{c:10.1f}" for c in cells]
            print(f"   {minutes:2d} min {shown[0]} {shown[1]:>11} {shown[2]}  m")

    def __repr__(self) -> str:
        return (f"<PFRun {self.name!r} {self.elapsed_s[-1] / 60:.1f} min, "
                f"final error {self.error_m[-1]:.0f} m vs INS {self.ins_error_m[-1]:.0f} m>")


# ----------------------------------------------------------------------
# The pieces, separately testable
# ----------------------------------------------------------------------

def systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Indices of the particles to keep, by systematic resampling.

    One uniform draw, N evenly spaced pointers: lower variance than
    multinomial resampling, and a particle of weight w is copied either
    floor(Nw) or ceil(Nw) times, never more.
    """
    n = len(weights)
    positions = (rng.uniform() + np.arange(n)) / n
    idx = np.searchsorted(np.cumsum(weights), positions)
    # Rounding can leave the cumulative sum a hair under 1.
    return np.minimum(idx, n - 1)


def log_likelihood(z: float, predicted: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian log-likelihood of one reading, -inf where the map has no value.

    -inf rather than a small number: a particle off the survey has no
    evidence for it, and a finite penalty would let it survive on luck.
    """
    ll = -0.5 * ((z - predicted) / sigma) ** 2
    return np.where(np.isfinite(ll), ll, -np.inf)


def normalise_log(logw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Log-sum-exp normalisation. Returns (normalised log weights, weights)."""
    m = np.max(logw)
    logw = logw - (m + np.log(np.sum(np.exp(logw - m))))
    return logw, np.exp(logw)


def ess(weights: np.ndarray) -> float:
    """Effective sample size, 1 / sum(w^2)."""
    return float(1.0 / np.sum(weights**2))


class ParticleFilter:
    """Error-state particle filter over a magnetic anomaly map."""

    def __init__(self, grid, config: PFConfig | None = None):
        self.grid = grid
        self.config = config or PFConfig()

    def initialise(self, rng: np.random.Generator) -> np.ndarray:
        c = self.config
        x = np.empty((c.n_particles, 4))
        x[:, :2] = rng.normal(0.0, c.sigma_pos0_m, (c.n_particles, 2))
        x[:, 2:] = rng.normal(0.0, c.sigma_vel0_mps, (c.n_particles, 2))
        return x

    def predict(self, x: np.ndarray, dt: float, rng: np.random.Generator) -> np.ndarray:
        x = x.copy()
        x[:, :2] += x[:, 2:] * dt
        x[:, 2:] += rng.normal(0.0, self.config.q * np.sqrt(dt), (len(x), 2))
        return x

    def weight(self, logw: np.ndarray, lat: np.ndarray, lon: np.ndarray,
               z: float) -> tuple[np.ndarray, bool]:
        """Apply one reading. Returns (new log weights, whether it was used).

        If every particle is off the map the reading carries no information
        about which is right, so the weights are left exactly as they were
        rather than all being set to -inf.
        """
        ll = log_likelihood(z, self.grid.sample_track(lat, lon), self.config.sigma_nt)
        new = logw + ll
        if not np.isfinite(new).any():
            return logw, False
        return normalise_log(new)[0], True

    def roughen(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Jitter after resampling, so duplicated particles spread out again."""
        c = self.config
        floor = np.array([c.roughen_floor_m] * 2 + [c.roughen_floor_mps] * 2)
        scale = np.maximum(c.roughen * x.std(axis=0), floor)
        return x + rng.normal(size=x.shape) * scale

    def run(self, ins: Trajectory, t_meas: np.ndarray, z_nt: np.ndarray,
            truth: Trajectory | None = None, name: str = "") -> PFRun:
        """Filter a whole flight.

            ins     the INS estimate from the GNSS cut onwards; its first
                    sample is taken as the cut
            t_meas  when each reading was *taken* (readings.t - latency)
            z_nt    measured anomaly, NaN where there is none
            truth   for scoring only, after the loop
        """
        c = self.config
        rng = np.random.default_rng(c.seed)
        t_meas = np.asarray(t_meas, dtype=float)
        z_nt = np.asarray(z_nt, dtype=float)

        # Pick update times by elapsed time rather than every k-th reading, so
        # a gap of dead-zone readings does not change the update rate after it.
        usable = np.flatnonzero(np.isfinite(z_nt) & (t_meas > ins.t[0])
                                & (t_meas <= ins.t[-1]))
        keep, last = [], -np.inf
        for i in usable:
            if t_meas[i] - last >= 1.0 / c.update_hz - 1e-9:
                keep.append(i)
                last = t_meas[i]
        keep = np.asarray(keep, dtype=int)
        t_upd, z_upd = t_meas[keep], z_nt[keep]

        where = ins.at(t_upd)
        m_lat, m_lon = metres_per_degree(float(ins.lat[0]))

        n_upd = len(t_upd)
        means = np.zeros((n_upd + 1, 4))
        covs = np.zeros((n_upd + 1, 2, 2))
        esss = np.full(n_upd + 1, float(c.n_particles))

        x = self.initialise(rng)
        logw = np.full(c.n_particles, -np.log(c.n_particles))
        means[0] = x.mean(axis=0)
        covs[0] = np.cov(x[:, :2].T)
        snapshots, skipped = [], 0
        t_prev = float(ins.t[0])

        for k in range(n_upd):
            x = self.predict(x, t_upd[k] - t_prev, rng)
            t_prev = t_upd[k]

            lat = where.lat[k] + x[:, 0] / m_lat
            lon = where.lon[k] + x[:, 1] / m_lon
            logw, used = self.weight(logw, lat, lon, z_upd[k])
            skipped += not used
            w = np.exp(logw)

            mean = w @ x
            d = x[:, :2] - mean[:2]
            means[k + 1] = mean
            covs[k + 1] = (d * w[:, None]).T @ d
            esss[k + 1] = ess(w)

            if c.snapshot_every and k % c.snapshot_every == 0:
                snapshots.append(dict(t=float(t_upd[k]), particles=x.copy(),
                                      weights=w.copy(), ins_lat=float(where.lat[k]),
                                      ins_lon=float(where.lon[k])))

            if esss[k + 1] < c.resample_frac * c.n_particles:
                x = self.roughen(x[systematic_resample(w, rng)], rng)
                logw = np.full(c.n_particles, -np.log(c.n_particles))

        # --- assemble the result; truth enters only from here on ----------
        t = np.concatenate([[ins.t[0]], t_upd])
        base = ins.at(t)
        vel = None
        if base.vel_ned is not None:
            vel = base.vel_ned.copy()
            vel[:, :2] += means[:, 2:]
        estimate = Trajectory(t, base.lat + means[:, 0] / m_lat,
                              base.lon + means[:, 1] / m_lon, base.alt_m, vel,
                              name=f"{name} INS+MagNav".strip())

        if truth is not None:
            error = estimate.horizontal_error_m(truth)
            ins_error = base.horizontal_error_m(truth)
        else:
            error = ins_error = np.full(len(t), np.nan)

        return PFRun(t, estimate, covs, esss, error, ins_error, snapshots,
                     t_cut=float(ins.t[0]), name=name, n_skipped=skipped)
