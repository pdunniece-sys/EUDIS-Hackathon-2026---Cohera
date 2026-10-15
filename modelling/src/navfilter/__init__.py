"""Navigation filters that correct an INS against the magnetic anomaly map.

    from navfilter import ParticleFilter, PFConfig, degrade_map

    filter_map = degrade_map(truth_map, seed=1)
    run = ParticleFilter(filter_map, PFConfig(grade="consumer_mems")).run(
        ins_run.estimate, taken_at, anomaly_nt, truth=truth)

Named `navfilter`, not `filter`, so it does not shadow the Python builtin.
Read particle.py for the design and maps.py for why the filter's map is
deliberately worse than the one the readings came from.
"""

from navfilter.maps import degrade_map, grid_from_array
from navfilter.particle import (
    ParticleFilter,
    PFConfig,
    PFRun,
    default_q_vel,
    ess,
    log_likelihood,
    systematic_resample,
)

__all__ = [
    "ParticleFilter", "PFConfig", "PFRun",
    "degrade_map", "grid_from_array",
    "systematic_resample", "log_likelihood", "ess", "default_q_vel",
]
