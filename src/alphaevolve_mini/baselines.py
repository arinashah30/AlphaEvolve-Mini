"""Fixed reference policies, evaluated in trusted host code."""

from __future__ import annotations

from . import simulator

# Common initial incumbent for both search conditions. It must place jobs exactly
# like simulator.best_fit_score (checked by tests and at search start).
INITIAL_SOURCE = '''def score(job_cpu, job_mem, free_cpu, free_mem):
    """Best fit: prefer the machine with the least total residual capacity after placement."""
    return -((free_cpu - job_cpu) + (free_mem - job_mem)) / 1000
'''

REFERENCE_POLICIES = {
    "first_fit": simulator.FIRST_FIT,
    "best_fit": simulator.best_fit_score,
}


def evaluate_reference(name: str, instances: list[dict]) -> dict[str, dict]:
    policy = REFERENCE_POLICIES[name]
    out = {}
    for inst in instances:
        assignment = simulator.place_jobs(inst["jobs"], policy)
        out[inst["instance_id"]] = simulator.verify_assignment(inst["jobs"], assignment)
    return out
