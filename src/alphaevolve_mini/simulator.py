"""Trusted placement simulator for online, append-only, two-resource bin packing.

Standard library only: this module is also mounted read-only into the evaluation
container. The evaluator (not the candidate) owns feasibility checks, machine
opening, capacity updates and metrics. A candidate only ranks feasible machines.
"""

from __future__ import annotations

import math
from typing import Callable, Iterable, Sequence

CAPACITY_CPU = 1000
CAPACITY_MEM = 1000

FIRST_FIT = "first_fit"

ScoreFn = Callable[[int, int, int, int], object]
Job = Sequence[int]


class CandidateError(Exception):
    """A candidate misbehaved (exception, bad return type, non-finite score)."""


class InvariantError(Exception):
    """A trusted evaluator invariant failed. This indicates an evaluator bug or tampering."""


def check_score(value: object) -> float:
    """Return the candidate score as a finite float, or raise CandidateError."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CandidateError(f"score must return int or float, got {type(value).__name__}")
    try:
        as_float = float(value)
    except OverflowError:
        raise CandidateError("score returned an integer too large to represent as a float") from None
    if not math.isfinite(as_float):
        raise CandidateError("score returned a non-finite value")
    return as_float


def _validate_job(job: Job, cap_cpu: int, cap_mem: int) -> tuple[int, int]:
    cpu, mem = int(job[0]), int(job[1])
    if cpu <= 0 or mem <= 0 or cpu > cap_cpu or mem > cap_mem:
        raise InvariantError(f"job ({cpu}, {mem}) does not fit an empty machine")
    return cpu, mem


def place_jobs(
    jobs: Iterable[Job],
    policy: ScoreFn | str,
    cap_cpu: int = CAPACITY_CPU,
    cap_mem: int = CAPACITY_MEM,
) -> list[int]:
    """Place every job in arrival order and return the machine index per job.

    `policy` is either FIRST_FIT (earliest feasible machine, no scoring) or a score
    function. The score function is called only for feasible existing machines;
    the highest finite score wins and ties go to the earliest-created machine.
    A new machine is opened only when no existing machine fits.
    """
    free: list[list[int]] = []
    assignment: list[int] = []
    for job in jobs:
        cpu, mem = _validate_job(job, cap_cpu, cap_mem)
        chosen = -1
        best = -math.inf
        for idx, (free_cpu, free_mem) in enumerate(free):
            if free_cpu < cpu or free_mem < mem:
                continue
            if policy == FIRST_FIT:
                chosen = idx
                break
            try:
                raw = policy(cpu, mem, free_cpu, free_mem)  # type: ignore[operator]
            except CandidateError:
                raise
            except RecursionError:
                raise CandidateError("score raised RecursionError") from None
            except Exception as exc:  # candidate code raised
                raise CandidateError(f"score raised {type(exc).__name__}: {_short(exc)}") from None
            value = check_score(raw)
            if chosen < 0 or value > best:  # strict '>' keeps the earliest machine on ties
                chosen, best = idx, value
        if chosen < 0:
            free.append([cap_cpu, cap_mem])
            chosen = len(free) - 1
        free[chosen][0] -= cpu
        free[chosen][1] -= mem
        assignment.append(chosen)
    return assignment


def _short(exc: Exception, limit: int = 120) -> str:
    text = str(exc).replace("\n", " ")
    return text[:limit]


def verify_assignment(
    jobs: Sequence[Job],
    assignment: Sequence[int],
    cap_cpu: int = CAPACITY_CPU,
    cap_mem: int = CAPACITY_MEM,
) -> dict:
    """Replay an assignment and check every evaluator invariant.

    Checks: each job placed exactly once, machines opened in order, a new machine
    opened only when no existing machine fits, residuals never negative, and
    resources conserved. Returns metrics. Policy-agnostic, so the host can check
    output produced inside the container without trusting it.
    """
    if len(assignment) != len(jobs):
        raise InvariantError(f"{len(assignment)} placements for {len(jobs)} jobs")
    free: list[list[int]] = []
    used: list[list[int]] = []
    for pos, (job, machine) in enumerate(zip(jobs, assignment)):
        cpu, mem = _validate_job(job, cap_cpu, cap_mem)
        if isinstance(machine, bool) or not isinstance(machine, int):
            raise InvariantError(f"job {pos}: machine index is not an int")
        fits_existing = any(fc >= cpu and fm >= mem for fc, fm in free)
        if machine == len(free):
            if fits_existing:
                raise InvariantError(f"job {pos}: opened a new machine although one fits")
            free.append([cap_cpu, cap_mem])
            used.append([0, 0])
        elif not 0 <= machine < len(free):
            raise InvariantError(f"job {pos}: machine index {machine} out of order")
        if free[machine][0] < cpu or free[machine][1] < mem:
            raise InvariantError(f"job {pos}: infeasible placement")
        free[machine][0] -= cpu
        free[machine][1] -= mem
        used[machine][0] += cpu
        used[machine][1] += mem
    for (fc, fm), (uc, um) in zip(free, used):
        if fc < 0 or fm < 0:
            raise InvariantError("negative residual resources")
        if fc + uc != cap_cpu or fm + um != cap_mem:
            raise InvariantError("resource conservation violated")
    total_cpu = sum(int(j[0]) for j in jobs)
    total_mem = sum(int(j[1]) for j in jobs)
    if sum(u[0] for u in used) != total_cpu or sum(u[1] for u in used) != total_mem:
        raise InvariantError("allocated resources do not match total demand")
    machines = len(free)
    return {
        "machines_used": machines,
        # Diagnostics only: for fixed demand these are determined by machines_used.
        "residual_cpu": machines * cap_cpu - total_cpu,
        "residual_mem": machines * cap_mem - total_mem,
        "lower_bound": max(math.ceil(total_cpu / cap_cpu), math.ceil(total_mem / cap_mem)),
    }


def best_fit_score(job_cpu: int, job_mem: int, free_cpu: int, free_mem: int) -> float:
    """Reference best-fit: smallest sum of residual CPU and memory fractions after placement.

    Both capacities are 1000, so the fraction sum is (residual integer sum) / 1000.
    Summing integers before one division keeps equal sums bit-identical, so ties are exact.
    """
    return -((free_cpu - job_cpu) + (free_mem - job_mem)) / CAPACITY_CPU
