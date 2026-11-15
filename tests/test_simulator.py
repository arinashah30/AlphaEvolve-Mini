import math

import pytest

from alphaevolve_mini import baselines, candidates, simulator as sim


def test_known_tiny_first_fit():
    jobs = [(600, 100), (500, 100), (400, 100), (100, 950)]
    # 600 -> m0; 500 doesn't fit m0 (400 free) -> m1; 400 fits m0 exactly; 100/950 fits neither (mem) -> m2
    assert sim.place_jobs(jobs, sim.FIRST_FIT) == [0, 1, 0, 2]


def test_known_tiny_best_fit_differs_from_first_fit():
    jobs = [(500, 500), (300, 300), (400, 400)]
    # m0 free 500/500 after job0. job1 fits m0. job2: m0 has 200/200 -> no fit -> m1
    assert sim.place_jobs(jobs, sim.best_fit_score) == [0, 0, 1]
    jobs = [(800, 800), (500, 500), (100, 100)]
    # job2 fits m0 (200 free, residual 100+100) and m1 (500 free, residual 400+400): best-fit picks m0
    assert sim.place_jobs(jobs, sim.best_fit_score) == [0, 1, 0]
    assert sim.place_jobs(jobs, sim.FIRST_FIT) == [0, 1, 0]
    jobs = [(500, 500), (800, 800), (100, 100)]
    assert sim.place_jobs(jobs, sim.FIRST_FIT) == [0, 1, 0]
    assert sim.place_jobs(jobs, sim.best_fit_score) == [0, 1, 1]


def test_ties_go_to_earliest_machine():
    jobs = [(900, 900), (900, 900), (50, 50)]
    assert sim.place_jobs(jobs, lambda *a: 0.0) == [0, 1, 0]
    # constant score with many machines -> always earliest feasible
    assert sim.place_jobs([(400, 400)] * 6, lambda *a: 7) == [0, 0, 1, 1, 2, 2]


def test_candidate_only_called_on_feasible_machines():
    seen = []

    def spy(jc, jm, fc, fm):
        seen.append((jc, jm, fc, fm))
        assert fc >= jc and fm >= jm
        return fc

    sim.place_jobs([(700, 100), (700, 100), (200, 100), (300, 900)], spy)
    assert seen  # was called


def test_new_machine_without_calling_candidate():
    def boom(*a):
        raise AssertionError("should not be called")

    assert sim.place_jobs([(600, 600), (600, 600)], boom) == [0, 1]


def test_verify_assignment_conservation_and_metrics():
    jobs = [(600, 100), (500, 100), (400, 100), (100, 950)]
    m = sim.verify_assignment(jobs, [0, 1, 0, 2])
    assert m["machines_used"] == 3
    assert m["residual_cpu"] == 3000 - 1600
    assert m["residual_mem"] == 3000 - 1250
    assert m["lower_bound"] == 2


@pytest.mark.parametrize("assignment,msg", [
    ([0, 0], "infeasible"),           # second 600-cpu job cannot share
    ([0], "placements"),              # missing job
    ([0, 2], "out of order"),         # skipped machine index
    ([0, -1], "out of order"),
])
def test_verify_assignment_rejects_bad(assignment, msg):
    with pytest.raises(sim.InvariantError, match=msg):
        sim.verify_assignment([(600, 100), (600, 100)], assignment)


def test_verify_rejects_unnecessary_new_machine():
    with pytest.raises(sim.InvariantError, match="although one fits"):
        sim.verify_assignment([(100, 100), (100, 100)], [0, 1])


def test_job_must_fit_fresh_machine():
    with pytest.raises(sim.InvariantError):
        sim.place_jobs([(1001, 10)], sim.FIRST_FIT)


@pytest.mark.parametrize("ret", [math.inf, -math.inf, math.nan, True, "1", None, 10**400])
def test_bad_scores_rejected(ret):
    with pytest.raises(sim.CandidateError):
        sim.place_jobs([(100, 100), (100, 100)], lambda *a: ret)


def test_candidate_exception_becomes_candidate_error():
    with pytest.raises(sim.CandidateError, match="ZeroDivisionError"):
        sim.place_jobs([(100, 100), (100, 100)], lambda *a: 1 / 0)


def test_initial_source_matches_trusted_best_fit(cfg):
    from alphaevolve_mini import workloads
    fn = candidates.compile_score(baselines.INITIAL_SOURCE)
    for split in ("dev", "test"):
        for inst in workloads.load_split(cfg, split):
            assert sim.place_jobs(inst["jobs"], fn) == sim.place_jobs(inst["jobs"], sim.best_fit_score)
