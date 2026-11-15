import json

import pytest

from alphaevolve_mini import finalize as F, report, search as S, workloads
from alphaevolve_mini.providers import Attempt, MockProvider, Provider, ProviderResponse
from alphaevolve_mini.runner import LocalTrustedRunner

HDR = "def score(job_cpu, job_mem, free_cpu, free_mem):\n"


def dev_runner(cfg):
    return LocalTrustedRunner(cfg, {"dev"})


def make_test_runner(cfg):
    return lambda: LocalTrustedRunner(cfg, {"test"})


def quiet(*a, **k):
    pass


class Scripted(Provider):
    """Returns a scripted sequence of sources; records prompts."""
    name, live, model = "mock", False, "mock-deterministic-v1"

    def __init__(self, sources):
        self.sources = list(sources)
        self.prompts = []

    def propose(self, prompt, slot_key, budget):
        budget.consume()
        self.prompts.append(prompt)
        src = self.sources.pop(0)
        return ProviderResponse(True, False, f"```python\n{src}```", [Attempt(1, "ok", 0.0, 0.0, usage=None)])


def test_full_mock_run_and_prompt_rules(cfg):
    run = S.run_search(cfg, "a", MockProvider(), dev_runner(cfg), log=quiet)
    assert run["status"] == "complete" and run["completed_slots"] == cfg.total_slots
    slots = S.load_slots(S.run_dir(cfg, "a"))
    first = {}
    for (rep, cond, slot), rec in slots.items():
        if cond == "independent":
            # independent always sees the initial program and the fixed initial feedback
            assert rec["parent_candidate_id"] == run["initial"]["candidate_id"]
            first.setdefault("ind", rec["prompt"])
            assert rec["prompt"] == first["ind"]
    for rep in range(cfg.search.replicates):
        assert slots[(rep, "independent", 0)]["prompt"] == slots[(rep, "evolution", 0)]["prompt"]


def test_evolution_parent_changes_only_on_strict_improvement(cfg):
    worse = HDR + "    return free_cpu + free_mem\n"           # worst-fit: more machines
    same = HDR + "    return -((free_cpu - job_cpu) + (free_mem - job_mem)) / 1000 + 0\n"  # tie
    invalid = "import os\n" + HDR + "    return 0\n"
    seq = [worse, same, invalid, worse, same, worse]
    cfg1 = cfg.__class__(**{**cfg.__dict__, "search": cfg.search.__class__(**{**cfg.search.__dict__, "replicates": 1})})
    prov = Scripted(seq * 2)
    S.run_search(cfg1, "b", prov, dev_runner(cfg1), log=quiet)
    slots = S.load_slots(S.run_dir(cfg1, "b"))
    evo = [slots[(0, "evolution", s)] for s in range(6)]
    assert not any(r["improved"] for r in evo)  # ties and regressions never replace
    assert len({r["parent_candidate_id"] for r in evo}) == 1
    # invalid outputs consume budget slots
    assert sum(1 for (r, c, s) in slots if c == "evolution") == cfg1.search.proposals_per_condition
    assert any(not r["eval"]["valid"] for r in evo)


def test_evolution_uses_improved_incumbent(cfg, monkeypatch):
    better = HDR + "    return 0.123\n"
    cfg1 = cfg.__class__(**{**cfg.__dict__, "search": cfg.search.__class__(**{**cfg.search.__dict__, "replicates": 1, "proposals_per_condition": 2})})
    # Force "better" to look like a strict improvement by patching its evaluation.
    real = LocalTrustedRunner.evaluate

    def fake(self, source, split):
        res = real(self, source, split)
        if "0.123" in source and res.get("valid"):
            res["total_machines"] -= 1
        return res

    monkeypatch.setattr(LocalTrustedRunner, "evaluate", fake)
    prov = Scripted([better] * 4)
    S.run_search(cfg1, "c", prov, dev_runner(cfg1), log=quiet)
    slots = S.load_slots(S.run_dir(cfg1, "c"))
    assert slots[(0, "evolution", 0)]["improved"]
    assert "0.123" in slots[(0, "evolution", 1)]["prompt"]
    assert "0.123" not in slots[(0, "independent", 1)]["prompt"]


def test_interrupt_and_resume_matches_uninterrupted(cfg):
    S.run_search(cfg, "full", MockProvider(), dev_runner(cfg), log=quiet)
    part = S.run_search(cfg, "split", MockProvider(), dev_runner(cfg), max_new_slots=7, log=quiet)
    assert part["status"] == "partial"
    with pytest.raises(S.SearchError, match="--resume"):
        S.run_search(cfg, "split", MockProvider(), dev_runner(cfg), log=quiet)
    S.run_search(cfg, "split", MockProvider(), dev_runner(cfg), resume=True, max_new_slots=5, log=quiet)
    done = S.run_search(cfg, "split", MockProvider(), dev_runner(cfg), resume=True, log=quiet)
    assert done["status"] == "complete"
    a, b = S.load_slots(S.run_dir(cfg, "full")), S.load_slots(S.run_dir(cfg, "split"))
    assert a.keys() == b.keys()
    for k in a:
        assert a[k]["prompt"] == b[k]["prompt"]
        assert a[k]["incumbent_after"] == b[k]["incumbent_after"]
    assert done["attempts_used"] == cfg.total_slots  # no slot resampled


def test_resume_rejects_changed_config(cfg):
    S.run_search(cfg, "r", MockProvider(), dev_runner(cfg), max_new_slots=2, log=quiet)
    changed = cfg.__class__(**{**cfg.__dict__, "search": cfg.search.__class__(**{**cfg.search.__dict__, "proposals_per_condition": 7})})
    with pytest.raises(S.SearchError, match="configuration"):
        S.run_search(changed, "r", MockProvider(), dev_runner(changed), resume=True, log=quiet)


def test_attempt_limit_stops_partial(cfg):
    small = cfg.__class__(**{**cfg.__dict__, "search": cfg.search.__class__(**{**cfg.search.__dict__, "max_total_attempts": 5})})
    run = S.run_search(small, "lim", MockProvider(), dev_runner(small), log=quiet)
    assert run["status"] == "partial" and "limit" in run["stop_reason"]
    assert run["completed_slots"] == 5 and run["attempts_used"] == 5


class FatalOnSlot(MockProvider):
    def __init__(self, n):
        self.n, self.calls = n, 0

    def propose(self, prompt, slot_key, budget):
        self.calls += 1
        if self.calls == self.n:
            budget.consume()
            return ProviderResponse(False, True, None, [Attempt(1, "fatal_error", 0.0, 0.1, 401, "ClientError", "UNAUTHENTICATED")])
        return super().propose(prompt, slot_key, budget)


def test_fatal_error_stops_without_consuming_slot(cfg):
    run = S.run_search(cfg, "f", FatalOnSlot(3), dev_runner(cfg), log=quiet)
    assert run["status"] == "partial" and "fatal" in run["stop_reason"]
    rdir = S.run_dir(cfg, "f")
    assert len(S.load_slots(rdir)) == 2
    assert S.count_attempts(rdir) == 3  # the fatal attempt still counts toward the attempt limit
    done = S.run_search(cfg, "f", MockProvider(), dev_runner(cfg), resume=True, log=quiet)
    assert done["status"] == "complete"


class FailedRequest(MockProvider):
    def propose(self, prompt, slot_key, budget):
        for _ in range(3):
            budget.consume()
        return ProviderResponse(False, False, None, [Attempt(i, "transient_error", 0.0, 0.1, 503) for i in (1, 2, 3)])


def test_failed_requests_count_as_slots(cfg):
    run = S.run_search(cfg, "t", FailedRequest(), dev_runner(cfg), log=quiet)
    slots = S.load_slots(S.run_dir(cfg, "t"))
    assert run["status"] == "complete" and len(slots) == cfg.total_slots
    assert all(r["request_status"] == "request_failed" for r in slots.values())
    assert run["attempts_used"] == 3 * cfg.total_slots


def test_search_never_reads_test(cfg, monkeypatch):
    real = workloads.load_split
    seen = []
    monkeypatch.setattr(workloads, "load_split", lambda c, split: seen.append(split) or real(c, split))
    S.run_search(cfg, "nt", MockProvider(), dev_runner(cfg), log=quiet)
    assert seen and set(seen) == {"dev"}
    with pytest.raises(S.SearchError):
        S.run_search(cfg, "nt2", MockProvider(), LocalTrustedRunner(cfg, {"dev", "test"}), log=quiet)


def test_finalize_freezes_before_test_and_blocks_search(cfg, monkeypatch):
    S.run_search(cfg, "fin", MockProvider(), dev_runner(cfg), log=quiet)
    rdir = S.run_dir(cfg, "fin")
    real = workloads.load_split

    def spy(c, split):
        if split == "test":
            assert (rdir / "final" / "winners.json").exists()
            assert json.loads((rdir / "run.json").read_text())["test_exposed"]
        return real(c, split)

    monkeypatch.setattr(workloads, "load_split", spy)
    res = F.finalize(cfg, "fin", make_test_runner(cfg), log=quiet)
    assert not res["partial"] and "best_fit" in res["policies"]
    assert {k for k in res["policies"] if k.startswith("winner:")} == {f"winner:{r}:{c}" for r in range(2) for c in S.CONDITIONS}
    with pytest.raises(S.SearchError, match="finalized"):
        S.run_search(cfg, "fin", MockProvider(), dev_runner(cfg), resume=True, log=quiet)
    # repeated finalize reuses results and does not re-test
    monkeypatch.setattr(workloads, "load_split", lambda *a: pytest.fail("re-tested"))
    assert F.finalize(cfg, "fin", make_test_runner(cfg), log=quiet) == res


def test_finalize_refuses_partial_unless_explicit(cfg):
    S.run_search(cfg, "p", MockProvider(), dev_runner(cfg), max_new_slots=3, log=quiet)
    with pytest.raises(S.SearchError, match="allow-partial"):
        F.finalize(cfg, "p", make_test_runner(cfg), log=quiet)
    res = F.finalize(cfg, "p", make_test_runner(cfg), allow_partial=True, log=quiet)
    assert res["partial"]
    with pytest.raises(S.SearchError):
        S.run_search(cfg, "p", MockProvider(), dev_runner(cfg), resume=True, log=quiet)


def test_winner_selection_uses_dev_only_with_earliest_tie(cfg):
    S.run_search(cfg, "w", MockProvider(), dev_runner(cfg), log=quiet)
    rdir = S.run_dir(cfg, "w")
    run = json.loads((rdir / "run.json").read_text())
    slots = S.load_slots(rdir)
    winners = F.select_winners(cfg, run, slots)
    for key, w in winners.items():
        rep, cond = key.split(":")
        recs = [slots[(int(rep), cond, s)] for s in range(cfg.search.proposals_per_condition)]
        best = min([run["initial"]["eval"]["total_machines"]] + [r["eval"]["total_machines"] for r in recs if r["eval"]["valid"]])
        assert w["dev"]["total_machines"] == best
        if run["initial"]["eval"]["total_machines"] == best:
            assert w["from_slot"] is None


def test_report_regenerates_from_logs_and_labels(cfg, monkeypatch):
    S.run_search(cfg, "rep", MockProvider(), dev_runner(cfg), max_new_slots=4, log=quiet)
    F.finalize(cfg, "rep", make_test_runner(cfg), allow_partial=True, log=quiet)
    # report must not evaluate or call providers
    monkeypatch.setattr(LocalTrustedRunner, "evaluate", lambda *a: pytest.fail("evaluated"))
    monkeypatch.setattr(MockProvider, "propose", lambda *a: pytest.fail("called provider"))
    out = report.generate(cfg, "rep", log=quiet)
    for f in ("search_progress.png", "heldout_comparison.png", "summary.csv", "heldout_per_instance.csv", "run_manifest.json"):
        assert (out / f).exists()
    manifest = json.loads((out / "run_manifest.json").read_text())
    assert manifest["mode"].startswith("mock") and manifest["completion"] == "partial"
    assert any((out / "candidate_changes").iterdir())
    assert "synthetic_mock" in (out / "summary.csv").read_text().splitlines()[0]


def test_resume_rejects_changed_prompt_template(cfg, monkeypatch):
    from alphaevolve_mini import prompts
    S.run_search(cfg, "pt", MockProvider(), dev_runner(cfg), max_new_slots=2, log=quiet)
    monkeypatch.setattr(prompts, "TEMPLATE", prompts.TEMPLATE + "\nextra")
    with pytest.raises(S.SearchError, match="prompt template"):
        S.run_search(cfg, "pt", MockProvider(), dev_runner(cfg), resume=True, log=quiet)
