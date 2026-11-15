"""Freeze winners using development scores only, then evaluate them on held-out test data.

Order is enforced: winners.json is written (and the run marked test_exposed) before
any test instance is read. Re-running finalize reuses frozen winners and existing
results; it never reselects or silently re-tests.
"""

from __future__ import annotations

from pathlib import Path

from . import baselines, workloads
from .config import Config
from .runner import BaseRunner, summarize
from .search import CONDITIONS, SearchError, load_slots, run_dir
from .util import atomic_write_json, read_json, utc_now


def select_winners(cfg: Config, run: dict, slots: dict) -> dict:
    """Best valid development score including the initial program; earliest wins ties."""
    winners = {}
    for rep in range(cfg.search.replicates):
        for cond in CONDITIONS:
            best = {"candidate_id": run["initial"]["candidate_id"], "source": run["initial"]["source"],
                    "from_slot": None, "dev": _dev(run["initial"]["eval"])}
            n_done = 0
            for slot in range(cfg.search.proposals_per_condition):
                rec = slots.get((rep, cond, slot))
                if rec is None:
                    continue
                n_done += 1
                ev = rec["eval"]
                if ev.get("valid") and ev["total_machines"] < best["dev"]["total_machines"]:
                    best = {"candidate_id": ev["candidate_id"], "source": rec["candidate"]["source"],
                            "from_slot": slot, "dev": _dev(ev)}
            # For evolution this must equal the final incumbent.
            last = max((s for (r, c, s) in slots if r == rep and c == cond), default=None)
            if last is not None and cond == "evolution":
                if slots[(rep, cond, last)]["incumbent_after"]["candidate_id"] != best["candidate_id"]:
                    raise SearchError(f"winner/incumbent mismatch for r{rep} evolution")
            best["completed_slots"] = n_done
            winners[f"{rep}:{cond}"] = best
    return winners


def _dev(ev: dict) -> dict:
    return {k: ev[k] for k in ("total_machines", "mean_machines", "family_means")}


def finalize(cfg: Config, run_id: str, make_test_runner, allow_partial: bool = False, log=print) -> dict:
    rdir = run_dir(cfg, run_id)
    if not (rdir / "run.json").exists():
        raise SearchError(f"no run {run_id}")
    run = read_json(rdir / "run.json")
    final_dir = rdir / "final"
    results_file = final_dir / "heldout.json"
    winners_file = final_dir / "winners.json"

    if results_file.exists():
        log(f"{run_id} already finalized; reusing {results_file} (no new test evaluation)")
        return read_json(results_file)

    if winners_file.exists():
        frozen = read_json(winners_file)
        log("winners already frozen; completing held-out evaluation for the frozen winners")
    else:
        partial = run["status"] != "complete"
        if partial and not allow_partial:
            raise SearchError(f"run {run_id} is {run['status']} ({run.get('stop_reason')}); "
                              "finish it with --resume or pass --allow-partial to finalize what exists")
        slots = load_slots(rdir)
        frozen = {
            "run_id": run_id,
            "frozen_at": utc_now(),
            "partial": partial,
            "selection_rule": "lowest total development machines (valid only, initial included); earliest on ties",
            "completed_slots": len(slots),
            "total_slots": run["total_slots"],
            "winners": select_winners(cfg, run, slots),
        }
        # Freeze and mark test exposure BEFORE reading any test data.
        atomic_write_json(winners_file, frozen)
        run["test_exposed"] = True
        run["finalization"] = {"status": "winners_frozen", "partial": partial, "frozen_at": frozen["frozen_at"]}
        if partial:
            run["status"] = "partial"
        atomic_write_json(rdir / "run.json", run)

    manifest = workloads.verify(cfg)
    if workloads.workload_hash(manifest) != run["workload_hash"]:
        raise SearchError("workload files changed since the search; refusing to evaluate")
    runner: BaseRunner = make_test_runner()
    if runner.allowed_splits != frozenset({"test"}):
        raise SearchError("finalization runner must be restricted to the test split")
    test = workloads.load_split(cfg, "test")

    policies = {}
    for name in baselines.REFERENCE_POLICIES:
        policies[name] = summarize(baselines.evaluate_reference(name, test), test) | {"kind": "reference"}
    by_candidate: dict[str, dict] = {}
    for key, w in frozen["winners"].items():
        cid = w["candidate_id"]
        if cid not in by_candidate:
            log(f"held-out evaluation of {cid} ({key})")
            res = runner.evaluate(w["source"], "test")
            if not res.get("valid"):
                raise SearchError(f"frozen winner {cid} failed on test: {res.get('error')}")
            by_candidate[cid] = res
        policies[f"winner:{key}"] = {k: by_candidate[cid][k] for k in ("per_instance", "total_machines", "mean_machines", "family_means")} | {
            "kind": "winner", "candidate_id": cid, "replicate": int(key.split(":")[0]), "condition": key.split(":")[1]}

    results = {
        "run_id": run_id,
        "evaluated_at": utc_now(),
        "partial": frozen["partial"],
        "live": run["live"],
        "runner": runner.kind,
        "test_split_hash": manifest["splits"]["test"]["split_hash"],
        "instances": [{"instance_id": i["instance_id"], "family": i["family"]} for i in test],
        "policies": policies,
    }
    atomic_write_json(results_file, results)
    run["finalization"] = run.get("finalization", {}) | {"status": "complete", "evaluated_at": results["evaluated_at"]}
    atomic_write_json(rdir / "run.json", run)
    return results
