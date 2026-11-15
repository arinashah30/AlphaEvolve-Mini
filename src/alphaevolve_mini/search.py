"""Proposal loop for the two conditions, with fair budgets, atomic logs and resume.

Run directory layout (runs/<run_id>/):
  run.json        identity, config snapshot, ordering, initial program, status
  state.json      incumbent / best-so-far per (replicate, condition) after the last slot
  slots/          one immutable JSON record per completed proposal slot
  aborted/        attempts made in slots that stopped on a fatal provider error
"""

from __future__ import annotations

import json
import platform
import random
import re
import sys
from importlib import metadata
from pathlib import Path

from . import baselines, candidates, prompts, workloads
from .config import Config
from .providers import AttemptBudget, BudgetExhausted, Provider
from .runner import BaseRunner, summarize
from .util import atomic_write_json, git_commit, read_json, utc_now

CONDITIONS = ("independent", "evolution")


class SearchError(Exception):
    pass


def run_dir(cfg: Config, run_id: str) -> Path:
    if not run_id or "/" in run_id or run_id.startswith("."):
        raise SearchError(f"invalid run id {run_id!r}")
    return Path(cfg.runs_dir) / run_id


def slot_name(rep: int, cond: str, slot: int) -> str:
    return f"r{rep}_{cond}_{slot:03d}.json"


def versions() -> dict:
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for pkg in ("numpy", "matplotlib", "pyyaml", "google-genai"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def schedule(cfg: Config) -> tuple[list[tuple[int, str, int]], dict[str, list[str]]]:
    """Interleaved slot order; condition order per replicate is randomized and recorded."""
    order: dict[str, list[str]] = {}
    seq = []
    for rep in range(cfg.search.replicates):
        conds = list(CONDITIONS)
        random.Random(f"{cfg.search.ordering_seed}:{rep}").shuffle(conds)
        order[str(rep)] = conds
        for slot in range(cfg.search.proposals_per_condition):
            for cond in conds:
                seq.append((rep, cond, slot))
    return seq, order


def _strip_eval(result: dict) -> dict:
    """Compact eval summary used as feedback/incumbent state."""
    keep = ("valid", "error_kind", "error", "candidate_id", "total_machines", "mean_machines", "family_means")
    return {k: result.get(k) for k in keep if k in result}


def is_strict_improvement(new: dict, incumbent: dict) -> bool:
    # Equal instance counts, so comparing integer totals is exact and equals comparing means.
    return bool(new.get("valid")) and new["total_machines"] < incumbent["total_machines"]


_SLOT_FILE = re.compile(r"^r\d+_(independent|evolution)_\d{3}\.json$")


def _log_files(directory: Path) -> list[Path]:
    """Our own log files only; skips sync-conflict copies such as 'x 2.json' (iCloud)."""
    return sorted(p for p in directory.glob("*.json") if " " not in p.name and not p.name.startswith("."))


def load_slots(rdir: Path) -> dict[tuple[int, str, int], dict]:
    out = {}
    for p in _log_files(rdir / "slots"):
        if not _SLOT_FILE.match(p.name):
            continue
        rec = read_json(p)
        out[(rec["replicate"], rec["condition"], rec["slot"])] = rec
    return out


def sent_attempts(response: dict) -> int:
    """Attempts actually sent (budget refusals are logged but never reached the provider)."""
    return sum(1 for a in response["attempts"] if a["status"] != "budget_exhausted")


def count_attempts(rdir: Path) -> int:
    n = sum(sent_attempts(r["response"]) for r in load_slots(rdir).values())
    for p in _log_files(rdir / "aborted"):
        n += sent_attempts(read_json(p)["response"])
    return n


def _initial_state(initial: dict) -> dict:
    entry = {"source": initial["source"], "candidate_id": initial["candidate_id"], "eval": initial["eval"],
             "from_slot": None, "feedback": initial["feedback"]}
    return entry


def replay_state(cfg: Config, run: dict, slots: dict) -> dict:
    """Rebuild incumbents from immutable slot records (the source of truth)."""
    state: dict = {}
    for rep in range(cfg.search.replicates):
        for cond in CONDITIONS:
            cur = _initial_state(run["initial"])
            for slot in range(cfg.search.proposals_per_condition):
                rec = slots.get((rep, cond, slot))
                if rec is None:
                    break
                if rec["improved"]:
                    cur = {"source": rec["candidate"]["source"], "candidate_id": rec["eval"]["candidate_id"],
                           "eval": rec["eval"], "from_slot": slot, "feedback": prompts.format_feedback(rec["eval"])}
                if rec["incumbent_after"]["candidate_id"] != cur["candidate_id"]:
                    raise SearchError(f"slot log inconsistent at r{rep} {cond} {slot}")
            state[f"{rep}:{cond}"] = cur
    return state


def start_or_resume(cfg: Config, run_id: str, provider: Provider, runner: BaseRunner, resume: bool) -> tuple[Path, dict]:
    manifest = workloads.verify(cfg)
    whash = workloads.workload_hash(manifest)
    rdir = run_dir(cfg, run_id)
    run_file = rdir / "run.json"
    if run_file.exists():
        if not resume:
            raise SearchError(f"run {run_id} already exists; pass --resume to continue it or pick a new --run-id")
        run = read_json(run_file)
        mismatches = []
        if run["config_hash"] != cfg.identity_hash():
            mismatches.append("configuration")
        if run["workload_hash"] != whash:
            mismatches.append("workload files")
        if run.get("prompt_template_hash") != prompts.template_hash():
            mismatches.append("prompt template")
        if run["provider"] != provider.name or run["live"] != provider.live or run["model"] != provider.model:
            mismatches.append("provider/model/live flag")
        if mismatches:
            raise SearchError(f"cannot resume {run_id}: {', '.join(mismatches)} differ from the saved run")
        if run.get("test_exposed"):
            raise SearchError(f"run {run_id} has been finalized (test split exposed); search cannot continue")
        return rdir, run
    if resume:
        raise SearchError(f"no run {run_id} to resume")

    # Common initial incumbent, evaluated through the same runner as candidates (dev only).
    init_eval = runner.evaluate(baselines.INITIAL_SOURCE, "dev")
    if not init_eval.get("valid"):
        raise SearchError(f"initial program failed evaluation: {init_eval.get('error')}")
    ref = summarize(baselines.evaluate_reference("best_fit", workloads.load_split(cfg, "dev")), workloads.load_split(cfg, "dev"))
    if ref["total_machines"] != init_eval["total_machines"]:
        raise SearchError("initial program disagrees with the trusted best-fit reference")
    _, order = schedule(cfg)
    run = {
        "run_id": run_id,
        "created_at": utc_now(),
        "provider": provider.name,
        "live": provider.live,
        "model": provider.model,
        "data_kind": "live" if provider.live else "mock (synthetic integration fixtures; not experimental evidence)",
        "config_name": cfg.name,
        "config_source": cfg.source_path,
        "config_hash": cfg.identity_hash(),
        "config": cfg.to_dict(),
        "workload_hash": whash,
        "prompt_template_hash": prompts.template_hash(),
        "split_hashes": {s: manifest["splits"][s]["split_hash"] for s in workloads.SPLITS},
        "git_commit": git_commit(),
        "versions": versions(),
        "runner": runner.kind,
        "condition_order": order,
        "total_slots": cfg.total_slots,
        "status": "in_progress",
        "stop_reason": None,
        "test_exposed": False,
        "initial": {
            "source": baselines.INITIAL_SOURCE,
            "candidate_id": init_eval["candidate_id"],
            "eval": init_eval,
            "feedback": prompts.format_feedback(init_eval),
        },
    }
    (rdir / "slots").mkdir(parents=True, exist_ok=True)
    (rdir / "aborted").mkdir(exist_ok=True)
    atomic_write_json(run_file, run)
    return rdir, run


def _save_run(rdir: Path, run: dict) -> None:
    atomic_write_json(rdir / "run.json", run)


def run_search(cfg: Config, run_id: str, provider: Provider, runner: BaseRunner, resume: bool = False,
               max_new_slots: int | None = None, log=print) -> dict:
    """Run (or continue) the search. `max_new_slots` lets tests simulate interruption."""
    if runner.allowed_splits != frozenset({"dev"}):
        raise SearchError("search runner must be restricted to the dev split")
    rdir, run = start_or_resume(cfg, run_id, provider, runner, resume)
    slots = load_slots(rdir)
    state = replay_state(cfg, run, slots)
    budget = AttemptBudget(cfg.search.max_total_attempts, used=count_attempts(rdir))
    seq, _ = schedule(cfg)
    run["status"], run["stop_reason"] = "in_progress", None
    _save_run(rdir, run)

    initial = _initial_state(run["initial"])
    done_now = 0
    for global_idx, (rep, cond, slot) in enumerate(seq):
        if (rep, cond, slot) in slots:
            continue  # never resample a completed slot
        if max_new_slots is not None and done_now >= max_new_slots:
            run["status"], run["stop_reason"] = "partial", "interrupted (slot cap for this invocation)"
            break
        if budget.remaining <= 0:
            run["status"], run["stop_reason"] = "partial", "total request-attempt limit reached"
            break
        key = f"{rep}:{cond}"
        parent = initial if cond == "independent" else state[key]
        prompt = prompts.build_prompt(parent["source"], parent["feedback"], cfg.candidate_limits)
        response = provider.propose(prompt, slot_key=f"{rep}:{slot}", budget=budget)

        if response.fatal:
            atomic_write_json(rdir / "aborted" / f"{utc_now().replace(':', '')}_{slot_name(rep, cond, slot)}",
                              {"replicate": rep, "condition": cond, "slot": slot, "response": response.to_dict()})
            run["status"], run["stop_reason"] = "partial", f"fatal provider error: {response.attempts[-1].error_message}"
            run["attempts_used"], run["completed_slots"] = budget.used, len(slots)
            _save_run(rdir, run)
            log(f"STOP: fatal provider error at r{rep} {cond} slot {slot}: {response.attempts[-1].error_message}")
            return run
        if not response.ok and response.attempts and response.attempts[0].status == "budget_exhausted":
            run["status"], run["stop_reason"] = "partial", "total request-attempt limit reached"
            break

        candidate = {"raw_response": response.text, "source": None, "extraction_error": None}
        if response.ok:
            try:
                candidate["source"] = candidates.extract_source(response.text)
            except candidates.ValidationError as exc:
                candidate["extraction_error"] = str(exc)
        if candidate["source"] is not None:
            result = runner.evaluate(candidate["source"], "dev")
        elif response.ok:
            result = {"valid": False, "error_kind": "extraction", "error": candidate["extraction_error"]}
        else:
            result = {"valid": False, "error_kind": "request_failed", "error": "request failed after retries"}

        incumbent_before = state[key]
        improved = is_strict_improvement(result, incumbent_before["eval"])
        if improved:
            state[key] = {"source": candidate["source"], "candidate_id": result["candidate_id"], "eval": result,
                          "from_slot": slot, "feedback": prompts.format_feedback(result)}
        record = {
            "replicate": rep,
            "condition": cond,
            "slot": slot,
            "global_index": global_idx,
            "completed_at": utc_now(),
            "synthetic": not provider.live,
            "parent_candidate_id": parent["candidate_id"],
            "prompt": prompt,
            "response": response.to_dict(),
            "request_status": "ok" if response.ok else "request_failed",
            "candidate": candidate,
            "eval": result,
            "improved": improved,
            "incumbent_before": {"candidate_id": incumbent_before["candidate_id"], "total_machines": incumbent_before["eval"]["total_machines"]},
            "incumbent_after": {"candidate_id": state[key]["candidate_id"], "total_machines": state[key]["eval"]["total_machines"],
                                 "mean_machines": state[key]["eval"]["mean_machines"], "from_slot": state[key]["from_slot"]},
        }
        atomic_write_json(rdir / "slots" / slot_name(rep, cond, slot), record)
        slots[(rep, cond, slot)] = record
        atomic_write_json(rdir / "state.json", {"updated_at": utc_now(), "attempts_used": budget.used,
                                                 "incumbents": {k: {kk: v[kk] for kk in ("candidate_id", "from_slot", "source")} | {"eval": _strip_eval(v["eval"])} for k, v in state.items()}})
        done_now += 1
        mark = "IMPROVED" if improved else ("valid" if result.get("valid") else f"invalid ({result.get('error_kind')})")
        score = f"{result['mean_machines']:.3f}" if result.get("valid") else "-"
        log(f"[{len(slots)}/{len(seq)}] r{rep} {cond:<11} slot {slot:>2}: {mark:<24} dev mean {score}  best {state[key]['eval']['mean_machines']:.3f}")
    else:
        run["status"], run["stop_reason"] = ("complete", None) if len(slots) == len(seq) else ("partial", "incomplete")

    if len(slots) == len(seq):
        run["status"], run["stop_reason"] = "complete", None
    run["attempts_used"] = budget.used
    run["completed_slots"] = len(slots)
    _save_run(rdir, run)
    return run
