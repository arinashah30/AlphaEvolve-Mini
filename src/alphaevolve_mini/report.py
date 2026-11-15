"""Reports generated purely from saved run files (no API calls, no evaluation)."""

from __future__ import annotations

import csv
import difflib
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .config import Config  # noqa: E402
from .search import CONDITIONS, _log_files, load_slots, run_dir, sent_attempts  # noqa: E402
from .util import atomic_write_json, read_json, utc_now  # noqa: E402
from .workloads import FAMILIES  # noqa: E402

COLORS = {"independent": "#2a78d6", "evolution": "#eb6834", "first_fit": "#8a8985", "best_fit": "#52514e"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
LABELS = {"first_fit": "First-fit", "best_fit": "Best-fit", "independent": "Independent", "evolution": "Evolution"}


# ----------------------------------------------------------------------------- usage / cost

def usage_totals(records: list[dict], cfg_pricing: dict) -> dict:
    """Sum usage over every sent attempt. Unknown usage stays unknown; it is never zero."""
    prompt = output = 0
    unknown = 0
    latency = 0.0
    attempts = 0
    for rec in records:
        for a in rec["response"]["attempts"]:
            if a["status"] == "budget_exhausted":
                continue
            attempts += 1
            latency += a.get("latency_s") or 0.0
            u = a.get("usage")
            if not u or u.get("prompt_token_count") is None or u.get("total_token_count") is None:
                unknown += 1
                continue
            prompt += u["prompt_token_count"]
            output += u["total_token_count"] - u["prompt_token_count"]  # candidates + thinking (+tool) tokens
    cost = None
    pin, pout = cfg_pricing.get("input_per_million"), cfg_pricing.get("output_per_million")
    if pin is not None and pout is not None and unknown == 0:
        # Conservative: cached prompt tokens are charged at the full input price.
        cost = prompt / 1e6 * pin + output / 1e6 * pout
    return {"attempts": attempts, "prompt_tokens": prompt, "output_tokens_incl_thinking": output,
            "attempts_with_unknown_usage": unknown, "latency_s_total": round(latency, 3),
            "est_cost_usd": None if cost is None else round(cost, 6)}


# ----------------------------------------------------------------------------- helpers

def _tag(run: dict, partial: bool) -> str:
    tags = []
    if not run["live"]:
        tags.append("MOCK - synthetic fixtures, not evidence")
    if partial:
        tags.append("PARTIAL")
    return f"\n[{' | '.join(tags)}]" if tags else ""


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _curves(cfg: Config, run: dict, slots: dict) -> dict:
    """Best-so-far dev mean machines after k proposals (k=0 is the initial program)."""
    init = run["initial"]["eval"]["mean_machines"]
    out = {}
    for cond in CONDITIONS:
        for rep in range(cfg.search.replicates):
            ys = [init]
            for s in range(cfg.search.proposals_per_condition):
                rec = slots.get((rep, cond, s))
                if rec is None:
                    break
                ys.append(rec["incumbent_after"]["mean_machines"])
            out[(cond, rep)] = ys
    return out


# ----------------------------------------------------------------------------- figures

def plot_progress(cfg: Config, run: dict, slots: dict, partial: bool, path: Path) -> None:
    curves = _curves(cfg, run, slots)
    fig, ax = plt.subplots(figsize=(7.5, 4.4), facecolor=SURFACE)
    _style(ax)
    ends = {}
    for cond in CONDITIONS:
        reps = [curves[(cond, r)] for r in range(cfg.search.replicates)]
        for ys in reps:
            ax.step(range(len(ys)), ys, where="post", color=COLORS[cond], linewidth=1, alpha=0.35)
        n = min(len(ys) for ys in reps)
        if len(reps) > 1 and n > 0:
            mean = [sum(ys[k] for ys in reps) / len(reps) for k in range(n)]
            ax.step(range(n), mean, where="post", color=COLORS[cond], linewidth=2, label=f"{LABELS[cond]} (mean of {len(reps)})")
            ends[cond] = (n - 1, mean[-1])
        else:
            ax.lines[-1].set_alpha(1)
            ax.lines[-1].set_linewidth(2)
            ax.lines[-1].set_label(LABELS[cond])
    if len(ends) == 2:  # direct labels, staggered so coincident curves stay readable
        (xa, ya), (xb, yb) = ends["independent"], ends["evolution"]
        hi, lo = ("independent", "evolution") if ya >= yb else ("evolution", "independent")
        for cond, dy in ((hi, 7), (lo, -7)):
            ax.annotate(LABELS[cond], ends[cond], xytext=(6, dy), textcoords="offset points", color=INK2, fontsize=9, va="center")
        ax.set_xlim(right=max(xa, xb) * 1.15 + 1)
    ax.set_xlabel("Proposals (model calls) in this condition", color=INK2)
    ax.set_ylabel("Best-so-far dev mean machines\n(lower is better)", color=INK2)
    ax.set_title(f"Search progress on development workloads{_tag(run, partial)}", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_heldout(heldout: dict, run: dict, partial: bool, path: Path) -> None:
    fam_of = {i["instance_id"]: i["family"] for i in heldout["instances"]}
    groups = list(FAMILIES) + ["all"]
    order = ["first_fit", "best_fit", "independent", "evolution"]

    def mean_for(pol: dict, group: str) -> float:
        vals = [m["machines_used"] for iid, m in pol["per_instance"].items() if group == "all" or fam_of[iid] == group]
        return sum(vals) / len(vals)

    fig, axes = plt.subplots(1, len(groups), figsize=(12, 4.1), facecolor=SURFACE)
    for ax, group in zip(axes, groups):
        _style(ax)
        for x, name in enumerate(order):
            if name in ("first_fit", "best_fit"):
                vals = [mean_for(heldout["policies"][name], group)]
            else:
                vals = [mean_for(p, group) for p in heldout["policies"].values() if p.get("condition") == name]
            if not vals:
                continue
            ax.scatter([x] * len(vals), vals, s=36, color=COLORS[name], edgecolors=SURFACE, linewidths=1.5, zorder=3)
            if len(vals) > 1:
                m = sum(vals) / len(vals)
                ax.hlines(m, x - 0.28, x + 0.28, color=COLORS[name], linewidth=2, zorder=2)
        ax.set_xticks(range(len(order)), [LABELS[o] for o in order], rotation=35, ha="right")
        ax.set_xlim(-0.6, len(order) - 0.4)
        ax.set_title(group.replace("_", "-"), color=INK, fontsize=10)
    axes[0].set_ylabel("Held-out mean machines\n(lower is better)", color=INK2)
    fig.suptitle(f"Held-out test comparison: dots = replicates, bar = mean; y-axes not zero-based{_tag(run, partial)}",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


# ----------------------------------------------------------------------------- tables

def write_summary(cfg: Config, run: dict, slots: dict, heldout: dict | None, winners: dict | None, path: Path) -> list[dict]:
    rows = []
    bf = heldout["policies"]["best_fit"]["mean_machines"] if heldout else None
    for rep in range(cfg.search.replicates):
        for cond in CONDITIONS:
            recs = [slots[k] for k in sorted(slots) if k[0] == rep and k[1] == cond]
            valid = sum(1 for r in recs if r["eval"].get("valid"))
            failed = sum(1 for r in recs if r["request_status"] != "ok")
            u = usage_totals(recs, run["config"]["pricing"])
            key = f"{rep}:{cond}"
            best_dev = (recs[-1]["incumbent_after"]["mean_machines"] if recs else run["initial"]["eval"]["mean_machines"])
            held = heldout["policies"].get(f"winner:{key}") if heldout else None
            rows.append({
                "replicate": rep,
                "condition": cond,
                "proposal_budget": cfg.search.proposals_per_condition,
                "completed_proposals": len(recs),
                "valid_proposals": valid,
                "valid_fraction": round(valid / len(recs), 4) if recs else "",
                "request_failures": failed,
                "best_dev_mean_machines": round(best_dev, 4),
                "winner_candidate_id": winners["winners"][key]["candidate_id"] if winners else "",
                "winner_from_slot": ("initial" if winners and winners["winners"][key]["from_slot"] is None
                                     else (winners["winners"][key]["from_slot"] if winners else "")),
                "heldout_mean_machines": round(held["mean_machines"], 4) if held else "",
                "heldout_pct_fewer_machines_vs_best_fit": round(100 * (bf - held["mean_machines"]) / bf, 3) if held else "",
                "request_attempts": u["attempts"],
                "prompt_tokens": u["prompt_tokens"],
                "output_tokens_incl_thinking": u["output_tokens_incl_thinking"],
                "attempts_with_unknown_usage": u["attempts_with_unknown_usage"],
                "latency_s_total": u["latency_s_total"],
                "mean_latency_s": round(u["latency_s_total"] / u["attempts"], 3) if u["attempts"] else "",
                "est_cost_usd": "unknown" if u["est_cost_usd"] is None else u["est_cost_usd"],
                "synthetic_mock": not run["live"],
            })
    _write_csv(path, rows)
    return rows


def write_heldout_per_instance(heldout: dict, path: Path) -> None:
    fam = {i["instance_id"]: i["family"] for i in heldout["instances"]}
    rows = []
    for name, pol in heldout["policies"].items():
        for iid in sorted(pol["per_instance"]):
            rows.append({
                "instance_id": iid,
                "family": fam[iid],
                "replicate": pol.get("replicate", ""),
                "policy": pol.get("condition", name),
                "candidate_id": pol.get("candidate_id", ""),
                "machines_used": pol["per_instance"][iid]["machines_used"],
                "lower_bound": pol["per_instance"][iid]["lower_bound"],
            })
    _write_csv(path, rows)


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as fh:
        if not rows:
            return
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _feedback_block(ev: dict) -> str:
    if not ev.get("valid"):
        return f"invalid ({ev.get('error_kind')}): {ev.get('error')}"
    fams = ", ".join(f"{f} {v:.3f}" for f, v in ev["family_means"].items())
    return f"mean {ev['mean_machines']:.3f} (total {ev['total_machines']}); {fams}"


def write_candidate_changes(run: dict, slots: dict, out: Path) -> int:
    """Diffs for strict incumbent improvements, chosen from dev logs only (never test results)."""
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    sources = {run["initial"]["candidate_id"]: (run["initial"]["source"], run["initial"]["eval"])}
    for rec in slots.values():
        if rec["eval"].get("valid"):
            sources.setdefault(rec["eval"]["candidate_id"], (rec["candidate"]["source"], rec["eval"]))
    banner = "" if run["live"] else "> **MOCK RUN — synthetic integration fixtures, not model output or evidence.**\n\n"
    n = 0
    for key in sorted(slots, key=lambda k: slots[k]["global_index"]):
        rec = slots[key]
        if not rec["improved"]:
            continue
        parent_src, parent_ev = sources[rec["incumbent_before"]["candidate_id"]]
        _write_change(out / f"r{rec['replicate']}_{rec['condition']}_slot{rec['slot']:03d}.md", banner, rec,
                      parent_src, parent_ev, "Strict improvement of the development incumbent")
        n += 1
    if n == 0:
        bad = next((slots[k] for k in sorted(slots, key=lambda k: slots[k]["global_index"])
                    if slots[k]["candidate"]["source"]), None)
        if bad:
            parent_src, parent_ev = sources[bad["incumbent_before"]["candidate_id"]]
            _write_change(out / "no_improvements_representative_change.md", banner, bad, parent_src, parent_ev,
                          "NO strict improvements occurred in this run. This is a representative (first) non-improving change")
    return n


def _write_change(path: Path, banner: str, rec: dict, parent_src: str, parent_ev: dict, heading: str) -> None:
    child = rec["candidate"]["source"] or ""
    diff = "".join(difflib.unified_diff(parent_src.splitlines(True), child.splitlines(True), "parent", "child"))
    path.write_text(
        f"# {heading}\n\n{banner}"
        f"- replicate {rec['replicate']}, condition `{rec['condition']}`, slot {rec['slot']}\n"
        f"- parent `{rec['incumbent_before']['candidate_id']}` -> child `{rec['eval'].get('candidate_id', 'invalid')}`\n\n"
        f"**Before (dev):** {_feedback_block(parent_ev)}\n\n**After (dev):** {_feedback_block(rec['eval'])}\n\n"
        f"```diff\n{diff}```\n\n## Model explanation / raw response\n\n~~~~text\n{rec['candidate']['raw_response']}\n~~~~\n"
    )


# ----------------------------------------------------------------------------- entry

def generate(cfg: Config, run_id: str, log=print) -> Path:
    rdir = run_dir(cfg, run_id)
    run = read_json(rdir / "run.json")
    slots = load_slots(rdir)
    final = rdir / "final"
    winners = read_json(final / "winners.json") if (final / "winners.json").exists() else None
    heldout = read_json(final / "heldout.json") if (final / "heldout.json").exists() else None
    partial = run["status"] != "complete" or bool(winners and winners["partial"])
    out = rdir / "report"
    out.mkdir(exist_ok=True)

    plot_progress(cfg, run, slots, partial, out / "search_progress.png")
    write_summary(cfg, run, slots, heldout, winners, out / "summary.csv")
    if heldout:
        plot_heldout(heldout, run, partial, out / "heldout_comparison.png")
        write_heldout_per_instance(heldout, out / "heldout_per_instance.csv")
    else:
        log("not finalized yet: skipping heldout_comparison.png and heldout_per_instance.csv")
    n_changes = write_candidate_changes(run, slots, out / "candidate_changes")
    all_records = list(slots.values()) + [read_json(p) for p in _log_files(rdir / "aborted")]
    atomic_write_json(out / "run_manifest.json", {
        "run_id": run_id,
        "report_generated_at": utc_now(),
        "mode": "live" if run["live"] else "mock (synthetic integration fixtures; not experimental evidence)",
        "completion": "partial" if partial else "complete",
        "status": run["status"],
        "stop_reason": run.get("stop_reason"),
        "provider": run["provider"],
        "model": run["model"],
        "config_hash": run["config_hash"],
        "config": run["config"],
        "workload_hash": run["workload_hash"],
        "split_hashes": run["split_hashes"],
        "git_commit": run["git_commit"],
        "versions": run["versions"],
        "runner": run["runner"],
        "condition_order": run["condition_order"],
        "completed_slots": len(slots),
        "total_slots": run["total_slots"],
        "request_attempts_sent": sum(sent_attempts(r["response"]) for r in all_records),
        "usage_all_attempts": usage_totals(all_records, run["config"]["pricing"]),
        "selection_frozen_at": winners["frozen_at"] if winners else None,
        "test_exposed": run.get("test_exposed", False),
        "finalization": run.get("finalization"),
        "strict_improvements_logged": n_changes,
    })
    log(f"report written to {out}")
    return out
