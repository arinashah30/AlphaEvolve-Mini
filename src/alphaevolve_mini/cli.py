"""Command-line interface: `alphaevolve-mini <command>`."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from . import baselines, config as config_mod, finalize as finalize_mod, report, search, workloads
from .config import Config, ConfigError
from .providers import MockProvider, ProviderSetupError, make_provider, AttemptBudget
from .runner import ContainerRunner, RunnerUnavailable, make_runner
from .util import atomic_write_json, read_json, utc_now

ROOT_HINT = "run commands from the project root"
LIVE_CHECKS = "_live_checks"


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Minimal .env loader (KEY=VALUE lines). Never overrides existing environment variables."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def _cfg_from_run(run_id: str, runs_dir: str) -> Config:
    path = Path(runs_dir) / run_id / "run.json"
    if not path.exists():
        raise ConfigError(f"no run {run_id} under {runs_dir}/ ({ROOT_HINT})")
    return config_mod.from_dict(read_json(path)["config"], source_path=str(path))


def _engine_status(cfg: Config) -> str:
    try:
        ContainerRunner(cfg, {"dev"}).check_available()
        return f"OK ({cfg.runner.engine}, image {cfg.runner.image})"
    except RunnerUnavailable as exc:
        return f"UNAVAILABLE: {exc}"
    except Exception as exc:  # noqa: BLE001
        return f"UNAVAILABLE: {type(exc).__name__}"


def _live_check_file(cfg: Config) -> Path:
    safe = (cfg.provider.model or "none").replace("/", "_")
    return Path(cfg.runs_dir) / LIVE_CHECKS / f"ok_{safe}.json"


# ----------------------------------------------------------------------------- commands

def cmd_preflight(args) -> int:
    _load_dotenv()
    cfg = config_mod.load(args.config)
    s, p = cfg.search, cfg.provider
    print(f"config:            {cfg.source_path}  (name={cfg.name}, hash={cfg.identity_hash()})")
    print(f"python:            {sys.version.split()[0]}")
    print(f"mode:              {'LIVE' if args.live else 'mock / offline (no API calls)'}  provider={args.provider}")
    print(f"proposal slots:    {cfg.total_slots} = 2 conditions x {s.proposals_per_condition} proposals x {s.replicates} replicates")
    print(f"attempt limit:     {s.max_total_attempts} total request attempts (retries count); max {s.max_retries} retries per slot")
    print(f"per-request:       max_output_tokens={p.max_output_tokens}, timeout={p.request_timeout_s:g}s, temperature={p.temperature}")
    print(f"model:             {p.model or 'NOT SET (required for live runs)'}")
    print(f"api key env:       {p.api_key_env} is {'set' if os.environ.get(p.api_key_env) else 'NOT set'} (value never printed)")
    try:
        m = workloads.verify(cfg)
        print(f"workloads:         OK {cfg.workloads.data_dir} (dev {m['splits']['dev']['n_instances']}, test {m['splits']['test']['n_instances']}, hash {workloads.workload_hash(m)})")
    except FileNotFoundError:
        print(f"workloads:         missing; run generate-data --config {args.config}")
    except RuntimeError as exc:
        print(f"workloads:         INVALID: {exc}")
    print(f"container runner:  {_engine_status(cfg)}")
    pr = cfg.pricing
    if pr.input_per_million is not None and pr.output_per_million is not None:
        # Rough upper bound: every attempt uses a ~2.5k-token prompt and the full output limit.
        ub = s.max_total_attempts * (2500 / 1e6 * pr.input_per_million + p.max_output_tokens / 1e6 * pr.output_per_million)
        print(f"cost upper bound:  ~${ub:.2f} if every allowed attempt used the full output limit (prices as of {pr.as_of}; rough, not enforced)")
    else:
        print("cost estimate:     unknown (no prices configured); set provider-side spend limits")
    print(f"live check:        {'done' if _live_check_file(cfg).exists() else 'not yet run (required before a live search)'}")
    return 0


def cmd_generate_data(args) -> int:
    cfg = config_mod.load(args.config)
    m = workloads.generate(cfg)
    print(f"workloads in {cfg.workloads.data_dir}: dev {m['splits']['dev']['n_instances']}, test {m['splits']['test']['n_instances']}; hash {workloads.workload_hash(m)}")
    return 0


def cmd_build_image(args) -> int:
    cfg = config_mod.load(args.config)
    engine = args.engine or cfg.runner.engine
    if shutil.which(engine) is None:
        print(f"{engine} not found on PATH", file=sys.stderr)
        return 2
    cmd = [engine, "build", "-t", cfg.runner.image, "-f", "docker/Dockerfile", "docker"]
    print(" ".join(cmd))
    return subprocess.run(cmd).returncode


def cmd_check_runner(args) -> int:
    """Implementation checkpoint: one valid and one intentionally invalid fixture in the container."""
    cfg = config_mod.load(args.config)
    workloads.verify(cfg)
    runner = ContainerRunner(cfg, {"dev"})
    runner.check_available()
    fixtures = {
        "valid best-fit": (baselines.INITIAL_SOURCE, True),
        "invalid (passes AST, crashes at runtime)": ("def score(job_cpu, job_mem, free_cpu, free_mem):\n    return 1 / (free_cpu - free_cpu)\n", False),
        "invalid (rejected by AST allowlist)": ("import os\ndef score(job_cpu, job_mem, free_cpu, free_mem):\n    return 0\n", False),
        "invalid (non-finite score)": ("def score(job_cpu, job_mem, free_cpu, free_mem):\n" + "    x = 1000000.0 ** 4\n    return x * x * x * x * x * x * x * x * x * x * x * x * x * x\n", False),
    }
    ok = True
    for name, (src, expect_valid) in fixtures.items():
        res = runner.evaluate(src, "dev")
        good = bool(res.get("valid")) == expect_valid
        ok &= good
        detail = f"mean {res['mean_machines']:.3f}" if res.get("valid") else f"{res.get('error_kind')}: {res.get('error')}"
        print(f"{'PASS' if good else 'FAIL'}  {name}: {detail}")
    return 0 if ok else 1


def cmd_check_live(args) -> int:
    _load_dotenv()
    cfg = config_mod.load(args.config)
    provider = make_provider(args.provider, cfg, args.live)
    workloads.verify(cfg)
    runner = make_runner(cfg, live=True, allowed_splits={"dev"})
    init = runner.evaluate(baselines.INITIAL_SOURCE, "dev")
    from . import candidates, prompts
    prompt = prompts.build_prompt(baselines.INITIAL_SOURCE, prompts.format_feedback(init), cfg.candidate_limits)
    resp = provider.propose(prompt, "live-check", AttemptBudget(1 + cfg.search.max_retries))
    record = {"checked_at": utc_now(), "model": provider.model, "prompt": prompt, "response": resp.to_dict()}
    if resp.ok:
        try:
            src = candidates.extract_source(resp.text)
            record["eval"] = runner.evaluate(src, "dev")
        except candidates.ValidationError as exc:
            record["eval"] = {"valid": False, "error_kind": "extraction", "error": str(exc)}
    out = Path(cfg.runs_dir) / LIVE_CHECKS / f"{datetime.now().strftime('%Y%m%dT%H%M%S')}.json"
    atomic_write_json(out, record)
    print(f"saved {out}")
    for a in resp.attempts:
        print(f"attempt {a.attempt}: {a.status} {a.http_code or ''} {a.error_message or ''} latency={a.latency_s} usage={a.usage}")
    if not resp.ok:
        print("live check FAILED: no response", file=sys.stderr)
        return 1
    ev = record.get("eval", {})
    print(f"candidate: {'valid, dev mean ' + format(ev['mean_machines'], '.3f') if ev.get('valid') else 'invalid: ' + str(ev.get('error'))}")
    atomic_write_json(_live_check_file(cfg), {"checked_at": record["checked_at"], "record": str(out)})
    print("live check OK (a response was received and evaluated); live search is now allowed for this model")
    return 0


def _do_search(cfg: Config, args, provider) -> int:
    runner = make_runner(cfg, live=provider.live, allowed_splits={"dev"}, prefer_container=getattr(args, "container", False))
    if not provider.live:
        print(f"NOTE: mock provider -> synthetic fixtures; runner={runner.kind}. Outputs are not experimental evidence.")
    run = search.run_search(cfg, args.run_id, provider, runner, resume=args.resume)
    print(f"run {args.run_id}: {run['status']} ({run.get('completed_slots')}/{run['total_slots']} slots, "
          f"{run.get('attempts_used')} attempts){' - ' + run['stop_reason'] if run.get('stop_reason') else ''}")
    return 0 if run["status"] == "complete" else 3


def cmd_search(args) -> int:
    _load_dotenv()
    cfg = config_mod.load(args.config)
    provider = make_provider(args.provider, cfg, args.live)
    if provider.live and not _live_check_file(cfg).exists():
        print(f"refusing live search: run `alphaevolve-mini check-live --config {args.config} --provider gemini --live` first",
              file=sys.stderr)
        return 2
    return _do_search(cfg, args, provider)


def _test_runner_factory(cfg: Config, run: dict, container: bool):
    return lambda: make_runner(cfg, live=run["live"], allowed_splits={"test"}, prefer_container=container)


def cmd_finalize(args) -> int:
    cfg = _cfg_from_run(args.run_id, args.runs_dir)
    run = read_json(Path(args.runs_dir) / args.run_id / "run.json")
    res = finalize_mod.finalize(cfg, args.run_id, _test_runner_factory(cfg, run, args.container), allow_partial=args.allow_partial)
    bf = res["policies"]["best_fit"]["mean_machines"]
    print(f"held-out mean machines (lower is better){' [PARTIAL]' if res['partial'] else ''}{'' if res['live'] else ' [MOCK]'}:")
    for name, pol in res["policies"].items():
        print(f"  {name:<24} {pol['mean_machines']:.3f}  ({100 * (bf - pol['mean_machines']) / bf:+.2f}% fewer machines vs best-fit)")
    return 0


def cmd_report(args) -> int:
    cfg = _cfg_from_run(args.run_id, args.runs_dir)
    report.generate(cfg, args.run_id)
    return 0


def cmd_smoke(args) -> int:
    if args.provider != "mock":
        print("smoke runs only with the mock provider", file=sys.stderr)
        return 2
    cfg = config_mod.load(args.config)
    workloads.generate(cfg)
    args.run_id = args.run_id or f"smoke-{datetime.now().strftime('%Y%m%dT%H%M%S')}"
    args.resume = False
    rc = _do_search(cfg, args, MockProvider())
    if rc not in (0,):
        return rc
    run = read_json(Path(cfg.runs_dir) / args.run_id / "run.json")
    finalize_mod.finalize(cfg, args.run_id, _test_runner_factory(cfg, run, args.container))
    report.generate(cfg, args.run_id)
    print(f"smoke OK: runs/{args.run_id} (synthetic; never use smoke output as evidence)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="alphaevolve-mini", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    def with_config(sp, default="configs/main.yaml"):
        sp.add_argument("--config", default=default)

    sp = sub.add_parser("preflight", help="show configuration, projected slots, limits and environment checks")
    with_config(sp)
    sp.add_argument("--provider", default="mock", choices=["mock", "gemini"])
    sp.add_argument("--live", action="store_true")
    sp.set_defaults(fn=cmd_preflight)

    sp = sub.add_parser("generate-data", help="generate immutable dev/test workloads and manifest")
    with_config(sp)
    sp.set_defaults(fn=cmd_generate_data)

    sp = sub.add_parser("build-image", help="build the evaluator container image")
    with_config(sp)
    sp.add_argument("--engine", choices=["docker", "podman"])
    sp.set_defaults(fn=cmd_build_image)

    sp = sub.add_parser("check-runner", help="checkpoint: run valid and invalid fixtures in the container")
    with_config(sp)
    sp.set_defaults(fn=cmd_check_runner)

    sp = sub.add_parser("check-live", help="send ONE live request, evaluate it, and save its metadata")
    with_config(sp)
    sp.add_argument("--provider", default="gemini", choices=["gemini"])
    sp.add_argument("--live", action="store_true")
    sp.set_defaults(fn=cmd_check_live)

    sp = sub.add_parser("smoke", help="offline end-to-end run with the mock provider (synthetic)")
    with_config(sp, "configs/smoke.yaml")
    sp.add_argument("--provider", default="mock")
    sp.add_argument("--run-id")
    sp.add_argument("--container", action="store_true", help="evaluate mock fixtures in the container too")
    sp.set_defaults(fn=cmd_smoke)

    sp = sub.add_parser("search", help="run or resume the search")
    with_config(sp)
    sp.add_argument("--provider", default="mock", choices=["mock", "gemini"])
    sp.add_argument("--live", action="store_true")
    sp.add_argument("--run-id", required=True)
    sp.add_argument("--resume", action="store_true")
    sp.add_argument("--container", action="store_true", help="use the container runner even for mock fixtures")
    sp.set_defaults(fn=cmd_search)

    for name, fn, help_ in (("finalize", cmd_finalize, "freeze winners on dev, then evaluate them on test"),
                            ("report", cmd_report, "regenerate figures and tables from saved files")):
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("--run-id", required=True)
        sp.add_argument("--runs-dir", default="runs")
        if name == "finalize":
            sp.add_argument("--allow-partial", action="store_true")
            sp.add_argument("--container", action="store_true")
        sp.set_defaults(fn=fn)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args) or 0
    except (ConfigError, ProviderSetupError, RunnerUnavailable, search.SearchError, FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
