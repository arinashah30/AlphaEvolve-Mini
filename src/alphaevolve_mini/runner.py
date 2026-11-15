"""Isolated batch evaluation of candidate programs.

ContainerRunner is the only runner allowed for model-generated code. It runs one
disposable container per candidate evaluation batch with no network, no
credentials, a read-only root filesystem, dropped capabilities, a non-root user,
and process/memory/CPU/tmpfs limits. Only the trusted evaluator package, the
candidate, and one split's instance files are mounted (all read-only).

LocalTrustedRunner runs the same entry point in a host subprocess. It exists only
for the deterministic mock provider's synthetic, hand-written fixtures and tests,
and the code refuses to use it for live runs.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

from . import candidates, simulator, workloads
from .config import Config

PACKAGE_DIR = Path(__file__).resolve().parent
ENTRY = PACKAGE_DIR / "sandbox_entry.py"
MAX_STDOUT = 5_000_000


class RunnerUnavailable(Exception):
    pass


class SplitAccessError(Exception):
    pass


def _sanitize(msg: str, limit: int = 300) -> str:
    for secret_path in {str(Path.cwd()), str(Path.home()), str(PACKAGE_DIR), tempfile.gettempdir()}:
        msg = msg.replace(secret_path, "<path>")
    return msg.replace("\n", " ")[:limit]


def invalid(kind: str, error: str, normalized: str | None = None) -> dict:
    return {"valid": False, "error_kind": kind, "error": _sanitize(error), "normalized_source": normalized}


class BaseRunner:
    kind = "base"

    def __init__(self, cfg: Config, allowed_splits: set[str]):
        self.cfg = cfg
        self.allowed_splits = frozenset(allowed_splits)
        self._instances: dict[str, list[dict]] = {}

    def _split_instances(self, split: str) -> list[dict]:
        if split not in self.allowed_splits:
            raise SplitAccessError(f"this runner may not read the {split!r} split")
        if split not in self._instances:
            self._instances[split] = workloads.load_split(self.cfg, split)
        return self._instances[split]

    def evaluate(self, source: str, split: str) -> dict:
        instances = self._split_instances(split)
        limits = self.cfg.candidate_limits
        try:
            normalized = candidates.validate(source, limits)  # parse only; nothing executes on the host
        except candidates.ValidationError as exc:
            return invalid("validation", str(exc))
        t0 = time.monotonic()
        raw = self._execute(normalized, split)
        elapsed = time.monotonic() - t0
        result = self._check(raw, instances, normalized)
        result["eval_seconds"] = round(elapsed, 3)
        result["runner"] = self.kind
        return result

    def _execute(self, normalized: str, split: str) -> dict:
        raise NotImplementedError

    def _check(self, raw: dict, instances: list[dict], normalized: str) -> dict:
        """Re-verify every assignment with the trusted simulator on the host."""
        if not raw.get("ok"):
            return invalid(raw.get("error_kind", "runtime"), raw.get("error", "evaluation failed"), normalized)
        by_id = {i["instance_id"]: i for i in instances}
        got = raw.get("instances")
        if not isinstance(got, list) or sorted(r.get("instance_id") for r in got) != sorted(by_id):
            return invalid("malformed", "evaluator output did not cover the expected instances", normalized)
        per_instance = {}
        try:
            for r in got:
                per_instance[r["instance_id"]] = simulator.verify_assignment(by_id[r["instance_id"]]["jobs"], r["assignment"])
        except (simulator.InvariantError, KeyError, TypeError, ValueError) as exc:
            return invalid("malformed", f"assignment failed verification: {exc}", normalized)
        return summarize(per_instance, instances) | {
            "valid": True,
            "candidate_id": candidates.candidate_id(normalized),
            "normalized_source": normalized,
        }


def summarize(per_instance: dict[str, dict], instances: list[dict]) -> dict:
    machines = {k: v["machines_used"] for k, v in per_instance.items()}
    return {
        "per_instance": per_instance,
        "total_machines": sum(machines.values()),
        "mean_machines": sum(machines.values()) / len(machines),
        "family_means": workloads.family_means(machines, instances),
    }


def _write_request(dirpath: Path, normalized: str, cfg: Config) -> Path:
    req = dirpath / "request.json"
    req.write_text(json.dumps({"source": normalized, "limits": cfg.candidate_limits.to_dict()}))
    os.chmod(dirpath, 0o755)
    os.chmod(req, 0o644)
    return req


def _parse_stdout(stdout: bytes) -> dict:
    if len(stdout) > MAX_STDOUT:
        return {"ok": False, "error_kind": "malformed", "error": "evaluator output too large"}
    try:
        data = json.loads(stdout.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        return {"ok": False, "error_kind": "malformed", "error": "evaluator output was not valid JSON"}
    if not isinstance(data, dict):
        return {"ok": False, "error_kind": "malformed", "error": "evaluator output was not an object"}
    return data


class ContainerRunner(BaseRunner):
    kind = "container"

    def scratch_root(self) -> Path:
        # Under the project (not /var/folders) so Podman/Docker VMs that share $HOME can see it.
        root = Path(self.cfg.runs_dir).resolve() / ".eval_tmp"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def check_available(self) -> None:
        r = self.cfg.runner
        if shutil.which(r.engine) is None:
            raise RunnerUnavailable(f"{r.engine} CLI not found on PATH")
        info = subprocess.run([r.engine, "info"], capture_output=True, timeout=30)
        if info.returncode != 0:
            raise RunnerUnavailable(f"{r.engine} is installed but its daemon/VM is not reachable")
        img = subprocess.run([r.engine, "image", "inspect", r.image], capture_output=True, timeout=30)
        if img.returncode != 0:
            raise RunnerUnavailable(f"image {r.image} not built; run `alphaevolve-mini build-image`")

    def command(self, name: str, request_dir: Path, data_dir: Path) -> list[str]:
        r = self.cfg.runner
        return [
            r.engine, "run", "--rm", "--name", name,
            "--network", "none",
            "--read-only",
            "--user", "65534:65534",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--pids-limit", str(r.pids_limit),
            "--memory", r.memory, "--memory-swap", r.memory,
            "--cpus", str(r.cpus),
            "--tmpfs", f"/tmp:rw,noexec,nosuid,size={r.tmpfs_size}",
            "--env", "PYTHONDONTWRITEBYTECODE=1",
            "-v", f"{PACKAGE_DIR}:/app/alphaevolve_mini:ro",
            "-v", f"{request_dir}:/candidate:ro",
            "-v", f"{data_dir}:/data:ro",
            r.image,
            "python", "-I", "/app/alphaevolve_mini/sandbox_entry.py", "/candidate/request.json", "/data",
        ]

    def _execute(self, normalized: str, split: str) -> dict:
        engine = self.cfg.runner.engine
        data_dir = workloads.split_dir(self.cfg, split).resolve()
        name = f"aem-eval-{uuid.uuid4().hex[:12]}"
        tmp = Path(tempfile.mkdtemp(dir=self.scratch_root()))
        try:
            _write_request(tmp, normalized, self.cfg)
            cmd = self.command(name, tmp, data_dir)
            try:
                proc = subprocess.run(cmd, capture_output=True, timeout=self.cfg.runner.timeout_s)
            except subprocess.TimeoutExpired:
                return {"ok": False, "error_kind": "timeout", "error": f"evaluation exceeded {self.cfg.runner.timeout_s:g}s"}
            if proc.returncode != 0 and not proc.stdout:
                if proc.returncode == 137:
                    return {"ok": False, "error_kind": "resource", "error": "evaluation was killed (memory limit)"}
                return {"ok": False, "error_kind": "container", "error": f"container exited with status {proc.returncode}"}
            return _parse_stdout(proc.stdout)
        finally:
            subprocess.run([engine, "rm", "-f", name], capture_output=True, timeout=30)
            shutil.rmtree(tmp, ignore_errors=True)


class LocalTrustedRunner(BaseRunner):
    """Host subprocess runner for synthetic mock fixtures only. Never for model output."""

    kind = "local-trusted-fixtures-only"

    def _execute(self, normalized: str, split: str) -> dict:
        data_dir = workloads.split_dir(self.cfg, split).resolve()
        with tempfile.TemporaryDirectory() as tmp:
            req = _write_request(Path(tmp), normalized, self.cfg)
            try:
                proc = subprocess.run(
                    [sys.executable, "-I", str(ENTRY), str(req), str(data_dir)],
                    capture_output=True, timeout=self.cfg.runner.timeout_s, env={},
                )
            except subprocess.TimeoutExpired:
                return {"ok": False, "error_kind": "timeout", "error": f"evaluation exceeded {self.cfg.runner.timeout_s:g}s"}
            return _parse_stdout(proc.stdout)


def make_runner(cfg: Config, live: bool, allowed_splits: set[str], prefer_container: bool = False) -> BaseRunner:
    """Live runs always get the container runner and fail loudly if it is unavailable."""
    if live or prefer_container:
        runner = ContainerRunner(cfg, allowed_splits)
        runner.check_available()
        return runner
    return LocalTrustedRunner(cfg, allowed_splits)
