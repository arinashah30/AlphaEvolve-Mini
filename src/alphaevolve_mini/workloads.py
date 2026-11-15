"""Deterministic workload generation and split manifests.

The distribution below is a transparent project design choice, not a real
(e.g. Google) cluster workload.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .config import Config, WorkloadConfig
from .util import atomic_write_json, utc_now

FAMILIES = ("balanced", "cpu_heavy", "mem_heavy", "mixed")
BASE_FAMILIES = FAMILIES[:3]
SPLITS = ("dev", "test")
GENERATOR_VERSION = 1


def _job(rng: np.random.Generator, family: str) -> tuple[int, int]:
    if family == "balanced":
        b = int(rng.integers(100, 501))
        cpu = b + int(rng.integers(-50, 51))
        mem = b + int(rng.integers(-50, 51))
        return int(np.clip(cpu, 50, 900)), int(np.clip(mem, 50, 900))
    if family == "cpu_heavy":
        return int(rng.integers(450, 851)), int(rng.integers(50, 251))
    if family == "mem_heavy":
        return int(rng.integers(50, 251)), int(rng.integers(450, 851))
    if family == "mixed":
        return _job(rng, BASE_FAMILIES[int(rng.integers(0, 3))])
    raise ValueError(family)


def generate_split(split: str, seed: int, per_family: int, n_jobs: int) -> list[dict]:
    rng = np.random.default_rng(seed)  # one Generator per split
    instances = []
    for family in FAMILIES:
        for k in range(per_family):
            jobs = [list(_job(rng, family)) for _ in range(n_jobs)]
            instances.append({"instance_id": f"{split}-{family}-{k:03d}", "split": split, "family": family, "jobs": jobs})
    return instances


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _encode(inst: dict) -> bytes:
    return (json.dumps(inst, sort_keys=True) + "\n").encode()


def split_dir(cfg: Config, split: str) -> Path:
    if split not in SPLITS:
        raise ValueError(split)
    return Path(cfg.workloads.data_dir) / split


def manifest_path(cfg: Config) -> Path:
    return Path(cfg.workloads.data_dir) / "manifest.json"


def generate(cfg: Config) -> dict:
    """Write dev/test instance files and a manifest. Refuses to overwrite different data."""
    w = cfg.workloads
    plan = {
        "dev": generate_split("dev", w.dev_seed, w.dev_instances_per_family, w.jobs_per_instance),
        "test": generate_split("test", w.test_seed, w.test_instances_per_family, w.jobs_per_instance),
    }
    manifest = {
        "generator_version": GENERATOR_VERSION,
        "generator_config": asdict(w),
        "capacity": {"cpu": 1000, "mem": 1000},
        "families": list(FAMILIES),
        "splits": {},
    }
    for split, insts in plan.items():
        files = {f"{i['instance_id']}.json": _sha(_encode(i)) for i in insts}
        manifest["splits"][split] = {
            "n_instances": len(insts),
            "files": files,
            "split_hash": _sha(json.dumps(files, sort_keys=True).encode()),
        }

    existing = load_manifest(cfg, required=False)
    if existing is not None:
        same = {k: existing.get(k) for k in manifest} == manifest
        if not same:
            raise RuntimeError(
                f"{manifest_path(cfg)} exists with different content. Workloads are immutable; "
                "use a new data_dir or delete the old data deliberately before any real search."
            )
        verify(cfg)
        return existing

    for split, insts in plan.items():
        d = split_dir(cfg, split)
        d.mkdir(parents=True, exist_ok=True)
        for inst in insts:
            (d / f"{inst['instance_id']}.json").write_bytes(_encode(inst))
    manifest["created_at"] = utc_now()
    atomic_write_json(manifest_path(cfg), manifest)
    verify(cfg)
    return manifest


def load_manifest(cfg: Config, required: bool = True) -> dict | None:
    p = manifest_path(cfg)
    if not p.exists():
        if required:
            raise FileNotFoundError(f"no workload manifest at {p}; run generate-data first")
        return None
    return json.loads(p.read_text())


def verify(cfg: Config) -> dict:
    """Check files on disk match the manifest hashes and splits are disjoint."""
    manifest = load_manifest(cfg)
    seen_jobs: dict[str, set] = {}
    for split in SPLITS:
        d = split_dir(cfg, split)
        expected = manifest["splits"][split]["files"]
        on_disk = {p.name for p in d.glob("*.json")}
        if on_disk != set(expected):
            raise RuntimeError(f"{split} files on disk do not match manifest")
        seen_jobs[split] = set()
        for name, digest in expected.items():
            raw = (d / name).read_bytes()
            if _sha(raw) != digest:
                raise RuntimeError(f"hash mismatch for {split}/{name}")
            seen_jobs[split].add(_sha(json.dumps(json.loads(raw)["jobs"]).encode()))
    if seen_jobs["dev"] & seen_jobs["test"]:
        raise RuntimeError("dev and test splits share an identical instance")
    return manifest


def workload_hash(manifest: dict) -> str:
    return _sha(json.dumps({s: manifest["splits"][s]["split_hash"] for s in SPLITS}, sort_keys=True).encode())[:16]


def load_split(cfg: Config, split: str) -> list[dict]:
    d = split_dir(cfg, split)
    return [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))]


def family_means(per_instance: dict[str, float], instances: list[dict]) -> dict[str, float]:
    fam = {i["instance_id"]: i["family"] for i in instances}
    out = {}
    for f in FAMILIES:
        vals = [v for k, v in per_instance.items() if fam[k] == f]
        if vals:
            out[f] = sum(vals) / len(vals)
    return out
