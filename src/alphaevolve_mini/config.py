"""Validated configuration and run identity."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .candidates import Limits


THINKING_LEVELS = ("MINIMAL", "LOW", "MEDIUM", "HIGH", "XHIGH")


class ConfigError(Exception):
    pass


@dataclass(frozen=True)
class WorkloadConfig:
    data_dir: str = "data/main"
    dev_seed: int = 1001
    test_seed: int = 2002
    dev_instances_per_family: int = 3
    test_instances_per_family: int = 10
    jobs_per_instance: int = 100


@dataclass(frozen=True)
class SearchConfig:
    proposals_per_condition: int = 20
    replicates: int = 3
    max_total_attempts: int = 200
    max_retries: int = 2
    backoff_initial_s: float = 2.0
    ordering_seed: int = 7


@dataclass(frozen=True)
class ProviderConfig:
    model: str | None = None
    api_key_env: str = "GEMINI_API_KEY"
    temperature: float = 1.0
    max_output_tokens: int = 4096
    request_timeout_s: float = 120.0
    thinking_budget: int | None = None
    thinking_level: str | None = None  # MINIMAL | LOW | MEDIUM | HIGH | XHIGH; null = model default


@dataclass(frozen=True)
class PricingConfig:
    """Optional USD prices per 1M tokens. Leave null if unknown; costs then report as unknown."""

    as_of: str | None = None
    source: str | None = None
    input_per_million: float | None = None
    output_per_million: float | None = None  # applied to candidate + thinking tokens
    cached_input_per_million: float | None = None


@dataclass(frozen=True)
class RunnerConfig:
    engine: str = "docker"  # "docker" or "podman" (same CLI flags)
    image: str = "alphaevolve-mini-eval:0.1"
    timeout_s: float = 60.0
    memory: str = "256m"
    cpus: float = 1.0
    pids_limit: int = 32
    tmpfs_size: str = "16m"


@dataclass(frozen=True)
class Config:
    name: str
    workloads: WorkloadConfig = field(default_factory=WorkloadConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    provider: ProviderConfig = field(default_factory=ProviderConfig)
    pricing: PricingConfig = field(default_factory=PricingConfig)
    runner: RunnerConfig = field(default_factory=RunnerConfig)
    candidate_limits: Limits = field(default_factory=Limits)
    runs_dir: str = "runs"
    source_path: str | None = None

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d.pop("source_path")
        return d

    def identity_hash(self) -> str:
        """Hash of every experimental setting. Resume requires an exact match."""
        d = self.to_dict()
        d.pop("runs_dir")
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]

    @property
    def total_slots(self) -> int:
        return 2 * self.search.proposals_per_condition * self.search.replicates


_SECTIONS = {
    "workloads": WorkloadConfig,
    "search": SearchConfig,
    "provider": ProviderConfig,
    "pricing": PricingConfig,
    "runner": RunnerConfig,
    "candidate_limits": Limits,
}


def _build(cls, data: dict | None, section: str):
    data = data or {}
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ConfigError(f"unknown keys in {section}: {sorted(unknown)}")
    return cls(**data)


def from_dict(raw: dict, source_path: str | None = None) -> Config:
    raw = dict(raw)
    name = raw.pop("name", None)
    if not name:
        raise ConfigError("config needs a 'name'")
    kwargs = {k: _build(cls, raw.pop(k, None), k) for k, cls in _SECTIONS.items()}
    runs_dir = raw.pop("runs_dir", "runs")
    if raw:
        raise ConfigError(f"unknown top-level keys: {sorted(raw)}")
    cfg = Config(name=name, runs_dir=runs_dir, source_path=source_path, **kwargs)
    _check(cfg)
    return cfg


def load(path: str | Path) -> Config:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    with path.open() as fh:
        return from_dict(yaml.safe_load(fh) or {}, str(path))


def _check(cfg: Config) -> None:
    w, s, p, r = cfg.workloads, cfg.search, cfg.provider, cfg.runner
    if w.dev_seed == w.test_seed:
        raise ConfigError("dev_seed and test_seed must differ")
    for n in (w.dev_instances_per_family, w.test_instances_per_family, w.jobs_per_instance,
              s.proposals_per_condition, s.replicates, s.max_total_attempts):
        if not isinstance(n, int) or n < 1:
            raise ConfigError("counts and limits must be positive integers")
    if not 0 <= s.max_retries <= 2:
        raise ConfigError("max_retries must be between 0 and 2")
    if r.engine not in ("docker", "podman"):
        raise ConfigError("runner.engine must be 'docker' or 'podman'")
    if p.thinking_level is not None and p.thinking_level not in THINKING_LEVELS:
        raise ConfigError(f"provider.thinking_level must be one of {THINKING_LEVELS} or null")
    if p.thinking_level is not None and p.thinking_budget is not None:
        raise ConfigError("set at most one of provider.thinking_level and provider.thinking_budget")
    if p.max_output_tokens < 1 or p.request_timeout_s <= 0:
        raise ConfigError("provider limits must be positive")
