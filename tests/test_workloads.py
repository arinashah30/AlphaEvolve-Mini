import json

import pytest

from alphaevolve_mini import config, workloads as W


def test_deterministic_splits():
    a = W.generate_split("dev", 5, 2, 30)
    b = W.generate_split("dev", 5, 2, 30)
    c = W.generate_split("dev", 6, 2, 30)
    assert a == b
    assert a != c


def test_family_bounds():
    for inst in W.generate_split("test", 9, 3, 200):
        for cpu, mem in inst["jobs"]:
            assert 50 <= cpu <= 900 and 50 <= mem <= 900
            if inst["family"] == "cpu_heavy":
                assert 450 <= cpu <= 850 and 50 <= mem <= 250
            if inst["family"] == "mem_heavy":
                assert 50 <= cpu <= 250 and 450 <= mem <= 850
            if inst["family"] == "balanced":
                assert abs(cpu - mem) <= 100


def test_equal_family_counts(cfg):
    for split, n in (("dev", 1), ("test", 2)):
        fams = [i["family"] for i in W.load_split(cfg, split)]
        assert all(fams.count(f) == n for f in W.FAMILIES)


def test_manifest_and_disjoint(cfg):
    m = W.verify(cfg)
    assert m["splits"]["dev"]["split_hash"] != m["splits"]["test"]["split_hash"]
    dev = {json.dumps(i["jobs"]) for i in W.load_split(cfg, "dev")}
    test = {json.dumps(i["jobs"]) for i in W.load_split(cfg, "test")}
    assert not dev & test


def test_regenerate_same_is_ok_but_different_refused(cfg):
    W.generate(cfg)  # idempotent
    changed = config.from_dict(cfg.to_dict() | {"workloads": cfg.to_dict()["workloads"] | {"jobs_per_instance": 41}})
    with pytest.raises(RuntimeError, match="immutable"):
        W.generate(changed)


def test_tamper_detected(cfg):
    p = next(W.split_dir(cfg, "dev").glob("*.json"))
    d = json.loads(p.read_text())
    d["jobs"][0][0] += 1
    p.write_text(json.dumps(d))
    with pytest.raises(RuntimeError, match="hash mismatch"):
        W.verify(cfg)


def test_family_means():
    insts = [{"instance_id": "a", "family": "balanced"}, {"instance_id": "b", "family": "balanced"},
             {"instance_id": "c", "family": "mixed"}]
    assert W.family_means({"a": 2, "b": 4, "c": 9}, insts) == {"balanced": 3, "mixed": 9}


def test_same_seeds_rejected():
    with pytest.raises(config.ConfigError):
        config.from_dict({"name": "x", "workloads": {"dev_seed": 1, "test_seed": 1}})
