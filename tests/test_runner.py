import pytest

from alphaevolve_mini import baselines, runner as R, workloads

HDR = "def score(job_cpu, job_mem, free_cpu, free_mem):\n"


def test_local_runner_valid(cfg):
    r = R.LocalTrustedRunner(cfg, {"dev"})
    res = r.evaluate(baselines.INITIAL_SOURCE, "dev")
    assert res["valid"]
    ref = R.summarize(baselines.evaluate_reference("best_fit", workloads.load_split(cfg, "dev")), workloads.load_split(cfg, "dev"))
    assert res["total_machines"] == ref["total_machines"]
    assert set(res["family_means"]) == set(workloads.FAMILIES)


@pytest.mark.parametrize("src,kind", [
    (HDR + "    return 1 / (free_cpu - free_cpu)\n", "runtime"),
    (HDR + "    x = 1000000.0 ** 4\n    return x * x * x * x * x * x * x * x * x * x * x * x * x * x\n", "runtime"),
    (HDR + "    return job_cpu > 3\n", "runtime"),
    ("import os\n" + HDR + "    return 0\n", "validation"),
])
def test_local_runner_invalid(cfg, src, kind):
    res = R.LocalTrustedRunner(cfg, {"dev"}).evaluate(src, "dev")
    assert not res["valid"] and res["error_kind"] == kind
    assert str(cfg.workloads.data_dir) not in res["error"]


def test_invalid_code_never_executes_on_host(cfg, tmp_path, monkeypatch):
    marker = tmp_path / "pwned"
    src = f"import os\nos.system('touch {marker}')\n" + HDR + "    return 0\n"
    called = []
    monkeypatch.setattr(R.LocalTrustedRunner, "_execute", lambda *a: called.append(1))
    res = R.LocalTrustedRunner(cfg, {"dev"}).evaluate(src, "dev")
    assert not res["valid"] and not called and not marker.exists()


def test_split_access_restricted(cfg):
    with pytest.raises(R.SplitAccessError):
        R.LocalTrustedRunner(cfg, {"dev"}).evaluate(baselines.INITIAL_SOURCE, "test")


def test_host_reverifies_container_output(cfg):
    r = R.LocalTrustedRunner(cfg, {"dev"})
    insts = workloads.load_split(cfg, "dev")
    forged = {"ok": True, "instances": [{"instance_id": i["instance_id"], "assignment": [0] * len(i["jobs"])} for i in insts]}
    res = r._check(forged, insts, "x")
    assert not res["valid"] and res["error_kind"] == "malformed"
    res = r._check({"ok": True, "instances": []}, insts, "x")
    assert not res["valid"]


def test_container_command_isolation_flags(cfg, tmp_path):
    cmd = R.ContainerRunner(cfg, {"dev"}).command("n", tmp_path, tmp_path / "dev")
    joined = " ".join(cmd)
    for flag in ("--network none", "--read-only", "--user 65534:65534", "--cap-drop ALL",
                 "--security-opt no-new-privileges", "--pids-limit", "--memory", "--cpus", "--tmpfs", "--rm"):
        assert flag in joined
    mounts = [cmd[i + 1] for i, c in enumerate(cmd) if c == "-v"]
    assert all(m.endswith(":ro") for m in mounts) and len(mounts) == 3
    assert "docker.sock" not in joined and "-e" not in cmd and "--env-file" not in joined


def test_live_requires_container(cfg, monkeypatch):
    monkeypatch.setattr(R.shutil, "which", lambda name: None)
    with pytest.raises(R.RunnerUnavailable):
        R.make_runner(cfg, live=True, allowed_splits={"dev"})
    assert isinstance(R.make_runner(cfg, live=False, allowed_splits={"dev"}), R.LocalTrustedRunner)
