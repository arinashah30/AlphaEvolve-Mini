import pytest

from alphaevolve_mini import config, workloads


@pytest.fixture
def cfg(tmp_path):
    c = config.from_dict({
        "name": "test",
        "workloads": {"data_dir": str(tmp_path / "data"), "dev_seed": 5, "test_seed": 6,
                      "dev_instances_per_family": 1, "test_instances_per_family": 2, "jobs_per_instance": 40},
        "search": {"proposals_per_condition": 6, "replicates": 2, "max_total_attempts": 100, "backoff_initial_s": 0.0},
        "runner": {"timeout_s": 30},
        "runs_dir": str(tmp_path / "runs"),
    })
    workloads.generate(c)
    return c
