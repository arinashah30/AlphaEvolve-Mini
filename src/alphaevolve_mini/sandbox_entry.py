"""Evaluation entry point executed inside the isolated container.

Usage: python -I sandbox_entry.py <request.json> <instances_dir>

Reads one candidate plus its limits, re-validates it, runs it on every instance in
the mounted directory, and prints one JSON object on stdout. The host re-verifies
every assignment with the trusted simulator, so this output is never trusted for
scores. Messages are kept short and never include paths, tracebacks or environment.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alphaevolve_mini import candidates, simulator  # noqa: E402


def run(request_path: str, data_dir: str) -> dict:
    with open(request_path) as fh:
        request = json.load(fh)
    limits = candidates.Limits.from_dict(request.get("limits"))
    try:
        score = candidates.compile_score(request["source"], limits)
    except candidates.ValidationError as exc:
        return {"ok": False, "error_kind": "validation", "error": str(exc)}
    except Exception as exc:
        return {"ok": False, "error_kind": "compile", "error": type(exc).__name__}

    results = []
    for name in sorted(os.listdir(data_dir)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(data_dir, name)) as fh:
            inst = json.load(fh)
        try:
            assignment = simulator.place_jobs(inst["jobs"], score)
        except simulator.CandidateError as exc:
            return {"ok": False, "error_kind": "runtime", "error": f"{inst['family']} instance: {exc}"}
        results.append({"instance_id": inst["instance_id"], "assignment": assignment})
    return {"ok": True, "instances": results}


def main() -> None:
    sys.setrecursionlimit(200)
    try:
        out = run(sys.argv[1], sys.argv[2])
    except Exception as exc:  # evaluator-level failure; keep it generic
        out = {"ok": False, "error_kind": "evaluator", "error": type(exc).__name__}
    sys.stdout.write(json.dumps(out))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
