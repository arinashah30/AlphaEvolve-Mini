# AlphaEvolve Mini implementation plan

## Goal and scope

Build `alphaevolve-mini`, a local Python experiment comparing LLM-guided program improvement with independent candidate sampling on two-resource scheduling. Produce reproducible logs, plots, and one explainable code-change example for a CS329Z paper presentation.

Research question: with equal numbers of model proposals, does supplying the best previous program and its evaluation feedback improve the final scheduling heuristic compared with repeatedly generating independent candidates?

This is a proposed small experiment inspired by AlphaEvolve, not its implementation or a reproduction of Google's production scheduling gain. Keeping one incumbent is a simplified search policy. The comparison measures the combined effect of program reuse and feedback; it does not isolate either component or establish superiority over population-based search.

Implement the experiment directly. The simulator is a required evaluation component, not a separate preliminary project. Do not build a website, train a model, implement a model ensemble, or create presentation slides or narration. The student will write their own analysis and presentation.

## Defaults and implementation decisions

- Python 3.11+, `uv`, package layout, CLI, pytest, NumPy, Matplotlib, and YAML configuration.
- Gemini is the first and only required live backend; use the current official Google Gen AI SDK. Read its current official documentation during implementation. Configure the exact model ID, do not guess one or depend on a deprecated ID.
- Provide a deterministic mock backend for offline development. Label all mock outputs as synthetic integration fixtures.
- Laptop CPU runs control, scheduling, evaluation, and plotting. No local GPU or training service is required. Tinker and OpenRouter are outside the minimum scope.
- Docker is the default isolation mechanism for live evaluation of generated code. Check its availability early. Do not silently execute untrusted candidates in the host process if Docker is absent.
- Environment variables hold API keys. Ignore `.env` and credentials in Git. Supply `.env.example` with placeholders only.
- Inspect an existing checkout and its AGENTS.md before edits. Preserve unrelated files. Do not create a remote repository or push unless separately requested.

## Experiment specification

### Task and candidate interface

Use online, append-only, two-dimensional bin packing as a simplified resource-placement problem. Each machine has CPU capacity 1000 and memory capacity 1000, expressed in integer units. Jobs arrive in a fixed sequence and never depart. Open a new identical machine only when no existing machine can fit the job. Finish placing every job.

The model supplies exactly one pure function:

```python
def score(job_cpu, job_mem, free_cpu, free_mem):
    return ...
```

Call it only for feasible existing machines. Higher finite numeric scores are better. Ties go to the earliest-created machine. The trusted evaluator owns feasibility checks, opening machines, updating capacities, and computing metrics. The candidate cannot change those rules. If no machine fits, open a machine without calling the candidate. Every job must fit a fresh machine.

Keep candidate programs short: arithmetic, comparisons, simple local assignments, conditional expressions, and return statements; optionally allow bare `abs`, `min`, and `max` calls. Reject imports, attributes, indexing, loops, comprehensions, nested functions, globals/nonlocals, decorators, arbitrary calls, and access to files, networking, or subprocesses. Set a source-length limit and bounds on literals and exponentiation. Maintain an explicit AST allowlist, not a blacklist. This restriction makes failures interpretable but must be reported as a limitation of the search space.

Use fixed first-fit and best-fit reference policies. First-fit bypasses scoring and selects the earliest feasible machine. Define best-fit unambiguously as choosing the feasible machine with the smallest sum of residual CPU and memory fractions after placement, with the same tie-break. Use this best-fit score as the common initial incumbent for both search conditions. Do not include the paper's heuristic in the required implementation; it is optional later.

### Workloads and splits

Pre-generate immutable development and test files, using separate RNG seeds and NumPy Generator instances. Use this transparent proposed distribution, not a claimed Google workload:

| Family | CPU demand | Memory demand |
| --- | --- | --- |
| Balanced | Sample integer b uniformly from 100 through 500, then independent integer jitter from -50 through 50 | Same b, separate jitter; clip both demands to [50, 900] |
| CPU-heavy | Uniform integer [450, 850] | Uniform integer [50, 250] |
| Memory-heavy | Uniform integer [50, 250] | Uniform integer [450, 850] |
| Mixed | For each job, choose uniformly among the three families above | Generate according to that family |

Default development set: 3 instances per family, 100 jobs each (12 instances).
Default test set: 10 instances per family, 100 jobs each (40 instances).
Offline smoke mode: 1 instance per family, 20 jobs each; never use smoke output as experimental evidence.

Write split hashes and generator configuration to a manifest. Search receives only development files. Do not mount test files in search-evaluation containers or expose test metrics to the model. Change these defaults only before any real search, then regenerate and version the manifest. Do not tune workloads after viewing test results.

### Fitness and diagnostics

Primary development objective: minimize mean machines used. All families have equal instance counts, so they receive equal weight. LLM feedback includes aggregate and per-family development machines used, valid/invalid status, and a short sanitized error when relevant. No arbitrary retrospective changes to the objective.

Report held-out mean machines used, per-family means, and percent reduction relative to the fixed best-fit policy. State the denominator and direction clearly. Record residual CPU/memory as optional diagnostics, not independent evidence: for fixed demands and capacities, aggregate unused resources are determined by machine count.

Verify that all jobs are placed once, residual resources never become negative, and allocations conserve resources. These are evaluator invariants, not claims delegated to generated code.

### Conditions and fairness

Both conditions use the exact same model ID, generation settings, source constraints, objective description, development instances, initial incumbent, and candidate count. Each model call returns one candidate.

- `independent`: every proposal gets the same initial program and its fixed initial development feedback. It never sees earlier generated candidates or their outcomes. Select its best valid candidate including the initial program.
- `evolution`: start with that same initial program and feedback. Each proposal gets only the current best-so-far program and its development feedback. Replace the incumbent only for a strict improvement; retain the incumbent on a tie or regression. No growing conversation history.

The first prompts should be identical. Use a stateless request per proposal. Fix tie selection to earliest candidate. Invalid candidates count against the proposal budget; do not ask for free repairs. Use the initial program if every proposal fails.

Equal proposal count is not equal token cost. Log actual usage, prompt lengths, latency, and failures. Do not claim cost matching. Interleave the conditions within each paired search replicate, randomizing which condition goes first. Record this ordering. Do not assume provider seeds guarantee deterministic generation.

Pilot: 5 proposals per condition, one replicate, to check operation and cost. Keep it separate from final experiment results. Any design change after the pilot must be frozen before the main run.
Main target: 20 proposals per condition, 3 independent search replicates, 120 proposal slots total. Allow a smaller predeclared run if time is limited and report its actual size.

### Costs, errors, and resumption

Require an explicit live flag and configured model; otherwise use mock mode. Provide a preflight command showing projected proposal slots and configuration. Verify one live response before launching a batch.

Hard limits: proposal slots, total API request attempts, per-response output limit, and bounded request timeout. Retry only transient service errors, at most twice, with exponential backoff; log every attempt and usage. Count all request attempts toward the total attempt limit. Authentication, billing, and invalid-model errors stop immediately. Failed slots are recorded and counted; never quietly replace them with extra candidates.

Provide an optional conservative dollar estimate using configured, timestamped prices and actual usage, including reasoning/cached tokens where applicable. Mark unknown usage as unknown, never zero. Do not advertise a guaranteed dollar cap unless the provider actually enforces one. An initial $10-20 project allowance is a user-selected budget, not a cost forecast. If spend enforcement cannot be made reliable, require provider-side limits and a small pilot before the main run.

Write logs atomically after each proposal. Resume only from a matching configuration and workload hash; never resample a completed slot. Record the incumbent after every slot so continuation is unambiguous. A partially completed run must remain clearly marked partial.

## Candidate execution and isolation

Validate AST before execution. AST validation reduces the allowed language; it is not a security sandbox. Evaluate candidates in a disposable non-root Docker container with no network, no credentials, no Docker socket, a read-only root filesystem, restricted capabilities, process/memory/CPU limits, and bounded temporary storage. Mount only trusted evaluator code, the candidate, and the appropriate development or final-test instance files read-only.

Run an entire small evaluation batch per container, not a container per machine/job. Execute only after validation; expose only the required builtins. On timeout, kill and remove the entire container. Reject exceptions, non-finite results, booleans, invalid result types, and malformed output. Errors must not reveal host paths or environment values to the model.

First implementation checkpoint: prove this runner works with one valid and one intentionally invalid fixture. If the coding environment cannot use Docker, implement and test the rest in mock mode and document the exact blocker. Do not weaken isolation to meet a deadline.

## Suggested project structure

| Path | Responsibility |
| --- | --- |
| `pyproject.toml`, `uv.lock` | Dependencies, CLI entry point, reproducible environment |
| `src/alphaevolve_mini/config.py` | Validated configuration and run identity |
| `src/alphaevolve_mini/workloads.py` | Deterministic generation and split manifests |
| `src/alphaevolve_mini/simulator.py` | Trusted placement and metrics |
| `src/alphaevolve_mini/baselines.py` | First-fit and best-fit policies |
| `src/alphaevolve_mini/candidates.py` | Parsing, extraction, AST validation, candidate IDs |
| `src/alphaevolve_mini/runner.py` | Isolated batch evaluation, resource limits, cleanup |
| `src/alphaevolve_mini/providers.py` | Gemini and deterministic mock adapters |
| `src/alphaevolve_mini/prompts.py` | Shared prompt and condition-specific state |
| `src/alphaevolve_mini/search.py` | Proposal loop, selection, budgets, resume |
| `src/alphaevolve_mini/finalize.py` | Freeze winners, held-out evaluation |
| `src/alphaevolve_mini/report.py` | CSV summaries, figures, candidate diffs |
| `src/alphaevolve_mini/cli.py` | Commands and useful errors |
| `configs/smoke.yaml`, `configs/main.yaml` | Explicit experimental settings |
| `docker/`, `tests/` | Evaluator image and focused correctness tests |
| `README.md`, `.env.example`, `.gitignore` | Setup, methods, constraints, commands |

Keep modules small; combine files if that makes implementation simpler without mixing trusted evaluation with generated code.

## Implementation order and acceptance checks

1. **Bootstrap and preflight.** Inspect the checkout, create the package/configuration/CLI, check Python and Docker, and implement mock responses. Done when installation and offline preflight work and missing credentials never trigger a live call.
2. **Evaluation foundation.** Implement deterministic workloads, simulator, reference policies, AST validation, and isolated execution. Done when fixtures produce known placements and invalid code cannot execute on the host.
3. **Experiment loop.** Implement prompts, conditions, selection, immutable logs, resumption, and proposal limits using the mock adapter first. Done when an end-to-end mock experiment can be interrupted and resumed without duplicated slots or test access.
4. **Live model adapter.** Use current official SDK documentation, configurable model IDs, usage logging, bounded retries, and output extraction. Done when one explicit live proposal can be evaluated and its real response metadata is saved. Do not run the main experiment automatically as part of implementation.
5. **Finalization and reporting.** Freeze winners based solely on development scores, then evaluate the same winners on shared held-out workloads. Done when report generation consumes saved files without further API calls.
6. **Documentation and handoff.** Explain assumptions, budget controls, safety setup, commands, log schema, and known limitations. Distinguish verified checks from live runs not performed. Do not fabricate experiment results or label mock results as real.

## CLI contract

Implement these commands or equivalent documented commands:

```bash
uv sync
uv run alphaevolve-mini preflight --config configs/main.yaml
uv run alphaevolve-mini generate-data --config configs/main.yaml
uv run alphaevolve-mini smoke --provider mock
uv run alphaevolve-mini search --config configs/main.yaml --provider gemini --live --run-id main01
uv run alphaevolve-mini search --config configs/main.yaml --provider gemini --live --run-id main01 --resume
uv run alphaevolve-mini finalize --run-id main01
uv run alphaevolve-mini report --run-id main01
uv run pytest
```

Finalization refuses incomplete runs by default. Explicit partial finalization must freeze the completed selections, label the output partial, and prevent continued search in that same run after test exposure. Repeating finalization should read existing final results rather than launching new tests silently.

## Logs and deliverables

Each run contains a configuration snapshot, workload hashes, Git commit if available, Python/dependency versions, prompts, raw responses, validated candidate source, validation errors, development scores, token usage, latency, request status, incumbent history, frozen winner identifiers, and held-out per-instance results. Never save API keys or request headers. Preserve source for failed candidates when available.

Produce these outputs automatically:

1. `search_progress.png`: best-so-far development machines used vs proposals, lower is better; individual replicate curves plus a mean if there are multiple runs.
2. `heldout_comparison.png`: reference policies and selected search winners across workload families, using the same test instances.
3. `summary.csv`: replicate, condition, proposal count, valid fraction, best development score, held-out mean, usage, and latency.
4. `heldout_per_instance.csv`: instance ID, workload family, replicate, policy, and machines used; support paired comparisons.
5. `candidate_changes/`: parent-child diffs with before/after development feedback for strict incumbent improvements, selected without looking at test performance. If there are no improvements, save a representative failed change and say so.
6. `run_manifest.json`: mock/live, complete/partial, configuration, model, split hashes, selection timestamps, and finalization status.

Do not invent confidence intervals from three search replicates. Show all replicates and descriptive means/ranges. If adding workload bootstrap intervals, explain that they measure workload variation conditional on selected candidates, not the uncertainty of the entire search procedure.

## Focused verification

Test resource conservation, feasibility, new-machine behavior, tie-breaking, known tiny placements, deterministic splits, split disjointness, per-family aggregation, invalid candidate handling, and non-finite scores. Test that independent prompts remain unchanged, evolution changes only after a strict improvement, equal proposal budgets include invalid outputs, and resumption cannot repeat slots. Test that search cannot read test instances and finalization freezes winners before test evaluation. Verify reports regenerate entirely from logs and visibly label mock/partial runs.

## Time control and non-goals

Aim for a focused half-day implementation, but assess after the isolated evaluator and mock loop work. The extra isolation, logging, and finalization requirements may exceed the earlier rough 3-5 hour estimate on an unfamiliar setup. If time is tight, reduce the predeclared candidate count or search replicates and state that limitation. Preserve the held-out split, fair proposal accounting, isolation, and real-result labeling.

Defer extra providers, model comparisons, model ensembles, Tinker training, larger population search, complex schedulers, dashboards, and statistical significance claims. Spend available time checking evaluator correctness and inspecting an actual code change.

## References for implementation context

- AlphaEvolve paper: https://arxiv.org/abs/2506.13131
- Official Google Gen AI Python SDK: https://github.com/googleapis/python-genai
- Gemini API documentation: https://ai.google.dev/gemini-api/docs

The workload design, restricted candidate interface, budgets, file layout, and experimental protocol above are project design choices. Do not present them as details of the AlphaEvolve paper.
