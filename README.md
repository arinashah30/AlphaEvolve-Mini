# alphaevolve-mini

A small, local experiment **inspired by** AlphaEvolve ([arXiv:2506.13131](https://arxiv.org/abs/2506.13131)).
It is not an implementation or reproduction of AlphaEvolve, and it does not reproduce Google's production scheduling gain.

**Research question.** With the same number of model proposals, does showing the model the best previous program and its evaluation feedback (`evolution`) give a better final scheduling heuristic than repeatedly asking for independent candidates (`independent`)?

**What it measures.** The combined effect of program reuse and feedback, under a single-incumbent search policy. It does not separate the two effects, and it says nothing about population-based search.

The workload design, restricted candidate interface, budgets, and protocol are project design choices. None of them come from the AlphaEvolve paper.

---

## Status: what has been verified

| Check | Status |
| --- | --- |
| `uv sync`, offline `preflight`, `generate-data` | verified |
| Unit/integration tests (`uv run pytest`, 108 tests) | verified, all pass |
| End-to-end mock run: search → interrupt/resume → finalize → report | verified (synthetic fixtures only) |
| Container runner checkpoint (`check-runner`) | verified with Docker 29.9.3: the valid fixture matches the host, and 3 invalid fixtures are rejected. Isolation was checked directly: no network, uid 65534, read-only mounts, no API key in the environment, test split not visible. |
| Live Gemini request (`check-live`) | verified with `gemini-3.8-flash` |
| Pilots | `pilot01` (8,192 output tokens, default thinking): 2/10 valid; most replies were cut off or empty because thinking used up the output budget. `pilot02` (thinking budget 2,048): 10/10 valid, but thinking was mostly unused. `pilot03` (budget 2,048 plus a written-analysis prompt): 10/10 valid, with analyses of about 240 words. Pilots are design checks, not results. One-off diagnostics are in `runs/_diagnostics/`. |
| Main experiment | frozen design below; see `runs/main01/` once it has run |

### Frozen live design (chosen after the pilots, before the main run)

- Model `gemini-3.8-flash`, temperature 1.0, `max_output_tokens` 16,384, request timeout 300 s, `thinking_budget` 2,048.
- **Thinking budgets are a ceiling, not a floor.** At 2,048, the model mostly chose not to think. A 4,096 budget was ignored once (15.7k thinking tokens used). `thinking_level` MEDIUM and the default setting used about 15.7k.
- **Proposals must include a written analysis.** The prompt asks for an analysis of up to about 250 words, based on the feedback, before the code. This is a project design choice, not something from the AlphaEvolve paper. That paper used Gemini 2.0 Flash and Pro, and its prompts asked for diffs. The analysis is the model's stated rationale and is not verified as the actual cause of any improvement.
- `gemini-2.5-flash` appears in `models.list` but returns 404 ("no longer available to new users").

---

## Task

Online, append-only, two-dimensional bin packing:

- Every machine has 1000 CPU and 1000 memory, in integer units.
- Jobs arrive in a fixed order and never leave.
- A new identical machine is opened only when no existing machine can fit the job.

The model writes exactly one pure function:

```python
def score(job_cpu, job_mem, free_cpu, free_mem):
    return ...
```

The trusted evaluator calls `score` only for feasible existing machines. The highest finite score wins, and ties go to the earliest-created machine. The evaluator alone owns feasibility, machine opening, capacity updates, and metrics.

**Reference policies:**

- **first-fit:** picks the earliest feasible machine and never calls `score`.
- **best-fit:** picks the feasible machine with the smallest sum of residual CPU and memory fractions after placement, with the same tie-break. It is computed as `-(integer residual sum) / 1000`, so equal sums tie exactly. Best-fit is also the common initial incumbent for both conditions (`baselines.INITIAL_SOURCE`). The code checks that this source places jobs exactly like the trusted implementation.

### Workloads (`configs/main.yaml`, generated into `data/main/`)

| Family | CPU | Memory |
| --- | --- | --- |
| balanced | `b ~ U{100..500}`, plus jitter `U{-50..50}` | same `b`, separate jitter; both clipped to [50, 900] |
| cpu_heavy | `U{450..850}` | `U{50..250}` |
| mem_heavy | `U{50..250}` | `U{450..850}` |
| mixed | per job, one of the three families above, chosen uniformly | |

- **Dev split:** 3 instances per family, 100 jobs each (12 instances), seed 1001.
- **Test split:** 10 instances per family, 100 jobs each (40 instances), seed 2002.
- Each split uses its own `numpy.random.Generator`.
- `data/main/manifest.json` records the generator config, per-file SHA-256 hashes, and split hashes. `generate-data` refuses to overwrite different data, and every search and finalization re-checks the hashes.
- This is a transparent, proposed distribution, not a real cluster trace.
- **Smoke data** (`data/smoke/`, 1 instance per family, 20 jobs each) is for plumbing checks only, never evidence.

### Objective and feedback

- **Objective:** minimize mean machines used over the dev instances. Families are weighted equally because each has the same number of instances.
- **Feedback to the model:** valid/invalid status, the aggregate dev mean, and per-family dev means.
- **Reporting:** held-out results give the mean, per-family means, and `% fewer machines vs best-fit = 100 × (best_fit_mean − policy_mean) / best_fit_mean`. Positive means the policy used fewer machines than best-fit.
- **Diagnostics only:** residual CPU/memory and a per-instance lower bound. For fixed demand, residuals are determined by the machine count, so they are not independent evidence.

### Conditions and fairness

Both conditions use the same model, settings, prompt template, constraints, dev instances, initial program, and proposal count. Each request is stateless and returns one candidate.

- **`independent`:** every proposal sees the initial program and its fixed initial feedback. The winner is the best valid candidate, with the initial program included; the earliest wins ties.
- **`evolution`:** each proposal sees only the current best-so-far program and its feedback. The incumbent is replaced only on a strict improvement in the dev score (integer total machines), never on a tie or a regression. There is no conversation history.

Fairness rules:

- First prompts are byte-identical across conditions. A test checks this.
- Invalid outputs and failed requests use up their proposal slot. There are no free repairs and no replacement candidates.
- In each replicate, the conditions alternate slot by slot. Which condition goes first is randomized from `search.ordering_seed` and recorded in `run.json`.
- Equal proposal count is **not** equal token cost. Usage, latency, and failures are logged so the costs can be compared, but no cost matching is claimed.

---

## Setup

```bash
uv sync
cp .env.example .env        # then put your key in .env (git-ignored); or export GEMINI_API_KEY
```

### Before any live run

1. **Set the model.** Set `provider.model` in `configs/pilot.yaml` and `configs/main.yaml` to an exact, current Gemini model ID from the [Gemini API docs](https://ai.google.dev/gemini-api/docs/models). It is `null` on purpose: no ID is guessed. The adapter uses the official `google-genai` SDK (v3.3.0 locked in `uv.lock`) through `client.models.generate_content`.
2. **Get a container engine.** Use either:
   - Docker Desktop, or
   - Podman: run `podman machine init && podman machine start`, then set `runner.engine: podman`.

   Then run:

   ```bash
   uv run alphaevolve-mini build-image --config configs/main.yaml
   uv run alphaevolve-mini check-runner --config configs/main.yaml   # valid + 3 invalid fixtures
   ```

3. **Set a spend limit and prices.** Set a spend limit on the Google side; this tool cannot enforce a dollar cap. Optionally fill in `pricing` with dated prices to get a conservative estimate.

---

## Commands

```bash
uv run alphaevolve-mini preflight --config configs/main.yaml          # slots, limits, key presence, data, runner
uv run alphaevolve-mini generate-data --config configs/main.yaml
uv run alphaevolve-mini smoke --provider mock                          # offline end-to-end; synthetic
uv run alphaevolve-mini check-live --config configs/pilot.yaml --provider gemini --live   # ONE request
uv run alphaevolve-mini search --config configs/pilot.yaml --provider gemini --live --run-id pilot01
uv run alphaevolve-mini search --config configs/main.yaml  --provider gemini --live --run-id main01
uv run alphaevolve-mini search --config configs/main.yaml  --provider gemini --live --run-id main01 --resume
uv run alphaevolve-mini finalize --run-id main01            # add --allow-partial to freeze an incomplete run
uv run alphaevolve-mini report --run-id main01
uv run pytest
```

- **`search` without `--live`** uses the mock provider. The Gemini provider refuses to start without `--live`, an exact model ID, the API key environment variable, a working container runner, and a prior successful `check-live` for that model.
- **Pilot vs main.** The pilot (`configs/pilot.yaml`: 5 proposals per condition, 1 replicate, at most 20 attempts) shares the main workloads. It has a different config hash, so it can never be resumed into a main run. Freeze any design change before the main run.
- **Finalization.**
  - It refuses incomplete runs unless you pass `--allow-partial`. In that case the output is labeled partial.
  - It writes `final/winners.json` and marks the run `test_exposed` **before** reading any test file. After that, search can no longer continue in that run.
  - Running finalize again reuses the existing results. It never reselects winners or silently re-tests.
- **`report`** reads only saved files. It makes no API calls and runs no evaluation.

---

## Budgets, errors, and resumption

- **Hard limits:**
  - proposal slots (`2 × proposals_per_condition × replicates`)
  - total request attempts (`search.max_total_attempts`; every attempt, including retries, is charged before it is sent)
  - `provider.max_output_tokens`, which also bounds thinking tokens on thinking models
  - `provider.request_timeout_s`
- **Retries:** only transient errors are retried (HTTP 408/429/5xx, timeouts, connection errors), at most `max_retries ≤ 2` times, with exponential backoff. The SDK's own retry is disabled (`HttpRetryOptions(attempts=1)`) so the counts stay exact.
- **Fatal errors:** HTTP 400/401/403/404 and unknown exceptions stop the run immediately. The aborted attempt is saved to `aborted/` and counts toward the attempt limit, but it uses no proposal slot. After you fix the cause, `--resume` retries that slot.
- **Failed requests:** a slot whose retries are all used up is recorded as `request_failed` and counts as a proposal.
- **Cost estimate:**
  - prompt tokens × input price + (total − prompt tokens) × output price. Thinking tokens are counted as output, and cached tokens are charged at the full input price, so the estimate is conservative.
  - If any attempt is missing usage data, the cost is reported as `unknown`, never zero.
  - This is an estimate, not a cap.
- **Logging:**
  - Each slot is written atomically to `slots/` as soon as it finishes, including the incumbent after that slot.
  - Resume requires the same config hash, workload hash, provider, model, and live flag.
  - Completed slots are never resampled.
  - An interrupted or limit-stopped run is marked `partial`.

---

## Candidate safety

1. **AST allowlist** (`candidates.py`; it narrows the language and is *not* a sandbox).
   - **Allowed:** one `def score(job_cpu, job_mem, free_cpu, free_mem)`, an optional docstring, arithmetic, comparisons, `and`/`or`/`not`, `if` statements, conditional expressions, simple name assignments and augmented assignments, `return`, and direct calls to `abs`, `min`, and `max`.
   - **Limits:** source length, AST node count, literal magnitude (≤ 1e6), exponent count, and exponents that must be literals of magnitude ≤ 4.
   - **Rejected:** everything else, including imports, attributes, indexing, loops, comprehensions, lambdas, nested functions, `global`/`nonlocal`, decorators, string literals, and other calls.
2. **Container** (`runner.ContainerRunner`). There is one disposable container per evaluation batch, with:
   - `--network none`, `--read-only`, user `65534:65534`, `--cap-drop ALL`, `no-new-privileges`
   - limits on processes, memory (no swap), and CPU, plus a small `noexec` tmpfs
   - no environment variables or credentials passed in, and no Docker socket
   - read-only mounts of only the evaluator package, the candidate, and **one** split's instance directory. Search mounts dev only; finalization mounts test only.
   - On timeout, the container is force-removed.
3. **Host re-verification.** The container returns only assignments. The host replays them with the trusted simulator and checks every invariant: each job placed once, no unnecessary new machines, no negative residuals, and resources conserved. A forged result is rejected.
4. **Errors and failures.** Error messages are short and scrubbed of host paths. Exceptions, booleans, non-numeric results, non-finite results, and malformed output all make a candidate invalid.

The mock provider's hand-written fixtures can run in a host subprocess (`LocalTrustedRunner`, labeled `local-trusted-fixtures-only`) so offline development works without Docker. The code never lets a live run use that runner. Pass `--container` to run mock fixtures in the container too.

The restricted language is a limitation of the search space. It should be reported as such.

---

## Run directory and log schema (`runs/<run_id>/`)

| File | Contents |
| --- | --- |
| `run.json` | identity (config snapshot and hash, workload and split hashes, git commit, Python and dependency versions), provider, model, live flag, runner kind, condition order, initial program and its dev evaluation, status (`in_progress`/`partial`/`complete`) with stop reason, `test_exposed`, finalization status |
| `slots/r{rep}_{cond}_{slot}.json` | prompt; full response (all attempts with status, HTTP code, sanitized error, latency, usage, finish reason, model version, response ID); raw text; extracted source; validation or runtime error; dev evaluation (per instance and per family); `improved`; incumbent before and after |
| `state.json` | incumbent per replicate and condition (derived; slot records are the source of truth) |
| `aborted/` | attempts from slots stopped by a fatal provider error |
| `final/winners.json` | frozen winners, selection rule, freeze timestamp, partial flag |
| `final/heldout.json` | test results per instance for first-fit, best-fit, and every winner |
| `report/` | `search_progress.png`, `heldout_comparison.png`, `summary.csv`, `heldout_per_instance.csv`, `candidate_changes/*.md`, `run_manifest.json` |

API keys and request headers are never saved. Mock runs are labeled in `run.json`, in every figure title, in the CSVs (`synthetic_mock`), in the candidate-change files, and in `run_manifest.json`. Partial runs are labeled the same way.

Each candidate-change file is a parent→child diff with before/after dev feedback for one strict improvement, selected without looking at test results. If no improvement happened, the report saves one representative non-improving change and says so.

`.gitignore` excludes smoke runs, evaluation scratch space, live-check logs, `.env`, and `data/smoke/`. Real runs under `runs/` and `data/main/` can be committed.

---

## Reporting guidance and known limitations

- With 3 search replicates, show every replicate along with descriptive means and ranges. Don't compute confidence intervals or significance tests over search replicates.
- Any workload bootstrap would describe variation across workloads *for the selected candidates*, not the uncertainty of the whole search procedure.
- Single-incumbent search is a simplification. Results don't speak to population-based search, and they don't separate the effect of feedback from the effect of program reuse.
- Provider-side sampling is not guaranteed to be deterministic, even with seeds, so live runs cannot be reproduced exactly from the config alone. Use the saved logs.
- The mock provider is deterministic and seeded by prompt and slot. Its "results" exist only to exercise the pipeline.
- macOS note: if `uv run alphaevolve-mini` reports `ModuleNotFoundError`, the editable-install `.pth` file got the macOS `hidden` flag, and Python 3.12+ skips hidden `.pth` files. Fix it with `chflags nohidden .venv/lib/python3.*/site-packages/*.pth`. Tests don't depend on this, because pytest sets `pythonpath = ["src"]`.
