"""Shared prompt template. Both conditions use it; only the program/feedback differ.

Independent: always the initial program and its fixed initial feedback.
Evolution: the current best-so-far program and its feedback. No conversation history.
"""

from __future__ import annotations

from .workloads import FAMILIES

PROGRAM_BEGIN = "### CURRENT PROGRAM BEGIN"
PROGRAM_END = "### CURRENT PROGRAM END"

TEMPLATE = """You are improving a heuristic for online two-dimensional bin packing (a simplified machine-placement problem).

Task rules (enforced by a trusted evaluator you cannot change):
- Every machine has CPU capacity 1000 and memory capacity 1000 (integer units).
- Jobs arrive one at a time in a fixed order and never leave. Each job has integer job_cpu and job_mem demands.
- For each job, the evaluator calls your function once for every EXISTING machine that can fit the job, passing that machine's current free_cpu and free_mem (before placement).
- The job goes to the machine with the highest score; ties go to the earliest-created machine.
- If no existing machine fits, a new machine is opened without calling your function.

Objective: minimize the mean number of machines used across the development workloads (lower is better).
Workload families: balanced (similar CPU and memory demand), cpu_heavy, mem_heavy, and mixed (each job drawn from one of the other three families).

Program constraints (programs that violate them are rejected and still count as an attempt):
- Define exactly one function: def score(job_cpu, job_mem, free_cpu, free_mem):
- Return a finite int or float (not a bool).
- Allowed: arithmetic (+ - * / // % **), comparisons, and/or/not, if statements, conditional expressions, simple local variable assignments, return statements, and calls to abs, min and max.
- Exponents must be numeric literals with magnitude at most {max_exp:g}. Numeric literals must have magnitude at most {max_lit:g}.
- Not allowed: imports, attribute access, indexing, loops, comprehensions, nested functions, lambdas, global/nonlocal, decorators, strings other than a docstring, or calls to anything except abs/min/max.
- Keep the source under {max_chars} characters.

{program_begin}
```python
{program}```
{program_end}

Evaluation feedback for the current program:
{feedback}

Propose one improved version of the score function that you expect to use fewer machines.
First, write a short analysis (up to about 250 words): using the feedback above, explain why the current program wastes machines, especially on the families with the most machines, and what placement behaviour would reduce that.
Then give exactly one ```python code block containing the complete improved function.
"""


def template_hash() -> str:
    """Identity of the prompt wording; resume requires it to match."""
    import hashlib
    return hashlib.sha256(TEMPLATE.encode()).hexdigest()[:16]


def format_feedback(result: dict) -> str:
    if not result.get("valid"):
        return f"Status: invalid. Error: {result.get('error', 'unknown')}"
    fam = result["family_means"]
    lines = [
        "Status: valid.",
        f"Mean machines used (all development instances): {result['mean_machines']:.3f}",
        "Mean machines used per family:",
    ]
    lines += [f"  {f}: {fam[f]:.3f}" for f in FAMILIES if f in fam]
    return "\n".join(lines)


def build_prompt(program_source: str, feedback: str, limits) -> str:
    program = program_source if program_source.endswith("\n") else program_source + "\n"
    return TEMPLATE.format(
        program=program,
        feedback=feedback,
        max_exp=limits.max_abs_exponent,
        max_lit=limits.max_abs_literal,
        max_chars=limits.max_source_chars,
        program_begin=PROGRAM_BEGIN,
        program_end=PROGRAM_END,
    )
