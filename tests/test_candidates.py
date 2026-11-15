import pytest

from alphaevolve_mini import candidates as C
from alphaevolve_mini.baselines import INITIAL_SOURCE

HDR = "def score(job_cpu, job_mem, free_cpu, free_mem):\n"


def body(*lines):
    return HDR + "".join(f"    {ln}\n" for ln in lines)


VALID = [
    INITIAL_SOURCE,
    body("rc = free_cpu - job_cpu", "rm = free_mem - job_mem", "if rc < rm:", "    return -rc", "return -max(rc, rm) - abs(rc - rm) ** 2"),
    body("x = 1.5", "x += job_cpu / 1000", "return x if job_cpu > job_mem and not free_cpu == 0 else -x"),
    body("return min(free_cpu, free_mem, 3) % 7 // 2 - (-job_cpu) ** -2"),
]


@pytest.mark.parametrize("src", VALID)
def test_valid_programs(src):
    norm = C.validate(src)
    assert "def score" in norm
    fn = C.compile_score(src)
    assert isinstance(fn(100, 100, 500, 500), (int, float))


INVALID = {
    "import": "import os\n" + body("return 0"),
    "import inside": body("import os", "return 0"),
    "attribute": body("return job_cpu.real"),
    "indexing": body("x = (1, 2)", "return 0"),
    "subscript": body("return free_cpu[0]"),
    "loop": body("for i in range(3):", "    pass", "return 0"),
    "while": body("while True:", "    return 0"),
    "comprehension": body("return max([x for x in (1, 2)])"),
    "lambda": body("f = lambda: 1", "return 0"),
    "nested def": body("def g():", "    return 1", "return 0"),
    "global": body("global y", "return 0"),
    "decorator": "@staticmethod\n" + body("return 0"),
    "arbitrary call": body("return open('x')"),
    "dunder import": body("return __import__('os')"),
    "string literal": body("x = 'hi'", "return 0"),
    "huge literal": body("return 10000000"),
    "big exponent": body("return job_cpu ** 5"),
    "variable exponent": body("return job_cpu ** job_mem"),
    "wrong name": "def f(job_cpu, job_mem, free_cpu, free_mem):\n    return 0\n",
    "wrong args": "def score(a, b, c, d):\n    return 0\n",
    "default arg": "def score(job_cpu, job_mem, free_cpu, free_mem=1):\n    return 0\n",
    "two functions": body("return 0") + "def other():\n    return 1\n",
    "top-level code": "x = 1\n" + body("return 0"),
    "unknown name": body("return y"),
    "no return": body("x = 1"),
    "underscore name": body("_x = 1", "return _x"),
    "keyword call": body("return max(1, 2, key=abs)"),
    "bare min": body("f = min", "return 0"),
    "assign builtin": body("abs = 1", "return 0"),
    "syntax": body("return (("),
    "try": body("try:", "    return 1", "except Exception:", "    return 0"),
    "with": body("with x:", "    return 1"),
    "fstring": body("return f'{job_cpu}'"),
    "walrus": body("return (x := 1)"),
    "too long": body("return 0" + " + 0" * 600),
    "bare expression": body("job_cpu", "return 0"),
}


@pytest.mark.parametrize("name", sorted(INVALID))
def test_invalid_programs(name):
    with pytest.raises(C.ValidationError):
        C.validate(INVALID[name])


def test_compile_refuses_invalid():
    with pytest.raises(C.ValidationError):
        C.compile_score(INVALID["import"])


def test_extract_prefers_fenced_block_with_score():
    text = "Here.\n```python\nprint('x')\n```\nBetter:\n```python\n" + body("return 1") + "```\n"
    assert C.extract_source(text).startswith("def score")


def test_extract_errors():
    with pytest.raises(C.ValidationError):
        C.extract_source("")
    with pytest.raises(C.ValidationError):
        C.extract_source("no code here")


def test_candidate_id_is_stable_and_normalized():
    a = C.validate(body("return  1"))
    b = C.validate(body("return 1  # comment"))
    assert C.candidate_id(a) == C.candidate_id(b)
