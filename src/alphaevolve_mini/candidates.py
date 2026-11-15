"""Candidate extraction, AST allowlist validation and compilation.

Standard library only: this module is also mounted read-only into the evaluation
container, which re-validates before executing. AST validation narrows the
language; it is NOT a security sandbox. Untrusted code runs only in a container.
"""

from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass

PARAMS = ("job_cpu", "job_mem", "free_cpu", "free_mem")
ALLOWED_CALLS = frozenset({"abs", "min", "max"})

# Explicit allowlist of AST node types. Anything not listed is rejected.
ALLOWED_NODES: frozenset[type] = frozenset(
    {
        ast.Module, ast.FunctionDef, ast.arguments, ast.arg,
        ast.Return, ast.Assign, ast.AugAssign, ast.If, ast.Expr,
        ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp, ast.Call,
        ast.Name, ast.Constant, ast.Load, ast.Store,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
        ast.UAdd, ast.USub, ast.Not, ast.And, ast.Or,
        ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq,
    }
)


@dataclass(frozen=True)
class Limits:
    max_source_chars: int = 2000
    max_ast_nodes: int = 400
    max_abs_literal: float = 1_000_000.0
    max_abs_exponent: float = 4.0
    max_pow_nodes: int = 4

    @classmethod
    def from_dict(cls, data: dict | None) -> "Limits":
        return cls(**(data or {}))

    def to_dict(self) -> dict:
        return {
            "max_source_chars": self.max_source_chars,
            "max_ast_nodes": self.max_ast_nodes,
            "max_abs_literal": self.max_abs_literal,
            "max_abs_exponent": self.max_abs_exponent,
            "max_pow_nodes": self.max_pow_nodes,
        }


class ValidationError(Exception):
    pass


_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\n(.*?)```", re.DOTALL)


def extract_source(response_text: str | None) -> str:
    """Pull the candidate function out of a model response.

    Prefers the last fenced python block that defines `score`; falls back to raw
    text that starts with `def score`. Raises ValidationError if none found.
    """
    if not response_text or not response_text.strip():
        raise ValidationError("empty response")
    blocks = [body for lang, body in _FENCE.findall(response_text) if lang.lower() in ("", "python", "py")]
    with_score = [b for b in blocks if re.search(r"^\s*def\s+score\s*\(", b, re.MULTILINE)]
    if with_score:
        return with_score[-1].strip() + "\n"
    stripped = response_text.strip()
    if stripped.startswith("def score"):
        return stripped + "\n"
    raise ValidationError("no fenced python code block defining score(...) found")


def validate(source: str, limits: Limits = Limits()) -> str:
    """Validate a candidate source. Returns normalized source (ast.unparse) or raises."""
    if len(source) > limits.max_source_chars:
        raise ValidationError(f"source is {len(source)} chars; limit is {limits.max_source_chars}")
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        raise ValidationError(f"syntax error on line {exc.lineno}: {exc.msg}") from None

    nodes = list(ast.walk(tree))
    if len(nodes) > limits.max_ast_nodes:
        raise ValidationError(f"program has {len(nodes)} AST nodes; limit is {limits.max_ast_nodes}")
    for node in nodes:
        if type(node) not in ALLOWED_NODES:
            raise ValidationError(f"disallowed syntax: {type(node).__name__}{_where(node)}")

    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValidationError("source must contain exactly one top-level function and nothing else")
    fn = tree.body[0]
    if sum(isinstance(n, ast.FunctionDef) for n in nodes) != 1:
        raise ValidationError("nested functions are not allowed")
    if fn.name != "score":
        raise ValidationError("function must be named score")
    if fn.decorator_list:
        raise ValidationError("decorators are not allowed")
    if fn.returns is not None or getattr(fn, "type_params", None):
        raise ValidationError("annotations and type parameters are not allowed")
    a = fn.args
    if (
        a.posonlyargs or a.vararg or a.kwonlyargs or a.kwarg or a.defaults or a.kw_defaults
        or tuple(arg.arg for arg in a.args) != PARAMS
        or any(arg.annotation is not None for arg in a.args)
    ):
        raise ValidationError(f"signature must be exactly score({', '.join(PARAMS)})")

    body = fn.body
    # Allow a leading docstring only.
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        body = body[1:]
    if not body:
        raise ValidationError("function body is empty")

    doc = _docstring_node(fn)
    for node in ast.walk(fn):
        if isinstance(node, ast.Expr) and (doc is None or node.value is not doc):
            raise ValidationError(f"bare expression statements are not allowed{_where(node)}")
        if isinstance(node, ast.Name):
            if node.id.startswith("_"):
                raise ValidationError(f"names may not start with underscore{_where(node)}")
            if isinstance(node.ctx, ast.Store) and node.id in ALLOWED_CALLS:
                raise ValidationError(f"may not assign to {node.id}{_where(node)}")
        if isinstance(node, (ast.Assign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if not all(isinstance(t, ast.Name) for t in targets):
                raise ValidationError(f"only simple name assignment is allowed{_where(node)}")
        if isinstance(node, ast.Constant) and node is not doc:
            v = node.value
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValidationError(f"only numeric literals are allowed{_where(node)}")
            if not abs(v) <= limits.max_abs_literal:
                raise ValidationError(f"literal {v!r} exceeds magnitude limit {limits.max_abs_literal:g}{_where(node)}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in ALLOWED_CALLS:
                raise ValidationError(f"only abs, min and max may be called{_where(node)}")
            if node.keywords:
                raise ValidationError(f"keyword arguments are not allowed{_where(node)}")
            want = (1, 1) if node.func.id == "abs" else (2, 8)
            if not want[0] <= len(node.args) <= want[1]:
                raise ValidationError(f"{node.func.id} called with {len(node.args)} arguments{_where(node)}")
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in ALLOWED_CALLS:
            if not any(isinstance(p, ast.Call) and p.func is node for p in ast.walk(fn)):
                raise ValidationError(f"{node.id} may only be called directly{_where(node)}")

    pows = [n for n in ast.walk(fn) if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Pow)]
    if len(pows) > limits.max_pow_nodes:
        raise ValidationError(f"at most {limits.max_pow_nodes} exponentiations allowed")
    for p in pows:
        exp = p.right
        if isinstance(exp, ast.UnaryOp) and isinstance(exp.op, (ast.USub, ast.UAdd)):
            exp = exp.operand
        if not (isinstance(exp, ast.Constant) and isinstance(exp.value, (int, float)) and not isinstance(exp.value, bool)):
            raise ValidationError(f"exponent must be a numeric literal{_where(p)}")
        if abs(exp.value) > limits.max_abs_exponent:
            raise ValidationError(f"exponent magnitude exceeds {limits.max_abs_exponent:g}{_where(p)}")

    _check_names_defined(fn)
    if not any(isinstance(n, ast.Return) and n.value is not None for n in ast.walk(fn)):
        raise ValidationError("function never returns a value")
    return ast.unparse(tree) + "\n"


def _docstring_node(fn: ast.FunctionDef):
    first = fn.body[0] if fn.body else None
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
        return first.value
    return None


def _check_names_defined(fn: ast.FunctionDef) -> None:
    """Every loaded name must be a parameter, an allowed call, or assigned somewhere."""
    assigned = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    known = set(PARAMS) | assigned | ALLOWED_CALLS
    for n in ast.walk(fn):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id not in known:
            raise ValidationError(f"unknown name {n.id!r}{_where(n)}")


def _where(node: ast.AST) -> str:
    line = getattr(node, "lineno", None)
    return f" (line {line})" if line else ""


SAFE_BUILTINS = {"abs": abs, "min": min, "max": max}


def compile_score(source: str, limits: Limits = Limits()):
    """Validate then compile. Only call this inside the isolated runner (or on trusted fixtures)."""
    normalized = validate(source, limits)
    code = compile(ast.parse(normalized), "<candidate>", "exec")
    namespace: dict = {"__builtins__": dict(SAFE_BUILTINS)}
    exec(code, namespace)  # noqa: S102 - validated, restricted, and run inside the sandbox
    return namespace["score"]


def candidate_id(normalized_source: str) -> str:
    return hashlib.sha256(normalized_source.encode()).hexdigest()[:12]
