"""QUANT LAB Strategy Spec v1 (`*.qls.yaml`): parsing, schema checks, content hash
and strategy construction (03 v2.0.0 §4.2–4.3).

A spec is the only way strategies written outside the app (by Claude or the user)
enter it. Checks here are deliberately strict: a spec that does not carry every
pre-registration item of protocol 제3조 is rejected, not patched.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

from quantlab.core.hashing import canonical_json
from quantlab.strategies.base import Strategy
from quantlab.strategies.declarative import DeclarativeStrategy, validate_rules

SPEC_VERSIONS = (1,)
ORIGINS = ("claude", "user")

# python-kind strategies: allowed imports and forbidden names (mistake/leak guard, not a sandbox)
ALLOWED_IMPORTS = {"math", "statistics", "numpy", "pandas", "collections", "itertools", "functools",
                   "dataclasses", "typing", "__future__", "quantlab.strategies.base"}
FORBIDDEN_NAMES = {"open", "eval", "exec", "compile", "__import__", "globals", "locals", "vars", "getattr",
                   "setattr", "delattr", "input", "breakpoint", "memoryview"}


class SpecError(ValueError):
    def __init__(self, problems: list[str]):
        super().__init__("; ".join(problems))
        self.problems = problems


@dataclass
class LoadedSpec:
    path: Path
    spec: dict
    code: str | None
    strategy_hash: str
    factory: Callable[..., Strategy] = field(repr=False)

    @property
    def strategy_id(self) -> str:
        return self.spec["strategy"]["id"]

    @property
    def version(self) -> str:
        return str(self.spec["strategy"]["version"])

    @property
    def origin(self) -> str:
        return self.spec["strategy"]["author"]

    @property
    def default_params(self) -> dict:
        return {k: v["default"] for k, v in (self.spec.get("parameters") or {}).items()}

    @property
    def search_space(self) -> dict:
        return {k: list(v["search"]) for k, v in (self.spec.get("parameters") or {}).items() if v.get("search")}


def _req(d: Any, path: str, problems: list[str], kind=None):
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur or cur[part] in (None, "", []):
            problems.append(f"missing required field '{path}'")
            return None
        cur = cur[part]
    if kind is not None and not isinstance(cur, kind):
        problems.append(f"'{path}' must be {kind.__name__ if isinstance(kind, type) else kind}")
        return None
    return cur


def check_schema(spec: Any) -> list[str]:
    p: list[str] = []
    if not isinstance(spec, dict):
        return ["spec must be a YAML mapping"]
    if spec.get("spec_version") not in SPEC_VERSIONS:
        p.append(f"spec_version must be one of {SPEC_VERSIONS}")
    for f in ("strategy.id", "strategy.version", "strategy.name", "strategy.type", "strategy.author",
              "hypothesis.class", "hypothesis.economic_rationale", "hypothesis.expected_risks",
              "hypothesis.falsification", "universe.market", "universe.symbols", "universe.frequency",
              "rules.kind", "execution.signal_time", "execution.fill", "execution.rebalance",
              "benchmark", "costs.one_way_bps", "validation.policy"):
        _req(spec, f, p)
    if p:
        return p
    st, hy, un, ex = spec["strategy"], spec["hypothesis"], spec["universe"], spec["execution"]
    if st["author"] not in ORIGINS:
        p.append(f"strategy.author must be one of {ORIGINS}")
    if "derived_after_holdout_view" not in st or not isinstance(st["derived_after_holdout_view"], bool):
        p.append("strategy.derived_after_holdout_view must be true or false (lineage declaration)")
    if st.get("parent") is not None and "@" not in str(st["parent"]):
        p.append("strategy.parent must be 'ID@version' or null")
    if hy["class"] not in ("theory", "data_mining"):
        p.append("hypothesis.class must be theory or data_mining (protocol 제4조)")
    if not isinstance(un["symbols"], list) or not all(isinstance(s, str) for s in un["symbols"]):
        p.append("universe.symbols must be a list of tickers")
    if un["frequency"] != "daily":
        p.append("universe.frequency: only 'daily' is supported in this app version")
    if ex["signal_time"] != "close" or ex["fill"] != "next_open":
        p.append("execution must be signal_time: close, fill: next_open (protocol 제8조)")
    params = spec.get("parameters") or {}
    if not isinstance(params, dict):
        p.append("parameters must be a mapping")
    else:
        for name, v in params.items():
            if not isinstance(v, dict) or "default" not in v:
                p.append(f"parameters.{name} needs a default")
            elif "search" in v and not isinstance(v["search"], list):
                p.append(f"parameters.{name}.search must be a list")
    rules = spec["rules"]
    if rules["kind"] == "declarative":
        p += validate_rules(rules, un["symbols"], ex["rebalance"])
        if "combo" in rules and isinstance(params, dict):
            from quantlab.strategies.combo import validate_params

            combo = rules["combo"] if isinstance(rules["combo"], dict) else {}
            p += validate_params(params, combo.get("catalog", 1))
    elif rules["kind"] == "python":
        if ":" not in str(rules.get("entry", "")):
            p.append("rules.entry must be 'file.py:ClassName' for python rules")
    else:
        p.append("rules.kind must be declarative or python")
    return p


def check_code(source: str) -> list[str]:
    """Static guard for python-kind strategies."""
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [f"syntax error: {e}"]
    p = []

    def immutable(v) -> bool:
        if isinstance(v, ast.Constant):
            return True
        if isinstance(v, ast.UnaryOp):
            return immutable(v.operand)
        if isinstance(v, ast.Tuple):
            return all(immutable(e) for e in v.elts)
        return False

    # Shared mutable state outlives one backtest run and can carry later prices into an
    # earlier-dated decision of the next run, which per-run leak tests cannot see.
    scopes = [tree.body] + [n.body for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    for body in scopes:
        for stmt in body:
            if isinstance(stmt, (ast.Assign, ast.AnnAssign)) and stmt.value is not None and not immutable(stmt.value):
                p.append(f"line {stmt.lineno}: module/class-level values must be constants (no shared mutable state)")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            p.append(f"line {node.lineno}: global/nonlocal not allowed")
        if isinstance(node, (ast.FunctionDef, ast.Lambda)):
            for d in node.args.defaults + [d for d in node.args.kw_defaults if d is not None]:
                if not immutable(d) and not (isinstance(d, ast.Name) and d.id == "None"):
                    p.append(f"line {node.lineno}: mutable default argument not allowed")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name not in ALLOWED_IMPORTS and a.name.split(".")[0] not in {"numpy", "pandas"}:
                    p.append(f"line {node.lineno}: import '{a.name}' not allowed")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod not in ALLOWED_IMPORTS and mod.split(".")[0] not in {"numpy", "pandas"}:
                p.append(f"line {node.lineno}: import from '{mod}' not allowed")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            p.append(f"line {node.lineno}: '{node.id}' not allowed")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("_") and node.attr != "__init__":
            p.append(f"line {node.lineno}: private attribute '{node.attr}' not allowed")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("read_"):
            p.append(f"line {node.lineno}: file reader '{node.attr}' not allowed")
    return p


def content_hash(spec: dict, code: str | None) -> str:
    h = hashlib.sha256(canonical_json(spec).encode())
    if code is not None:
        h.update(b"\0code\0" + code.replace("\r\n", "\n").encode())
    return h.hexdigest()


def _load_class(source: str, class_name: str, module_name: str) -> type:
    mod = types.ModuleType(module_name)
    mod.__dict__["__builtins__"] = __builtins__
    exec(compile(source, module_name, "exec"), mod.__dict__)  # noqa: S102 - checked by check_code
    cls = mod.__dict__.get(class_name)
    if not isinstance(cls, type) or not issubclass(cls, Strategy):
        raise SpecError([f"{class_name} is not a quantlab Strategy subclass"])
    return cls


def load_spec(path: str | Path) -> LoadedSpec:
    path = Path(path)
    try:
        spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise SpecError([f"YAML parse error: {e}"]) from e
    problems = check_schema(spec)
    if problems:
        raise SpecError(problems)
    st, rules = spec["strategy"], spec["rules"]
    sid, ver = st["id"], str(st["version"])
    code = None
    if rules["kind"] == "python":
        fname, cls_name = rules["entry"].split(":", 1)
        code_path = path.parent / fname
        if not code_path.is_file():
            raise SpecError([f"code file not found next to spec: {fname}"])
        code = code_path.read_text(encoding="utf-8")
        problems = check_code(code)
        if problems:
            raise SpecError(problems)
        cls = _load_class(code, cls_name, f"qls_{sid}_{ver}".replace("-", "_").replace(".", "_"))

        def factory(**params):
            s = cls(**params)
            s.strategy_id, s.version = sid, ver
            return s
    else:
        symbols, rebalance = spec["universe"]["symbols"], spec["execution"]["rebalance"]

        def factory(**params):
            if "combo" in rules:
                from quantlab.strategies.combo import build_rules
                from quantlab.strategies.entry_exit import EntryExitStrategy

                return EntryExitStrategy(spec_id=sid, spec_version_str=ver, symbols=symbols,
                                         rules=build_rules(rules["combo"], params), fast=True, **params)
            if "entry_exit" in rules:
                from quantlab.strategies.declarative import _apply_params
                from quantlab.strategies.entry_exit import EntryExitStrategy, resolve_params

                return EntryExitStrategy(spec_id=sid, spec_version_str=ver, symbols=symbols,
                                         rules=_apply_params(resolve_params(rules, params), params), **params)
            return DeclarativeStrategy(spec_id=sid, spec_version_str=ver, symbols=symbols,
                                       rules=rules, rebalance=rebalance, **params)
    return LoadedSpec(path, spec, code, content_hash(spec, code), factory)
