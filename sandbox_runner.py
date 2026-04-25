"""Child-process runner for submitted theory modules."""

from __future__ import annotations

import ast
import builtins
import contextlib
import importlib
import importlib.util
import io
import json
import math
import statistics
import sys
import time
import traceback
from typing import Any


BANNED_CALLS = {"breakpoint", "compile", "eval", "exec", "input", "open", "__import__"}


def main() -> int:
    payload = json.loads(sys.stdin.read())
    result = run_payload(payload)
    sys.stdout.write(json.dumps(result))
    return 0


def run_payload(payload: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    stdout = io.StringIO()
    try:
        raw = payload["theory_module"]
        # Transparently render DSL JSON into a Theory class before execution
        try:
            from theory_dsl import render_theory_module
            code = render_theory_module(raw)
        except Exception:
            code = raw
        allowed_imports = tuple(payload.get("allowed_imports", ()))
        _preload_modules(allowed_imports)
        _validate_ast(code, allowed_imports)

        namespace = {
            "__builtins__": _safe_builtins(allowed_imports),
            "__name__": "submitted_theory",
            "__package__": None,
            "math": math,
            "statistics": statistics,
        }
        compiled = compile(code, "<submitted_theory>", "exec")
        with contextlib.redirect_stdout(stdout):
            exec(compiled, namespace, namespace)

            theory_cls = namespace.get("Theory")
            if theory_cls is None:
                raise ValueError("Submitted module must define class Theory")
            theory = theory_cls()

            history = payload.get("history", [])
            future = payload.get("future", [])
            window = {
                "history": history,
                "steps": len(future),
                "sensor_names": payload.get("sensor_names", []),
            }

            if not hasattr(theory, "fit"):
                raise ValueError("Theory must implement fit(history)")
            if not hasattr(theory, "predict"):
                raise ValueError("Theory must implement predict(window)")
            if not hasattr(theory, "log_prob"):
                raise ValueError("Theory must implement log_prob(observations)")

            fit_result = theory.fit(history)
            predictions = theory.predict(window)
            reported_log_prob = theory.log_prob(future)
            drift_detected = bool(theory.detect_drift(history)) if hasattr(theory, "detect_drift") else False
            description = str(theory.describe()) if hasattr(theory, "describe") else ""

        return {
            "ok": True,
            "error": "",
            "stdout": stdout.getvalue()[-4000:],
            "predictions": _jsonable(predictions),
            "reported_log_prob": float(reported_log_prob),
            "fit_result": _jsonable(fit_result),
            "description": description[:2000],
            "drift_detected": drift_detected,
            "runtime_ms": round((time.perf_counter() - started) * 1000, 3),
        }
    except Exception as exc:  # pragma: no cover - details are asserted by parent tests.
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "stdout": stdout.getvalue()[-4000:],
            "traceback": traceback.format_exc(limit=4)[-4000:],
            "predictions": [],
            "reported_log_prob": float("-inf"),
            "description": "",
            "drift_detected": False,
            "runtime_ms": round((time.perf_counter() - started) * 1000, 3),
        }


def _preload_modules(allowed_imports: tuple[str, ...]) -> None:
    for name in ("numpy", "scipy"):
        if name in allowed_imports:
            try:
                importlib.import_module(name)
            except Exception:
                pass


def _validate_ast(code: str, allowed_imports: tuple[str, ...]) -> None:
    tree = ast.parse(code)
    module_aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                _require_allowed_import(alias.name, allowed_imports)
                module_aliases[alias.asname or alias.name.split(".", 1)[0]] = alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                raise ImportError("Relative imports are not allowed")
            _require_allowed_import(node.module, allowed_imports)
            _require_allowed_fromlist(node.module, [alias.name for alias in node.names], allowed_imports)
            for alias in node.names:
                imported_name = f"{node.module}.{alias.name}"
                if imported_name in allowed_imports:
                    module_aliases[alias.asname or alias.name] = imported_name
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in BANNED_CALLS:
                raise PermissionError(f"Call to {node.func.id} is not allowed")
            if node.func.id == "getattr":
                _require_allowed_getattr(node, module_aliases, allowed_imports)
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise PermissionError("Dunder attribute access is not allowed")
        elif isinstance(node, ast.Attribute):
            _require_allowed_attribute(node, module_aliases, allowed_imports)


def _require_allowed_import(module_name: str, allowed_imports: tuple[str, ...]) -> None:
    if module_name not in allowed_imports:
        raise ImportError(f"Import '{module_name}' is not allowed")


def _require_allowed_fromlist(module_name: str, fromlist: list[str] | tuple[str, ...], allowed_imports: tuple[str, ...]) -> None:
    for item in fromlist:
        if item == "*":
            raise ImportError("Wildcard imports are not allowed")
        submodule_name = f"{module_name}.{item}"
        if _looks_like_submodule(submodule_name) and submodule_name not in allowed_imports:
            raise ImportError(f"Import '{submodule_name}' is not allowed")


def _looks_like_submodule(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, AttributeError, ModuleNotFoundError, ValueError):
        return False


def _require_allowed_attribute(
    node: ast.Attribute,
    module_aliases: dict[str, str],
    allowed_imports: tuple[str, ...],
) -> None:
    if not isinstance(node.value, ast.Name):
        return
    module_name = module_aliases.get(node.value.id)
    if not module_name:
        return
    submodule_name = f"{module_name}.{node.attr}"
    if _looks_like_submodule(submodule_name) and submodule_name not in allowed_imports:
        raise ImportError(f"Import '{submodule_name}' is not allowed")


def _require_allowed_getattr(
    node: ast.Call,
    module_aliases: dict[str, str],
    allowed_imports: tuple[str, ...],
) -> None:
    if len(node.args) < 2 or not isinstance(node.args[1], ast.Constant) or not isinstance(node.args[1].value, str):
        return
    attr = node.args[1].value
    if attr.startswith("__"):
        raise PermissionError("Dunder attribute access is not allowed")
    if not isinstance(node.args[0], ast.Name):
        return
    module_name = module_aliases.get(node.args[0].id)
    if not module_name:
        return
    submodule_name = f"{module_name}.{attr}"
    if _looks_like_submodule(submodule_name) and submodule_name not in allowed_imports:
        raise ImportError(f"Import '{submodule_name}' is not allowed")


def _safe_builtins(allowed_imports: tuple[str, ...]) -> dict[str, Any]:
    safe_names = [
        "abs",
        "all",
        "any",
        "ArithmeticError",
        "AssertionError",
        "BaseException",
        "bool",
        "dict",
        "enumerate",
        "Exception",
        "filter",
        "float",
        "getattr",
        "hasattr",
        "int",
        "isinstance",
        "issubclass",
        "len",
        "list",
        "map",
        "max",
        "min",
        "object",
        "pow",
        "print",
        "property",
        "range",
        "repr",
        "round",
        "set",
        "slice",
        "sorted",
        "staticmethod",
        "str",
        "sum",
        "super",
        "tuple",
        "type",
        "ValueError",
        "zip",
    ]
    safe = {name: getattr(builtins, name) for name in safe_names}
    safe["__build_class__"] = builtins.__build_class__
    safe["__import__"] = _limited_import(allowed_imports)
    return safe


def _limited_import(allowed_imports: tuple[str, ...]):
    def import_hook(name, globals=None, locals=None, fromlist=(), level=0):
        if level != 0:
            raise ImportError("Relative imports are not allowed")
        _require_allowed_import(name, allowed_imports)
        _require_allowed_fromlist(name, fromlist or (), allowed_imports)
        return builtins.__import__(name, globals, locals, fromlist, level)

    return import_hook


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return _jsonable(value.tolist())
        except Exception:
            pass
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


if __name__ == "__main__":
    raise SystemExit(main())
