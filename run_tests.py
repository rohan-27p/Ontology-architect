"""
Test runner: executes all pytest suites + smoke baselines, writes a single report file.

Usage:
    python run_tests.py                        # report saved to test_results/
    python run_tests.py --out results/run.txt  # custom path
    python run_tests.py --no-smoke             # skip smoke runs
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import textwrap
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "test_results"

SEP_THICK = "=" * 80
SEP_THIN = "-" * 80


def _run(
    cmd: list[str], cwd: Path, timeout: int = 120, extra_env: dict[str, str] | None = None
) -> tuple[str, str, int]:
    """Run a subprocess, return (stdout, stderr, returncode)."""
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )
    return proc.stdout, proc.stderr, proc.returncode


# Workspace root is the parent of this file's directory (contains ontology_architect as a sub-package)
WORKSPACE_ROOT = str(HERE.parent)


def _python() -> str:
    return sys.executable


# ---------------------------------------------------------------------------
# Pytest runner
# ---------------------------------------------------------------------------

def _parse_plain_pytest(stdout: str, stderr: str, rc: int) -> dict:
    """Parse pytest -v plain text output into a structured dict."""
    tests = []
    failures_raw: list[str] = []

    # Match lines like: tests/test_foo.py::test_bar PASSED [  6%]
    test_line_re = re.compile(
        r"^(?P<nodeid>\S+::(?:\S+))\s+(?P<outcome>PASSED|FAILED|SKIPPED|ERROR|XFAIL|XPASS)"
        r"(?:\s+\[.*?\])?(?:\s+(?P<dur>[\d.]+)s)?",
        re.MULTILINE,
    )

    for m in test_line_re.finditer(stdout):
        tests.append({
            "nodeid": m.group("nodeid"),
            "outcome": m.group("outcome"),
            "duration": float(m.group("dur") or 0.0),
        })

    # Extract summary line: "3 passed, 1 failed in 0.42s"
    summary_re = re.compile(
        r"(?P<passed>\d+) passed|(?P<failed>\d+) failed|"
        r"(?P<skipped>\d+) skipped|(?P<errors>\d+) error|"
        r"in (?P<dur>[\d.]+)s",
        re.IGNORECASE,
    )
    summary: dict[str, int | float] = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0, "duration": 0.0}
    for line in stdout.splitlines():
        if re.search(r"\d+ (passed|failed|error|skipped)", line, re.IGNORECASE):
            for m in summary_re.finditer(line):
                for key in ("passed", "failed", "skipped", "errors"):
                    if m.group(key):
                        summary[key] = int(m.group(key))
                if m.group("dur"):
                    summary["duration"] = float(m.group("dur"))

    summary["collected"] = len(tests)

    # Collect failure sections (text between FAILURES header and the summary)
    in_failure = False
    current: list[str] = []
    for line in stdout.splitlines():
        if re.match(r"_{5,}\s+FAILURES?\s+_{5,}", line) or re.match(r"={5,}\s+FAILURES?\s+={5,}", line):
            in_failure = True
            continue
        if in_failure:
            if re.match(r"={5,}", line) and current:
                failures_raw.append("\n".join(current))
                current = []
            else:
                current.append(line)
    if current:
        failures_raw.append("\n".join(current))

    return {
        "mode": "plain",
        "stdout": stdout,
        "stderr": stderr,
        "returncode": rc,
        "report": {
            "tests": tests,
            "summary": summary,
            "duration": summary["duration"],
            "failures_raw": failures_raw,
        },
    }


def run_pytest(cwd: Path) -> dict:
    """Run pytest with JSON output and return a parsed result dict."""
    report_path = cwd / "test_results" / "_pytest_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    cmd_json = [
        _python(), "-m", "pytest",
        "tests/",
        "-v",
        "--tb=short",
        "--no-header",
        "--json-report",
        f"--json-report-file={report_path}",
        "--json-report-indent=2",
    ]
    stdout, stderr, rc = _run(cmd_json, cwd, timeout=300)

    # rc=4 means usage error (plugin not installed); rc=3 means internal error
    plugin_missing = rc == 4 or "unrecognized arguments" in stderr or (
        "no module named" in stderr.lower() and "json_report" in stderr.lower()
    )
    if plugin_missing:
        cmd_plain = [_python(), "-m", "pytest", "tests/", "-v", "--tb=short", "--no-header"]
        stdout, stderr, rc = _run(cmd_plain, cwd, timeout=300)
        return _parse_plain_pytest(stdout, stderr, rc)

    report = None
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass

    return {
        "mode": "json",
        "stdout": stdout,
        "stderr": stderr,
        "returncode": rc,
        "report": report,
    }


# ---------------------------------------------------------------------------
# Smoke runner
# ---------------------------------------------------------------------------

SMOKE_BASELINES = ["static", "linear", "teacher"]
SMOKE_STEPS = 3


def run_smoke(baseline: str, cwd: Path) -> dict:
    """Run scripts/smoke.py for one baseline, return result dict."""
    cmd = [
        _python(), "scripts/smoke.py",
        "--config", "configs/tiny_smoke.json",
        "--baseline", baseline,
        "--steps", str(SMOKE_STEPS),
    ]
    pythonpath = os.environ.get("PYTHONPATH", "")
    sep = os.pathsep
    new_pythonpath = f"{WORKSPACE_ROOT}{sep}{pythonpath}" if pythonpath else WORKSPACE_ROOT
    try:
        stdout, stderr, rc = _run(cmd, cwd, timeout=60, extra_env={"PYTHONPATH": new_pythonpath})
    except subprocess.TimeoutExpired:
        return {"baseline": baseline, "ok": False, "stdout": "", "stderr": "TIMEOUT", "returncode": -1}
    return {
        "baseline": baseline,
        "ok": rc == 0,
        "stdout": stdout,
        "stderr": stderr,
        "returncode": rc,
    }


# ---------------------------------------------------------------------------
# Report builders
# ---------------------------------------------------------------------------

def _header(label: str) -> str:
    return f"\n{SEP_THICK}\n  {label}\n{SEP_THICK}\n"


def _subheader(label: str) -> str:
    return f"\n{SEP_THIN}\n  {label}\n{SEP_THIN}"


def build_pytest_section(result: dict) -> str:
    lines = [_header("1. PYTEST SUITE")]
    report = result.get("report") or {}
    summary = report.get("summary", {})

    collected = summary.get("collected", "?")
    passed = summary.get("passed", 0)
    failed = summary.get("failed", 0)
    skipped = summary.get("skipped", 0)
    errors = summary.get("errors", 0)
    duration = report.get("duration", 0)

    if collected != "?":
        lines.append(
            f"  Collected: {collected}  |  Passed: {passed}  |  "
            f"Failed: {failed}  |  Skipped: {skipped}  |  Errors: {errors}  |  "
            f"Duration: {duration:.2f}s\n"
        )

    tests = report.get("tests", [])
    if tests:
        lines.append(f"  {'TEST':<65} {'OUTCOME':<10} {'DURATION':>8}")
        lines.append(f"  {'-'*65} {'-'*10} {'-'*8}")
        for test in tests:
            node = test.get("nodeid", "")
            outcome = test.get("outcome", "").upper()
            dur = test.get("duration", 0.0)
            marker = {
                "PASSED": "PASS", "FAILED": "FAIL", "SKIPPED": "SKIP",
                "ERROR": "ERR ", "XFAIL": "XFAIL", "XPASS": "XPASS",
            }.get(outcome, outcome[:4])
            lines.append(f"  {node:<65} {marker:<10} {dur:>7.3f}s")

    # Failure detail — JSON mode has longrepr, plain mode has failures_raw
    if result["mode"] == "json":
        failures = [t for t in tests if t.get("outcome") in ("failed", "error")]
        if failures:
            lines.append(_subheader("FAILURES / ERRORS"))
            for test in failures:
                lines.append(f"\n  >> {test['nodeid']}")
                call = test.get("call") or test.get("setup") or {}
                longrepr = call.get("longrepr", "")
                if longrepr:
                    lines.append(textwrap.indent(str(longrepr), "     "))
    else:
        failures_raw = report.get("failures_raw", [])
        if failures_raw:
            lines.append(_subheader("FAILURES / ERRORS"))
            for block in failures_raw:
                lines.append(block)

    lines.append(_subheader("FULL PYTEST OUTPUT"))
    lines.append(result["stdout"] or "  (none)")

    if result["stderr"].strip():
        lines.append(_subheader("STDERR"))
        lines.append(result["stderr"])

    return "\n".join(lines)


def build_smoke_section(smoke_results: list[dict]) -> str:
    lines = [_header("2. SMOKE RUNS (configs/tiny_smoke.json)")]
    lines.append(f"  Baselines tested: {', '.join(SMOKE_BASELINES)}  |  Steps per run: {SMOKE_STEPS}\n")

    for r in smoke_results:
        status = "OK" if r["ok"] else f"FAILED (rc={r['returncode']})"
        lines.append(_subheader(f"Baseline: {r['baseline'].upper()}  [{status}]"))
        if r["stdout"].strip():
            lines.append(r["stdout"])
        if r["stderr"].strip():
            lines.append("  --- stderr ---")
            lines.append(r["stderr"])

    return "\n".join(lines)


def build_summary_section(pytest_result: dict, smoke_results: list[dict]) -> str:
    lines = [_header("3. SUMMARY")]

    # Pytest summary
    report = pytest_result.get("report") or {}
    s = report.get("summary", {})
    total = s.get("collected", 0)
    passed = s.get("passed", 0)
    failed = s.get("failed", 0) + s.get("errors", 0)
    if total:
        pct = f"{100 * passed // total}%"
        pytest_ok = failed == 0 and pytest_result["returncode"] in (0, 1)
        pytest_line = f"PASS ({pct})" if pytest_ok else f"FAIL  ({failed} failures, {passed}/{total} passed)"
    else:
        rc = pytest_result["returncode"]
        pytest_ok = rc == 0
        pytest_line = "PASS" if pytest_ok else f"FAIL (rc={rc})"

    lines.append(f"  {'Component':<30} {'Result'}")
    lines.append(f"  {'-'*30} {'-'*30}")
    lines.append(f"  {'pytest suite':<30} {pytest_line}")

    all_ok = pytest_ok
    for r in smoke_results:
        tag = "OK" if r["ok"] else f"FAIL (rc={r['returncode']})"
        lines.append(f"  {'smoke:' + r['baseline']:<30} {tag}")
        if not r["ok"]:
            all_ok = False

    verdict = "ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED -- see sections above"
    lines.append(f"\n  Overall: {verdict}")

    return "\n".join(lines)


def build_env_section() -> str:
    lines = [_header("0. ENVIRONMENT")]
    lines.append(f"  Date/time : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Python    : {sys.version.splitlines()[0]}")
    lines.append(f"  Platform  : {platform.platform()}")
    lines.append(f"  CWD       : {HERE}")

    # Installed package versions of interest
    pkgs = ["numpy", "scipy", "pytest", "openenv_core", "fastapi", "pydantic"]
    lines.append("  Packages  :")
    for pkg in pkgs:
        try:
            import importlib.metadata
            ver = importlib.metadata.version(pkg.replace("_", "-"))
        except Exception:
            ver = "not installed"
        lines.append(f"    {pkg:<20} {ver}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run all tests and write a single report file.")
    parser.add_argument("--out", default=None, help="Output file path (default: test_results/report_<timestamp>.txt)")
    parser.add_argument("--no-smoke", action="store_true", help="Skip smoke baseline runs")
    args = parser.parse_args()

    if args.out:
        out_path = Path(args.out)
    else:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = RESULTS_DIR / f"report_{timestamp}.txt"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[run_tests] Writing report to: {out_path}")
    print("[run_tests] Running pytest ...")
    pytest_result = run_pytest(HERE)
    print(f"[run_tests] pytest exit code: {pytest_result['returncode']}")

    smoke_results: list[dict] = []
    if not args.no_smoke:
        for bl in SMOKE_BASELINES:
            print(f"[run_tests] Smoke run: {bl} ...")
            r = run_smoke(bl, HERE)
            smoke_results.append(r)
            print(f"[run_tests]   -> {'ok' if r['ok'] else 'FAILED'}")

    sections = [
        build_env_section(),
        build_pytest_section(pytest_result),
    ]
    if smoke_results:
        sections.append(build_smoke_section(smoke_results))
    sections.append(build_summary_section(pytest_result, smoke_results))

    report = "\n".join(sections) + f"\n\n{SEP_THICK}\n  END OF REPORT\n{SEP_THICK}\n"
    out_path.write_text(report, encoding="utf-8")

    print(f"\n[run_tests] Report saved -> {out_path}")

    # Mirror summary to terminal
    print(build_summary_section(pytest_result, smoke_results))

    rc = 0 if pytest_result["returncode"] == 0 and all(r["ok"] for r in smoke_results) else 1
    sys.exit(rc)


if __name__ == "__main__":
    main()
