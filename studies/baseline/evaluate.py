"""Evaluate generated PyTorch attention modules against a failure taxonomy.

Reads ``studies/baseline/raw/{model}/*.py``, runs each through progressively
stricter checks, records the first failure encountered. Writes
``results.json`` and ``results.md``.

Each generated module is executed in a *subprocess* so that crashes,
infinite loops, and stray ``sys.exit`` calls do not take down the evaluator.
A 30-second per-sample timeout caps runaway generations.

Usage::

    python studies/baseline/evaluate.py
    python studies/baseline/evaluate.py --models claude-haiku-4-5
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
RAW_DIR = HERE / "raw"
RESULTS_JSON = HERE / "results.json"
RESULTS_MD = HERE / "results.md"

# Per-sample evaluation timeout. Generous enough for slow imports on cold
# Python starts, tight enough that a stray ``while True:`` doesn't block the
# whole study.
TIMEOUT_SECONDS = 30

# The full failure taxonomy, ordered. Categories appear in this order in
# tables and JSON output.
TAXONOMY = [
    "syntax",
    "import",
    "instantiate",
    "forward_runtime",
    "shape_contract",
    "nan_inf",
    "backward_runtime",
    "dead_grads",
    "pass",
]


# --------------------------------------------------------------------------- #
# The actual single-sample probe — runs in a subprocess so we can isolate
# crashes. Emits a single JSON line on stdout: ``{"category": "..."}``.
# --------------------------------------------------------------------------- #

PROBE_SCRIPT = r"""
import ast, json, sys, traceback
from pathlib import Path

src_path = Path(sys.argv[1])
src = src_path.read_text(encoding="utf-8")


def emit(cat):
    print(json.dumps({"category": cat}))
    sys.exit(0)


# 1. Syntax.
try:
    ast.parse(src)
except SyntaxError:
    emit("syntax")

# 2. Import / module-level execution.
ns = {"__name__": "__probe__"}
try:
    exec(compile(src, str(src_path), "exec"), ns)
except Exception:
    emit("import")

CustomAttention = ns.get("CustomAttention")
if CustomAttention is None:
    emit("import")  # symbol missing post-exec

# 3. Instantiate.
try:
    model = CustomAttention(d_model=64, n_heads=8)
except Exception:
    emit("instantiate")

# 4. Forward runtime.
import torch
torch.manual_seed(0)
q = torch.randn(2, 5, 64, requires_grad=True)
k = torch.randn(2, 5, 64, requires_grad=True)
v = torch.randn(2, 5, 64, requires_grad=True)
mask = torch.ones(2, 5, 5, dtype=torch.bool)
try:
    out = model(q, k, v, mask)
except Exception:
    emit("forward_runtime")

# 5. Shape contract.
expected = (2, 5, 64)
if not isinstance(out, torch.Tensor) or tuple(out.shape) != expected:
    emit("shape_contract")

# 6. NaN/Inf.
if torch.isnan(out).any().item() or torch.isinf(out).any().item():
    emit("nan_inf")

# 7. Backward runtime.
try:
    loss = out.sum()
    loss.backward()
except Exception:
    emit("backward_runtime")

# 8. Dead grads — at least one parameter must have a non-zero grad.
any_grad = False
for p in model.parameters():
    if p.grad is not None and p.grad.abs().sum().item() > 0:
        any_grad = True
        break
if not any_grad:
    emit("dead_grads")

emit("pass")
"""


def evaluate_sample(py_path: Path) -> str:
    """Run the probe on one sample. Returns the failure category."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", PROBE_SCRIPT, str(py_path)],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        # An infinite loop in the model's code counts as forward_runtime
        # (it never returned a tensor). Categorize separately if you want
        # to surface this; for now bucket with forward_runtime.
        return "forward_runtime"
    except Exception:
        return "forward_runtime"

    # Parse the last JSON line from stdout. Anything else means the probe
    # itself crashed — treat as forward_runtime.
    for line in reversed(result.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                payload = json.loads(line)
                cat = payload.get("category", "forward_runtime")
                if cat in TAXONOMY:
                    return cat
            except json.JSONDecodeError:
                continue
    return "forward_runtime"


# --------------------------------------------------------------------------- #
# Aggregation + output
# --------------------------------------------------------------------------- #


def evaluate_model(model: str) -> dict:
    """Evaluate every .py file under raw/{model}/."""
    model_dir = RAW_DIR / model
    if not model_dir.exists():
        return {"n_total": 0, "failure_distribution": {}, "pass_rate": 0.0}

    paths = sorted(model_dir.glob("*.py"))
    failure_counts = dict.fromkeys(TAXONOMY, 0)

    print(f"=== {model} ({len(paths)} samples) ===", flush=True)
    for i, py_path in enumerate(paths, 1):
        cat = evaluate_sample(py_path)
        failure_counts[cat] += 1
        if i % 10 == 0 or i == len(paths):
            print(f"  {i}/{len(paths)}", flush=True)

    n_total = len(paths)
    n_pass = failure_counts["pass"]
    return {
        "n_total": n_total,
        "n_pass": n_pass,
        "failure_distribution": failure_counts,
        "pass_rate": n_pass / n_total if n_total else 0.0,
    }


def render_markdown(results: dict) -> str:
    """Human-readable summary for the blog post."""
    lines = ["# Baseline Study Results", ""]
    meta = results["metadata"]
    lines.append(f"- Generated at: `{meta['completed_at']}`")
    lines.append(f"- aero-eval: `{meta['aero_eval_version']}`")
    lines.append(f"- Samples per model: {meta['n_per_model']}")
    lines.append("")
    lines.append("## Pass rate by model")
    lines.append("")
    lines.append("| Model | n | Pass | Pass rate |")
    lines.append("|-------|---|------|-----------|")
    for model, data in results["per_model"].items():
        lines.append(
            f"| `{model}` | {data['n_total']} | {data['n_pass']} | {data['pass_rate']:.1%} |"
        )
    lines.append("")
    lines.append("## Failure distribution")
    lines.append("")
    header = "| Model | " + " | ".join(TAXONOMY) + " |"
    sep = "|" + "---|" * (len(TAXONOMY) + 1)
    lines.append(header)
    lines.append(sep)
    for model, data in results["per_model"].items():
        cells = [str(data["failure_distribution"].get(cat, 0)) for cat in TAXONOMY]
        lines.append(f"| `{model}` | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Restrict evaluation to these model directories. Default: every subdirectory of raw/.",
    )
    args = parser.parse_args()

    if not RAW_DIR.exists():
        print(f"ERROR: {RAW_DIR} does not exist. Run generate.py first.", file=sys.stderr)
        return 2

    if args.models:
        models = args.models
    else:
        models = sorted(p.name for p in RAW_DIR.iterdir() if p.is_dir())

    if not models:
        print(f"ERROR: no model directories under {RAW_DIR}.", file=sys.stderr)
        return 2

    per_model = {model: evaluate_model(model) for model in models}

    # Pull aero-eval version if available.
    try:
        from aero_eval import __version__ as aero_version
    except ImportError:
        aero_version = "unknown"

    results = {
        "metadata": {
            "n_per_model": max((d["n_total"] for d in per_model.values()), default=0),
            "models": list(per_model.keys()),
            "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "aero_eval_version": aero_version,
        },
        "per_model": per_model,
    }

    RESULTS_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    RESULTS_MD.write_text(render_markdown(results), encoding="utf-8")

    print(f"\nWrote {RESULTS_JSON}")
    print(f"Wrote {RESULTS_MD}")

    # Echo the pass-rate summary at the bottom.
    print("\n=== summary ===")
    for model, data in per_model.items():
        print(f"  {model}: {data['n_pass']}/{data['n_total']} = {data['pass_rate']:.1%}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
