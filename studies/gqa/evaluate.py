"""Evaluate generated GroupedQueryAttention modules against the failure taxonomy.

Mirrors ``studies/baseline/evaluate.py`` but uses a GQA-aware reference
implementation in the semantic-correctness check. The probe expands the
KV heads to match the query head count using the model's own ``n_rep``
attribute, then compares to ``F.scaled_dot_product_attention``.

Usage::

    python studies/gqa/evaluate.py
    python studies/gqa/evaluate.py --models claude-haiku-4-5
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

TIMEOUT_SECONDS = 30

TAXONOMY = [
    "syntax",
    "import",
    "instantiate",
    "forward_runtime",
    "shape_contract",
    "nan_inf",
    "backward_runtime",
    "dead_grads",
    "semantic_mismatch",
    "pass",
]


# --------------------------------------------------------------------------- #
# Probe — GQA variant. Differences from baseline probe:
#   1. Constructor takes (d_model, n_heads, n_kv_heads).
#   2. Reference attention expands K/V along head axis by n_rep.
#   3. We use a non-trivial group ratio (n_heads=8, n_kv_heads=2) so an
#      implementation that accidentally treats this as MHA (no expansion)
#      fails on the shape contract or semantic check.
# --------------------------------------------------------------------------- #

PROBE_SCRIPT = r"""
import ast, json, sys
from pathlib import Path

src_path = Path(sys.argv[1])
src = src_path.read_text(encoding="utf-8")


def emit(cat, **extra):
    payload = {"category": cat}
    payload.update(extra)
    print(json.dumps(payload))
    sys.exit(0)


# 1. Syntax.
try:
    ast.parse(src)
except SyntaxError:
    emit("syntax")

# 2. Import.
ns = {"__name__": "__probe__"}
try:
    exec(compile(src, str(src_path), "exec"), ns)
except Exception:
    emit("import")

GQA = ns.get("GroupedQueryAttention")
if GQA is None:
    emit("import")

# 3. Instantiate. n_heads=8, n_kv_heads=2 -> n_rep=4.
try:
    model = GQA(d_model=64, n_heads=8, n_kv_heads=2)
except Exception:
    emit("instantiate")

# 4. Forward runtime.
import torch
import torch.nn.functional as F
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

# 8. Dead grads.
any_grad = False
for p in model.parameters():
    if p.grad is not None and p.grad.abs().sum().item() > 0:
        any_grad = True
        break
if not any_grad:
    emit("dead_grads")

# 9. Semantic correctness with GQA-aware reference.
def _reference_gqa(model, q, k, v, mask):
    bsz, seq, _ = q.shape
    H = model.n_heads
    HKV = model.n_kv_heads
    D = model.head_dim
    n_rep = model.n_rep

    qh = model.q_proj(q).view(bsz, seq, H, D).transpose(1, 2)        # [B, H, S, D]
    kh = model.k_proj(k).view(bsz, seq, HKV, D).transpose(1, 2)      # [B, HKV, S, D]
    vh = model.v_proj(v).view(bsz, seq, HKV, D).transpose(1, 2)      # [B, HKV, S, D]

    # Repeat KV heads along the head axis to match Q heads. Use repeat_interleave
    # so each KV head is consumed by n_rep contiguous Q heads.
    kh = kh.repeat_interleave(n_rep, dim=1)                          # [B, H, S, D]
    vh = vh.repeat_interleave(n_rep, dim=1)                          # [B, H, S, D]

    if mask is not None:
        attn_mask = mask.unsqueeze(1)                                 # [B, 1, S, S]
    else:
        attn_mask = None

    ref_h = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=attn_mask)
    ref = ref_h.transpose(1, 2).contiguous().view(bsz, seq, H * D)
    return model.out_proj(ref)


# Fresh module for the semantic test.
torch.manual_seed(0)
m2 = GQA(d_model=64, n_heads=8, n_kv_heads=2)
m2.eval()

regimes = [
    {"seed": 1, "use_mask": False, "label": "no_mask_seed1"},
    {"seed": 2, "use_mask": True, "label": "full_mask_seed2"},
    {"seed": 3, "use_mask": True, "label": "causal_mask_seed3", "causal": True},
]

mismatch_evidence = []
for regime in regimes:
    g = torch.Generator().manual_seed(regime["seed"])
    qi = torch.randn(2, 5, 64, generator=g)
    ki = torch.randn(2, 5, 64, generator=g)
    vi = torch.randn(2, 5, 64, generator=g)

    if regime["use_mask"]:
        if regime.get("causal", False):
            m_ = torch.tril(torch.ones(5, 5, dtype=torch.bool)).expand(2, 5, 5).contiguous()
        else:
            m_ = torch.ones(2, 5, 5, dtype=torch.bool)
    else:
        m_ = None

    try:
        with torch.no_grad():
            llm_out = m2(qi, ki, vi, m_)
            ref_out = _reference_gqa(m2, qi, ki, vi, m_)
    except Exception as e:
        mismatch_evidence.append({
            "regime": regime["label"],
            "reason": "exception_during_semantic_eval",
            "type": type(e).__name__,
        })
        continue

    if not torch.allclose(llm_out, ref_out, atol=1e-4, rtol=1e-4):
        max_abs = (llm_out - ref_out).abs().max().item()
        mismatch_evidence.append({
            "regime": regime["label"],
            "max_abs_diff": max_abs,
        })

if mismatch_evidence:
    emit("semantic_mismatch", regimes=mismatch_evidence)

emit("pass")
"""


def evaluate_sample(py_path: Path) -> tuple[str, dict]:
    try:
        result = subprocess.run(
            [sys.executable, "-c", PROBE_SCRIPT, str(py_path)],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return "forward_runtime", {}
    except Exception:
        return "forward_runtime", {}

    for line in reversed(result.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            cat = payload.pop("category", "forward_runtime")
            if cat in TAXONOMY:
                return cat, payload
    return "forward_runtime", {}


def evaluate_model(model: str) -> dict:
    model_dir = RAW_DIR / model
    if not model_dir.exists():
        return {"n_total": 0, "failure_distribution": {}, "pass_rate": 0.0}

    paths = sorted(model_dir.glob("*.py"))
    failure_counts = dict.fromkeys(TAXONOMY, 0)
    semantic_evidence: list[dict] = []

    print(f"=== {model} ({len(paths)} samples) ===", flush=True)
    for i, py_path in enumerate(paths, 1):
        cat, extra = evaluate_sample(py_path)
        failure_counts[cat] += 1
        if cat == "semantic_mismatch" and extra:
            semantic_evidence.append({"sample": py_path.name, **extra})
        if i % 10 == 0 or i == len(paths):
            print(f"  {i}/{len(paths)}", flush=True)

    n_total = len(paths)
    n_pass = failure_counts["pass"]
    return {
        "n_total": n_total,
        "n_pass": n_pass,
        "failure_distribution": failure_counts,
        "pass_rate": n_pass / n_total if n_total else 0.0,
        "semantic_mismatch_evidence": semantic_evidence,
    }


def render_markdown(results: dict) -> str:
    lines = ["# GQA Study Results", ""]
    meta = results["metadata"]
    lines.append(f"- Generated at: `{meta['completed_at']}`")
    lines.append(f"- aero-eval: `{meta['aero_eval_version']}`")
    lines.append(f"- Samples per model: {meta['n_per_model']}")
    lines.append("- Configuration: `n_heads=8, n_kv_heads=2, n_rep=4, head_dim=8`")
    lines.append("")
    lines.append("## Pass rate by model")
    lines.append("")
    lines.append("| Model | n | Naive pass | Semantic pass | Naive→semantic gap |")
    lines.append("|-------|---|------------|---------------|--------------------|")
    for model, data in results["per_model"].items():
        n = data["n_total"]
        n_pass = data["n_pass"]
        n_semantic_fail = data["failure_distribution"].get("semantic_mismatch", 0)
        n_naive_pass = n_pass + n_semantic_fail
        naive_rate = n_naive_pass / n if n else 0.0
        semantic_rate = n_pass / n if n else 0.0
        gap = naive_rate - semantic_rate
        lines.append(
            f"| `{model}` | {n} | {n_naive_pass} ({naive_rate:.1%}) | "
            f"{n_pass} ({semantic_rate:.1%}) | {gap:.1%} |"
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
    parser.add_argument("--models", nargs="+", default=None)
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
            "task_config": {"n_heads": 8, "n_kv_heads": 2, "head_dim": 8, "d_model": 64},
        },
        "per_model": per_model,
    }

    RESULTS_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    RESULTS_MD.write_text(render_markdown(results), encoding="utf-8")

    print(f"\nWrote {RESULTS_JSON}")
    print(f"Wrote {RESULTS_MD}")

    print("\n=== summary ===")
    for model, data in per_model.items():
        n = data["n_total"]
        n_pass = data["n_pass"]
        n_semantic_fail = data["failure_distribution"].get("semantic_mismatch", 0)
        n_naive = n_pass + n_semantic_fail
        print(
            f"  {model}: naive={n_naive}/{n} ({n_naive / n:.1%}) "
            f"semantic={n_pass}/{n} ({n_pass / n:.1%})"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
