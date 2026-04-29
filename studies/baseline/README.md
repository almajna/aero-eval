# Baseline Study: LLM-Generated PyTorch Failure Distribution

## Research question

When an LLM is asked to implement a small but non-trivial PyTorch component
under a fixed interface contract, what fraction of generations are silently
broken — i.e. compile and run without raising, but fail one or more semantic
correctness checks?

This study produces the empirical baseline that motivates `aero-eval`. The
results inform the first public blog post.

## Methodology

### Task

Implement the `forward` method of a `CustomAttention` module under a fixed
class signature. The signature, expected tensor shapes, and behavioral
contract are pinned in `prompts/attention_task.md`. The model fills in the
body; everything else (imports, class header, harness) is provided.

This deliberately removes "the LLM didn't know what we wanted" as a failure
mode. Anything broken about the output is broken about the *implementation*,
not the spec.

### Models

- `claude-haiku-4-5`
- `claude-sonnet-4-6`
- `claude-opus-4-7`

100 generations from each, total N=300. Temperature 1.0 (the production
default for this provider) so we measure realistic distribution-of-output
quality, not deterministic single-sample quality.

### Failure taxonomy

Each generation is graded by progressively stricter checks. The first failure
mode encountered is recorded; later checks are skipped for that sample.

1. **`syntax`** — refuses to parse via `ast.parse`.
2. **`import`** — parses, but `exec` raises (undefined name, bad import).
3. **`instantiate`** — class body executes, but `CustomAttention(d_model=64,
   n_heads=8)` raises.
4. **`forward_runtime`** — instance constructed, but `forward(q, k, v, mask)`
   raises (shape mismatch, dtype error, OOM).
5. **`shape_contract`** — forward returns, but output shape ≠ input shape
   contract.
6. **`nan_inf`** — output contains NaN or Inf.
7. **`backward_runtime`** — forward result clean, but `loss.backward()`
   raises.
8. **`dead_grads`** — backward succeeds but every parameter has zero gradient
   (no learning signal at all).
9. **`pass`** — survived every check.

This taxonomy is conservative — a sample classified as `pass` here might
still fail under heavier checks (overfit, capacity-pairing). The point is to
get a lower-bound on the failure rate, not an upper-bound on quality.

### Reproduction

```bash
# 1. Set your Anthropic API key in the environment (do NOT commit it).
export ANTHROPIC_API_KEY="sk-ant-..."

# 2. Generate samples (writes to studies/baseline/raw/{model}/{idx}.py).
#    Skipped per-model if --models is passed, otherwise runs all three.
python studies/baseline/generate.py --n-per-model 100

# 3. Evaluate. Reads raw/, writes results.json + results.md.
python studies/baseline/evaluate.py
```

Cost estimate at the time of writing (Apr 2026): roughly USD 5–15 for the
full N=300 run, dominated by Opus. Haiku and Sonnet account for under USD 2
combined.

### Output

`results.json` schema:

```json
{
  "metadata": {
    "n_per_model": 100,
    "models": ["claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-7"],
    "completed_at": "2026-04-29T12:34:56Z",
    "aero_eval_version": "0.1.0a1"
  },
  "per_model": {
    "claude-haiku-4-5": {
      "n_total": 100,
      "n_pass": 41,
      "failure_distribution": {
        "syntax": 2, "import": 1, "instantiate": 8,
        "forward_runtime": 23, "shape_contract": 12,
        "nan_inf": 3, "backward_runtime": 4, "dead_grads": 6
      },
      "pass_rate": 0.41
    }
  }
}
```

`results.md` is a human-readable summary table, suitable for the blog post.
Both files are git-ignored — re-run the scripts to regenerate.

## Caveats

- Single-prompt benchmark. We do not vary the task. A different prompt might
  shift the distribution materially.
- Temperature 1.0 measures realistic *production* output. Lowering to 0.0
  would understate failure rates as users see them.
- The taxonomy short-circuits at the first failure. A sample failing at
  `forward_runtime` might also have NaN-prone numerics, but we don't know.
- API keys are read from `ANTHROPIC_API_KEY`. The scripts never log, write,
  or transmit the key beyond the Anthropic SDK call.
