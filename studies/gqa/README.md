# GQA Study: LLM-Generated Grouped-Query Attention

## Research question

The baseline study showed that frontier models (Haiku 4.5, Sonnet 4.6, Opus
4.7) implement textbook multi-head attention with 100% semantic correctness
on the samples that compile. This is a *negative* result for the
"LLMs silently break PyTorch" thesis.

This follow-up study tests whether the negative result generalizes to a
strictly harder task: **Grouped-Query Attention** (GQA). GQA has more
moving parts than vanilla MHA — asymmetric projection sizes, KV head
expansion, group-ratio bookkeeping — and is the form actually used by
modern decoder LLMs (Llama 2/3, Mistral, Qwen).

## Methodology

### Task

Implement `GroupedQueryAttention.forward` under a fixed signature with
`n_heads=8, n_kv_heads=2, n_rep=4`. Full prompt at
[`prompts/gqa_task.md`](prompts/gqa_task.md).

### Models

Same three: `claude-haiku-4-5`, `claude-sonnet-4-6`, `claude-opus-4-7`.
100 samples each, temperature 1.0.

### Failure taxonomy

Identical to the baseline study (`syntax`, `import`, `instantiate`,
`forward_runtime`, `shape_contract`, `nan_inf`, `backward_runtime`,
`dead_grads`, `semantic_mismatch`, `pass`). The semantic check uses a
GQA-aware reference that expands KV heads via `repeat_interleave(n_rep)`.

### Reproduction

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
python studies/gqa/generate.py --n-per-model 100
python studies/gqa/evaluate.py
```

## Predicted outcomes (recorded before running)

If the baseline result generalizes, semantic pass rates will again be
≥95%. If GQA exposes silent breakage, expect drops to 50–80% range, with
failures concentrated in `shape_contract` (forgetting to expand KV) and
`semantic_mismatch` (expanding incorrectly — wrong axis or `repeat`
instead of `repeat_interleave`).

## Caveats

Same as baseline: single-prompt benchmark, temperature 1.0, taxonomy
short-circuits at first failure. The specific group ratio (`n_rep=4`) is
chosen because it's wide enough that "skip the expand step" produces
clearly broken outputs — a ratio of 1 (i.e. plain MHA) would let buggy
implementations pass by accident.
