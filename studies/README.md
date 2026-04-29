# Studies

Empirical research artifacts that ship outside the published `aero-eval`
package. Each study is fully reproducible from the scripts and prompts in its
own subdirectory.

## Index

- [`baseline/`](baseline/) — *"How often does an LLM produce silently broken
  PyTorch?"* The empirical baseline that motivates `aero-eval`. Generates N
  attention-mechanism implementations from each model, runs them through the
  evaluation pipeline, reports the failure distribution.

## Reproducibility contract

Each study directory contains:

- `README.md` — research question, methodology, exact reproduction steps.
- `prompts/` — the verbatim prompts sent to LLMs. Edits are versioned in git.
- `*.py` — generation and evaluation scripts. Side-effect-free where possible.
- Output paths are `.gitignore`d. Re-running a study reproduces results from
  scratch given the same model versions and seeds.

Studies do not import internal-only `aero-eval` APIs — they use the same
public surface external users would.
