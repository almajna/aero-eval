# aero-eval

> Adversarial evaluation harness for LLM-generated PyTorch models.

Catches the structural bugs LLMs silently introduce — NaN propagation, dead neurons,
frozen parameters, attention collapse, capacity-clamping — before they burn GPU hours.

## Status

**Pre-release.** v0.1 in active development. Not yet on PyPI.

## Why this exists

LLMs generate PyTorch code that compiles and runs, then silently fails:

- Dimension broadcasts that produce wrong shapes without erroring
- Attention masks applied to the wrong axis
- Activation functions that kill gradients at init
- `torch.clamp` calls that fake gradient stability and pass naive checks

Existing tools (`torch.compile`, lint, type-checkers) catch *syntactic* bugs.
`aero-eval` catches *semantic* bugs by running adversarial training-time probes
against your model and asserting expected dynamics.

## Install

Coming soon.

```bash
pip install aero-eval  # not yet published
```

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
