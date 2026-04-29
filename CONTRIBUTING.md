# Contributing to aero-eval

## Development setup

Requires Python 3.10 or 3.11. Python 3.12 is **not yet supported** (waiting on
upstream PyTorch + transformers wheel coverage).

```bash
git clone https://github.com/almajna/aero-eval.git
cd aero-eval
python -m venv .venv
source .venv/bin/activate   # Windows: .\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

## Running tests

```bash
pytest                          # full suite
pytest tests/fpr                # false-positive suite (must pass before merge)
pytest -k overfit               # filter by keyword
```

The **false-positive (FPR) suite** runs `aero-eval` against known-good HuggingFace
models (`bert-tiny`, `TinyStories-1M`, `SmolLM-135M`) on a real training scenario
with pinned seeds. Any check that flags a healthy model is treated as a tuning
failure and must be fixed before merge.

## Code style

- Formatter + linter: [`ruff`](https://docs.astral.sh/ruff/)
- Line length: 100
- Target: Python 3.10

```bash
ruff format .
ruff check . --fix
```

CI rejects PRs that don't pass `ruff format --check` and `ruff check`.

## Commit messages

Conventional Commits, e.g.:

- `feat(checks): add attention entropy collapse detector`
- `fix(callback): handle empty grad on frozen layers`
- `chore: bump pytorch matrix to 2.6`
- `docs: clarify capacity-pairing thresholds`

## Pull requests

1. Branch from `main`: `git checkout -b feat/your-thing`
2. Make changes + add tests.
3. Ensure `pytest` passes locally including `tests/fpr`.
4. Open PR with a clear description of *what changed* and *why*.
5. CI must pass before merge.

## Reporting issues

When filing a bug, include:

- `aero-eval` version, PyTorch version, Python version, OS
- Minimal reproducing snippet (model + check that misbehaves)
- Expected vs. actual behavior
- For false positives: which known-good model triggered the check

## Security

Do not file public issues for security concerns. Email the maintainers
(see repo metadata).
