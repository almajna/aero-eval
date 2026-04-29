"""Generate N attention implementations from each Anthropic model.

Reads ``prompts/attention_task.md``, sends it to each model ``--n-per-model``
times, saves raw responses to ``raw/{model}/{idx:03d}.txt`` and the assembled
runnable Python module to ``raw/{model}/{idx:03d}.py``.

Resumable: skips an index if its ``.py`` file already exists. Re-running
after a partial crash just fills in the gaps.

Usage::

    python studies/baseline/generate.py --n-per-model 100

Requires ``ANTHROPIC_API_KEY`` in the environment. The key is never logged.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

try:
    from anthropic import AsyncAnthropic
    from anthropic._exceptions import APIError, APIStatusError

    _HAS_ANTHROPIC = True
except ImportError:
    AsyncAnthropic = None  # type: ignore[assignment,misc]
    APIError = APIStatusError = Exception  # type: ignore[misc,assignment]
    _HAS_ANTHROPIC = False

DEFAULT_MODELS = [
    "claude-haiku-4-5",
    "claude-sonnet-4-6",
    "claude-opus-4-7",
]

HERE = Path(__file__).parent
PROMPT_PATH = HERE / "prompts" / "attention_task.md"
RAW_DIR = HERE / "raw"

# Concurrency cap per model. Anthropic's per-account rate limits are well
# above this for the models we use; 10 concurrent requests is conservative
# and avoids triggering 429s on a shared dev account.
CONCURRENCY = 10

# Per-call cap. Even malformed code blocks fit comfortably in 2k tokens.
MAX_TOKENS = 2048


# --------------------------------------------------------------------------- #
# Code assembly: turn raw model output into an importable .py file.
# --------------------------------------------------------------------------- #

CLASS_HEADER = """\
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CustomAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
"""


def assemble_module(forward_body: str) -> str:
    """Stitch a model's forward-body output into a runnable .py module.

    Strips fenced code blocks if the model returned them despite instructions
    and re-indents the body to match the 8-space level of ``def forward``'s
    body inside the class.
    """
    import textwrap

    # Trailing-only strip. We must NOT strip leading whitespace from the whole
    # block: if the model returned an already-indented body, ``.strip()``
    # would remove the common prefix from the first line only and break
    # textwrap.dedent's common-prefix detection.
    body = forward_body.rstrip()

    # Remove a leading blank-line preface if any.
    body_lines = body.splitlines()
    while body_lines and not body_lines[0].strip():
        body_lines.pop(0)
    body = "\n".join(body_lines)

    # Remove markdown fences if present.
    if body.lstrip().startswith("```"):
        # Find the first fence line and strip everything up to and including it.
        lines = body.splitlines()
        # Drop leading blank lines and the opening fence.
        while lines and not lines[0].strip():
            lines.pop(0)
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        # Drop closing fence if present.
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        body = "\n".join(lines)

    if not body.strip():
        return CLASS_HEADER + "        pass\n"

    # Normalize: dedent removes any common leading whitespace; re-indent to
    # the 8-space level expected inside ``def forward``.
    dedented = textwrap.dedent(body)
    indented = textwrap.indent(dedented, " " * 8)

    return CLASS_HEADER + indented + "\n"


# --------------------------------------------------------------------------- #
# Generation.
# --------------------------------------------------------------------------- #


async def _generate_one(
    client: AsyncAnthropic,
    model: str,
    prompt: str,
    out_dir: Path,
    idx: int,
    semaphore: asyncio.Semaphore,
) -> tuple[int, str]:
    py_path = out_dir / f"{idx:03d}.py"
    if py_path.exists():
        return idx, "skipped (already exists)"

    txt_path = out_dir / f"{idx:03d}.txt"
    err_path = out_dir / f"{idx:03d}.err"

    async with semaphore:
        try:
            resp = await client.messages.create(
                model=model,
                max_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )
        except (APIStatusError, APIError) as e:
            err_path.write_text(f"{type(e).__name__}: {e}\n", encoding="utf-8")
            return idx, f"api error: {type(e).__name__}"

    # Concatenate all text blocks (rare for non-tool calls but be safe).
    text_blocks = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
    raw = "".join(text_blocks)

    txt_path.write_text(raw, encoding="utf-8")
    py_path.write_text(assemble_module(raw), encoding="utf-8")

    return idx, "generated"


async def _generate_for_model(
    client: AsyncAnthropic,
    model: str,
    prompt: str,
    n: int,
) -> None:
    out_dir = RAW_DIR / model
    out_dir.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(CONCURRENCY)

    tasks = [_generate_one(client, model, prompt, out_dir, idx, semaphore) for idx in range(n)]

    # Progress: print as each task completes, not just at the end.
    n_done = 0
    n_skipped = 0
    n_errors = 0
    for coro in asyncio.as_completed(tasks):
        _idx, status = await coro
        n_done += 1
        if status.startswith("skipped"):
            n_skipped += 1
        elif status.startswith("api error"):
            n_errors += 1
        if n_done % 10 == 0 or n_done == n:
            print(
                f"  [{model}] {n_done}/{n} (skipped={n_skipped}, errors={n_errors})",
                flush=True,
            )


async def _main(args: argparse.Namespace) -> int:
    if not _HAS_ANTHROPIC:
        print(
            "ERROR: anthropic SDK not installed. Run: pip install 'anthropic>=0.40'",
            file=sys.stderr,
        )
        return 2

    if "ANTHROPIC_API_KEY" not in os.environ:
        print("ERROR: ANTHROPIC_API_KEY not set.", file=sys.stderr)
        return 2

    if not PROMPT_PATH.exists():
        print(f"ERROR: prompt file missing at {PROMPT_PATH}", file=sys.stderr)
        return 2

    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    models = args.models or DEFAULT_MODELS

    client = AsyncAnthropic()
    try:
        for model in models:
            print(f"=== {model} (n={args.n_per_model}) ===")
            await _generate_for_model(client, model, prompt, args.n_per_model)
    finally:
        await client.close()

    print("Done.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-per-model",
        type=int,
        default=100,
        help="Generations per model (default: 100).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=f"Override model list. Default: {' '.join(DEFAULT_MODELS)}",
    )
    args = parser.parse_args()
    return asyncio.run(_main(args))


if __name__ == "__main__":
    sys.exit(main())
