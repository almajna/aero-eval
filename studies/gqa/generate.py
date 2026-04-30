"""Generate N grouped-query attention implementations from each Anthropic model.

Concurrency drops to 3 for Opus to stay under default rate limits.

Usage::

    python studies/gqa/generate.py --n-per-model 100

Requires ``ANTHROPIC_API_KEY``.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Reuse the shared assembler from studies/_assembler.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _assembler import assemble_module as _assemble

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
PROMPT_PATH = HERE / "prompts" / "gqa_task.md"
RAW_DIR = HERE / "raw"

PER_MODEL_CONCURRENCY: dict[str, int] = {
    "claude-opus-4-7": 3,
}
DEFAULT_CONCURRENCY = 10
MAX_TOKENS = 2048

CLASS_HEADER = """\
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if n_heads % n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim)
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
    return _assemble(CLASS_HEADER, forward_body)


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
    concurrency = PER_MODEL_CONCURRENCY.get(model, DEFAULT_CONCURRENCY)
    print(f"  (concurrency={concurrency})", flush=True)
    semaphore = asyncio.Semaphore(concurrency)

    tasks = [_generate_one(client, model, prompt, out_dir, idx, semaphore) for idx in range(n)]

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
    parser.add_argument("--n-per-model", type=int, default=100)
    parser.add_argument("--models", nargs="+", default=None)
    args = parser.parse_args()
    return asyncio.run(_main(args))


if __name__ == "__main__":
    sys.exit(main())
