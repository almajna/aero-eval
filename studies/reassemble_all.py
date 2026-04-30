"""Reassemble all existing .txt files in studies/{baseline,gqa}/raw/* using the
current assembler. Use this after fixing the assembler to refresh .py files
without spending more API budget.

Reads ``raw/{model}/{idx:03d}.txt`` (the verbatim model response) and
overwrites ``raw/{model}/{idx:03d}.py``. Skips any index that has no .txt
(e.g. one that crashed at API time and only has a .err file).

Usage::

    python studies/reassemble_all.py
    python studies/reassemble_all.py --study baseline
    python studies/reassemble_all.py --study gqa
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def reassemble_study(study: str) -> tuple[int, int]:
    """Reassemble every .txt under raw/. Returns (n_assembled, n_skipped)."""
    study_dir = HERE / study
    if not study_dir.exists():
        print(f"  no such study: {study_dir}", file=sys.stderr)
        return 0, 0

    raw_dir = study_dir / "raw"
    if not raw_dir.exists():
        print(f"  no raw dir under {study_dir}", file=sys.stderr)
        return 0, 0

    # Import the study's generate.py to get its assemble_module (the headers
    # differ between studies). Cache the module per study.
    sys.path.insert(0, str(study_dir))
    try:
        gen = importlib.import_module("generate")
        # If generate was already imported with a different study's path, force a fresh
        # import so we get the right CLASS_HEADER.
        gen = importlib.reload(gen)
        assemble = gen.assemble_module
    finally:
        sys.path.pop(0)

    n_assembled = 0
    n_skipped = 0
    for model_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        for txt_path in sorted(model_dir.glob("*.txt")):
            py_path = txt_path.with_suffix(".py")
            raw = txt_path.read_text(encoding="utf-8")
            new_py = assemble(raw)
            old_py = py_path.read_text(encoding="utf-8") if py_path.exists() else None
            if new_py == old_py:
                n_skipped += 1
                continue
            py_path.write_text(new_py, encoding="utf-8")
            n_assembled += 1
        print(f"  {model_dir.name}: scanned {len(list(model_dir.glob('*.txt')))} txt files")

    return n_assembled, n_skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study",
        choices=["baseline", "gqa", "all"],
        default="all",
    )
    args = parser.parse_args()

    studies = ["baseline", "gqa"] if args.study == "all" else [args.study]

    total_a = total_s = 0
    for s in studies:
        print(f"=== {s} ===")
        a, s_count = reassemble_study(s)
        print(f"  reassembled: {a}, unchanged: {s_count}")
        total_a += a
        total_s += s_count

    print(f"\nTotal reassembled: {total_a}; unchanged: {total_s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
