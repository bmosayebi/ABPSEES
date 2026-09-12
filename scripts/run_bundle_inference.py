#!/usr/bin/env python3
"""Load an ABPSEES model bundle (.zip) and run inference locally.

This is the "run it here" counterpart to Colab training: after
``train_model()`` finishes on Colab it automatically writes a single
portable file at ``outputs/model_bundle.zip`` (see
``project/bundle.py::save_model_bundle``). Download that one file (e.g.
via ``google.colab.files.download`` -- see ``COLAB.md``) to any machine
that has this repository installed (``pip install -e .`` or
``pip install -r requirements.txt``), then run this script against it.

The base Qwen model itself is *not* bundled (only the small LoRA adapter +
aspect heads are); the base weights are re-downloaded from the Hugging
Face Hub the first time you run this, exactly like every other step in
this project. Make sure ``HF_TOKEN`` (or ``HUGGING_FACE_HUB_TOKEN``) is set
in your environment and that you have accepted the base model's license on
huggingface.co.

Usage:
    # One-shot: classify a single Persian text and exit.
    python scripts/run_bundle_inference.py --bundle outputs/model_bundle.zip \\
        --text "من دانشجوی مهندسی کامپیوتر هستم و یک لپ‌تاپ قدرتمند و قابل حمل می‌خواهم."

    # Interactive REPL: keep the model loaded and try multiple inputs.
    python scripts/run_bundle_inference.py --bundle outputs/model_bundle.zip

Options:
    --bundle PATH   Path to the .zip bundle (required).
    --text TEXT     Persian input text. If omitted, starts an interactive
                     REPL instead of exiting after one prediction.
    --extract-dir   Directory to extract the bundle into (default: a fresh
                     temporary directory).
    --cuda          Require a CUDA GPU (fails fast instead of falling back
                     to slow CPU inference).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `import project` work when this script is run directly (e.g.
# `python scripts/run_bundle_inference.py`) without a prior `pip install -e .`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Load an ABPSEES model bundle (.zip) and run inference.",
    )
    parser.add_argument("--bundle", required=True, help="Path to the .zip model bundle.")
    parser.add_argument("--text", default=None, help="Persian input text (omit for interactive mode).")
    parser.add_argument(
        "--extract-dir", default=None, help="Directory to extract the bundle into (default: temp dir)."
    )
    parser.add_argument(
        "--cuda", action="store_true", help="Require a CUDA GPU instead of falling back to CPU."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point: load the bundle once, then classify one text or run a REPL."""
    from project.bundle import load_model_bundle
    from project.inference import predict, pretty_print_prediction

    args = parse_args(argv)

    print(f"Loading model bundle from {args.bundle} ...")
    model, tokenizer, config = load_model_bundle(
        args.bundle, extract_dir=args.extract_dir, require_cuda=args.cuda
    )
    device = next(model.parameters()).device
    print(f"Model loaded on device: {device}")
    print(f"Aspect-guided attention heads enabled: {config.aspect.enabled}")

    if args.text:
        result = predict(args.text, config, model=model, tokenizer=tokenizer)
        print(pretty_print_prediction(result))
        for warning in result.get("_warnings", []):
            print(f"[warning] {warning}")
        return 0

    print("\nInteractive mode — type Persian text and press Enter.")
    print("(Ctrl+D or Ctrl+C to quit)\n")
    while True:
        try:
            text = input("متن ورودی > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not text:
            continue
        result = predict(text, config, model=model, tokenizer=tokenizer)
        print(pretty_print_prediction(result))
        for warning in result.get("_warnings", []):
            print(f"[warning] {warning}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
