"""Load a Colab-trained ABPSEES bundle and run inference locally.

Download ``outputs/model_bundle.zip`` from Colab (see COLAB.md), put it
next to this repo (default path below), then:

    python3 run_local.py

Edit ``TEXT`` or pass a sentence as a command-line argument.
"""

from __future__ import annotations

import sys
from pathlib import Path

from project.bundle import load_model_bundle
from project.inference import predict, pretty_print_prediction

BUNDLE = Path("outputs/model_bundle.zip")

TEXT = (
    "من دانشجوی مهندسی کامپیوتر هستم و می‌خواهم مدل‌های هوش مصنوعی آموزش بدهم. "
    "هر روز لپ‌تاپم را با خودم به دانشگاه می‌برم."
)


def main() -> int:
    bundle = Path(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].endswith(".zip") else BUNDLE
    text = " ".join(a for a in sys.argv[1:] if not a.endswith(".zip")) or TEXT

    if not bundle.exists():
        print(f"Bundle not found: {bundle}")
        print("Download model_bundle.zip from Colab and place it at that path.")
        return 1

    print(f"Loading {bundle} ...")
    model, tokenizer, config = load_model_bundle(bundle)
    result = predict(text, config, model=model, tokenizer=tokenizer)
    print(pretty_print_prediction(result))
    for warning in result.get("_warnings", []):
        print(f"[warning] {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
