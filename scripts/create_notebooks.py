"""Helper to build orchestration notebooks."""

from __future__ import annotations

import json
from pathlib import Path


def nb(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11.0"},
            "colab": {"provenance": []},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": source.splitlines(True),
        "outputs": [],
        "execution_count": None,
    }


COLAB_SETUP = """import os
import sys
from pathlib import Path

# Colab: project lives at /content/ABPSEES
if Path('/content/ABPSEES/project').exists():
    ROOT = Path('/content/ABPSEES')
else:
    ROOT = Path.cwd()
    if not (ROOT / 'project').exists():
        ROOT = ROOT.parent

sys.path.insert(0, str(ROOT))
os.environ['ABPSEES_ROOT'] = str(ROOT)

%load_ext autoreload
%autoreload 2

from project.config import load_config, is_colab
from project.utils import setup_logging, setup_colab_environment

setup_logging()
config = load_config()  # auto-selects colab.yaml on Colab
setup_colab_environment(
    hf_token_env=config.colab.hf_token_env,
    use_colab_secrets=config.colab.use_colab_secrets,
    mount_google_drive=config.colab.mount_google_drive,
)

print('Project root:', ROOT)
print('Colab:', is_colab())
print('Data dir:', config.paths.data_dir)
"""


def write_notebooks(root: Path) -> None:
    notebooks = {
        "01_environment.ipynb": [
            md(
                "# 01 — Environment Setup\n\n"
                "Verify dependencies, CUDA GPU, and HuggingFace access.\n\n"
                "**Colab:** Enable GPU (Runtime → Change runtime type → T4 GPU) "
                "and add `HF_TOKEN` in Secrets (🔑)."
            ),
            code(
                """# Run once on Colab
import os
if os.path.exists('/content'):
    get_ipython().system('pip install -q -r /content/ABPSEES/requirements.txt')
    get_ipython().system('pip install -q -e /content/ABPSEES')
"""
            ),
            code(COLAB_SETUP),
            code(
                """import importlib
import torch

packages = [
    'torch', 'transformers', 'peft', 'accelerate', 'datasets', 'bitsandbytes',
    'sklearn', 'pandas', 'numpy', 'matplotlib', 'yaml', 'scipy',
]
for pkg in packages:
    try:
        importlib.import_module(pkg)
        print(f'OK  {pkg}')
    except ImportError as exc:
        print(f'FAIL {pkg}: {exc}')

print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('VRAM (GB):', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
"""
            ),
            code(
                """from project.utils import get_device_config, set_seed

device_cfg = get_device_config(require_cuda=is_colab())
set_seed(config.seed)
print('Device:', device_cfg.device)
print('QLoRA:', device_cfg.use_qlora)
"""
            ),
            code(
                """from project.model import load_tokenizer

tokenizer = load_tokenizer(config)
print('Tokenizer:', tokenizer.__class__.__name__)
"""
            ),
        ],
        "02_dataset_analysis.ipynb": [
            md("# 02 — Dataset Analysis"),
            code(COLAB_SETUP),
            code(
                """from project.dataset import load_raw_dataset, prepare_dataset_splits
import pandas as pd

split_paths = prepare_dataset_splits(config)
raw = load_raw_dataset(config)
print(f'Total samples: {len(raw)}')
"""
            ),
            code(
                """rows = []
for sample in raw:
    for aspect, label in sample['labels'].items():
        rows.append({
            'aspect': aspect,
            'score': label['score'],
            'has_evidence': bool(label.get('evidence', '')),
        })
df = pd.DataFrame(rows)
df.groupby('aspect').agg(mean_score=('score', 'mean'), pct_evidence=('has_evidence', 'mean'))
"""
            ),
        ],
        "03_preprocessing.ipynb": [
            md("# 03 — Preprocessing"),
            code(COLAB_SETUP),
            code(
                """from project.dataset import load_split
from project.model import load_tokenizer
from project.preprocessing import preprocess_samples, save_enriched_split

tokenizer = load_tokenizer(config)
train_raw = load_split(config, 'train')
enriched = preprocess_samples(train_raw, tokenizer, config)
print(f'Enriched: {len(enriched)} / {len(train_raw)}')

for split in ('train', 'val', 'test'):
    save_enriched_split(load_split(config, split), tokenizer, config, split)
"""
            ),
        ],
        "04_training.ipynb": [
            md(
                "# 04 — Training\n\n"
                "Fine-tune Qwen2.5-3B-Instruct with **4-bit QLoRA** on Colab CUDA GPU."
            ),
            code(COLAB_SETUP),
            code(
                """from project.trainer import build_trainer, train_model
from project.utils import get_device_config

device_cfg = get_device_config(require_cuda=True)
print(f'Training on {device_cfg.device} (QLoRA={device_cfg.use_qlora})')
print(f'max_seq_length={config.model.max_seq_length}')
print(f'Checkpoints -> {config.paths.checkpoint_dir}')
"""
            ),
            code(
                """trainer, model, tokenizer = build_trainer(config)
print('Train:', len(trainer.train_dataset), '| Val:', len(trainer.eval_dataset))
"""
            ),
            code(
                """trainer = train_model(config)

# Resume:
# trainer = train_model(config, resume_from_checkpoint=config.paths.checkpoint_dir / 'checkpoint-XXX')
"""
            ),
            code(
                """# Save outputs to Google Drive (optional):
# from google.colab import drive
# drive.mount('/content/drive')
# !cp -r outputs /content/drive/MyDrive/ABPSEES/
"""
            ),
        ],
        "05_evaluation.ipynb": [
            md("# 05 — Evaluation"),
            code(COLAB_SETUP),
            code("from project.evaluation import run_evaluation\nresults = run_evaluation(config, split='test')"),
            code(
                """import pandas as pd
display(pd.DataFrame(results['score_metrics']).T)
display(pd.DataFrame(results['evidence_metrics']).T)
"""
            ),
        ],
        "06_inference.ipynb": [
            md("# 06 — Inference"),
            code(COLAB_SETUP),
            code(
                """from project.inference import predict, pretty_print_prediction

text = 'من دانشجوی مهندسی کامپیوتر هستم و می خواهم مدل‌های هوش مصنوعی آموزش بدهم.'
result = predict(text, config)
print(pretty_print_prediction(result))
"""
            ),
        ],
        "07_error_analysis.ipynb": [
            md("# 07 — Error Analysis"),
            code(COLAB_SETUP),
            code(
                """from project.evaluation import run_evaluation
from project.visualization import generate_all_visualizations

results = run_evaluation(config, split='test')
paths = generate_all_visualizations(results, config.paths.reports_dir / 'figures')
for k, v in paths.items():
    print(k, '->', v)
"""
            ),
        ],
    }

    out_dir = root / "notebooks"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, cells in notebooks.items():
        path = out_dir / name
        path.write_text(json.dumps(nb(cells), ensure_ascii=False, indent=1), encoding="utf-8")
        print("Wrote", path)


if __name__ == "__main__":
    write_notebooks(Path(__file__).resolve().parent.parent)
