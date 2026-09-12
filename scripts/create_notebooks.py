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


COLAB_SETUP = """# Bootstrap path (required before import project)
exec(open("/content/ABPSEES/colab_bootstrap.py").read())
ROOT = setup_colab_path()

%load_ext autoreload
%autoreload 2

from project.config import load_config, is_colab
from project.utils import setup_logging, setup_colab_environment

setup_logging()
config = load_config()
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
                """# SEL 1 — Run once: clone + install (use %pip, not !pip)
# !rm -rf /content/ABPSEES
# !git clone https://github.com/bmosayebi/ABPSEES.git /content/ABPSEES
# %cd /content/ABPSEES
# %pip install -q -r requirements.txt
# %pip install -q -e .
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
paths = generate_all_visualizations(
    results, config.paths.reports_dir / 'figures',
    aspect_results=results.get('aspect_results'),
)
for k, v in paths.items():
    print(k, '->', v)
"""
            ),
        ],
        "08_aspect_attention.ipynb": [
            md(
                "# 08 — Aspect-Guided Attention (Phase 1 + Phase 2)\n\n"
                "Demos and ablations for the auxiliary aspect-conditioned attention "
                "heads: per-aspect score regression, evidence-span extraction, and "
                "faithfulness diagnostics. See `docs/ASPECT_ATTENTION_GUIDE.md` for "
                "the full scientific write-up.\n\n"
                "**Requires:** `aspect.enabled: true` in the active config, and a "
                "checkpoint trained with `04_training.ipynb` (which saves "
                "`aspect_heads.pt` alongside the LoRA adapter when enabled)."
            ),
            code(COLAB_SETUP),
            code(
                """print('Aspect attention enabled:', config.aspect.enabled)
print('Hidden layer:', config.aspect.hidden_layer)
print('Lambdas: ce={} score={} span={} faith={}'.format(
    config.aspect.lambda_ce, config.aspect.lambda_score,
    config.aspect.lambda_span, config.aspect.lambda_faith,
))
"""
            ),
            md(
                "## Load the hybrid model\n\n"
                "Loads the base Qwen model + LoRA adapter + the auxiliary "
                "`AspectHeads` (score/span/has-evidence) from "
                "`outputs/checkpoints/best/`."
            ),
            code(
                """from project.hybrid_model import load_hybrid_model_for_inference

hybrid_model, tokenizer = load_hybrid_model_for_inference(config)
print(hybrid_model.aspect_heads)
"""
            ),
            md(
                "## Single-example demo: score head vs generated JSON\n\n"
                "Compares the primary generative JSON prediction with the "
                "auxiliary score/span heads on one Persian example."
            ),
            code(
                """from project.inference import predict, pretty_print_prediction

text = (
    'من دانشجوی مهندسی کامپیوتر هستم و می خواهم مدل‌های هوش مصنوعی آموزش بدهم. '
    'هر روز لپ‌تاپم را با خودم به دانشگاه می‌برم.'
)
result = predict(text, config, model=hybrid_model, tokenizer=tokenizer)
print(pretty_print_prediction(result))
print()
print('Aspect head diagnostics:')
for aspect, diag in result.get('_aspect_attention', {}).items():
    print(f'  {aspect}: score_head={diag[\"score_head\"]:.3f} '
          f'has_evidence_prob={diag[\"has_evidence_prob\"]:.3f} '
          f'span_evidence={diag[\"span_evidence\"]!r}')
"""
            ),
            md(
                "## Attention heatmap for one aspect\n\n"
                "Visualizes the aspect-conditioned attention distribution "
                "`alpha_a` over the input tokens, for direct inspection of "
                "*where* the model is looking when scoring each aspect."
            ),
            code(
                """from project.hybrid_model import run_aspect_heads_on_text
from project.visualization import plot_aspect_attention_heatmap

diag = run_aspect_heads_on_text(hybrid_model, tokenizer, text, config)
fig_path = plot_aspect_attention_heatmap(
    diag['tokens'], diag['attn_weights'],
    config.paths.reports_dir / 'figures' / 'attention' / 'demo_sample.png',
)
print('Saved heatmap to', fig_path)
"""
            ),
            md(
                "## Full test-set aspect-head evaluation\n\n"
                "Runs the same head-based evaluation used by "
                "`run_evaluation` (score MAE/RMSE, span Token F1/EM, "
                "faithfulness rates) standalone, for ablation experiments."
            ),
            code(
                """from project.evaluation import evaluate_aspect_heads
from project.dataset import load_split
from project.preprocessing import save_enriched_split

test_raw = load_split(config, 'test')
test_enriched = save_enriched_split(test_raw, tokenizer, config, 'test')
aspect_results = evaluate_aspect_heads(hybrid_model, tokenizer, test_enriched, config)

import pandas as pd
display(pd.DataFrame(aspect_results['score_metrics']).T)
display(pd.DataFrame(aspect_results['span_metrics']).T)
display(pd.DataFrame(aspect_results['faithfulness_rates']).T)
"""
            ),
            md(
                "## Ablation checklist\n\n"
                "Suggested runs to compare against the full "
                "CE + score + span + faith objective (see "
                "`docs/ASPECT_ATTENTION_GUIDE.md` §7 for details):\n\n"
                "1. `lambda_score=0, lambda_span=0, lambda_faith=0` (CE-only baseline)\n"
                "2. `+lambda_score` only (Phase 1)\n"
                "3. `+lambda_span` (Phase 1 + 2, no faithfulness)\n"
                "4. Full objective (`+lambda_faith`)\n\n"
                "Compare `outputs/reports/test_metrics.json` (generative) and "
                "`outputs/reports/test_metrics_aspect.json` (head-based) across runs."
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
