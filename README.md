# ABPSEES

**Aspect-Based Preference Scoring + Evidence Extraction System**

Fine-tune [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) with **QLoRA** on Google Colab GPU to extract structured JSON from Persian laptop-purchase descriptions.

## Quick Start (Colab)

See **[COLAB.md](COLAB.md)** for the full step-by-step guide in Persian.

```python
# In Colab — after cloning to /content/ABPSEES
import sys
sys.path.insert(0, "/content/ABPSEES")

from project.config import load_config
from project.utils import setup_colab_environment

config = load_config()  # auto-loads colab.yaml
setup_colab_environment(config.colab.hf_token_env)
```

## Pipeline Notebooks

| Notebook | Purpose |
|----------|---------|
| `01_environment.ipynb` | Verify GPU, deps, HuggingFace access |
| `02_dataset_analysis.ipynb` | Explore dataset statistics |
| `03_preprocessing.ipynb` | Validate evidence, build spans |
| `04_training.ipynb` | QLoRA fine-tuning on CUDA |
| `05_evaluation.ipynb` | Score + evidence metrics |
| `06_inference.ipynb` | Interactive prediction |
| `07_error_analysis.ipynb` | Visual error analysis |

## Training Stack

| Component | Setting |
|-----------|---------|
| Model | Qwen2.5-3B-Instruct |
| Method | 4-bit QLoRA (bitsandbytes) |
| GPU | Colab T4 (CUDA) |
| Loss | Standard causal LM cross-entropy |
| Output | `outputs/checkpoints/best/` |

## License

MIT
