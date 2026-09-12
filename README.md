# ABPSEES

**Aspect-Based Preference Scoring + Evidence Extraction System**

Fine-tune [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) with **QLoRA** on Google Colab GPU to extract structured JSON from Persian laptop-purchase descriptions.

## Quick Start (Colab)

See **[COLAB.md](COLAB.md)** for the full step-by-step guide in Persian.  
For **inference only** (load a trained bundle from Drive, no training): **[COLAB-RUN.md](COLAB-RUN.md)**.

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
| `08_aspect_attention.ipynb` | Aspect-guided attention demos + ablations |

## Training Stack

| Component | Setting |
|-----------|---------|
| Model | Qwen2.5-3B-Instruct |
| Method | 4-bit QLoRA (bitsandbytes) |
| GPU | Colab T4 (CUDA) |
| Loss | Causal LM cross-entropy (+ optional aspect-guided auxiliary losses, see below) |
| Output | `outputs/checkpoints/best/` |

## Aspect-Guided Attention (optional hybrid module)

When `aspect.enabled: true` in the active config, the model is augmented with
auxiliary aspect-conditioned attention heads (per-aspect score regression +
evidence span extraction + a faithfulness regularizer) on top of the unchanged
generative JSON pipeline. See **[docs/ASPECT_ATTENTION_GUIDE.md](docs/ASPECT_ATTENTION_GUIDE.md)**
for the full scientific write-up, from problem formulation to end-to-end
implementation.

After Colab training, the adapter is packed into a single
`outputs/model_bundle.zip`. Download that file and run it locally:

```bash
python3 run_local.py outputs/model_bundle.zip "متن فارسی کاربر..."
```

See **[COLAB.md](COLAB.md)** for the download + local-inference steps.

## License

MIT
