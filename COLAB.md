# راهنمای اجرای ABPSEES روی Google Colab

این راهنما قدم‌به‌قدم نحوه اجرای پروژه **Aspect-Based Preference Scoring + Evidence Extraction** را روی Google Colab با GPU توضیح می‌دهد.

---

## پیش‌نیازها

1. **حساب Google** (برای Colab)
2. **حساب HuggingFace** — [huggingface.co](https://huggingface.co)
3. **Access Token** از HuggingFace:
   - برو به [Settings → Access Tokens](https://huggingface.co/settings/tokens)
   - یک token با دسترسی **Read** بساز
4. **مجوز مدل Qwen** — در صفحه [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) روی «Agree and access repository» کلیک کن

---

## قدم ۱ — فعال‌سازی GPU در Colab

1. به [Google Colab](https://colab.research.google.com) برو
2. یک **Notebook جدید** بساز
3. از منو: **Runtime → Change runtime type**
4. **Hardware accelerator** را روی **T4 GPU** بگذار
5. Save کن

---

## قدم ۲ — آپلود پروژه به Colab

### روش A — Clone از GitHub (پیشنهادی)

اگر پروژه را روی GitHub دارید:

```python
!git clone https://github.com/YOUR_USERNAME/ABPSEES.git /content/ABPSEES
%cd /content/ABPSEES
```

### روش B — آپلود ZIP

1. پوشه `ABPSEES` را zip کن
2. در Colab از Files panel فایل را آپلود کن
3. سپس:

```python
!unzip -q ABPSEES.zip -d /content/
%cd /content/ABPSEES
```

### روش C — Google Drive (برای ذخیره دائمی)

1. پروژه را در Drive کپی کن (مثلاً `MyDrive/ABPSEES`)
2. در Colab:

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/ABPSEES
```

> اگر از Drive استفاده می‌کنی، در `config/colab.yaml` مقدار `mount_google_drive: true` بگذار.

---

## قدم ۳ — نصب وابستگی‌ها

> ⚠️ **مهم:** از `%pip` استفاده کن (نه `!pip`) و نصب را در **سل جدا** از import انجام بده.

**سل ۱ — Clone و نصب:**

```python
# اگر قبلاً clone کرده‌ای، پاک کن:
!rm -rf /content/ABPSEES

!git clone https://github.com/bmosayebi/ABPSEES.git /content/ABPSEES
%cd /content/ABPSEES

# %pip همان Python کرنل Colab را هدف می‌گیرد
%pip install -q -r requirements.txt
%pip install -q -e .
```

**سل ۲ — Bootstrap و import:**

```python
exec(open("/content/ABPSEES/colab_bootstrap.py").read())
ROOT = setup_colab_path()
print("Project root:", ROOT)

from project.config import load_config
from project.utils import setup_colab_environment

config = load_config()
setup_colab_environment("HF_TOKEN")
```

اگر `import project` باز هم خطا داد، **Runtime → Restart session** بزن و فقط **سل ۲** را دوباره اجرا کن (نیازی به clone مجدد نیست).

---

## قدم ۴ — تنظیم HuggingFace Token

### روش Secrets (پیشنهادی)

1. در Colab روی آیکون **🔑 Secrets** (سمت چپ) کلیک کن
2. Secret جدید بساز:
   - **Name:** `HF_TOKEN`
   - **Value:** توکن HuggingFace
3. دسترسی Notebook را فعال کن

### تست توکن

```python
exec(open("/content/ABPSEES/colab_bootstrap.py").read())
setup_colab_path()

from project.config import load_config, is_colab
from project.utils import setup_colab_environment, get_device_config

config = load_config()
setup_colab_environment("HF_TOKEN")

print("Colab:", is_colab())
print("GPU:", get_device_config(require_cuda=True).device)
```

اگر GPU و توکن درست باشند، خطایی نمی‌بینی.

---

## قدم ۵ — آماده‌سازی داده

### گزینه A — داده synthetic (پیش‌فرض)

پروژه خودش ۲۰۰ نمونه فارسی synthetic می‌سازد. نیازی به کار اضافه نیست.

### گزینه B — داده واقعی خودت

1. فایل JSON را در این مسیر بگذار:

```
/content/ABPSEES/data/raw/your_data.json
```

2. در `config/colab.yaml` مسیر را عوض کن:

```yaml
paths:
  data_dir: data/raw
```

فرمت هر نمونه:

```json
{
  "text": "متن فارسی کاربر...",
  "labels": {
    "performance": {"score": 0.9, "evidence": "زیررشته از متن"},
    "portability": {"score": 0.5, "evidence": ""},
    "design": {"score": 0.2, "evidence": ""},
    "durability": {"score": 0.1, "evidence": ""},
    "cost_effectiveness": {"score": 0.3, "evidence": ""}
  }
}
```

---

## قدم ۶ — اجرای نوت‌بوک‌ها

نوت‌بوک‌های پروژه را به ترتیب اجرا کن:

| # | نوت‌بوک | کار |
|---|---------|-----|
| 01 | `01_environment.ipynb` | بررسی GPU و وابستگی‌ها |
| 02 | `02_dataset_analysis.ipynb` | تحلیل داده |
| 03 | `03_preprocessing.ipynb` | اعتبارسنجی evidence و span |
| 04 | `04_training.ipynb` | **آموزش QLoRA** |
| 05 | `05_evaluation.ipynb` | ارزیابی |
| 06 | `06_inference.ipynb` | پیش‌بینی |
| 07 | `07_error_analysis.ipynb` | تحلیل خطا |

### آپلود نوت‌بوک‌ها در Colab

- **File → Upload notebook** → هر نوت‌بوک از پوشه `notebooks/` را آپلود کن
- یا مستقیم از GitHub باز کن

### سل راه‌اندازی مشترک (در ابتدای هر نوت‌بوک)

```python
exec(open("/content/ABPSEES/colab_bootstrap.py").read())
ROOT = setup_colab_path()

from project.config import load_config, is_colab
from project.utils import setup_logging, setup_colab_environment

setup_logging()
config = load_config()
setup_colab_environment(config.colab.hf_token_env)
```

---

## قدم ۷ — آموزش مدل

در `04_training.ipynb` یا مستقیم:

```python
from project.trainer import train_model

trainer = train_model(config)
print("Done! Adapter saved to:", config.paths.checkpoint_dir / "best")
```

**زمان تقریبی:** ۳۰–۹۰ دقیقه (بسته به تعداد epoch و اندازه داده)

---

## قدم ۸ — ذخیره خروجی‌ها

⚠️ **فایل‌های `/content/` بعد از بستن session پاک می‌شوند.**

### روش A — دانلود مستقیم

```python
from google.colab import files
import shutil

shutil.make_archive("/content/abpsees_outputs", "zip", "/content/ABPSEES/outputs")
files.download("/content/abpsees_outputs.zip")
```

### روش B — Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')

!cp -r /content/ABPSEES/outputs /content/drive/MyDrive/ABPSEES/
!cp -r /content/ABPSEES/outputs/checkpoints/best /content/drive/MyDrive/ABPSEES/checkpoints/
```

---

## قدم ۹ — Inference

```python
from project.inference import predict, pretty_print_prediction

text = (
    "من دانشجوی مهندسی کامپیوتر هستم و می خواهم مدل‌های هوش مصنوعی آموزش بدهم. "
    "هر روز لپ‌تاپم را با خودم به دانشگاه می‌برم."
)
result = predict(text, config)
print(pretty_print_prediction(result))
```

---

## ساختار مسیرها در Colab

| مسیر | محتوا |
|------|--------|
| `/content/ABPSEES/` | ریشه پروژه |
| `data/synthetic/` | داده synthetic |
| `data/processed/` | train/val/test splits |
| `outputs/checkpoints/best/` | LoRA adapter آموزش‌دیده |
| `outputs/reports/` | گزارش ارزیابی |
| `outputs/tensorboard/` | لاگ TensorBoard |

---

## رفع مشکلات رایج

| مشکل | راه‌حل |
|------|--------|
| `CUDA GPU is required` | Runtime → Change runtime type → T4 GPU |
| خطای دانلود مدل | Secret `HF_TOKEN` را اضافه کن؛ مجوز Qwen را بپذیر |
| `OutOfMemoryError` | `max_seq_length` را به `1024` کاهش بده یا `per_device_train_batch_size` را `1` کن |
| session قطع شد | checkpoint‌ها را روی Drive ذخیره کن؛ با `resume_from_checkpoint` ادامه بده |
| `bitsandbytes` error | `!pip install -q bitsandbytes` را دوباره اجرا کن |

### ادامه آموزش از checkpoint

```python
from project.trainer import train_model

trainer = train_model(
    config,
    resume_from_checkpoint="/content/ABPSEES/outputs/checkpoints/checkpoint-XXX",
)
```

---

## تنظیمات GPU (پیش‌فرض Colab T4)

| پارامتر | مقدار |
|---------|-------|
| Quantization | 4-bit QLoRA |
| max_seq_length | 2048 |
| batch_size | 2 |
| gradient_accumulation | 4 |
| fp16 | فعال |
| LoRA rank | 16 |

---

## خلاصه سریع (دو سل)

**سل ۱ — نصب:**
```python
!rm -rf /content/ABPSEES
!git clone https://github.com/bmosayebi/ABPSEES.git /content/ABPSEES
%cd /content/ABPSEES
%pip install -q -r requirements.txt
%pip install -q -e .
```

**سل ۲ — آموزش:**
```python
exec(open("/content/ABPSEES/colab_bootstrap.py").read())
setup_colab_path()

from project.config import load_config
from project.utils import setup_colab_environment
from project.trainer import train_model

config = load_config()
setup_colab_environment("HF_TOKEN")
train_model(config)
```

> قبل از اجرا: GPU را فعال کن و Secret `HF_TOKEN` را اضافه کن.
