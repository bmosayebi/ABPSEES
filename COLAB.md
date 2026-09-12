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
| 08 | `08_aspect_attention.ipynb` | Aspect-Guided Attention (دمو + ablation) |

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
print("Single-file bundle:", config.paths.output_dir / "model_bundle.zip")
```

**زمان تقریبی:** ۳۰–۹۰ دقیقه (بسته به تعداد epoch و اندازه داده)

بعد از اتمام آموزش، پروژه **خودکار** این‌ها را می‌سازد:

| خروجی | مسیر |
|--------|------|
| پوشه LoRA adapter (+ هدهای aspect) | `outputs/checkpoints/best/` |
| **یک فایل قابل دانلود** | `outputs/model_bundle.zip` |

روی Colab، دانلود مرورگر برای `model_bundle.zip` هم خودکار شروع می‌شود. اگر پنجره دانلود را ندیدی، قدم ۸ را اجرا کن.

این فایل zip فقط adapter کوچک LoRA و (در صورت فعال بودن) `aspect_heads.pt` را دارد، نه وزن‌های چندگیگابایتی Qwen. مدل پایه موقع inference از HuggingFace دوباره لود می‌شود.

---

## قدم ۸ — دانلود مدل از Colab

⚠️ **فایل‌های `/content/` بعد از بستن session پاک می‌شوند.** حتماً `model_bundle.zip` را دانلود کن یا روی Drive کپی کن.

### روش A — دانلود همان یک فایل (پیشنهادی)

```python
from pathlib import Path
from google.colab import files

bundle = Path("/content/ABPSEES/outputs/model_bundle.zip")
assert bundle.exists(), "train_model() را اول اجرا کن تا bundle ساخته شود."
print("Bundle size (MB):", round(bundle.stat().st_size / 1e6, 1))
files.download(str(bundle))
```

فایل را جایی روی لپ‌تاپ نگه دار، مثلاً:

```
/Users/webravo/Desktop/ABPSEES/outputs/model_bundle.zip
```

### روش B — کپی روی Google Drive

```python
from google.colab import drive
from pathlib import Path

drive.mount("/content/drive")
dest = Path("/content/drive/MyDrive/ABPSEES")
dest.mkdir(parents=True, exist_ok=True)
!cp /content/ABPSEES/outputs/model_bundle.zip /content/drive/MyDrive/ABPSEES/
print("Copied to", dest / "model_bundle.zip")
```

### روش C — آرشیو همهٔ خروجی‌ها (گزارش‌ها + checkpoint)

اگر علاوه بر مدل، گزارش ارزیابی و TensorBoard را هم می‌خواهی:

```python
from google.colab import files
import shutil

shutil.make_archive("/content/abpsees_outputs", "zip", "/content/ABPSEES/outputs")
files.download("/content/abpsees_outputs.zip")
```

---

## اجرای مدل دانلودشده روی لپ‌تاپ

بعد از دانلود `model_bundle.zip`، روی همین ماشین (بدون Colab) مدل را لود کن، متن فارسی بده و JSON خروجی را ببین.

### پیش‌نیاز لوکال

1. کلون همین ریپو و نصب وابستگی‌ها (یک‌بار):

```bash
cd /Users/webravo/Desktop/ABPSEES
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

2. توکن HuggingFace را در محیط ست کن و مجوز Qwen را در huggingface.co بپذیر:

```bash
export HF_TOKEN="hf_..."
```

3. فایل دانلودشده را مثلاً اینجا بگذار:

```
outputs/model_bundle.zip
```

وزن پایهٔ Qwen2.5-3B اولین بار از Hub دانلود می‌شود (چند گیگ). روی CPU بدون GPU کند است و RAM زیادی می‌خواهد (حدود ۱۲ گیگ برای float32). اگر GPU محلی داری همان اسکریپت CUDA را برمی‌دارد.

### روش A — اسکریپت آماده (پیشنهادی)

یک ورودی و خروج:

```bash
python3 scripts/run_bundle_inference.py \
  --bundle outputs/model_bundle.zip \
  --text "من دانشجوی مهندسی کامپیوتر هستم و می‌خواهم مدل‌های هوش مصنوعی آموزش بدهم. هر روز لپ‌تاپم را با خودم به دانشگاه می‌برم."
```

حالت تعاملی (مدل یک‌بار لود می‌شود؛ چند متن پشت سر هم بده):

```bash
python3 scripts/run_bundle_inference.py --bundle outputs/model_bundle.zip
```

اگر GPU محلی داری: `--cuda` را هم اضافه کن.

### روش B — تکه کد پایتون جدا (`run_local.py`)

فایل `run_local.py` در ریشهٔ ریپو همین کار را می‌کند. مسیر zip و متن را عوض کن، یا مستقیم اجرا کن:

```bash
cd /Users/webravo/Desktop/ABPSEES
python3 run_local.py
# یا:
python3 run_local.py outputs/model_bundle.zip "متن فارسی خودت اینجا"
```

معادل همان فایل:

```python
from pathlib import Path

from project.bundle import load_model_bundle
from project.inference import predict, pretty_print_prediction

BUNDLE = Path("outputs/model_bundle.zip")  # مسیر فایل دانلودشده از Colab

model, tokenizer, config = load_model_bundle(BUNDLE)

text = (
    "من دانشجوی مهندسی کامپیوتر هستم و می‌خواهم مدل‌های هوش مصنوعی آموزش بدهم. "
    "هر روز لپ‌تاپم را با خودم به دانشگاه می‌برم."
)
result = predict(text, config, model=model, tokenizer=tokenizer)
print(pretty_print_prediction(result))
```

خروجی یک JSON با پنج جنبه است (`performance`, `portability`, `design`, `durability`, `cost_effectiveness`)؛ هر کدام `score` و `evidence`. اگر aspect heads فعال باشد، فیلد `_aspect_attention` هم در `result` هست.

برای چند ورودی پشت سر هم، مدل را **یک‌بار** لود کن و فقط `predict` را تکرار کن:

```python
while True:
    text = input("متن ورودی > ").strip()
    if not text:
        break
    result = predict(text, config, model=model, tokenizer=tokenizer)
    print(pretty_print_prediction(result))
```

---

## قدم ۹ — Inference روی Colab (لود از `outputs`)

یک‌بار این سل را اجرا کن تا مدل از `outputs` لود شود و تابع `ask` ساخته شود. بعد فقط متن بده.

> ⚠️ متن را **حتماً داخل گیومه** بنویس: `ask("...")`.  
> نام تابع را `ask` گذاشته‌ایم (نه `run`) چون در Colab/IPython دستور `%run` با نام `run` قاطی می‌شود و متن فارسی را به‌اشتباه به‌عنوان نام فایل `.py` می‌گیرد.

```python
from pathlib import Path

from project.bundle import load_model_bundle
from project.inference import predict, pretty_print_prediction

OUTPUTS = Path("/content/ABPSEES/outputs")
BUNDLE = OUTPUTS / "model_bundle.zip"
CHECKPOINT = OUTPUTS / "checkpoints" / "best"

if BUNDLE.exists():
    _model, _tokenizer, _config = load_model_bundle(BUNDLE)
    print("Loaded:", BUNDLE)
elif CHECKPOINT.exists():
    from project.config import load_config
    from project.hybrid_model import load_hybrid_model_for_inference

    _config = load_config()
    _model, _tokenizer = load_hybrid_model_for_inference(_config, str(CHECKPOINT))
    print("Loaded:", CHECKPOINT)
else:
    raise FileNotFoundError(
        "خروجی مدل در outputs پیدا نشد. اول train_model را اجرا کن."
    )


def ask(text: str) -> dict:
    """متن کاربر را می‌گیرد، پیش‌بینی می‌کند و JSON مرتب چاپ می‌کند."""
    result = predict(text, _config, model=_model, tokenizer=_tokenizer)
    print(pretty_print_prediction(result))
    for warning in result.get("_warnings", []):
        print("[warning]", warning)
    return result


# مثال — گیومه الزامی است:
ask("من دانشجوی مهندسی کامپیوتر هستم و هر روز لپ‌تاپم را به دانشگاه می‌برم.")
```

از این به بعد در سل‌های بعدی فقط:

```python
ask("من یه لپ‌تاپ می‌خوام که ارزون و قیمت مناسب باشه. هر روز باید با خودم ببرمش محل کار.")
```

---

## ساختار مسیرها در Colab

| مسیر | محتوا |
|------|--------|
| `/content/ABPSEES/` | ریشه پروژه |
| `data/synthetic/` | داده synthetic |
| `data/processed/` | train/val/test splits |
| `outputs/checkpoints/best/` | LoRA adapter آموزش‌دیده |
| `outputs/model_bundle.zip` | **یک فایل** برای دانلود و اجرای لوکال |
| `outputs/reports/` | گزارش ارزیابی |
| `outputs/tensorboard/` | لاگ TensorBoard |

---

## رفع مشکلات رایج

| مشکل | راه‌حل |
|------|--------|
| `CUDA GPU is required` | Runtime → Change runtime type → T4 GPU |
| خطای دانلود مدل | Secret `HF_TOKEN` را اضافه کن؛ مجوز Qwen را بپذیر |
| `OutOfMemoryError` | `max_seq_length` را به `1024` کاهش بده یا `per_device_train_batch_size` را `1` کن |
| session قطع شد | `model_bundle.zip` یا checkpoint‌ها را روی Drive ذخیره کن؛ با `resume_from_checkpoint` ادامه بده |
| `model_bundle.zip` پیدا نشد | اول `train_model(config)` را تمام کن؛ فایل در `outputs/model_bundle.zip` است |
| inference لوکال خیلی کند / RAM کم | مدل ۳B روی CPU سنگین است؛ GPU محلی یا همان Colab را برای inference استفاده کن |
| `NameError: name '__file__' is not defined` | `colab_bootstrap.py` را آپدیت کن (یا دوباره `git clone` / `git pull` بزن) سپس سل bootstrap را دوباره اجرا کن |
| `Some tensors share memory` / `lm_head.weight` + `embed_tokens.weight` | با نسخهٔ جدید `AspectGuidedTrainer._save` رفع شده؛ `git pull` بزن و آموزش را از اول یا از آخرین checkpoint معتبر ادامه بده |
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

## قدم ۱۰ — Aspect-Guided Attention (اختیاری)

اگر در `config/colab.yaml` مقدار `aspect.enabled: true` باشد (پیش‌فرض)،
مدل علاوه بر خروجی JSON، هدهای کمکی attention/امتیاز/evidence هم دارد.
راهنمای کامل علمی: **[docs/ASPECT_ATTENTION_GUIDE.md](docs/ASPECT_ATTENTION_GUIDE.md)**.
دمو و ablation در `08_aspect_attention.ipynb`.

```python
from project.hybrid_model import load_hybrid_model_for_inference, run_aspect_heads_on_text

hybrid_model, tokenizer = load_hybrid_model_for_inference(config)
diag = run_aspect_heads_on_text(hybrid_model, tokenizer, text, config)
print(diag["per_aspect"])
```

---

## قدم ۱۱ — مقایسه قبل/بعد از رفع مشکل Evidence Misattribution

اگر برای رفع مشکل «evidence اشتباه به aspect غلط نسبت داده می‌شود» طبق
**[docs/ASPECT_ATTENTION_GUIDE.md §4.1/§5.7](docs/ASPECT_ATTENTION_GUIDE.md#41-cross-aspect-disjointness-the-evidence-misattribution-failure-mode)**
دوباره train می‌کنی (دیتاست جدید + `lambda_disjoint` + `lambda_span` بالاتر)،
این سل‌ها یک مقایسهٔ کمّی قبل/بعد روی همان تقسیم `test` می‌سازند.

### سل ۱ — اسنپ‌شات «قبل» (روی مدل فعلی، قبل از retrain)

اگر از قبل یک `outputs/checkpoints/best` (نسخهٔ قدیمی، آموزش‌دیده با کد/دیتاست قبلی) داری، این را همین حالا اجرا کن — قبل از این‌که دیتاست را دوباره بسازی یا دوباره train کنی:

```python
from pathlib import Path

from project.config import load_config
from project.evaluation import run_evaluation
from project.utils import load_json, save_json

_config = load_config()
_reports = _config.paths.reports_dir
_reports.mkdir(parents=True, exist_ok=True)

run_evaluation(_config, split="test")  # می‌نویسد: outputs/reports/test_metrics_aspect.json

# کپی با پسوند before تا سل بعدی (retrain) آن را overwrite نکند
before_metrics = load_json(_reports / "test_metrics_aspect.json")
before_preds = load_json(_reports / "test_predictions_aspect.json")
save_json(before_metrics, _reports / "test_metrics_aspect_before.json")
save_json(before_preds, _reports / "test_predictions_aspect_before.json")
print("Saved BEFORE snapshot:", _reports / "test_metrics_aspect_before.json")
```

### سل ۲ — بازسازی دیتاست + retrain

دیتاست را با ژنراتور جدید (بدون bias جای‌گاهی، الگوهای متنوع‌تر، جملات hard-negative) دوباره بساز و مدل را دوباره train کن — کدهای قدم ۵ و قدم ۷ همین بالا را دوباره اجرا کن (کانفیگ `config/colab.yaml` از قبل به‌روز است: `synthetic_num_samples: 1200`, `lambda_span: 0.8`, `lambda_disjoint: 0.2`):

```python
import shutil
from pathlib import Path

from project.config import load_config
from project.dataset import prepare_dataset_splits

_config = load_config()

# دیتاست synthetic قدیمی را پاک کن تا با ژنراتور جدید از صفر ساخته شود
shutil.rmtree(_config.paths.processed_dir, ignore_errors=True)
for f in Path(_config.paths.data_dir).glob("*.json"):
    f.unlink()

prepare_dataset_splits(_config)
print("Dataset regenerated:", _config.data.synthetic_num_samples, "samples")

# بعد از این، سل‌های «قدم ۷ — آموزش مدل» را دوباره اجرا کن.
```

### سل ۳ — اسنپ‌شات «بعد» (روی مدل تازه retrain‌شده)

بعد از این‌که آموزش قدم ۷ با موفقیت تمام شد:

```python
from project.config import load_config
from project.evaluation import run_evaluation
from project.utils import load_json, save_json

_config = load_config()
_reports = _config.paths.reports_dir

run_evaluation(_config, split="test")

after_metrics = load_json(_reports / "test_metrics_aspect.json")
after_preds = load_json(_reports / "test_predictions_aspect.json")
save_json(after_metrics, _reports / "test_metrics_aspect_after.json")
save_json(after_preds, _reports / "test_predictions_aspect_after.json")
print("Saved AFTER snapshot:", _reports / "test_metrics_aspect_after.json")
```

### سل ۴ — مقایسهٔ کمّی قبل/بعد

جدول per-aspect span F1/EM + faithfulness rates، به‌همراه یک متریک اختصاصی
برای همین باگ — **نرخ هم‌پوشانی evidence بین دو aspect مختلف** (چند درصد
نمونه‌ها حداقل یک جفت aspect دارند که span‌های پیش‌بینی‌شده‌شان با هم
overlap می‌کنند؛ هر چه این عدد بعد از fix کمتر شود، یعنی مشکل misattribution
واقعاً حل شده):

```python
from itertools import combinations

from project.config import load_config
from project.constants import ASPECTS
from project.metrics import token_span_to_set
from project.utils import load_json

_reports = load_config().paths.reports_dir

before = load_json(_reports / "test_metrics_aspect_before.json")
after = load_json(_reports / "test_metrics_aspect_after.json")
before_preds = load_json(_reports / "test_predictions_aspect_before.json")
after_preds = load_json(_reports / "test_predictions_aspect_after.json")


def cross_aspect_overlap_rate(predictions: list[dict]) -> float:
    """% of samples with >=1 pair of aspects whose predicted spans overlap."""
    flagged = 0
    for sample in predictions:
        spans = {
            a: sample["per_aspect"][a].get("span_token")
            for a in ASPECTS
            if sample["per_aspect"][a].get("span_token") is not None
        }
        overlapping = False
        for a1, a2 in combinations(spans, 2):
            if token_span_to_set(spans[a1]) & token_span_to_set(spans[a2]):
                overlapping = True
                break
        flagged += int(overlapping)
    return flagged / max(len(predictions), 1)


print(f"{'Aspect':<20}{'F1 before':>12}{'F1 after':>12}{'EM before':>12}{'EM after':>12}")
for aspect in ASPECTS:
    b = before["span_metrics"][aspect]
    a = after["span_metrics"][aspect]
    print(
        f"{aspect:<20}{b['f1']:>12.4f}{a['f1']:>12.4f}"
        f"{b['exact_match']:>12.4f}{a['exact_match']:>12.4f}"
    )

print()
print("Cross-aspect evidence overlap rate:")
print("  before:", f"{cross_aspect_overlap_rate(before_preds):.4f}")
print("  after: ", f"{cross_aspect_overlap_rate(after_preds):.4f}")
```

سپس دو مثال گزارش‌شدهٔ اصلی (لپ‌تاپ ارزون/روزمره، و لپ‌تاپ بی‌خرابی/مقرون‌به‌صرفه) را دستی با `ask(...)` (قدم ۹) دوباره تست کن تا مطمئن شوی `cost_effectiveness`/`durability` دیگر evidence اشتباه از aspectهای دیگر نمی‌گیرند.

---

## فقط inference (بدون آموزش)

اگر مدل را روی Drive گذاشته‌ای و فقط می‌خواهی `ask("...")` بزنی، راهنمای جدا را ببین:

**[COLAB-RUN.md](COLAB-RUN.md)** — لود از `/content/drive/MyDrive/ABPSEES/model_bundle.zip`

---

## خلاصه سریع — همهٔ سل‌های Colab (از صفر تا ذخیره مدل)

> قبل از اجرا: **Runtime → T4 GPU** و Secret **`HF_TOKEN`** را تنظیم کن.

**سل ۱ — Clone و نصب وابستگی‌ها**

```python
!rm -rf /content/ABPSEES
!git clone https://github.com/bmosayebi/ABPSEES.git /content/ABPSEES
%cd /content/ABPSEES
%pip install -q -r requirements.txt
%pip install -q -e .
```

**سل ۲ — Bootstrap، config، HuggingFace**

```python
exec(open("/content/ABPSEES/colab_bootstrap.py").read())
ROOT = setup_colab_path()

from project.config import load_config, is_colab
from project.utils import setup_logging, setup_colab_environment, get_device_config

setup_logging()
config = load_config()
setup_colab_environment(
    hf_token_env=config.colab.hf_token_env,
    use_colab_secrets=config.colab.use_colab_secrets,
    mount_google_drive=config.colab.mount_google_drive,
)

print("Root:", ROOT)
print("Colab:", is_colab())
print("GPU:", get_device_config(require_cuda=True).device)
```

**سل ۳ — آموزش مدل**

```python
from project.trainer import train_model

trainer = train_model(config)

# خروجی‌های خودکار بعد از آموزش:
#   /content/ABPSEES/outputs/checkpoints/best/   ← LoRA + aspect heads
#   /content/ABPSEES/outputs/model_bundle.zip    ← یک فایل قابل دانلود
print("Adapter:", config.paths.checkpoint_dir / "best")
print("Bundle: ", config.paths.output_dir / "model_bundle.zip")
```

**سل ۴ — ذخیره مدل در آدرس مشخص (دانلود + Drive)**

```python
from pathlib import Path
from google.colab import files, drive
import shutil

BUNDLE = Path("/content/ABPSEES/outputs/model_bundle.zip")
assert BUNDLE.exists(), "اول سل ۳ (train_model) را کامل اجرا کن."

# A) دانلود مستقیم روی لپ‌تاپ
files.download(str(BUNDLE))

# B) کپی روی Google Drive (مسیر دائمی)
drive.mount("/content/drive")
DRIVE_DIR = Path("/content/drive/MyDrive/ABPSEES")
DRIVE_DIR.mkdir(parents=True, exist_ok=True)
shutil.copy(BUNDLE, DRIVE_DIR / "model_bundle.zip")
print("Saved to:", DRIVE_DIR / "model_bundle.zip")
```

**سل ۵ — لود مدل از `outputs` و تعریف `ask(text)`**

> متن را داخل گیومه بنویس. از نام `run` استفاده نکن (با `%run` در Colab قاطی می‌شود).

```python
from pathlib import Path

from project.bundle import load_model_bundle
from project.inference import predict, pretty_print_prediction

OUTPUTS = Path("/content/ABPSEES/outputs")
BUNDLE = OUTPUTS / "model_bundle.zip"
CHECKPOINT = OUTPUTS / "checkpoints" / "best"

if BUNDLE.exists():
    _model, _tokenizer, _config = load_model_bundle(BUNDLE)
    print("Loaded:", BUNDLE)
elif CHECKPOINT.exists():
    from project.config import load_config
    from project.hybrid_model import load_hybrid_model_for_inference

    _config = load_config()
    _model, _tokenizer = load_hybrid_model_for_inference(_config, str(CHECKPOINT))
    print("Loaded:", CHECKPOINT)
else:
    raise FileNotFoundError(
        "خروجی مدل در outputs پیدا نشد. اول سل ۳ (train_model) را اجرا کن."
    )


def ask(text: str) -> dict:
    """متن کاربر را می‌گیرد، پیش‌بینی می‌کند و JSON مرتب چاپ می‌کند."""
    result = predict(text, _config, model=_model, tokenizer=_tokenizer)
    print(pretty_print_prediction(result))
    for warning in result.get("_warnings", []):
        print("[warning]", warning)
    return result


ask("من دانشجوی مهندسی کامپیوتر هستم و هر روز لپ‌تاپم را به دانشگاه می‌برم.")
```

بعد فقط صدا بزن (با گیومه):

```python
ask("من یه لپ‌تاپ می‌خوام که ارزون و قیمت مناسب باشه. هر روز باید با خودم ببرمش محل کار.")
```

بعد از دانلود، روی لپ‌تاپ:

```bash
cd /Users/webravo/Desktop/ABPSEES
python3 run_local.py outputs/model_bundle.zip "متن فارسی کاربر..."
```
