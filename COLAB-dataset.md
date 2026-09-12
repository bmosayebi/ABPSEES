# آموزش مدل با دیتاست جدید (`project/dataset.py`) روی Colab

این راهنما مدل را **از صفر** با دیتاست synthetic تولیدشده توسط
[`project/dataset.py`](project/dataset.py) (نسخهٔ بدون bias جای‌گاهی، الگوهای
متنوع، جملات hard-negative) آموزش می‌دهد و خروجی را با **نام جدا** ذخیره می‌کند
تا با `model_bundle.zip` قبلی قاطی نشود.

| خروجی | مسیر پیش‌فرض |
|--------|----------------|
| Checkpoint آموزش | `outputs/checkpoints/best/` |
| Bundle جدید (نام جدا) | `outputs/model_bundle_dataset.zip` |
| کپی روی Drive | `/content/drive/MyDrive/ABPSEES/model_bundle_dataset.zip` |

> قبل از شروع: **Runtime → Change runtime type → T4 GPU** و Secret **`HF_TOKEN`** را تنظیم کن.  
> زمان تقریبی آموزش: حدود ۱–۲ ساعت (بسته به GPU و `synthetic_num_samples: 1200`).

---

## سل ۱ — Clone پروژه و نصب وابستگی‌ها

```python
!rm -rf /content/ABPSEES
!git clone https://github.com/bmosayebi/ABPSEES.git /content/ABPSEES
%cd /content/ABPSEES
%pip install -q -r requirements.txt
%pip install -q -e .
```

---

## سل ۲ — Bootstrap، config، HuggingFace، GPU

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
    mount_google_drive=False,  # Drive را فقط موقع ذخیره (سل ۵) mount می‌کنیم
)

print("Root:", ROOT)
print("Colab:", is_colab())
print("GPU:", get_device_config(require_cuda=True).device)
print("synthetic_num_samples:", config.data.synthetic_num_samples)
print("lambda_span:", config.aspect.lambda_span)
print("lambda_disjoint:", getattr(config.aspect, "lambda_disjoint", None))
```

---

## سل ۳ — تولید دیتاست با `project/dataset.py`

دیتاست قدیمی (اگر وجود داشته باشد) پاک می‌شود و از نو با ژنراتور فعلی ساخته می‌شود.

```python
import shutil
from pathlib import Path

from project.dataset import prepare_dataset_splits, load_raw_dataset

# پاک‌سازی دادهٔ قبلی تا حتماً از ژنراتور جدید استفاده شود
shutil.rmtree(config.paths.processed_dir, ignore_errors=True)
config.paths.data_dir.mkdir(parents=True, exist_ok=True)
for f in Path(config.paths.data_dir).glob("*.json"):
    f.unlink()

paths = prepare_dataset_splits(config)
samples = load_raw_dataset(config)

print("Samples:", len(samples))
print("Splits:", {k: str(v) for k, v in paths.items()})
print("Example text:", samples[0]["text"][:180], "...")
print("Example labels keys:", list(samples[0]["labels"].keys()))
```

---

## سل ۴ — آموزش مدل

```python
from project.trainer import train_model

trainer = train_model(config)

print("Adapter:", config.paths.checkpoint_dir / "best")
print("Default bundle (auto):", config.paths.output_dir / "model_bundle.zip")
```

بعد از اتمام، پروژه به‌صورت خودکار هم `outputs/checkpoints/best/` و هم
`outputs/model_bundle.zip` را می‌سازد. در سل بعد، یک **کپی با نام جدا** می‌سازیم.

---

## سل ۵ — ذخیره با نام جدا (`model_bundle_dataset.zip`)

```python
from pathlib import Path
import shutil

from google.colab import files, drive
from project.bundle import save_model_bundle

ADAPTER = config.paths.checkpoint_dir / "best"
assert ADAPTER.exists(), "اول سل ۴ (train_model) را کامل اجرا کن."

# نام جدا — با model_bundle.zip قبلی قاطی نمی‌شود
BUNDLE_NAME = "model_bundle_dataset.zip"
BUNDLE_LOCAL = config.paths.output_dir / BUNDLE_NAME

bundle_path = save_model_bundle(config, ADAPTER, output_path=BUNDLE_LOCAL)
print("Local bundle:", bundle_path, f"({bundle_path.stat().st_size / 1e6:.1f} MB)")

# A) دانلود مستقیم روی لپ‌تاپ
files.download(str(bundle_path))

# B) کپی روی Google Drive
drive.mount("/content/drive")
DRIVE_DIR = Path("/content/drive/MyDrive/ABPSEES")
DRIVE_DIR.mkdir(parents=True, exist_ok=True)
drive_dest = DRIVE_DIR / BUNDLE_NAME
shutil.copy(bundle_path, drive_dest)
print("Drive copy:", drive_dest)
```

---

## سل ۶ — لود مدل جدید و تعریف `ask(text)`

> متن را **حتماً داخل گیومه** بنویس: `ask("...")`.  
> نام تابع را `ask` گذاشته‌ایم (نه `run`) چون در Colab/IPython دستور `%run` با نام `run` قاطی می‌شود.

```python
from pathlib import Path

from project.bundle import load_model_bundle
from project.inference import predict, pretty_print_prediction

BUNDLE = Path("/content/ABPSEES/outputs/model_bundle_dataset.zip")
assert BUNDLE.exists(), f"فایل پیدا نشد: {BUNDLE} — اول سل ۵ را اجرا کن."

print("Loading:", BUNDLE)
_model, _tokenizer, _config = load_model_bundle(BUNDLE)
print("Model ready.")


def ask(text: str) -> dict:
    """متن کاربر را می‌گیرد، پیش‌بینی می‌کند و JSON مرتب چاپ می‌کند."""
    result = predict(text, _config, model=_model, tokenizer=_tokenizer)
    print(pretty_print_prediction(result))
    for warning in result.get("_warnings", []):
        print("[warning]", warning)
    return result
```

---

## سل ۷ — تست چند متن نمونه

```python
ask("من یه لپ تاپ میخام که ارزون و قیمت مناسب باشه. من باید هر روز لپ تاپمو تا محل کارم ببرم. کا خاصی با لپ تاپم نمی‌کنم و اهل بازی یا کارای گرافیکی نیستم.")
```

```python
ask("من یک خانم میانسال هستم و می‌خواهم لپ تاپی بخرم که برای کارهای روزمره مثل دیدن فیلم و گشت و گذار در اینترنت مناسب باشد. انتظار من از یک لپ تاپ این است که خرابی نداشته باشد و بدون نیاز به تعمیر از آن استفاده کنم. ترجیح من خرید یک لپ‌تاپ مقرون به صرفه است.")
```

```python
ask("من دانشجوی مهندسی کامپیوتر هستم و می‌خواهم مدل‌های هوش مصنوعی آموزش بدهم. هر روز لپ‌تاپم را با خودم به دانشگاه می‌برم.")
```

از این به بعد در سل‌های بعدی فقط:

```python
ask("متن فارسی دلخواهت اینجا...")
```

---

## نکته‌ها

| موضوع | توضیح |
|--------|--------|
| منبع دیتاست | ژنراتور `project/dataset.py` (سل ۳ از نو می‌سازد) |
| تعداد نمونه | از `config/colab.yaml` → `data.synthetic_num_samples` (پیش‌فرض ۱۲۰۰) |
| نام bundle جدید | `model_bundle_dataset.zip` |
| bundle قبلی | `model_bundle.zip` دست‌نخورده می‌ماند (مگر خودت overwrite کنی) |
| لود بعدی از Drive | مسیر: `/content/drive/MyDrive/ABPSEES/model_bundle_dataset.zip` |
| inference بدون آموزش | می‌توانی در [`COLAB-RUN.md`](COLAB-RUN.md) مسیر `BUNDLE` را به همین نام جدید عوض کنی |
| مقایسه قبل/بعد | برای متریک‌های کمّی ببین [`COLAB.md` قدم ۱۱](COLAB.md) |

---

## خلاصهٔ مسیرها بعد از آموزش

```text
/content/ABPSEES/outputs/checkpoints/best/          ← LoRA + aspect heads
/content/ABPSEES/outputs/model_bundle.zip           ← bundle خودکار پیش‌فرض
/content/ABPSEES/outputs/model_bundle_dataset.zip   ← bundle با نام جدا (این راهنما)
/content/drive/MyDrive/ABPSEES/model_bundle_dataset.zip
```
