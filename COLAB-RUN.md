# راهنمای اجرای مدل آموزش‌دیده روی Colab (بدون آموزش)

این فایل فقط برای **لود مدل از قبل آموزش‌دیده** و تست چندباره با تابع `ask` است.  
آموزش (`train_model`) اینجا اجرا نمی‌شود.

مدل از این مسیر خوانده می‌شود:

```text
/content/drive/MyDrive/ABPSEES/model_bundle.zip
```

> قبل از شروع: **Runtime → Change runtime type → T4 GPU** و Secret **`HF_TOKEN`** را تنظیم کن.

---

## سل ۱ — اتصال به Google Drive

```python
from google.colab import drive

drive.mount("/content/drive")

from pathlib import Path

BUNDLE = Path("/content/drive/MyDrive/ABPSEES/model_bundle.zip")
assert BUNDLE.exists(), f"فایل مدل پیدا نشد: {BUNDLE}"
print("Bundle OK:", BUNDLE, f"({BUNDLE.stat().st_size / 1e6:.1f} MB)")
```

---

## سل ۲ — Clone پروژه و نصب وابستگی‌ها (بدون آموزش)

```python
!rm -rf /content/ABPSEES
!git clone https://github.com/bmosayebi/ABPSEES.git /content/ABPSEES
%cd /content/ABPSEES
%pip install -q -r requirements.txt
%pip install -q -e .
```

---

## سل ۳ — Bootstrap، HuggingFace، بررسی GPU

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
    mount_google_drive=False,  # Drive را در سل ۱ جداگانه mount کرده‌ایم
)

print("Root:", ROOT)
print("Colab:", is_colab())
print("GPU:", get_device_config(require_cuda=True).device)
```

---

## سل ۴ — لود مدل از Drive و تعریف `ask(text)`

مدل **یک‌بار** از آدرس ثابت Drive لود می‌شود. بعد فقط `ask("...")` را صدا بزن.

> متن را **حتماً داخل گیومه** بنویس. از نام `run` استفاده نکن (با `%run` در Colab قاطی می‌شود).

```python
from pathlib import Path

from project.bundle import load_model_bundle
from project.inference import predict, pretty_print_prediction

BUNDLE = Path("/content/drive/MyDrive/ABPSEES/model_bundle.zip")
assert BUNDLE.exists(), f"فایل مدل پیدا نشد: {BUNDLE}"

print("Loading model from:", BUNDLE)
_model, _tokenizer, _config = load_model_bundle(BUNDLE)
print("Model ready.")


def ask(text: str) -> dict:
    """متن فارسی کاربر را می‌گیرد، پیش‌بینی می‌کند و JSON مرتب چاپ می‌کند."""
    result = predict(text, _config, model=_model, tokenizer=_tokenizer)
    print(pretty_print_prediction(result))
    for warning in result.get("_warnings", []):
        print("[warning]", warning)
    return result
```

---

## سل ۵ — صدا زدن `ask` (هر چند بار که بخواهی)

هر بار فقط یک سلول جدید با متن دلخواهت اجرا کن. مدل دوباره لود نمی‌شود.

```python
ask("من یه لپ‌تاپ می‌خوام که ارزون و قیمت مناسب باشه. هر روز باید با خودم ببرمش محل کار. کار خاصی با لپ‌تاپم نمی‌کنم و اهل بازی یا کارهای گرافیکی نیستم.")
```

```python
ask("من دانشجوی مهندسی کامپیوتر هستم و می‌خواهم مدل‌های هوش مصنوعی آموزش بدهم. هر روز لپ‌تاپم را با خودم به دانشگاه می‌برم.")
```

```python
ask("بیشتر به ظاهر و طراحی لپ‌تاپ اهمیت می‌دهم و بودجه محدودی ندارم.")
```

---

## نکته‌ها

| موضوع | توضیح |
|--------|--------|
| مسیر مدل | همیشه ` /content/drive/MyDrive/ABPSEES/model_bundle.zip ` |
| آموزش | در این راهنما انجام نمی‌شود |
| GPU | برای سرعت بیشتر T4 لازم است؛ بدون GPU کند است |
| HF Token | برای دانلود وزن پایهٔ Qwen لازم است |
| چندبار تست | فقط `ask("...")` را تکرار کن؛ سل ۴ را دوباره نزن مگر session قطع شده باشد |
| قطع session | Drive را دوباره mount کن و از سل ۳–۴ ادامه بده (clone اگر باقی مانده باشد لازم نیست) |

اگر پروژه از قبل در `/content/ABPSEES` هست و وابستگی‌ها نصب‌اند، می‌توانی سل ۲ را رد کنی و از سل ۱ + ۳ + ۴ شروع کنی.

---

## مقایسهٔ کمّی قبل/بعد از رفع مشکل evidence

این فایل فقط برای inference دستی با `ask(...)` است و متریک/gold label ندارد.
برای مقایسهٔ عددی دقیق (span F1/EM per-aspect، نرخ هم‌پوشانی evidence بین
aspectها، و غیره) قبل و بعد از retrain با دیتاست/loss جدید، به
**[COLAB.md - قدم ۱۱](COLAB.md#قدم-۱۱--مقایسه-قبلبعد-از-رفع-مشکل-evidence-misattribution)**
مراجعه کن — آن سل‌ها روی مجموعهٔ `test` واقعی (با labelهای گلد) اجرا می‌شوند،
نه فقط چند نمونهٔ دستی.
