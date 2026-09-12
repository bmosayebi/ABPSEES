# راهنمای رابط گرافیکی ABPSEES

برنامهٔ دسکتاپ (Tkinter) برای وارد کردن متن فارسی کاربر و نمایش گرافیکی خروجی مدل.

## محل مدل

برنامه به‌ترتیب این مسیرها را جستجو می‌کند:

1. `ABPSEES/UI/model_bundle.zip`  ← مسیر فعلی شما
2. `ABPSEES/model_bundle.zip`
3. `ABPSEES/outputs/model_bundle.zip`

مسیر دلخواه را هم می‌توانید هنگام اجرا بدهید:

```bash
python app.py /path/to/model_bundle.zip
```

## پیش‌نیازها — محیط مجازی `UI/venv`

```bash
cd /path/to/ABPSEES/UI
python3 -m venv venv          # اگر هنوز نساخته‌اید
source venv/bin/activate
pip install -r requirements.txt
pip install -e ..             # نصب پکیج project از ریشه
```

داخل `requirements.txt` این‌ها هست: torch، transformers، peft، accelerate، arabic-reshaper، python-bidi و بقیهٔ وابستگی‌های inference.

## فونت Peyda

فایل‌های TTF باید اینجا باشند (از روی `fonts/*.woff` ساخته شده‌اند):

```text
fonts/ttf/PeydaWebFaNum-Regular.ttf
fonts/ttf/PeydaWebFaNum-Medium.ttf
fonts/ttf/PeydaWebFaNum-SemiBold.ttf
fonts/ttf/PeydaWebFaNum-Bold.ttf
fonts/ttf/PeydaWebFaNum-ExtraBold.ttf
```

اگر نبود، از ریشهٔ پروژه:

```bash
python3 - <<'PY'
from pathlib import Path
from fontTools.ttLib import TTFont
src, dst = Path("fonts"), Path("fonts/ttf")
dst.mkdir(exist_ok=True)
for woff in src.glob("*.woff"):
    font = TTFont(woff)
    font.flavor = None
    font.save(dst / f"{woff.stem}.ttf")
    print("ok", woff.stem)
PY
```

## اجرا

با venv فعال، از پوشهٔ `UI`:

```bash
cd /path/to/ABPSEES/UI
source venv/bin/activate
python app.py
```

یا از ریشهٔ پروژه:

```bash
source UI/venv/bin/activate
python UI/app.py
```

بار اول، وزن‌های پایهٔ Qwen از Hugging Face دانلود می‌شود (نیاز به اینترنت). روی CPU ممکن است چند دقیقه طول بکشد.

## نمایش فارسی

Tkinter جدول‌های OpenType فونت Peyda را اجرا نمی‌کند؛ برای همین حروف اگر مستقیم در Label/Text گذاشته شوند جدا می‌مانند. برنامه متن را با `arabic-reshaper` به فرم متصل تبدیل می‌کند، با Pillow روی فایل TTF خود Peyda می‌کشد، و همان تصویر را نشان می‌دهد. به مدل همیشه متن منطقی (logical) فرستاده می‌شود.

## کار با برنامه

1. با باز شدن پنجره، مدل در پس‌زمینه بارگذاری می‌شود (۲–۳ دقیقه برای بار اول عادی است). در هدر «مدل آماده است» را ببینید.
2. متن فارسی را در کادر بالا بنویسید.
3. روی **اجرای مدل** کلیک کنید. هر اجرا یک forward از سرهای جنبه است (معمولاً چند ثانیه)، نه تولید ۵۱۲ توکن که روی مک می‌تواند نیم‌ساعت طول بکشد.
4. برای هر جنبه امتیاز، نوار پیشرفت و شواهد نمایش داده می‌شود.
5. متن را عوض کنید و دوباره اجرا کنید؛ مدل در حافظه می‌ماند.

## ساختار پوشهٔ UI

| فایل | نقش |
|------|-----|
| `app.py` | پنجرهٔ اصلی |
| `persian_text.py` | شکل‌دهی اصولی فارسی + ویجت ورودی/خروجی |
| `runner.py` | بارگذاری bundle و inference |
| `widgets.py` | کارت هر جنبه |
| `fonts.py` | ثبت فونت Peyda |
| `theme.py` | رنگ‌ها و برچسب‌های فارسی |
| `requirements.txt` | وابستگی‌های venv |
| `model_bundle.zip` | فایل مدل (اختیاری؛ همین مسیر ترجیح داده می‌شود) |
| `README.md` | همین راهنما |
