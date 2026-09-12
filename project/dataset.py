"""Dataset loading, validation, splitting, and synthetic data generation."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

from sklearn.model_selection import train_test_split

from project.config import AppConfig, validate_sample_scores
from project.constants import ASPECTS
from project.utils import load_json, save_json

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Synthetic Persian laptop-preference generator.
#
# Design notes (see docs/ASPECT_ATTENTION_GUIDE.md and the evidence-
# misattribution investigation for the full rationale):
#
# 1. No aspect is structurally privileged. Earlier versions always placed a
#    "performance" clause in the same fixed opening slot of every sample and
#    always gave it non-empty evidence -- the model could then learn a
#    *positional* shortcut ("first clause = performance") instead of a
#    semantic one. Every aspect below (including performance) is now
#    included independently at random, and the chosen clauses are shuffled
#    before being joined into the final text, so position never correlates
#    with aspect identity.
# 2. Wide phrasing diversity per aspect, mixing formal and colloquial
#    register (e.g. "می‌خواهم" and "میخوام"/"می‌خوام"), so the span head sees
#    many ways of expressing the same concept instead of memorizing a
#    handful of fixed template sentences.
# 3. A small set of explicit "combo" sentences pack two aspects' concepts
#    into one sentence with clearly separable sub-spans (hard negatives),
#    forcing the model to disambiguate content instead of relying on
#    sentence-level position.
# ---------------------------------------------------------------------------

_INTRO_TEMPLATES: list[str] = [
    "من {role} هستم.",
    "به عنوان {role} کار می‌کنم.",
    "شغلم {role} است.",
    "من یک {role} هستم و دنبال لپ‌تاپ جدید می‌گردم.",
    "من {role} هستم و می‌خوام یه لپ‌تاپ جدید بخرم.",
]

_ROLES: list[str] = [
    "دانشجوی مهندسی کامپیوتر",
    "برنامه‌نویس",
    "گیم‌پلیر",
    "طراح گرافیک",
    "کارمند اداری",
    "فریلنسر",
    "دانشجوی پزشکی",
    "عکاس",
    "معلم",
    "حسابدار",
    "خانم خانه‌دار",
    "بازنشسته",
    "دانشجوی هنر",
    "مهندس عمران",
    "وکیل",
    "پرستار",
]

# 15-25+ varied sentences per aspect, mixing formal and colloquial
# (spoken-register) Persian. Each string is a complete clause without a
# trailing period; the generator appends punctuation when joining.
_ASPECT_CLAUSES: dict[str, list[str]] = {
    "performance": [
        "می‌خواهم عملکرد بالایی برای کارهای سنگین داشته باشد",
        "پردازنده و رم قوی برایم مهم است",
        "باید بتونه بازی‌های سنگین رو خوب اجرا کنه",
        "سرعتش برام مهمه، نمی‌خوام کند باشه",
        "دنبال لپ‌تاپی هستم که در اجرای برنامه‌های سنگین کند نشود",
        "می‌خوام مدل‌های هوش مصنوعی رو روش آموزش بدم، پس قدرت پردازش زیاد لازم دارم",
        "کارایی بالا برام از همه چیز مهم‌تره",
        "باید بتونه چند برنامه رو همزمان بدون افت سرعت اجرا کنه",
        "نرم‌افزارهای سنگین ویرایش ویدیو رو باید راحت اجرا کنه",
        "توی رندر و پردازش گرافیکی نباید کند باشه",
        "کدهای سنگین را بدون کندی اجرا کند",
        "بازی‌های جدید را با گرافیک بالا اجرا کند",
        "نرم‌افزارهای طراحی را روان اجرا کند",
        "برام مهمه که موقع باز بودن چند تب و برنامه هنگ نکنه",
        "قدرت پردازشش باید بالا باشه",
        "می‌خوام سریع بوت بشه و کند نباشه",
        "برای کارهای گرافیکی سنگین قدرت پردازنده مهمه",
        "نمی‌خوام موقع اجرای برنامه‌های سنگین کند و داغ کنه",
        "دنبال یه لپ‌تاپ قدرتمند برای تحلیل داده هستم",
        "کارت گرافیک قوی برام ضروریه",
    ],
    "portability": [
        "هر روز با اتوبوس به دانشگاه می‌روم و لپ‌تاپم را با خودم حمل می‌کنم",
        "سفرهای کاری زیادی دارم و لپ‌تاپ باید سبک و قابل حمل باشد",
        "اغلب در کافه کار می‌کنم و لپ‌تاپ را همیشه با خودم می‌برم",
        "بین خانه و محل کار جابه‌جا می‌شوم و وزن لپ‌تاپ برایم مهم است",
        "لپ‌تاپم باید سبک باشه چون هر روز با خودم می‌برمش",
        "دنبال یه لپ‌تاپ نازک و سبک هستم که راحت جا بشه تو کیفم",
        "زیاد این‌ور و اون‌ور می‌برمش، پس نباید سنگین باشه",
        "باید بتونم راحت با خودم ببرمش سر کار",
        "دنبال یه لپ تاپ کوچیک و سبک هستم",
        "زیاد سفر می‌رم و نمی‌خوام لپ‌تاپم سنگین باشه",
        "هر روز باید لپ‌تاپمو با خودم ببرم محل کار",
        "می‌خوام راحت تو کیفم جا بشه",
        "وزنش برام مهمه چون هر روز حملش می‌کنم",
        "دنبال یه گزینه کم‌حجم و قابل‌حمل هستم",
        "باید بشه راحت با خودم این‌ور اون‌ور ببرمش",
        "اندازه و وزنش باید مناسب سفر باشه",
    ],
    "design": [
        "ظاهر لپ‌تاپ برایم اهمیت دارد و دوست دارم طراحی مدرن و زیبایی داشته باشد",
        "رنگ و طراحی بدنه لپ‌تاپ در انتخاب من تأثیرگذار است",
        "دوست دارم لپ‌تاپم ظاهر حرفه‌ای و شیکی داشته باشد",
        "برام مهمه که ظاهرش شیک و امروزی باشه",
        "دنبال یه لپ‌تاپ خوش‌رنگ و خوش‌طرح هستم",
        "دوست دارم بدنه‌ش فلزی و شیک باشه",
        "ظاهرش برام مهمه، می‌خوام جلوی بقیه خوب دیده بشه",
        "طراحی باریک و مینیمال رو ترجیح می‌دم",
        "دنبال یه لپ‌تاپ خوش‌ظاهر برای محیط کارم هستم",
        "دوست دارم رنگ و ظاهرش با استایلم هماهنگ باشه",
        "زیبایی ظاهری لپ‌تاپ برام مهمه",
        "می‌خوام لپ‌تاپم خوش‌تیپ و شیک باشه",
        "طراحی بدنه و کیفیت ساختش برام مهمه",
        "دنبال یه ظاهر ساده و شیک هستم",
        "دوست دارم لپ‌تاپم جلب توجه کنه از نظر ظاهری",
    ],
    "durability": [
        "لپ‌تاپ قبلی‌ام بعد از دو سال خراب شد و می‌خواهم این بار محکم‌تر باشد",
        "محیط کارم پر از گرد و غبار است و به بدنه مقاوم نیاز دارم",
        "می‌خواهم لپ‌تاپم چند سال بدون مشکل کار کند",
        "انتظار من از یک لپ تاپ این است که خرابی نداشته باشد",
        "نمی‌خوام مدام خراب بشه و بره تعمیرگاه",
        "دنبال یه لپ‌تاپ محکم و بادوام هستم",
        "می‌خوام سال‌های سال بدون خرابی کار کنه",
        "بدنه‌ش باید در برابر ضربه مقاوم باشه",
        "نمی‌خوام زود خراب بشه و پول تعمیرش رو بدم",
        "دوام و طول عمر باتری و قطعاتش برام مهمه",
        "بدون نیاز به تعمیر می‌خوام ازش استفاده کنم",
        "کیفیت ساخت و مقاومتش برام مهم‌تر از هر چیزیه",
        "می‌خوام لپ‌تاپی بخرم که زود از کار نیفته",
        "استحکام بدنه و کیبوردش برام مهمه",
        "نمی‌خوام هر چند وقت یک‌بار برای تعمیر ببرمش",
    ],
    "cost_effectiveness": [
        "بودجه محدودی دارم و بهترین ارزش را در ازای پول می‌خواهم",
        "نمی‌خواهم بیش از حد هزینه کنم ولی کیفیت مناسب می‌خواهم",
        "به دنبال گزینه‌ای مقرون‌به‌صرفه هستم",
        "ارزون و قیمت مناسب باشه",
        "قیمتش باید مناسب باشه، پول زیادی برای خرید لپ‌تاپ ندارم",
        "دنبال یه گزینه ارزون هستم که زیاد به جیبم فشار نیاره",
        "بودجه‌ام کمه، می‌خوام لپ‌تاپ ارزون‌قیمت بخرم",
        "ترجیح من خرید یک لپ‌تاپ مقرون به صرفه است",
        "نمی‌خوام گزینه گرون قیمت بخرم",
        "دنبال بهترین قیمت به نسبت کیفیت هستم",
        "پول زیادی برای این کار کنار نگذاشتم",
        "می‌خوام با کمترین هزینه بهترین گزینه رو بخرم",
        "قیمت برام از هر چیزی مهم‌تره",
        "دنبال یه لپ تاپ ارزون هستم",
        "بودجه‌م محدوده و نمی‌تونم گزینه گرون بخرم",
        "به دنبال کالایی با نسبت قیمت به کیفیت خوب هستم",
        "نمی‌خوام برای این خرید خیلی هزینه کنم",
    ],
}

# Hard-negative "combo" sentences: two aspects' concepts appear next to each
# other in a single sentence, with clearly disjoint sub-spans, so the model
# must key off content -- not position -- to tell them apart. Each ``spans``
# value must be an exact (findable) substring of ``text``.
_COMBO_CLAUSES: list[dict[str, Any]] = [
    {
        "text": "کارایی بالا برام مهمه، ولی قیمتش هم باید مناسب باشه",
        "spans": {
            "performance": "کارایی بالا برام مهمه",
            "cost_effectiveness": "قیمتش هم باید مناسب باشه",
        },
    },
    {
        "text": "میخوام هم سریع باشه هم زیاد گرون نباشه",
        "spans": {
            "performance": "سریع باشه",
            "cost_effectiveness": "زیاد گرون نباشه",
        },
    },
    {
        "text": "دنبال لپ‌تاپی هستم که هم بادوام باشه هم قیمت مناسبی داشته باشه",
        "spans": {
            "durability": "بادوام باشه",
            "cost_effectiveness": "قیمت مناسبی داشته باشه",
        },
    },
    {
        "text": "می‌خوام هم ظاهر شیکی داشته باشه هم زیر بار سنگین کم نیاره",
        "spans": {
            "design": "ظاهر شیکی داشته باشه",
            "performance": "زیر بار سنگین کم نیاره",
        },
    },
    {
        "text": "هم باید سبک باشه هم زیاد گرون نباشه",
        "spans": {
            "portability": "سبک باشه",
            "cost_effectiveness": "زیاد گرون نباشه",
        },
    },
    {
        "text": "هم بادوام باشه هم ارزون باشه",
        "spans": {
            "durability": "بادوام باشه",
            "cost_effectiveness": "ارزون باشه",
        },
    },
    {
        "text": "می‌خوام هم قدرت پردازش خوبی داشته باشه هم زیاد سنگین و بزرگ نباشه",
        "spans": {
            "performance": "قدرت پردازش خوبی داشته باشه",
            "portability": "زیاد سنگین و بزرگ نباشه",
        },
    },
    {
        "text": "دنبال لپ‌تاپی هستم که هم ظاهر خوبی داشته باشه هم زیاد گرون نباشه",
        "spans": {
            "design": "ظاهر خوبی داشته باشه",
            "cost_effectiveness": "زیاد گرون نباشه",
        },
    },
]

# Probability that each aspect independently gets a clause in a sample.
_ASPECT_INCLUDE_PROB = 0.6
# Probability of splicing in one hard-negative combo sentence per sample.
_COMBO_PROB = 0.25


def _sample_score(rng: random.Random, has_evidence: bool) -> float:
    """Sample a relevance score, biased high when evidence is expected."""
    if has_evidence:
        return round(rng.uniform(0.55, 0.98), 2)
    return round(rng.uniform(0.05, 0.30), 2)


def generate_synthetic_sample(rng: random.Random) -> dict[str, Any]:
    """Generate a single synthetic labeled sample with valid evidence spans.

    No aspect (including ``performance``) is structurally privileged: every
    aspect is independently included at random, clause order is shuffled,
    and a chance hard-negative "combo" sentence packs two aspects together
    to force content-based (not positional) disambiguation.

    Args:
        rng: Random number generator instance.

    Returns:
        Sample dict with ``text`` and ``labels`` keys.
    """
    role = rng.choice(_ROLES)
    text = rng.choice(_INTRO_TEMPLATES).format(role=role)

    evidence_by_aspect: dict[str, str] = {}
    sentences: list[str] = []

    if rng.random() < _COMBO_PROB:
        combo = rng.choice(_COMBO_CLAUSES)
        sentences.append(combo["text"])
        for aspect, span_text in combo["spans"].items():
            evidence_by_aspect[aspect] = span_text

    aspect_included = {
        aspect: (aspect in evidence_by_aspect) or (rng.random() < _ASPECT_INCLUDE_PROB)
        for aspect in ASPECTS
    }
    if not any(aspect_included.values()):
        aspect_included[rng.choice(ASPECTS)] = True

    for aspect in ASPECTS:
        if aspect in evidence_by_aspect:
            continue
        if aspect_included[aspect]:
            clause = rng.choice(_ASPECT_CLAUSES[aspect])
            sentences.append(clause)
            evidence_by_aspect[aspect] = clause

    rng.shuffle(sentences)
    if sentences:
        text = f"{text} " + ". ".join(sentences) + "."

    ordered_labels: dict[str, dict[str, Any]] = {}
    for aspect in ASPECTS:
        if aspect in evidence_by_aspect:
            score = _sample_score(rng, has_evidence=True)
            ordered_labels[aspect] = {"score": score, "evidence": evidence_by_aspect[aspect]}
        else:
            score = _sample_score(rng, has_evidence=False)
            ordered_labels[aspect] = {"score": score, "evidence": ""}

    validate_sample_scores(ordered_labels)
    return {"text": text.strip(), "labels": ordered_labels}


def generate_synthetic_dataset(num_samples: int, seed: int = 42) -> list[dict[str, Any]]:
    """Generate a list of synthetic Persian laptop-preference samples.

    Args:
        num_samples: Number of samples to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of validated sample dictionaries.
    """
    rng = random.Random(seed)
    return [generate_synthetic_sample(rng) for _ in range(num_samples)]


def write_synthetic_dataset(path: Path, num_samples: int, seed: int = 42) -> Path:
    """Generate and persist synthetic data to a JSON file.

    Args:
        path: Output JSON file path.
        num_samples: Number of samples.
        seed: Random seed.

    Returns:
        Path to the written file.
    """
    data = generate_synthetic_dataset(num_samples, seed=seed)
    save_json(data, path)
    logger.info("Wrote %d synthetic samples to %s", len(data), path)
    return path


def discover_data_files(data_dir: Path, file_pattern: str) -> list[Path]:
    """Discover JSON data files in a directory.

    Args:
        data_dir: Directory to search.
        file_pattern: Glob pattern (e.g. ``*.json``).

    Returns:
        Sorted list of matching file paths.

    Raises:
        FileNotFoundError: If no files are found.
    """
    if data_dir.is_file():
        return [data_dir]
    files = sorted(data_dir.glob(file_pattern))
    if not files:
        raise FileNotFoundError(f"No files matching '{file_pattern}' in {data_dir}")
    return files


def load_raw_dataset(config: AppConfig) -> list[dict[str, Any]]:
    """Load and validate all samples from configured data directory.

    Args:
        config: Application configuration.

    Returns:
        List of validated raw samples.
    """
    data_dir = config.paths.data_dir
    files = discover_data_files(data_dir, config.data.file_pattern)
    samples: list[dict[str, Any]] = []

    for file_path in files:
        payload = load_json(file_path)
        if isinstance(payload, list):
            batch = payload
        elif isinstance(payload, dict) and "samples" in payload:
            batch = payload["samples"]
        else:
            raise ValueError(f"Unsupported JSON structure in {file_path}")

        for idx, sample in enumerate(batch):
            if "text" not in sample or "labels" not in sample:
                raise ValueError(f"Sample {idx} in {file_path} missing 'text' or 'labels'")
            validate_sample_scores(sample["labels"])
            samples.append(sample)

    logger.info("Loaded %d samples from %d file(s) in %s", len(samples), len(files), data_dir)
    return samples


def split_dataset(
    samples: list[dict[str, Any]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split samples into train, validation, and test sets.

    Args:
        samples: Full dataset.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        test_ratio: Fraction for testing.
        seed: Random seed.

    Returns:
        Tuple of (train, val, test) sample lists.
    """
    if len(samples) < 3:
        raise ValueError("Need at least 3 samples to create train/val/test splits")

    test_size = test_ratio
    temp, test = train_test_split(samples, test_size=test_size, random_state=seed)
    relative_val = val_ratio / (train_ratio + val_ratio)
    train, val = train_test_split(temp, test_size=relative_val, random_state=seed)
    return train, val, test


def save_splits(
    train: list[dict[str, Any]],
    val: list[dict[str, Any]],
    test: list[dict[str, Any]],
    processed_dir: Path,
) -> dict[str, Path]:
    """Persist train/val/test splits to JSON files.

    Args:
        train: Training samples.
        val: Validation samples.
        test: Test samples.
        processed_dir: Output directory.

    Returns:
        Mapping of split name to file path.
    """
    paths = {
        "train": processed_dir / "train.json",
        "val": processed_dir / "val.json",
        "test": processed_dir / "test.json",
    }
    save_json(train, paths["train"])
    save_json(val, paths["val"])
    save_json(test, paths["test"])
    logger.info(
        "Saved splits: train=%d, val=%d, test=%d -> %s",
        len(train),
        len(val),
        len(test),
        processed_dir,
    )
    return paths


def prepare_dataset_splits(config: AppConfig) -> dict[str, Path]:
    """Load raw data, split, and save processed JSON files.

    Args:
        config: Application configuration.

    Returns:
        Mapping of split name to file path.
    """
    synthetic_path = config.paths.data_dir / "laptop_prefs.json"
    if not synthetic_path.exists() and not any(config.paths.data_dir.glob("*.json")):
        write_synthetic_dataset(
            synthetic_path,
            num_samples=config.data.synthetic_num_samples,
            seed=config.seed,
        )

    samples = load_raw_dataset(config)
    train, val, test = split_dataset(
        samples,
        config.data.train_ratio,
        config.data.val_ratio,
        config.data.test_ratio,
        config.seed,
    )
    return save_splits(train, val, test, config.paths.processed_dir)


def load_split(config: AppConfig, split: str) -> list[dict[str, Any]]:
    """Load a processed split by name.

    Args:
        config: Application configuration.
        split: One of ``train``, ``val``, or ``test``.

    Returns:
        List of samples for the requested split.
    """
    path = config.paths.processed_dir / f"{split}.json"
    if not path.exists():
        prepare_dataset_splits(config)
    return load_json(path)
