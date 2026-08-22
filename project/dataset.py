"""Dataset loading, validation, splitting, and synthetic data generation."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

from sklearn.model_selection import train_test_split

from project.config import AppConfig, validate_sample_scores
from project.constants import ASPECTS, LOW_SCORE_EVIDENCE_THRESHOLD
from project.utils import load_json, save_json

logger = logging.getLogger(__name__)

# Persian sentence templates for synthetic laptop-preference descriptions.
_INTRO_TEMPLATES: list[str] = [
    "من {role} هستم و به لپ‌تاپی نیاز دارم که {need}.",
    "به عنوان {role}، دنبال لپ‌تاپی می‌گردم که {need}.",
    "من {role} هستم و می‌خواهم لپ‌تاپی بخرم که {need}.",
]

_ROLE_NEED_PAIRS: list[tuple[str, str]] = [
    ("دانشجوی مهندسی کامپیوتر", "مدل‌های هوش مصنوعی آموزش بدهم و تحلیل داده انجام بدهم"),
    ("برنامه‌نویس", "کدهای سنگین را بدون کندی اجرا کند"),
    ("گیم‌پلیر", "بازی‌های جدید را با گرافیک بالا اجرا کند"),
    ("طراح گرافیک", "نرم‌افزارهای طراحی را روان اجرا کند"),
    ("کارمند اداری", "کارهای روزمره و جلسات آنلاین را انجام دهم"),
    ("فریلنسر", "چند پروژه همزمان را مدیریت کنم"),
    ("دانشجوی پزشکی", "مطالعه و یادداشت‌برداری روزانه داشته باشم"),
    ("عکاس", "عکس‌های با کیفیت را ویرایش کنم"),
]

_EXTRA_CLAUSES: dict[str, list[str]] = {
    "portability": [
        "هر روز با اتوبوس به دانشگاه می‌روم و لپ‌تاپم را با خودم حمل می‌کنم.",
        "سفرهای کاری زیادی دارم و لپ‌تاپ باید سبک و قابل حمل باشد.",
        "اغلب در کافه کار می‌کنم و لپ‌تاپ را همیشه با خودم می‌برم.",
        "بین خانه و محل کار جابه‌جا می‌شوم و وزن لپ‌تاپ برایم مهم است.",
    ],
    "design": [
        "ظاهر لپ‌تاپ برایم اهمیت دارد و دوست دارم طراحی مدرن و زیبایی داشته باشد.",
        "رنگ و طراحی بدنه لپ‌تاپ در انتخاب من تأثیرگذار است.",
        "دوست دارم لپ‌تاپم ظاهر حرفه‌ای و شیکی داشته باشد.",
    ],
    "durability": [
        "لپ‌تاپ قبلی‌ام بعد از دو سال خراب شد و می‌خواهم این بار محکم‌تر باشد.",
        "محیط کارم پر از گرد و غبار است و به بدنه مقاوم نیاز دارم.",
        "می‌خواهم لپ‌تاپم چند سال بدون مشکل کار کند.",
    ],
    "cost_effectiveness": [
        "بودجه محدودی دارم و بهترین ارزش را در ازای پول می‌خواهم.",
        "نمی‌خواهم بیش از حد هزینه کنم ولی کیفیت مناسب می‌خواهم.",
        "به دنبال گزینه‌ای مقرون‌به‌صرفه با کارایی خوب هستم.",
    ],
}

_PERFORMANCE_CLAUSE = "می‌خواهم عملکرد بالایی برای کارهای سنگین داشته باشد."


def _make_base_text(rng: random.Random) -> tuple[str, str]:
    """Build a base Persian description and return text plus performance evidence."""
    role, need = rng.choice(_ROLE_NEED_PAIRS)
    intro = rng.choice(_INTRO_TEMPLATES).format(role=role, need=need)
    text = f"{intro} {need}."
    performance_evidence = need
    return text, performance_evidence


def _sample_score(rng: random.Random, has_evidence: bool) -> float:
    """Sample a relevance score, biased high when evidence is expected."""
    if has_evidence:
        return round(rng.uniform(0.55, 0.98), 2)
    return round(rng.uniform(0.05, 0.30), 2)


def generate_synthetic_sample(rng: random.Random) -> dict[str, Any]:
    """Generate a single synthetic labeled sample with valid evidence spans.

    Args:
        rng: Random number generator instance.

    Returns:
        Sample dict with ``text`` and ``labels`` keys.
    """
    text, performance_evidence = _make_base_text(rng)
    labels: dict[str, dict[str, Any]] = {}

    # Performance is always relevant in synthetic samples.
    perf_score = _sample_score(rng, has_evidence=True)
    labels["performance"] = {"score": perf_score, "evidence": performance_evidence}

    # Randomly add clauses for other aspects.
    for aspect in ["portability", "design", "durability", "cost_effectiveness"]:
        clauses = _EXTRA_CLAUSES[aspect]
        if rng.random() < 0.55:
            clause = rng.choice(clauses)
            text = f"{text} {clause}"
            score = _sample_score(rng, has_evidence=True)
            labels[aspect] = {"score": score, "evidence": clause.rstrip(".")}
        else:
            score = _sample_score(rng, has_evidence=False)
            labels[aspect] = {
                "score": score,
                "evidence": "" if score < LOW_SCORE_EVIDENCE_THRESHOLD else "",
            }

    # Ensure key order matches ASPECTS.
    ordered_labels = {aspect: labels[aspect] for aspect in ASPECTS}
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
