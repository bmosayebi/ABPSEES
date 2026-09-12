"""Background model load + inference for the desktop UI."""

from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
UI_DIR = Path(__file__).resolve().parent


def resolve_default_bundle() -> Path:
    """Prefer ``UI/model_bundle.zip``, then repo-root ``model_bundle.zip``."""
    candidates = [
        UI_DIR / "model_bundle.zip",
        REPO_ROOT / "model_bundle.zip",
        REPO_ROOT / "outputs" / "model_bundle.zip",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


DEFAULT_BUNDLE_PATH = resolve_default_bundle()

logger = logging.getLogger(__name__)


def ensure_project_on_path() -> None:
    """Make ``project`` importable when launching ``UI/app.py`` directly."""
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


class ModelRunner:
    """Lazy-loads ``model_bundle.zip`` once and runs ``predict`` off the UI thread."""

    def __init__(self, bundle_path: Path | None = None) -> None:
        self.bundle_path = Path(bundle_path) if bundle_path else DEFAULT_BUNDLE_PATH
        self.model: Any | None = None
        self.tokenizer: Any | None = None
        self.config: Any | None = None
        self._lock = threading.Lock()
        self._load_error: str | None = None

    @property
    def is_ready(self) -> bool:
        return self.model is not None and self.tokenizer is not None and self.config is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def load_sync(self) -> None:
        """Load the bundle on the current thread (may take minutes on CPU)."""
        ensure_project_on_path()
        from project.bundle import load_model_bundle

        if not self.bundle_path.exists():
            raise FileNotFoundError(
                f"فایل مدل پیدا نشد:\n{self.bundle_path}\n\n"
                "خروجی Colab را با نام model_bundle.zip در ریشهٔ پروژه قرار دهید."
            )

        with self._lock:
            if self.is_ready:
                return
            logger.info("Loading model bundle from %s", self.bundle_path)
            model, tokenizer, config = load_model_bundle(self.bundle_path)
            self.model = model
            self.tokenizer = tokenizer
            self.config = config
            self._prepare_for_fast_inference()
            self._load_error = None

    def _prepare_for_fast_inference(self) -> None:
        """Merge LoRA and warm up one heads-only forward (interactive UI)."""
        model = self.model
        inner = getattr(model, "base_model", model)
        if hasattr(inner, "merge_and_unload"):
            try:
                logger.info("Merging LoRA adapters for faster local inference")
                merged = inner.merge_and_unload()
                if hasattr(model, "base_model"):
                    model.base_model = merged
                else:
                    self.model = merged
            except Exception:
                logger.exception("LoRA merge skipped; inference will still use the fast heads path.")

        self.model.eval()
        cfg = getattr(self.model, "config", None)
        if cfg is not None:
            cfg.use_cache = True

        if hasattr(self.model, "aspect_heads"):
            from project.hybrid_model import run_aspect_heads_on_text

            try:
                logger.info("Warming up aspect-head inference")
                run_aspect_heads_on_text(self.model, self.tokenizer, "سلام", self.config)
                logger.info("Aspect-head warmup done")
            except Exception:
                logger.exception("Warmup failed; first Run click may be slower.")

    def load_async(
        self,
        on_done: Callable[[], None] | None = None,
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        """Load the bundle in a daemon thread."""

        def _work() -> None:
            try:
                self.load_sync()
                if on_done:
                    on_done()
            except Exception as exc:
                self._load_error = str(exc)
                logger.exception("Failed to load model bundle")
                if on_error:
                    on_error(str(exc))

        threading.Thread(target=_work, daemon=True).start()

    def predict_async(
        self,
        text: str,
        on_done: Callable[[dict[str, Any]], None] | None = None,
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        """Run inference in a daemon thread."""

        def _work() -> None:
            try:
                if not self.is_ready:
                    self.load_sync()
                ensure_project_on_path()
                from project.inference import predict

                result = predict(
                    text,
                    self.config,
                    model=self.model,
                    tokenizer=self.tokenizer,
                    use_generation=False,
                )
                if on_done:
                    on_done(result)
            except Exception as exc:
                logger.exception("Inference failed")
                if on_error:
                    on_error(str(exc))

        threading.Thread(target=_work, daemon=True).start()
