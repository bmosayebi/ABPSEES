"""Load Peyda TTF faces for Tkinter (macOS / Linux / Windows)."""

from __future__ import annotations

import ctypes
import logging
import sys
import tkinter.font as tkfont
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
FONTS_TTF_DIR = REPO_ROOT / "fonts" / "ttf"

# Logical role → (filename, Tk family name after OS registration, weight)
_FACES: dict[str, tuple[str, str, str]] = {
    "regular": ("PeydaWebFaNum-Regular.ttf", "PeydaWeb(FaNum)", "normal"),
    "medium": ("PeydaWebFaNum-Medium.ttf", "PeydaWeb(FaNum) Medium", "normal"),
    "semibold": ("PeydaWebFaNum-SemiBold.ttf", "PeydaWeb(FaNum) SemiBold", "normal"),
    "bold": ("PeydaWebFaNum-Bold.ttf", "PeydaWeb(FaNum)", "bold"),
    "extrabold": ("PeydaWebFaNum-ExtraBold.ttf", "PeydaWeb(FaNum) ExtraBold", "normal"),
}


def _register_font_file(path: Path) -> None:
    """Register a TTF with the OS so Tk can resolve its family name."""
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    if sys.platform == "darwin":
        _register_macos(path)
    elif sys.platform.startswith("linux"):
        _register_linux(path)
    elif sys.platform == "win32":
        _register_windows(path)
    else:
        logger.warning("No font-registration backend for %s; using system fonts.", sys.platform)


def _register_macos(path: Path) -> None:
    import ctypes.util

    core_foundation = ctypes.cdll.LoadLibrary(ctypes.util.find_library("CoreFoundation"))
    core_text = ctypes.cdll.LoadLibrary(ctypes.util.find_library("CoreText"))

    CFStringCreateWithCString = core_foundation.CFStringCreateWithCString
    CFStringCreateWithCString.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_uint32,
    ]
    CFStringCreateWithCString.restype = ctypes.c_void_p

    CFURLCreateWithFileSystemPath = core_foundation.CFURLCreateWithFileSystemPath
    CFURLCreateWithFileSystemPath.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_bool,
    ]
    CFURLCreateWithFileSystemPath.restype = ctypes.c_void_p

    CTFontManagerRegisterFontsForURL = core_text.CTFontManagerRegisterFontsForURL
    CTFontManagerRegisterFontsForURL.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_void_p,
    ]
    CTFontManagerRegisterFontsForURL.restype = ctypes.c_bool

    kCFStringEncodingUTF8 = 0x08000100
    kCFURLPOSIXPathStyle = 0
    kCTFontManagerScopeProcess = 1

    cf_path = CFStringCreateWithCString(None, str(path).encode("utf-8"), kCFStringEncodingUTF8)
    if not cf_path:
        raise RuntimeError(f"CFStringCreateWithCString failed for {path}")
    url = CFURLCreateWithFileSystemPath(None, cf_path, kCFURLPOSIXPathStyle, False)
    if not url:
        raise RuntimeError(f"CFURLCreateWithFileSystemPath failed for {path}")
    ok = CTFontManagerRegisterFontsForURL(url, kCTFontManagerScopeProcess, None)
    if not ok:
        # Already registered in this process is common / harmless.
        logger.debug("CTFontManagerRegisterFontsForURL returned false for %s", path)


def _register_linux(path: Path) -> None:
    try:
        fontconfig = ctypes.cdll.LoadLibrary("libfontconfig.so.1")
    except OSError:
        logger.warning("libfontconfig not found; Peyda may fall back to a system font.")
        return
    fontconfig.FcConfigAppFontAddFile.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    fontconfig.FcConfigAppFontAddFile.restype = ctypes.c_int
    fontconfig.FcConfigGetCurrent.restype = ctypes.c_void_p
    cfg = fontconfig.FcConfigGetCurrent()
    fontconfig.FcConfigAppFontAddFile(cfg, str(path).encode("utf-8"))


def _register_windows(path: Path) -> None:
    gdi32 = ctypes.windll.gdi32  # type: ignore[attr-defined]
    FR_PRIVATE = 0x10
    added = gdi32.AddFontResourceExW(str(path), FR_PRIVATE, 0)
    if not added:
        logger.warning("AddFontResourceExW failed for %s", path)


def load_peyda(root: Any) -> dict[str, tkfont.Font]:
    """Register Peyda weights and return ready-to-use Font objects.

    Falls back to a generic Tk family if Peyda cannot be registered (still
    usable, but not the branded look).
    """
    missing: list[Path] = []
    for filename, _family, _weight in _FACES.values():
        path = FONTS_TTF_DIR / filename
        if not path.exists():
            missing.append(path)
            continue
        try:
            _register_font_file(path)
        except Exception:
            logger.exception("Failed to register font %s", path)

    if missing:
        raise FileNotFoundError(
            "Peyda TTF files missing under fonts/ttf/:\n"
            + "\n".join(f"  - {p}" for p in missing)
            + "\nSee UI/README.md for conversion from fonts/*.woff."
        )

    # Prefer registered Peyda; otherwise Tk will substitute.
    def make(role: str, size: int) -> tkfont.Font:
        _filename, family, weight = _FACES[role]
        return tkfont.Font(root=root, family=family, size=size, weight=weight)

    # Probe once so we can log a clear warning.
    probe = make("regular", 12)
    actual = probe.actual("family")
    if "Peyda" not in actual and "FaNum" not in actual:
        logger.warning(
            "Peyda family not resolved by Tk (got %r). UI will use a fallback face.",
            actual,
        )

    return {
        "brand": make("extrabold", 22),
        "title": make("bold", 16),
        "subtitle": make("medium", 11),
        "body": make("regular", 13),
        "body_bold": make("semibold", 13),
        "input": make("regular", 14),
        "button": make("semibold", 13),
        "score": make("bold", 18),
        "caption": make("regular", 10),
        "mono": make("regular", 11),
    }
