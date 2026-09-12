"""Persian text for Tkinter via Peyda TTF + presentation-form shaping.

Tk 8.6 does not run OpenType GSUB/GPOS, so Peyda letters stay isolated even
when the TTF is registered. The reliable path is:

1. ``arabic_reshaper`` → joined presentation forms (Peyda has these glyphs)
2. ``python-bidi`` → visual order for an LTR blit
3. Pillow draws those glyphs from the Peyda TTF file
4. Tk only shows the resulting ``PhotoImage``
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageTk

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "برای نمایش درست فارسی این بسته‌ها لازم است:\n"
        "  pip install arabic-reshaper python-bidi pillow\n"
        "یا: pip install -r UI/requirements.txt"
    ) from exc

from UI.fonts import FONTS_TTF_DIR

_RESHAPER = arabic_reshaper.ArabicReshaper(
    configuration={
        "delete_harakat": True,
        "support_ligatures": True,
        "RIAL SIGN": False,
    }
)

_FACE_FILE = {
    "regular": "PeydaWebFaNum-Regular.ttf",
    "medium": "PeydaWebFaNum-Medium.ttf",
    "semibold": "PeydaWebFaNum-SemiBold.ttf",
    "bold": "PeydaWebFaNum-Bold.ttf",
    "extrabold": "PeydaWebFaNum-ExtraBold.ttf",
}

ROLE_FACE: dict[str, tuple[str, int]] = {
    "brand": ("extrabold", 22),
    "title": ("bold", 16),
    "subtitle": ("medium", 12),
    "body": ("regular", 14),
    "body_bold": ("semibold", 14),
    "input": ("regular", 15),
    "button": ("semibold", 14),
    "caption": ("regular", 11),
    "score": ("bold", 18),
}

_FONT_CACHE: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}


def peyda_ttf(face: str = "regular") -> Path:
    path = FONTS_TTF_DIR / _FACE_FILE[face]
    if not path.exists():
        raise FileNotFoundError(f"Peyda TTF not found: {path}")
    return path


def _pil_font(face: str, size: int) -> ImageFont.FreeTypeFont:
    key = (face, size)
    cached = _FONT_CACHE.get(key)
    if cached is None:
        cached = ImageFont.truetype(str(peyda_ttf(face)), size)
        _FONT_CACHE[key] = cached
    return cached


def reshape(text: str) -> str:
    """Visual Persian string (presentation forms + bidi). Used by Pillow."""
    if not text:
        return ""
    try:
        return get_display(_RESHAPER.reshape(text))
    except Exception:
        return text


def reshape_paragraph(text: str) -> str:
    if not text:
        return ""
    return "\n".join(reshape(line) for line in text.split("\n"))


def _hex_rgb(color: str) -> tuple[int, int, int]:
    h = color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _wrap_logical(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    if max_width <= 0:
        return [text] if text else [""]
    words = text.split(" ")
    lines: list[str] = []
    current = ""
    for word in words:
        trial = word if not current else f"{current} {word}"
        width = font.getlength(reshape(trial))
        if width <= max_width or not current:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]


def render_persian(
    text: str,
    *,
    role: str = "body",
    fg: str = "#14202B",
    bg: str = "#FFFFFF",
    max_width: int | None = None,
    padding_x: int = 2,
    padding_y: int = 2,
    align: str = "right",
    min_height: int = 0,
) -> Image.Image:
    """Rasterize logical Persian with Peyda into an RGB image."""
    face, size = ROLE_FACE.get(role, ROLE_FACE["body"])
    font = _pil_font(face, size)
    line_h = size + 10
    wrap_at = (max_width - 2 * padding_x) if max_width else 10_000

    raw_lines: list[str] = []
    for para in (text or "").split("\n"):
        if para == "":
            raw_lines.append("")
        else:
            raw_lines.extend(_wrap_logical(para, font, max(1, int(wrap_at))))
    if not raw_lines:
        raw_lines = [""]

    visuals = [reshape(line) if line else "" for line in raw_lines]
    widths = [int(font.getlength(v)) if v else 0 for v in visuals]
    content_w = max(widths + [1])
    img_w = max(1, max_width if max_width else content_w + 2 * padding_x)
    img_h = max(min_height, line_h * len(visuals) + 2 * padding_y)

    image = Image.new("RGB", (img_w, img_h), _hex_rgb(bg))
    draw = ImageDraw.Draw(image)
    fill = _hex_rgb(fg)
    y = padding_y
    for visual, visual_w in zip(visuals, widths):
        if visual:
            x = img_w - padding_x - visual_w if align == "right" else padding_x
            draw.text((x, y), visual, font=font, fill=fill)
        y += line_h
    return image


def photo_persian(text: str, **kwargs: Any) -> ImageTk.PhotoImage:
    return ImageTk.PhotoImage(render_persian(text, **kwargs))


class PersianLabel(tk.Label):
    """Static Peyda label drawn as an image (joined Persian)."""

    def __init__(
        self,
        master: tk.Misc,
        text: str = "",
        *,
        role: str = "body",
        fg: str = "#14202B",
        bg: str = "#FFFFFF",
        wrap: bool = False,
        anchor: str = "e",
        padx: int = 0,
        pady: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(master, bg=bg, bd=0, highlightthickness=0, padx=padx, pady=pady, **kwargs)
        self._logical = text
        self._role = role
        self._fg = fg
        self._bg = bg
        self._wrap = wrap
        self._anchor = anchor
        self._photo: ImageTk.PhotoImage | None = None
        self._last_w = 0
        if wrap:
            self.bind("<Configure>", self._on_configure)
        self._paint()

    def set_text(self, text: str) -> None:
        self._logical = text
        self._paint()

    def _on_configure(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        if event.width < 20 or abs(event.width - self._last_w) < 6:
            return
        self._last_w = event.width
        self._paint(max_width=event.width)

    def _paint(self, max_width: int | None = None) -> None:
        width = max_width if self._wrap else None
        self._photo = photo_persian(
            self._logical,
            role=self._role,
            fg=self._fg,
            bg=self._bg,
            max_width=width,
        )
        self.configure(image=self._photo, anchor=self._anchor)


class PersianButton(tk.Button):
    """Button whose caption is a Peyda-rendered image."""

    def __init__(
        self,
        master: tk.Misc,
        text: str,
        command: Any,
        *,
        primary: bool = True,
        fg: str = "#FFFFFF",
        bg: str = "#0E6B6E",
        activebackground: str = "#0A5558",
        **kwargs: Any,
    ) -> None:
        self._logical = text
        self._fg = fg
        self._bg = bg
        self._photo = photo_persian(
            text,
            role="button",
            fg=fg,
            bg=bg,
            padding_x=22,
            padding_y=9,
        )
        super().__init__(
            master,
            image=self._photo,
            command=command,
            bg=bg,
            activebackground=activebackground,
            fg=fg,
            activeforeground=fg,
            relief="flat",
            bd=0,
            highlightthickness=0,
            cursor="hand2",
            padx=0,
            pady=0,
            **kwargs,
        )


class PersianText(tk.Frame):
    """Editable / read-only Persian field: logical buffer + Peyda image."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        role: str = "input",
        height: int = 4,
        readonly: bool = False,
        bg: str = "#FFFFFF",
        fg: str = "#000000",
        highlightbackground: str = "#CCCCCC",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            master,
            bg=bg,
            highlightthickness=1,
            highlightbackground=highlightbackground,
        )
        self._logical = ""
        self._readonly = readonly
        self._role = role
        self._fg = fg
        self._bg = bg
        self._photo: ImageTk.PhotoImage | None = None
        face, size = ROLE_FACE.get(role, ROLE_FACE["input"])
        self._line_h = size + 10
        min_h = max(height, 1) * self._line_h + 16

        self.canvas = tk.Canvas(
            self,
            height=min_h,
            bg=bg,
            highlightthickness=0,
            bd=0,
            cursor="xterm" if not readonly else "arrow",
        )
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self._on_configure)
        if not readonly:
            self.canvas.bind("<Button-1>", self._focus)
            self.canvas.bind("<KeyPress>", self._on_keypress)
            self.canvas.bind("<<Paste>>", self._on_paste)
            self.canvas.bind("<Control-v>", self._on_paste)
            self.canvas.bind("<Command-v>", self._on_paste)

    def get_logical(self) -> str:
        return self._logical

    def get(self, *_args: Any) -> str:
        return self._logical

    def set_logical(self, text: str, cursor: int | None = None) -> None:
        del cursor
        self._logical = text or ""
        self._paint()

    def clear(self) -> None:
        self.set_logical("")

    def _focus(self, _event: tk.Event | None = None) -> None:  # type: ignore[type-arg]
        self.canvas.focus_set()

    def _on_configure(self, _event: tk.Event) -> None:  # type: ignore[type-arg]
        self._paint()

    def _paint(self) -> None:
        width = max(self.canvas.winfo_width(), 40)
        height = max(self.canvas.winfo_height(), 40)
        self._photo = photo_persian(
            self._logical or " ",
            role=self._role,
            fg=self._fg,
            bg=self._bg,
            max_width=width,
            padding_x=14,
            padding_y=10,
            min_height=height,
        )
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo)

    def _insert(self, s: str) -> None:
        self._logical += s
        self._paint()

    def _on_keypress(self, event: tk.Event) -> str:  # type: ignore[type-arg]
        if self._readonly:
            return "break"
        key = event.keysym
        if key == "BackSpace":
            self._logical = self._logical[:-1]
            self._paint()
            return "break"
        if key in ("Return", "KP_Enter"):
            self._insert("\n")
            return "break"
        if key in (
            "Shift_L", "Shift_R", "Control_L", "Control_R",
            "Alt_L", "Alt_R", "Meta_L", "Meta_R", "Command", "Caps_Lock",
            "Left", "Right", "Up", "Down", "Home", "End", "Tab", "Escape",
            "Delete",
        ):
            return "break"
        if event.state & 0x4 and key.lower() in ("c", "a"):
            return "break"
        if event.char and event.char.isprintable():
            self._insert(event.char)
        return "break"

    def _on_paste(self, _event: tk.Event | None = None) -> str:  # type: ignore[type-arg]
        if self._readonly:
            return "break"
        try:
            clip = self.clipboard_get()
        except tk.TclError:
            return "break"
        self._insert(clip)
        return "break"
