"""Custom Tk widgets for aspect results."""

from __future__ import annotations

import tkinter as tk
from typing import Any

from UI.persian_text import PersianLabel, PersianText
from UI.theme import ASPECT_FA, ASPECT_HINT, COLORS


def _bar_color(score: float) -> str:
    if score < 0.34:
        return COLORS["bar_low"]
    if score < 0.67:
        return COLORS["bar_mid"]
    return COLORS["bar_high"]


class AspectCard(tk.Frame):
    """One aspect result: title, numeric score, bar, evidence."""

    def __init__(
        self,
        master: tk.Misc,
        aspect_key: str,
        fonts: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        super().__init__(
            master,
            bg=COLORS["surface"],
            highlightthickness=1,
            highlightbackground=COLORS["border"],
            **kwargs,
        )
        self.aspect_key = aspect_key
        self.fonts = fonts

        pad = tk.Frame(self, bg=COLORS["surface"])
        pad.pack(fill="both", expand=True, padx=16, pady=14)

        header = tk.Frame(pad, bg=COLORS["surface"])
        header.pack(fill="x")

        PersianLabel(
            header,
            ASPECT_FA.get(aspect_key, aspect_key),
            role="body_bold",
            fg=COLORS["ink"],
            bg=COLORS["surface"],
        ).pack(side="right")

        self.score_lbl = tk.Label(
            header,
            text="—",
            font=fonts["score"],
            fg=COLORS["accent"],
            bg=COLORS["surface"],
            anchor="w",
        )
        self.score_lbl.pack(side="left")

        PersianLabel(
            pad,
            ASPECT_HINT.get(aspect_key, ""),
            role="caption",
            fg=COLORS["muted"],
            bg=COLORS["surface"],
        ).pack(anchor="e", pady=(4, 10))

        self.canvas = tk.Canvas(
            pad,
            height=10,
            bg=COLORS["surface"],
            highlightthickness=0,
            bd=0,
        )
        self.canvas.pack(fill="x")
        self._bar_bg = self.canvas.create_rectangle(
            0, 0, 10, 10, fill=COLORS["bar_track"], outline="", width=0
        )
        self._bar_fg = self.canvas.create_rectangle(
            0, 0, 0, 10, fill=COLORS["bar_mid"], outline="", width=0
        )
        self.canvas.bind("<Configure>", self._on_resize)
        self._score = 0.0

        PersianLabel(
            pad,
            "شواهد در متن کاربر",
            role="caption",
            fg=COLORS["muted"],
            bg=COLORS["surface"],
        ).pack(anchor="e", pady=(12, 4))

        self.evidence_box = PersianText(
            pad,
            height=3,
            readonly=True,
            role="body",
            fg=COLORS["ink_soft"],
            bg=COLORS["surface_alt"],
            highlightbackground=COLORS["border"],
        )
        self.evidence_box.pack(fill="x")

    def _on_resize(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        self._draw_bar(event.width)

    def _draw_bar(self, width: int | None = None) -> None:
        w = width if width is not None else max(self.canvas.winfo_width(), 1)
        h = 10
        fill_w = max(0, int(w * max(0.0, min(1.0, self._score))))
        self.canvas.coords(self._bar_bg, 0, 0, w, h)
        self.canvas.coords(self._bar_fg, 0, 0, fill_w, h)
        self.canvas.itemconfigure(self._bar_fg, fill=_bar_color(self._score))

    def set_result(self, score: float | None, evidence: str) -> None:
        if score is None or (isinstance(score, float) and score != score):  # NaN
            self._score = 0.0
            self.score_lbl.configure(text="—", fg=COLORS["muted"])
        else:
            self._score = float(max(0.0, min(1.0, score)))
            self.score_lbl.configure(
                text=f"{self._score:.2f}",
                fg=_bar_color(self._score),
            )
        self._draw_bar()
        text = (evidence or "").strip() or "—"
        self.evidence_box.set_logical(text)

    def clear(self) -> None:
        self.set_result(None, "")
