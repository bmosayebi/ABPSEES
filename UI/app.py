#!/usr/bin/env python3
"""ABPSEES desktop GUI — Persian RTL inference viewer (Tkinter).

Default model path:

    UI/model_bundle.zip   (then repo-root / outputs/)

Run from the UI folder with the project venv:

    source venv/bin/activate
    python app.py
"""

from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from project.constants import ASPECTS  # noqa: E402

from UI.fonts import load_peyda  # noqa: E402
from UI.persian_text import PersianButton, PersianLabel, PersianText  # noqa: E402
from UI.runner import ModelRunner, resolve_default_bundle  # noqa: E402
from UI.theme import COLORS  # noqa: E402
from UI.widgets import AspectCard  # noqa: E402


class AbpseesApp(tk.Tk):
    def __init__(self, bundle_path: Path | None = None) -> None:
        super().__init__()
        self.title("ABPSEES")
        self.geometry("980x820")
        self.minsize(820, 700)
        self.configure(bg=COLORS["bg"])

        self.fonts = load_peyda(self)
        self.bundle_path = Path(bundle_path) if bundle_path else resolve_default_bundle()
        self.runner = ModelRunner(self.bundle_path)
        self._busy = False
        self.aspect_cards: dict[str, AspectCard] = {}

        self._build_layout()
        self._build_overlay()
        self._show_overlay("در حال بارگذاری مدل…")
        self.after(120, self._start_model_load)

    # ------------------------------------------------------------------ layout
    def _build_layout(self) -> None:
        self._build_header()

        body = tk.Frame(self, bg=COLORS["bg"])
        body.pack(fill="both", expand=True, padx=24, pady=(18, 20))

        self._build_input_panel(body)
        self._build_actions(body)
        self._build_results_panel(body)
        self._build_footer()

    def _build_header(self) -> None:
        header = tk.Frame(self, bg=COLORS["header_bg"], height=78)
        header.pack(fill="x")
        header.pack_propagate(False)

        inner = tk.Frame(header, bg=COLORS["header_bg"])
        inner.pack(fill="both", expand=True, padx=28, pady=14)

        tk.Label(
            inner,
            text="ABPSEES",
            font=self.fonts["brand"],
            fg=COLORS["header_fg"],
            bg=COLORS["header_bg"],
            anchor="e",
        ).pack(side="right")

        PersianLabel(
            inner,
            "استخراج ترجیحات جنبه‌محور از متن فارسی",
            role="subtitle",
            fg=COLORS["header_muted"],
            bg=COLORS["header_bg"],
        ).pack(side="right", padx=(0, 16), pady=(8, 0))

        self.status_lbl = PersianLabel(
            inner,
            "",
            role="caption",
            fg=COLORS["header_muted"],
            bg=COLORS["header_bg"],
        )
        self.status_lbl.pack(side="left", pady=(8, 0))

    def _panel(self, parent: tk.Misc, title: str) -> tk.Frame:
        wrap = tk.Frame(
            parent,
            bg=COLORS["surface"],
            highlightthickness=1,
            highlightbackground=COLORS["border"],
        )
        wrap.pack(fill="x", pady=(0, 14))

        head = tk.Frame(wrap, bg=COLORS["surface"])
        head.pack(fill="x", padx=18, pady=(14, 6))
        PersianLabel(
            head,
            title,
            role="title",
            fg=COLORS["ink"],
            bg=COLORS["surface"],
        ).pack(side="right")
        return wrap

    def _build_input_panel(self, parent: tk.Misc) -> None:
        wrap = self._panel(parent, "متن کاربر")
        pad = tk.Frame(wrap, bg=COLORS["surface"])
        pad.pack(fill="both", expand=True, padx=18, pady=(0, 16))

        PersianLabel(
            pad,
            "نیازها، سبک زندگی و ترجیحات لپ‌تاپ را به فارسی بنویسید.",
            role="caption",
            fg=COLORS["muted"],
            bg=COLORS["surface"],
        ).pack(anchor="e", pady=(0, 8))

        self.input_text = PersianText(
            pad,
            height=6,
            readonly=False,
            role="input",
            fg=COLORS["ink"],
            bg=COLORS["surface_alt"],
            highlightbackground=COLORS["border"],
        )
        self.input_text.pack(fill="x")

        sample = (
            "من دانشجوی مهندسی کامپیوتر هستم و می‌خواهم مدل‌های هوش مصنوعی آموزش بدهم. "
            "هر روز لپ‌تاپم را با خودم به دانشگاه می‌برم و به صفحه‌نمایش باکیفیت هم اهمیت می‌دهم."
        )
        self.input_text.set_logical(sample)

    def _build_actions(self, parent: tk.Misc) -> None:
        row = tk.Frame(parent, bg=COLORS["bg"])
        row.pack(fill="x", pady=(0, 14))

        self.run_btn = PersianButton(
            row,
            "اجرای مدل",
            command=self.on_run,
            primary=True,
            fg="#FFFFFF",
            bg=COLORS["accent"],
            activebackground=COLORS["accent_hover"],
        )
        self.run_btn.pack(side="right")

        self.clear_btn = PersianButton(
            row,
            "پاک کردن نتایج",
            command=self.on_clear_results,
            primary=False,
            fg=COLORS["ink_soft"],
            bg=COLORS["surface"],
            activebackground=COLORS["surface_alt"],
        )
        self.clear_btn.pack(side="right", padx=(0, 10))

        self.bundle_lbl = PersianLabel(
            row,
            f"مدل: {self.bundle_path.name}",
            role="caption",
            fg=COLORS["muted"],
            bg=COLORS["bg"],
        )
        self.bundle_lbl.pack(side="left")

    def _build_results_panel(self, parent: tk.Misc) -> None:
        wrap = tk.Frame(
            parent,
            bg=COLORS["surface"],
            highlightthickness=1,
            highlightbackground=COLORS["border"],
        )
        wrap.pack(fill="both", expand=True)

        head = tk.Frame(wrap, bg=COLORS["surface"])
        head.pack(fill="x", padx=18, pady=(14, 6))
        PersianLabel(
            head,
            "نتایج جنبه‌ها",
            role="title",
            fg=COLORS["ink"],
            bg=COLORS["surface"],
        ).pack(side="right")

        self.warnings_lbl = PersianLabel(
            head,
            "",
            role="caption",
            fg=COLORS["warning"],
            bg=COLORS["surface"],
        )
        self.warnings_lbl.pack(side="left")

        canvas = tk.Canvas(wrap, bg=COLORS["surface"], highlightthickness=0, bd=0)
        scrollbar = tk.Scrollbar(wrap, orient="vertical", command=canvas.yview)
        self.cards_frame = tk.Frame(canvas, bg=COLORS["surface"])

        self.cards_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        self._cards_window = canvas.create_window((0, 0), window=self.cards_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.bind(
            "<Configure>",
            lambda e: canvas.itemconfigure(self._cards_window, width=e.width),
        )

        scrollbar.pack(side="left", fill="y", padx=(0, 4), pady=(0, 12))
        canvas.pack(side="right", fill="both", expand=True, padx=18, pady=(0, 14))
        canvas.bind_all(
            "<MouseWheel>",
            lambda e: canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"),
        )

        grid = tk.Frame(self.cards_frame, bg=COLORS["surface"])
        grid.pack(fill="both", expand=True, padx=4, pady=4)

        for i, aspect in enumerate(ASPECTS):
            card = AspectCard(grid, aspect, self.fonts)
            r, c = divmod(i, 2)
            card.grid(row=r, column=c, sticky="nsew", padx=6, pady=6)
            self.aspect_cards[aspect] = card

        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)

    def _build_footer(self) -> None:
        foot = tk.Frame(self, bg=COLORS["bg"])
        foot.pack(fill="x", padx=24, pady=(0, 12))
        PersianLabel(
            foot,
            f"مسیر مدل: {self.bundle_path}",
            role="caption",
            fg=COLORS["muted"],
            bg=COLORS["bg"],
        ).pack(side="right")

    def _build_overlay(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(
            "ABP.Horizontal.TProgressbar",
            troughcolor=COLORS["bar_track"],
            background=COLORS["accent"],
            bordercolor=COLORS["border"],
            lightcolor=COLORS["accent"],
            darkcolor=COLORS["accent"],
            thickness=12,
        )

        self.overlay = tk.Frame(self, bg=COLORS["header_bg"])
        card = tk.Frame(
            self.overlay,
            bg=COLORS["surface"],
            highlightthickness=1,
            highlightbackground=COLORS["border"],
        )
        card.place(relx=0.5, rely=0.45, anchor="center")

        inner = tk.Frame(card, bg=COLORS["surface"])
        inner.pack(padx=36, pady=28)

        self.overlay_title = PersianLabel(
            inner,
            "لطفاً صبر کنید",
            role="title",
            fg=COLORS["ink"],
            bg=COLORS["surface"],
        )
        self.overlay_title.pack()

        self.overlay_msg = PersianLabel(
            inner,
            "",
            role="body",
            fg=COLORS["ink_soft"],
            bg=COLORS["surface"],
        )
        self.overlay_msg.pack(pady=(10, 18))

        self.progress = ttk.Progressbar(
            inner,
            style="ABP.Horizontal.TProgressbar",
            mode="indeterminate",
            length=280,
        )
        self.progress.pack()

        PersianLabel(
            inner,
            "این مرحله ممکن است چند دقیقه طول بکشد",
            role="caption",
            fg=COLORS["muted"],
            bg=COLORS["surface"],
        ).pack(pady=(14, 0))

    def _show_overlay(self, message: str) -> None:
        self.overlay_msg.set_text(message)
        self.overlay.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.overlay.lift()
        self.progress.start(12)

    def _hide_overlay(self) -> None:
        self.progress.stop()
        self.overlay.place_forget()

    # ----------------------------------------------------------------- helpers
    def _set_status(self, text: str, kind: str = "muted") -> None:
        color = {
            "muted": COLORS["header_muted"],
            "ok": "#7DDBB0",
            "warn": "#F0C674",
            "err": "#F0A0A0",
        }.get(kind, COLORS["header_muted"])
        self.status_lbl._fg = color
        self.status_lbl.set_text(text)

    def _set_busy(self, busy: bool, message: str | None = None) -> None:
        self._busy = busy
        state = "disabled" if busy else "normal"
        self.run_btn.configure(state=state)
        self.clear_btn.configure(state=state)
        if busy:
            self._show_overlay(message or "لطفاً صبر کنید…")
        else:
            self._hide_overlay()

    def _start_model_load(self) -> None:
        self._set_busy(True, "در حال بارگذاری مدل…")
        self.runner.load_async(
            on_done=lambda: self.after(0, self._on_model_ready),
            on_error=lambda err: self.after(0, lambda: self._on_model_error(err)),
        )

    def _on_model_ready(self) -> None:
        self._set_busy(False)
        self._set_status("مدل آماده است", kind="ok")
        self.bundle_lbl.set_text(f"مدل آماده · {self.runner.bundle_path.name}")

    def _on_model_error(self, err: str) -> None:
        self._set_busy(False)
        self._set_status("خطا در بارگذاری مدل", kind="err")
        messagebox.showerror(
            "خطا در بارگذاری مدل",
            f"{err}\n\nمسیر پیش‌فرض:\n{self.runner.bundle_path}",
        )

    # ----------------------------------------------------------------- actions
    def on_clear_results(self) -> None:
        for card in self.aspect_cards.values():
            card.clear()
        self.warnings_lbl.set_text("")

    def on_run(self) -> None:
        if self._busy:
            return
        text = self.input_text.get_logical().strip()
        if not text:
            messagebox.showwarning("متن خالی", "لطفاً متن فارسی را وارد کنید.")
            return

        self._set_busy(True, "در حال استخراج ترجیحات…")
        self._set_status("در حال استخراج ترجیحات…", kind="warn")
        self.warnings_lbl.set_text("")
        self.runner.predict_async(
            text,
            on_done=lambda result: self.after(0, lambda: self._on_predict_done(result)),
            on_error=lambda err: self.after(0, lambda: self._on_predict_error(err)),
        )

    def _on_predict_done(self, result: dict[str, Any]) -> None:
        self._set_busy(False)
        self._set_status("نتیجه آماده است", kind="ok")

        labels = result.get("labels", {})
        for aspect, card in self.aspect_cards.items():
            entry = labels.get(aspect, {})
            score = entry.get("score")
            evidence = entry.get("evidence", "")
            try:
                score_f = float(score) if score is not None else None
            except (TypeError, ValueError):
                score_f = None
            card.set_result(score_f, str(evidence or ""))

        warnings = result.get("_warnings") or []
        if warnings:
            self.warnings_lbl.set_text(f"{len(warnings)} هشدار — جزئیات در لاگ")
        else:
            self.warnings_lbl.set_text("")

    def _on_predict_error(self, err: str) -> None:
        self._set_busy(False)
        self._set_status("خطا در اجرا", kind="err")
        messagebox.showerror("خطا در اجرای مدل", err)


def main() -> int:
    bundle: Path | None = None
    if len(sys.argv) > 1 and sys.argv[1].endswith(".zip"):
        bundle = Path(sys.argv[1]).expanduser().resolve()

    app = AbpseesApp(bundle_path=bundle)
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
