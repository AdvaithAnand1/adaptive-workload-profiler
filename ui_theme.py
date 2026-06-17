"""Shared Google-inspired light theme and rounded Tk widgets."""

from __future__ import annotations

import tkinter as tk
import tkinter.font as tkfont
from collections.abc import Callable
from tkinter import ttk

APP_BG = "#f8f9fa"
SURFACE = "#ffffff"
SURFACE_ALT = "#f1f3f4"
BORDER = "#dadce0"
TEXT = "#202124"
MUTED = "#5f6368"
PRIMARY = "#1a73e8"
PRIMARY_HOVER = "#1765cc"
PRIMARY_PRESSED = "#185abc"
PRIMARY_SOFT = "#e8f0fe"
DANGER = "#d93025"
SUCCESS = "#188038"
SHADOW = "#e8eaed"


def apply_google_theme(root: tk.Misc) -> ttk.Style:
    style = ttk.Style(root)
    if "vista" in style.theme_names():
        style.theme_use("vista")
    elif "clam" in style.theme_names():
        style.theme_use("clam")

    root.configure(background=APP_BG)
    default_font = tkfont.nametofont("TkDefaultFont")
    default_font.configure(family="Arial", size=10)
    text_font = tkfont.nametofont("TkTextFont")
    text_font.configure(family="Arial", size=10)
    root.option_add("*Font", default_font)
    root.option_add("*Menu.Font", default_font)

    style.configure(".", font=("Arial", 10), foreground=TEXT)
    style.configure("TFrame", background=APP_BG)
    style.configure("Card.TFrame", background=SURFACE)
    style.configure("Inset.TFrame", background=SURFACE_ALT)
    style.configure("TLabel", background=APP_BG, foreground=TEXT)
    style.configure("Card.TLabel", background=SURFACE, foreground=TEXT)
    style.configure("Inset.TLabel", background=SURFACE_ALT, foreground=TEXT)
    style.configure(
        "Title.TLabel",
        background=APP_BG,
        foreground=TEXT,
        font=("Arial", 18, "bold"),
    )
    style.configure(
        "Subtitle.TLabel",
        background=APP_BG,
        foreground=MUTED,
        font=("Arial", 10),
    )
    style.configure(
        "CardTitle.TLabel",
        background=SURFACE,
        foreground=TEXT,
        font=("Arial", 11, "bold"),
    )
    style.configure(
        "InsetTitle.TLabel",
        background=SURFACE_ALT,
        foreground=TEXT,
        font=("Arial", 10, "bold"),
    )
    style.configure(
        "Muted.Card.TLabel",
        background=SURFACE,
        foreground=MUTED,
    )
    style.configure(
        "Muted.Inset.TLabel",
        background=SURFACE_ALT,
        foreground=MUTED,
    )

    style.configure(
        "TButton",
        background=SURFACE_ALT,
        foreground=TEXT,
        bordercolor=BORDER,
        lightcolor=BORDER,
        darkcolor=BORDER,
        relief="flat",
        padding=(14, 8),
    )
    style.configure(
        "TLabelframe",
        background=SURFACE,
        bordercolor=BORDER,
        lightcolor=BORDER,
        darkcolor=BORDER,
        relief="flat",
        padding=10,
    )
    style.configure(
        "TLabelframe.Label",
        background=SURFACE,
        foreground=TEXT,
        font=("Arial", 10, "bold"),
    )
    style.map(
        "TButton",
        background=[("pressed", "#e3e6e8"), ("active", "#e8eaed")],
        bordercolor=[("focus", PRIMARY), ("active", "#bdc1c6")],
    )
    style.configure(
        "TMenubutton",
        background=SURFACE,
        foreground=MUTED,
        bordercolor=BORDER,
        relief="flat",
        padding=(8, 6),
    )
    style.map("TMenubutton", background=[("active", SURFACE_ALT)])

    style.configure(
        "TEntry",
        fieldbackground=SURFACE,
        background=SURFACE,
        foreground=TEXT,
        bordercolor=BORDER,
        lightcolor=BORDER,
        darkcolor=BORDER,
        insertcolor=TEXT,
        relief="flat",
        padding=(9, 7),
    )
    style.map(
        "TEntry",
        fieldbackground=[
            ("readonly", SURFACE),
            ("disabled", SURFACE),
            ("focus", SURFACE),
        ],
        foreground=[
            ("readonly", TEXT),
            ("disabled", MUTED),
        ],
        bordercolor=[("focus", PRIMARY)],
        lightcolor=[("focus", PRIMARY)],
        darkcolor=[("focus", PRIMARY)],
    )
    style.configure(
        "TCombobox",
        fieldbackground=SURFACE,
        background=SURFACE,
        foreground=TEXT,
        arrowcolor=MUTED,
        bordercolor=BORDER,
        lightcolor=BORDER,
        darkcolor=BORDER,
        relief="flat",
        padding=(8, 6),
    )
    style.map(
        "TCombobox",
        fieldbackground=[("readonly", SURFACE)],
        bordercolor=[("focus", PRIMARY)],
    )
    style.configure(
        "TCheckbutton",
        background=SURFACE,
        foreground=TEXT,
        indicatorcolor=SURFACE,
        bordercolor=BORDER,
        padding=(2, 4),
    )
    style.map(
        "TCheckbutton",
        background=[("active", SURFACE)],
        indicatorcolor=[("selected", PRIMARY)],
    )
    style.configure(
        "Horizontal.TProgressbar",
        troughcolor=SURFACE_ALT,
        background=PRIMARY,
        bordercolor=SURFACE_ALT,
        lightcolor=PRIMARY,
        darkcolor=PRIMARY,
        thickness=8,
    )
    style.configure(
        "TNotebook",
        background=APP_BG,
        borderwidth=0,
        tabmargins=(0, 4, 0, 0),
    )
    style.configure(
        "TNotebook.Tab",
        background=APP_BG,
        foreground=MUTED,
        borderwidth=0,
        padding=(16, 8),
        font=("Arial", 10),
    )
    style.map(
        "TNotebook.Tab",
        background=[("selected", SURFACE), ("active", SURFACE_ALT)],
        foreground=[("selected", PRIMARY), ("active", TEXT)],
    )
    style.configure("TSeparator", background=BORDER)
    return style


def _rounded_polygon(
    canvas: tk.Canvas,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    radius: float,
    **kwargs,
):
    radius = min(radius, (x2 - x1) / 2, (y2 - y1) / 2)
    points = (
        x1 + radius,
        y1,
        x2 - radius,
        y1,
        x2,
        y1,
        x2,
        y1 + radius,
        x2,
        y2 - radius,
        x2,
        y2,
        x2 - radius,
        y2,
        x1 + radius,
        y2,
        x1,
        y2,
        x1,
        y2 - radius,
        x1,
        y1 + radius,
        x1,
        y1,
    )
    return canvas.create_polygon(
        points,
        smooth=True,
        splinesteps=24,
        **kwargs,
    )


class RoundedCard(tk.Frame):
    def __init__(
        self,
        parent,
        *,
        title: str | None = None,
        subtitle: str | None = None,
        radius: int = 16,
        surface: str = SURFACE,
        outside: str = APP_BG,
        border: str = BORDER,
        content_style: str = "Card.TFrame",
        title_style: str = "CardTitle.TLabel",
        subtitle_style: str = "Muted.Card.TLabel",
        padding: tuple[int, int] = (10, 8),
    ):
        super().__init__(parent, background=outside, borderwidth=0)
        self._radius = radius
        self._surface = surface
        self._border = border

        self._canvas = tk.Canvas(
            self,
            background=outside,
            borderwidth=0,
            highlightthickness=0,
        )
        self._canvas.place(x=0, y=0, relwidth=1, relheight=1)

        inset_x, inset_y = padding
        self.content = ttk.Frame(self, style=content_style)
        self.content.pack(
            fill=tk.BOTH,
            expand=True,
            padx=radius + inset_x,
            pady=radius + inset_y,
        )

        if title:
            ttk.Label(
                self.content,
                text=title,
                style=title_style,
            ).pack(anchor="w")
        if subtitle:
            ttk.Label(
                self.content,
                text=subtitle,
                style=subtitle_style,
                wraplength=840,
                justify=tk.LEFT,
            ).pack(anchor="w", pady=(3, 0))

        self.body = ttk.Frame(self.content, style=content_style)
        self.body.pack(
            fill=tk.BOTH,
            expand=True,
            pady=(12 if title or subtitle else 0, 0),
        )
        self.bind("<Configure>", self._redraw)

    def _redraw(self, _event=None):
        width = max(1, self.winfo_width())
        height = max(1, self.winfo_height())
        self._canvas.delete("all")
        _rounded_polygon(
            self._canvas,
            2,
            3,
            width - 1,
            height - 1,
            self._radius,
            fill=SHADOW,
            outline="",
        )
        _rounded_polygon(
            self._canvas,
            1,
            1,
            width - 2,
            height - 3,
            self._radius,
            fill=self._surface,
            outline=self._border,
            width=1,
        )


class AutoScrollbar(ttk.Scrollbar):
    """Grid-managed scrollbar that disappears when its full range is visible."""

    def set(self, first, last):
        if float(first) <= 0.0 and float(last) >= 1.0:
            self.grid_remove()
        else:
            self.grid()
        super().set(first, last)


class Tooltip:
    def __init__(self, widget: tk.Widget, text: str = ""):
        self.widget = widget
        self.text = text
        self.window: tk.Toplevel | None = None
        widget.bind("<Enter>", self._show, add="+")
        widget.bind("<Leave>", self._hide, add="+")

    def set_text(self, text: str):
        self.text = text
        if self.window is not None:
            self._hide()

    def _show(self, _event=None):
        if not self.text or self.window is not None:
            return
        self.window = tk.Toplevel(self.widget)
        self.window.wm_overrideredirect(True)
        self.window.configure(background=TEXT)
        self.window.geometry(
            f"+{self.widget.winfo_rootx() + 12}+"
            f"{self.widget.winfo_rooty() + self.widget.winfo_height() + 6}"
        )
        tk.Label(
            self.window,
            text=self.text,
            background=TEXT,
            foreground=SURFACE,
            justify=tk.LEFT,
            wraplength=360,
            padx=10,
            pady=7,
            font=("Arial", 9),
        ).pack()

    def _hide(self, _event=None):
        if self.window is not None:
            self.window.destroy()
            self.window = None


class RoundedButton(tk.Canvas):
    def __init__(
        self,
        parent,
        *,
        text: str,
        command: Callable[[], None],
        variant: str = "secondary",
        width: int | None = None,
        height: int = 38,
        radius: int = 11,
        canvas_bg: str = SURFACE,
    ):
        self._font = tkfont.Font(
            family="Arial",
            size=10,
            weight="bold" if variant == "primary" else "normal",
        )
        measured_width = self._font.measure(text) + 34
        super().__init__(
            parent,
            width=width or max(88, measured_width),
            height=height,
            background=canvas_bg,
            borderwidth=0,
            highlightthickness=0,
            cursor="hand2",
            takefocus=1,
        )
        self._text = text
        self._command = command
        self._radius = radius
        self._enabled = True
        self._selected = False
        self._pressed = False
        self._hovered = False
        self._variant = variant

        self.bind("<Configure>", self._redraw)
        self.bind("<Enter>", self._enter)
        self.bind("<Leave>", self._leave)
        self.bind("<ButtonPress-1>", self._press)
        self.bind("<ButtonRelease-1>", self._release)
        self.bind("<Return>", self._keyboard_invoke)
        self.bind("<space>", self._keyboard_invoke)
        self.bind("<FocusIn>", self._redraw)
        self.bind("<FocusOut>", self._redraw)
        self._redraw()

    def _colors(self) -> tuple[str, str, str]:
        if self._selected:
            return PRIMARY_SOFT, "#174ea6", PRIMARY_SOFT
        if not self._enabled:
            return "#eef0f1", "#9aa0a6", "#eef0f1"
        if self._variant == "primary":
            fill = (
                PRIMARY_PRESSED
                if self._pressed
                else PRIMARY_HOVER if self._hovered else PRIMARY
            )
            return fill, SURFACE, fill
        if self._variant == "danger":
            fill = "#fce8e6" if self._hovered else SURFACE
            return fill, DANGER, "#f3b8b3"
        fill = "#e8eaed" if self._pressed else SURFACE_ALT if self._hovered else SURFACE
        return fill, TEXT, BORDER

    def _redraw(self, _event=None):
        width = max(4, self.winfo_width(), self.winfo_reqwidth())
        height = max(4, self.winfo_height(), self.winfo_reqheight())
        fill, foreground, border = self._colors()
        self.delete("all")
        _rounded_polygon(
            self,
            1,
            1,
            width - 2,
            height - 2,
            self._radius,
            fill=fill,
            outline=PRIMARY if self.focus_get() is self else border,
            width=2 if self.focus_get() is self else 1,
        )
        self.create_text(
            width / 2,
            height / 2,
            text=self._text,
            fill=foreground,
            font=self._font,
        )

    def _enter(self, _event):
        if self._enabled:
            self._hovered = True
            self._redraw()

    def _leave(self, _event):
        self._hovered = False
        self._pressed = False
        self._redraw()

    def _press(self, _event):
        if self._enabled:
            self.focus_set()
            self._pressed = True
            self._redraw()

    def _release(self, event):
        should_invoke = (
            self._enabled
            and self._pressed
            and 0 <= event.x <= self.winfo_width()
            and 0 <= event.y <= self.winfo_height()
        )
        self._pressed = False
        self._redraw()
        if should_invoke:
            self._command()

    def _keyboard_invoke(self, _event):
        if self._enabled:
            self._command()
        return "break"

    def configure(self, cnf=None, **kwargs):
        state = None
        selected = None
        if isinstance(cnf, dict):
            state = cnf.pop("state", None)
            selected = cnf.pop("selected", None)
        if "state" in kwargs:
            state = kwargs.pop("state")
        if "selected" in kwargs:
            selected = kwargs.pop("selected")
        result = super().configure(cnf, **kwargs)
        if selected is not None:
            self._selected = bool(selected)
        if state is not None:
            self._enabled = str(state) != str(tk.DISABLED)
            self.configure(cursor="hand2" if self._enabled else "")
        if state is not None or selected is not None:
            self._redraw()
        return result

    config = configure
