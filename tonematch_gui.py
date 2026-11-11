"""
ToneMatch – GUI
Start:  python tonematch_gui.py   (tonematch_core.py muss im selben Ordner liegen)
"""
import json
import math
import os
import subprocess
import sys
import queue
import threading
import time
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog

import numpy as np
import tonematch_core as core

try:  # scharfe Darstellung auf Windows mit Skalierung
    import ctypes
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass

# Farbwelt: Tolex, Messing-Frontplatte, Creme-Beschriftung, rotes Jewel-Lämpchen
TOLEX = "#1f1b18"
PANEL = "#2a2521"
FIELD = "#36302a"
LINE = "#4a4139"
BRASS = "#b8995e"
BRASS_HI = "#cdb27a"
BRASS_LO = "#8a7142"
INK = "#1d1812"
CREAM = "#efe6d2"
MUTED = "#a89c88"
AMBER = "#e8a33d"
TEAL = "#6fb7a8"
JEWEL_ON = "#e0452f"
JEWEL_OFF = "#4a2420"

DISPLAY = "Bahnschrift"
BODY = "Segoe UI"
# Als EXE (PyInstaller) liegen die Daten neben ToneMatch.exe, nicht im _internal-Ordner
if getattr(sys, "frozen", False):
    APP_DIR = Path(sys.executable).resolve().parent
else:
    APP_DIR = Path(__file__).resolve().parent
TONES_DIR = APP_DIR / "tones"
RECORDINGS_DIR = APP_DIR / "recordings"
STEMS_DIR = APP_DIR / "stems"


def db_fmt(v):
    return f"{v:+.1f} dB"


KNOBS = [
    ("input_trim_db", "Trim", -24, 36, 0.0, db_fmt, False),
    ("gate_db", "Gate", -90, -30, -90.0, lambda v: "aus" if v <= -89.5 else f"{v:.0f} dB", False),
    ("gain_db", "Gain", -10, 50, 20.0, lambda v: f"{v:.1f} dB", False),
    ("tight_hz", "Tight", 20, 250, 80.0, lambda v: f"{v:.0f} Hz", True),
    ("mid_push_db", "Push", 0, 12, 2.0, lambda v: f"{v:.1f} dB", False),
    ("bass_db", "Bass", -12, 12, 0.0, db_fmt, False),
    ("mid_db", "Middle", -12, 12, 0.0, db_fmt, False),
    ("treble_db", "Treble", -12, 12, 0.0, db_fmt, False),
    ("cab_lp_hz", "Presence", 2500, 9000, 5500.0, lambda v: f"{v / 1000:.1f} kHz", True),
    ("match_strength", "Match", 0, 1, 0.8, lambda v: f"{v * 100:.0f} %", False),
    ("master_db", "Master", -40, 12, -12.0, db_fmt, False),
]


def pct(v):
    return f"{v * 100:.0f} %"


PEDALS = [
    ("overdrive", "Overdrive", "#4e8a4a", [
        ("drive", "Drive", 0, 1, 0.5, pct, False), ("tone", "Tone", 0, 1, 0.5, pct, False),
        ("level", "Level", -12, 12, 0.0, db_fmt, False)]),
    ("distortion", "Distortion", "#b8502a", [
        ("gain", "Gain", 0, 1, 0.5, pct, False), ("tone", "Tone", 0, 1, 0.5, pct, False),
        ("level", "Level", -12, 12, 0.0, db_fmt, False)]),
    ("chorus", "Chorus", "#3d7aa0", [
        ("rate", "Rate", 0.1, 4, 0.8, lambda v: f"{v:.1f} Hz", True),
        ("depth", "Depth", 0, 1, 0.5, pct, False), ("mix", "Mix", 0, 1, 0.5, pct, False)]),
    ("delay", "Delay", "#55606e", [
        ("time_ms", "Time", 30, 1500, 380.0, lambda v: f"{v:.0f} ms", True),
        ("feedback", "Repeats", 0, 0.9, 0.35, pct, False), ("mix", "Mix", 0, 1, 0.3, pct, False),
        ("tone", "Tone", 0, 1, 0.6, pct, False)]),
    ("reverb", "Reverb", "#6c4e86", [
        ("size", "Size", 0, 1, 0.6, pct, False), ("tone", "Tone", 0, 1, 0.5, pct, False),
        ("mix", "Mix", 0, 1, 0.25, pct, False)]),
]
DELAY_DIVISIONS = {"1/2": 2.0, "1/4": 1.0, "1/8 punktiert": 0.75, "1/8": 0.5, "1/16": 0.25}


def mix(c1, c2, t):
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{int(x + (y - x) * t):02x}" for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------

class Knob(tk.Canvas):
    """Drehknopf: ziehen (Shift = fein), Mausrad, Doppelklick = Standardwert."""

    def __init__(self, master, label, vmin, vmax, default, fmt, command=None, log=False, size=58,
                 bg=BRASS, ink=INK, pad=34):
        self.small = size < 50
        super().__init__(master, width=size + pad, height=size + (44 if self.small else 52), bg=bg,
                         highlightthickness=0, cursor="hand2")
        self.label, self.vmin, self.vmax = label, vmin, vmax
        self.default, self.fmt, self.command, self.log, self.size = default, fmt, command, log, size
        self.bg_col, self.ink, self.pad = bg, ink, pad
        self.value = default
        self.bind("<ButtonPress-1>", self._press)
        self.bind("<B1-Motion>", self._drag)
        self.bind("<Double-Button-1>", lambda e: self.set(self.default))
        self.bind("<MouseWheel>", self._wheel)
        self._draw()

    def _to_norm(self, v):
        if self.log:
            return math.log(v / self.vmin) / math.log(self.vmax / self.vmin)
        return (v - self.vmin) / (self.vmax - self.vmin)

    def _from_norm(self, n):
        n = min(max(n, 0.0), 1.0)
        if self.log:
            return float(self.vmin * (self.vmax / self.vmin) ** n)
        return float(self.vmin + n * (self.vmax - self.vmin))

    def _press(self, e):
        self._y0, self._n0 = e.y, self._to_norm(self.value)

    def _drag(self, e):
        scale = 900.0 if e.state & 0x0001 else 220.0
        self.set(self._from_norm(self._n0 + (self._y0 - e.y) / scale))

    def _wheel(self, e):
        self.set(self._from_norm(self._to_norm(self.value) + (0.02 if e.delta > 0 else -0.02)))

    def _draw(self):
        self.delete("all")
        s = self.size
        cx, cy, r = (s + self.pad) / 2, 10 + s / 2 + 4, s / 2
        ink = self.ink
        for i in range(11):
            a = math.radians(225 - 27 * i)
            r1, r2 = r + 3, r + (8 if i % 5 == 0 else 6)
            self.create_line(cx + r1 * math.cos(a), cy - r1 * math.sin(a),
                             cx + r2 * math.cos(a), cy - r2 * math.sin(a), fill=ink, width=1.5)
        self.create_oval(cx - r + 2, cy - r + 4, cx + r + 2, cy + r + 4,
                         fill=mix(self.bg_col, "#000000", 0.35), outline="")
        self.create_oval(cx - r, cy - r, cx + r, cy + r, fill="#16130f", outline="#000")
        for i in range(24):
            a = math.radians(i * 15)
            self.create_line(cx + (r - 1) * math.cos(a), cy - (r - 1) * math.sin(a),
                             cx + (r - 4) * math.cos(a), cy - (r - 4) * math.sin(a), fill="#2c2620")
        ri = r - (5 if self.small else 7)
        self.create_oval(cx - ri, cy - ri, cx + ri, cy + ri, fill="#221d18", outline="#0d0b09")
        a = math.radians(225 - 270 * self._to_norm(self.value))
        self.create_line(cx + ri * 0.15 * math.cos(a), cy - ri * 0.15 * math.sin(a),
                         cx + ri * 0.95 * math.cos(a), cy - ri * 0.95 * math.sin(a),
                         fill=CREAM, width=2 if self.small else 3, capstyle="round")
        if self.small:
            self.create_text(cx, cy + r + 12, text=self.label, fill=ink, font=(DISPLAY, 8, "bold"))
            self.create_text(cx, cy + r + 24, text=self.fmt(self.value), fill=ink, font=(BODY, 7))
        else:
            self.create_text(cx, cy + r + 14, text=self.label, fill=ink, font=(DISPLAY, 10, "bold"))
            self.create_text(cx, cy + r + 29, text=self.fmt(self.value), fill=ink, font=(BODY, 8))

    def set(self, v, notify=True):
        v = min(max(float(v), self.vmin), self.vmax)
        if v == self.value:
            return
        self.value = v
        self._draw()
        if notify and self.command:
            self.command(v)


class Meter(tk.Canvas):
    def __init__(self, master, label, width=220):
        super().__init__(master, width=width, height=18, bg=PANEL, highlightthickness=0)
        self.label, self.w, self.disp = label, width, -60.0
        self._draw()

    def set(self, peak):
        db = 20 * math.log10(peak + 1e-9)
        self.disp = max(db, self.disp - 1.8)
        self._draw()

    def _draw(self):
        self.delete("all")
        self.create_text(4, 9, text=self.label, anchor="w", fill=MUTED, font=(BODY, 8))
        w = self.winfo_width() if self.winfo_width() > 60 else self.w
        x0, x1 = 34, w - 4
        self.create_rectangle(x0, 4, x1, 14, fill=FIELD, outline="")
        n = min(max((self.disp + 60) / 60, 0), 1)
        col = "#7fb069" if self.disp < -12 else AMBER if self.disp < -2 else JEWEL_ON
        if n > 0:
            self.create_rectangle(x0, 4, x0 + (x1 - x0) * n, 14, fill=col, outline="")
        for db in (-12, -2):
            x = x0 + (x1 - x0) * (db + 60) / 60
            self.create_line(x, 3, x, 15, fill=TOLEX)


class CharacterBar(tk.Canvas):
    """Zeigt, wie stark verzerrt der Referenz-Sound ist."""

    def __init__(self, master, width=250):
        super().__init__(master, width=width, height=48, bg=PANEL, highlightthickness=0)
        self.w, self.value = width, None
        self._draw()

    def set(self, v):
        self.value = v
        self._draw()

    def _draw(self):
        self.delete("all")
        x0, x1, y0, y1 = 6, self.w - 6, 14, 22
        steps = 50
        for i in range(steps):
            t = i / (steps - 1)
            col = mix("#7fb069", AMBER, t * 2) if t < 0.5 else mix(AMBER, JEWEL_ON, (t - 0.5) * 2)
            self.create_rectangle(x0 + (x1 - x0) * i / steps, y0,
                                  x0 + (x1 - x0) * (i + 1) / steps, y1, fill=col, outline="")
        for t, lab in ((0.0, "Clean"), (0.45, "Crunch"), (1.0, "High Gain")):
            anchor = "w" if t == 0 else "e" if t == 1 else "center"
            self.create_text(x0 + (x1 - x0) * t, 36, text=lab, anchor=anchor, fill=MUTED, font=(BODY, 8))
        if self.value is not None:
            x = x0 + (x1 - x0) * self.value
            self.create_polygon(x - 6, 3, x + 6, 3, x, 12, fill=CREAM, outline="")
            self.create_line(x, 12, x, y1 + 2, fill=CREAM, width=2)


class SpectrumView(tk.Canvas):
    FMIN, FMAX, DBMAX, DBMIN = 40.0, 12000.0, 15.0, -35.0

    def __init__(self, master):
        super().__init__(master, bg="#181512", highlightthickness=0, height=220)
        self.ref = self.live = self.match = None
        self.bind("<Configure>", lambda e: self.redraw())

    def _x(self, f, w):
        return 44 + np.log(f / self.FMIN) / np.log(self.FMAX / self.FMIN) * (w - 58)

    def _y(self, db, h):
        return 14 + (self.DBMAX - np.clip(db, self.DBMIN, self.DBMAX)) / (self.DBMAX - self.DBMIN) * (h - 40)

    def redraw(self):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w < 120:
            return
        for f, lab in ((50, "50"), (100, "100"), (200, "200"), (500, "500"), (1000, "1k"),
                       (2000, "2k"), (5000, "5k"), (10000, "10k")):
            x = float(self._x(f, w))
            self.create_line(x, 14, x, h - 26, fill="#2b2622")
            self.create_text(x, h - 13, text=lab, fill=MUTED, font=(BODY, 8))
        for db in range(-30, 16, 10):
            y = float(self._y(db, h))
            self.create_line(44, y, w - 14, y, fill="#3a332c" if db == 0 else "#2b2622")
            self.create_text(22, y, text=f"{db:+d}", fill=MUTED, font=(BODY, 8))

        m = (core.LOG_F >= self.FMIN) & (core.LOG_F <= self.FMAX)
        fs = core.LOG_F[m]

        def curve(db, color, width, dash=None):
            pts = np.column_stack([self._x(fs, w), self._y(np.asarray(db)[m], h)]).ravel().tolist()
            self.create_line(*pts, fill=color, width=width, smooth=True, dash=dash)

        if self.match is not None:
            curve(self.match, TEAL, 2, (5, 3))
        if self.live is not None:
            curve(self.live, CREAM, 1.5)
        if self.ref is not None:
            curve(self.ref, AMBER, 2.5)
        if self.ref is None and self.live is None:
            self.create_text(w / 2, h / 2 - 8, fill=MUTED, font=(BODY, 10),
                             text="Lade einen Song – hier erscheint sein Klangbild.")


class BeatDots(tk.Canvas):
    def __init__(self, master):
        super().__init__(master, width=200, height=26, bg=PANEL, highlightthickness=0)
        self.beats, self.active = 4, None
        self._draw()

    def set(self, beats, active):
        if (beats, active) != (self.beats, self.active):
            self.beats, self.active = beats, active
            self._draw()

    def _draw(self):
        self.delete("all")
        step = min(24, 196 / max(self.beats, 1))
        x0 = 200 - step * self.beats
        for i in range(self.beats):
            x = x0 + step * i + step / 2
            on = i == self.active
            col = (JEWEL_ON if i == 0 else AMBER) if on else FIELD
            r = 8 if i == 0 else 6
            self.create_oval(x - r, 13 - r, x + r, 13 + r, fill=col, outline=LINE)


class TunerView(tk.Canvas):
    """Zeiger-Stimmgerät: -50 bis +50 Cent."""

    def __init__(self, master):
        super().__init__(master, bg="#181512", highlightthickness=0, height=260)
        self.state = None        # (Notenname, Cent, Frequenz)
        self.needle = 0.0
        self.bind("<Configure>", lambda e: self.redraw())

    def set(self, state):
        self.state = state
        if state is not None:
            self.needle += (state[1] - self.needle) * 0.4
        self.redraw()

    def redraw(self):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w < 120 or h < 120:
            return
        cx, cy = w / 2, h * 0.58
        r = min(w * 0.40, h * 0.48)
        for c in range(-50, 51, 5):
            a = math.radians(90 - c * 1.2)
            major = c % 25 == 0
            r1 = r - (16 if major else 8)
            self.create_line(cx + r1 * math.cos(a), cy - r1 * math.sin(a),
                             cx + r * math.cos(a), cy - r * math.sin(a),
                             fill=TEAL if abs(c) <= 5 else MUTED, width=2 if major else 1)
            if major:
                self.create_text(cx + (r + 16) * math.cos(a), cy - (r + 16) * math.sin(a),
                                 text="0" if c == 0 else f"{c:+d}", fill=MUTED, font=(BODY, 9))
        self.create_arc(cx - r - 4, cy - r - 4, cx + r + 4, cy + r + 4, start=84, extent=12,
                        style="arc", outline=TEAL, width=4)
        if self.state is None:
            self.create_text(cx, cy + h * 0.14, text="Schlag eine Saite an", fill=MUTED, font=(BODY, 13))
            return
        name, cents, freq = self.state
        ok = abs(cents) < 3
        col = TEAL if ok else AMBER if abs(cents) < 15 else JEWEL_ON
        a = math.radians(90 - max(-50.0, min(50.0, self.needle)) * 1.2)
        self.create_line(cx, cy, cx + (r - 6) * math.cos(a), cy - (r - 6) * math.sin(a),
                         fill=col, width=4, capstyle="round")
        self.create_oval(cx - 8, cy - 8, cx + 8, cy + 8, fill=col, outline="")
        self.create_text(cx, cy + h * 0.15, text=name, fill=col if ok else CREAM,
                         font=(DISPLAY, max(24, int(h * 0.13)), "bold"))
        hint = "gestimmt" if ok else ("zu tief" if cents < 0 else "zu hoch")
        self.create_text(cx, cy + h * 0.29, text=f"{freq:.1f} Hz   {cents:+.0f} Cent   {hint}",
                         fill=MUTED, font=(BODY, 10))


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ToneMatch")
        self.configure(bg=TOLEX)
        self.geometry("1320x900")
        self.minsize(1200, 820)

        self.engine = core.Engine()
        self.params = dict(core.DEFAULT_PARAMS)
        self.ref_log = None
        self.features = None
        self.song_name = None
        self.song_path = None
        self.live_db = None
        self.devices = []
        self.busy = False
        self._pending = False
        self._tick = 0
        self.taps = []
        self.player_mode = None
        self.backing_cache = None
        self.stems = None            # (Gitarre, Song ohne Gitarre)
        self.pitch_hist = []
        self.pitch_last = 0.0
        self.q = queue.Queue()

        self._style()
        self._build()
        self._refresh_devices()
        self._refresh_tones()
        self._refresh_takes()
        self.after(50, self._poll)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------- Aufbau ----------

    def _style(self):
        st = ttk.Style(self)
        st.theme_use("clam")
        for w in ("TCombobox", "TSpinbox"):
            st.configure(w, fieldbackground=FIELD, background=FIELD, foreground=CREAM,
                         arrowcolor=CREAM, bordercolor=LINE, lightcolor=FIELD, darkcolor=FIELD)
            st.map(w, fieldbackground=[("readonly", FIELD)], foreground=[("readonly", CREAM)])
        st.configure("Horizontal.TProgressbar", troughcolor=FIELD, background=AMBER,
                     bordercolor=FIELD, lightcolor=AMBER, darkcolor=AMBER)
        st.configure("TNotebook", background=TOLEX, borderwidth=0)
        st.configure("TNotebook.Tab", background=PANEL, foreground=MUTED, padding=(18, 7),
                     font=(DISPLAY, 11, "bold"), borderwidth=0)
        st.map("TNotebook.Tab", background=[("selected", FIELD)], foreground=[("selected", CREAM)])
        self.option_add("*TCombobox*Listbox.background", FIELD)
        self.option_add("*TCombobox*Listbox.foreground", CREAM)
        self.option_add("*TCombobox*Listbox.selectBackground", AMBER)

    def _card(self, parent, title):
        outer = tk.Frame(parent, bg=PANEL, highlightbackground=LINE, highlightthickness=1)
        tk.Label(outer, text=title, bg=PANEL, fg=CREAM, font=(DISPLAY, 12, "bold"),
                 anchor="w").pack(fill="x", padx=14, pady=(10, 4))
        inner = tk.Frame(outer, bg=PANEL)
        inner.pack(fill="both", expand=True, padx=14, pady=(0, 12))
        return outer, inner

    def _button(self, parent, text, cmd, strong=False):
        return tk.Button(parent, text=text, command=cmd, relief="flat", bd=0, cursor="hand2",
                         bg=AMBER if strong else FIELD, fg=INK if strong else CREAM,
                         activebackground=BRASS_HI if strong else LINE,
                         activeforeground=INK if strong else CREAM,
                         font=(BODY, 9, "bold"), padx=12, pady=6)

    def _scale(self, parent, value, command, length=110):
        sc = tk.Scale(parent, from_=0, to=100, orient="horizontal", length=length, showvalue=0,
                      bg=BRASS, troughcolor=FIELD, activebackground=BRASS_HI, highlightthickness=0,
                      sliderrelief="flat", sliderlength=16, width=10, bd=1, relief="flat",
                      command=lambda v: command())
        sc.set(value)
        return sc

    def _check(self, parent, text, var, command):
        return tk.Checkbutton(parent, text=text, variable=var, command=command, bg=PANEL, fg=CREAM,
                              selectcolor=FIELD, activebackground=PANEL, activeforeground=CREAM,
                              font=(BODY, 9), highlightthickness=0, bd=0)

    def _label(self, parent, text, muted=True, **kw):
        return tk.Label(parent, text=text, bg=PANEL, fg=MUTED if muted else CREAM,
                        font=(BODY, 9), anchor="w", **kw)

    def _build(self):
        # Kopfzeile
        top = tk.Frame(self, bg=TOLEX)
        top.pack(fill="x", padx=18, pady=(14, 8))
        tk.Label(top, text="ToneMatch", bg=TOLEX, fg=CREAM, font=(DISPLAY, 22, "bold")).pack(side="left")
        tk.Label(top, text="Spiel wie der Song klingt", bg=TOLEX, fg=MUTED,
                 font=(BODY, 10)).pack(side="left", padx=(14, 0), pady=(8, 0))
        self.status_var = tk.StringVar(value="Lade einen Referenz-Song und starte das Audio.")
        tk.Label(top, textvariable=self.status_var, bg=TOLEX, fg=MUTED, font=(BODY, 9)).pack(side="right")

        main = tk.Frame(self, bg=TOLEX)
        main.pack(fill="both", expand=True, padx=18)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        # Links: Song + Bibliothek
        left = tk.Frame(main, bg=TOLEX)
        left.grid(row=0, column=0, sticky="ns", padx=(0, 12))

        card, c = self._card(left, "Referenz-Song")
        card.pack(fill="x")
        self._button(c, "Song laden …", self._choose_song, strong=True).pack(fill="x")
        self.song_var = tk.StringVar(value="Noch kein Song geladen")
        tk.Label(c, textvariable=self.song_var, bg=PANEL, fg=CREAM, font=(BODY, 9),
                 wraplength=250, justify="left", anchor="w").pack(fill="x", pady=(8, 6))
        row = tk.Frame(c, bg=PANEL)
        row.pack(fill="x")
        self._label(row, "Nur Abschnitt von").pack(side="left")
        self.from_var, self.to_var = tk.StringVar(), tk.StringVar()
        for var, txt in ((self.from_var, "bis"), (self.to_var, "s")):
            tk.Entry(row, textvariable=var, width=5, bg=FIELD, fg=CREAM, insertbackground=CREAM,
                     relief="flat", font=(BODY, 9)).pack(side="left", padx=4)
            self._label(row, txt).pack(side="left")
        self._label(c, "Leer lassen = ganzer Song. Am besten eine Stelle mit viel Gitarre.",
                    wraplength=250, justify="left").pack(fill="x", pady=(4, 6))
        self.isolate_var = tk.BooleanVar(value=False)
        iso = self._check(c, "Gitarre aus dem Song isolieren", self.isolate_var, lambda: None)
        iso.pack(anchor="w", pady=(0, 2))
        if not core.demucs_available():
            iso.configure(state="disabled")
            self._label(c, "Dafür einmalig im Terminal:  pip install demucs",
                        wraplength=250, justify="left").pack(fill="x", pady=(0, 4))
        row = tk.Frame(c, bg=PANEL)
        row.pack(fill="x")
        self._button(row, "Neu analysieren", self._analyze).pack(side="left", expand=True, fill="x")
        self._button(row, "Gitarre anhören", self._play_stem).pack(side="left", expand=True, fill="x", padx=(4, 0))
        self.char_bar = CharacterBar(c)
        self.char_bar.pack(pady=(10, 0))
        self.feat_var = tk.StringVar(value="")
        self._label(c, "", textvariable=self.feat_var, wraplength=250, justify="left").pack(fill="x")

        card, c = self._card(left, "Meine Tones")
        card.pack(fill="both", expand=True, pady=(12, 0))
        self.tone_list = tk.Listbox(c, bg=FIELD, fg=CREAM, selectbackground=AMBER, selectforeground=INK,
                                    relief="flat", highlightthickness=0, font=(BODY, 10), height=7,
                                    activestyle="none")
        self.tone_list.pack(fill="both", expand=True)
        self.tone_list.bind("<<ListboxSelect>>", self._on_tone_select)
        row = tk.Frame(c, bg=PANEL)
        row.pack(fill="x", pady=(8, 0))
        self._button(row, "Speichern", self._save_tone).pack(side="left", expand=True, fill="x")
        self._button(row, "Öffnen …", self._open_tone).pack(side="left", expand=True, fill="x", padx=4)
        self._button(row, "Löschen", self._delete_tone).pack(side="left", expand=True, fill="x")
        self._button(c, "Als REAPER-Plugin exportieren …", self._export_reaper).pack(fill="x", pady=(6, 0))

        # Mitte: Tabs + Metronom
        center = tk.Frame(main, bg=TOLEX)
        center.grid(row=0, column=1, sticky="nsew")
        self.tabs = ttk.Notebook(center)
        self.tabs.pack(fill="both", expand=True)
        self.tabs.bind("<<NotebookTabChanged>>", lambda e: self._on_tab())
        tab = tk.Frame(self.tabs, bg=PANEL)
        self.tabs.add(tab, text="Klangbild")
        tuner_tab = tk.Frame(self.tabs, bg=PANEL)
        self.tabs.add(tuner_tab, text="Stimmgerät")
        self._build_tuner(tuner_tab)
        rec_tab = tk.Frame(self.tabs, bg=PANEL)
        self.tabs.add(rec_tab, text="Aufnahme")
        self._build_recorder(rec_tab)
        fx_tab = tk.Frame(self.tabs, bg=PANEL)
        self.tabs.add(fx_tab, text="Effekte")
        self._build_pedals(fx_tab)
        card, mc = self._card(center, "Metronom")
        card.pack(fill="x", pady=(12, 0))
        self._build_metronome(mc)

        c = tk.Frame(tab, bg=PANEL)
        c.pack(fill="both", expand=True, padx=14, pady=12)
        self.spectrum = SpectrumView(c)
        self.spectrum.pack(fill="both", expand=True)
        legend = tk.Frame(c, bg=PANEL)
        legend.pack(fill="x", pady=(8, 0))
        for col, txt in ((AMBER, "Song"), (CREAM, "Dein Signal (live)"), (TEAL, "Angleichung")):
            tk.Label(legend, text="━━", bg=PANEL, fg=col, font=(BODY, 10, "bold")).pack(side="left")
            self._label(legend, txt).pack(side="left", padx=(4, 16))

        # Rechts: Audio + Tone Match
        right = tk.Frame(main, bg=TOLEX)
        right.grid(row=0, column=2, sticky="ns", padx=(12, 0))

        card, c = self._card(right, "Audio-Interface")
        card.pack(fill="x")
        self._label(c, "Eingang (ASIO)").pack(fill="x")
        self.dev_var = tk.StringVar()
        self.dev_combo = ttk.Combobox(c, textvariable=self.dev_var, state="readonly", width=30)
        self.dev_combo.pack(fill="x")
        self.dev_combo.bind("<<ComboboxSelected>>", lambda e: self._on_device())
        self._label(c, "Ausgabe").pack(fill="x", pady=(6, 0))
        self.outdev_var = tk.StringVar()
        self.outdev_combo = ttk.Combobox(c, textvariable=self.outdev_var, state="readonly", width=30)
        self.outdev_combo.pack(fill="x")
        self.outdev_combo.bind("<<ComboboxSelected>>", lambda e: self._on_outdev())
        grid = tk.Frame(c, bg=PANEL)
        grid.pack(fill="x", pady=(8, 0))
        self.in_var = tk.StringVar(value="1")
        self.out_var = tk.StringVar(value="1/2")
        self.sr_var = tk.StringVar(value="48000")
        self.block_var = tk.StringVar(value="128")
        self.in_spin = ttk.Spinbox(grid, from_=1, to=2, textvariable=self.in_var, width=8, state="readonly")
        self.out_combo = ttk.Combobox(grid, textvariable=self.out_var, values=["1/2"], width=8, state="readonly")
        rows = [("Gitarren-Eingang", self.in_spin), ("Ausgangskanäle", self.out_combo),
                ("Samplerate", ttk.Combobox(grid, textvariable=self.sr_var, width=8, state="readonly",
                                            values=["44100", "48000", "88200", "96000"])),
                ("Puffer", ttk.Combobox(grid, textvariable=self.block_var, width=8, state="readonly",
                                        values=["64", "128", "256", "512"]))]
        for r, (txt, w) in enumerate(rows):
            self._label(grid, txt).grid(row=r, column=0, sticky="w", pady=2)
            w.grid(row=r, column=1, sticky="e", pady=2)
        grid.columnconfigure(0, weight=1)

        row = tk.Frame(c, bg=PANEL)
        row.pack(fill="x", pady=(10, 6))
        self.jewel = tk.Canvas(row, width=22, height=22, bg=PANEL, highlightthickness=0)
        self.jewel.pack(side="left")
        self._draw_jewel(False)
        self.engine_btn = self._button(row, "Audio starten", self._toggle_engine, strong=True)
        self.engine_btn.pack(side="left", fill="x", expand=True, padx=(8, 0))
        self.meter_in = Meter(c, "In")
        self.meter_in.pack(fill="x")
        self.meter_out = Meter(c, "Out")
        self.meter_out.pack(fill="x", pady=(2, 0))
        self.xrun_var = tk.StringVar(value="")
        self._label(c, "", textvariable=self.xrun_var).pack(fill="x")

        card, c = self._card(right, "Tone Match")
        card.pack(fill="both", expand=True, pady=(12, 0))
        self._label(c, "Spiel Akkorde, Riffs und Einzeltöne. Danach wird dein Signal "
                       "an den Song angeglichen.", wraplength=250, justify="left").pack(fill="x")
        row = tk.Frame(c, bg=PANEL)
        row.pack(fill="x", pady=(8, 8))
        self._label(row, "Aufnahmedauer").pack(side="left")
        self.secs_var = tk.StringVar(value="30")
        ttk.Spinbox(row, from_=10, to=120, increment=5, textvariable=self.secs_var,
                    width=6).pack(side="right")
        self.match_btn = tk.Button(c, text="Tone Match starten", command=self._tone_match,
                                   relief="flat", bd=0, cursor="hand2", bg=JEWEL_ON, fg=CREAM,
                                   activebackground="#f05a44", activeforeground=CREAM,
                                   font=(DISPLAY, 13, "bold"), pady=10)
        self.match_btn.pack(fill="x")
        self.progress = ttk.Progressbar(c, maximum=100)
        self.progress.pack(fill="x", pady=(8, 6))
        self.match_var = tk.BooleanVar(value=True)
        tk.Checkbutton(c, text="Angleichung aktiv", variable=self.match_var, command=self._on_match_toggle,
                       bg=PANEL, fg=CREAM, selectcolor=FIELD, activebackground=PANEL,
                       activeforeground=CREAM, font=(BODY, 9), anchor="w",
                       highlightthickness=0, bd=0).pack(fill="x")

        # Unten: Messing-Frontplatte mit Reglern
        plate = tk.Frame(self, bg=BRASS, highlightbackground=BRASS_LO, highlightthickness=2)
        plate.pack(fill="x", padx=18, pady=(12, 16))
        tk.Frame(plate, bg=BRASS_HI, height=2).pack(fill="x")
        strip = tk.Frame(plate, bg=BRASS)
        strip.pack(pady=(6, 4))
        self.knobs = {}
        for key, label, vmin, vmax, default, fmt, log in KNOBS:
            k = Knob(strip, label, vmin, vmax, default, fmt,
                     command=lambda v, key=key: self._on_knob(key, v), log=log)
            k.pack(side="left", padx=2)
            self.knobs[key] = k
            if key in ("gate_db", "mid_push_db", "cab_lp_hz", "match_strength"):
                tk.Frame(strip, bg=BRASS_LO, width=2, height=80).pack(side="left", padx=6)

    def _draw_jewel(self, on):
        self.jewel.delete("all")
        col = JEWEL_ON if on else JEWEL_OFF
        if on:
            self.jewel.create_oval(1, 1, 21, 21, fill=mix(JEWEL_ON, PANEL, 0.6), outline="")
        self.jewel.create_oval(4, 4, 18, 18, fill=col, outline="#000")
        self.jewel.create_oval(7, 6, 11, 10, fill=mix(col, "#ffffff", 0.5), outline="")

    def _build_metronome(self, c):
        row = tk.Frame(c, bg=PANEL)
        row.pack(fill="x")
        self.metro_btn = self._button(row, "Start", self._toggle_metro, strong=True)
        self.metro_btn.configure(width=7)
        self.metro_btn.pack(side="left")
        self._button(row, "−", lambda: self._bpm_step(-1)).pack(side="left", padx=(14, 0))
        self.bpm_var = tk.StringVar(value="120")
        e = tk.Entry(row, textvariable=self.bpm_var, width=4, justify="center", bg=FIELD, fg=CREAM,
                     insertbackground=CREAM, relief="flat", font=(DISPLAY, 16, "bold"))
        e.pack(side="left", padx=4, ipady=1)
        e.bind("<Return>", lambda ev: self._apply_metro())
        e.bind("<FocusOut>", lambda ev: self._apply_metro())
        e.bind("<MouseWheel>", lambda ev: self._bpm_step(1 if ev.delta > 0 else -1))
        self._button(row, "+", lambda: self._bpm_step(1)).pack(side="left")
        self._label(row, "BPM").pack(side="left", padx=(6, 14))
        self._button(row, "Tap", self._tap).pack(side="left")
        self.beat_dots = BeatDots(row)
        self.beat_dots.pack(side="right")

        row = tk.Frame(c, bg=PANEL)
        row.pack(fill="x", pady=(8, 0))
        self._label(row, "Takt").pack(side="left")
        self.sig_var = tk.StringVar(value="4/4")
        cb = ttk.Combobox(row, textvariable=self.sig_var, width=5, state="readonly",
                          values=["2/4", "3/4", "4/4", "5/4", "6/8", "7/8"])
        cb.pack(side="left", padx=(6, 16))
        cb.bind("<<ComboboxSelected>>", lambda e: self._apply_metro())
        self.accent_var = tk.BooleanVar(value=True)
        self._check(row, "Betonung auf 1", self.accent_var, self._apply_metro).pack(side="left")
        self.metro_vol = self._scale(row, 50, self._apply_metro)
        self.metro_vol.pack(side="right")
        self._label(row, "Lautstärke").pack(side="right", padx=(16, 6))

    def _build_pedals(self, tab):
        c = tk.Frame(tab, bg=PANEL)
        c.pack(fill="both", expand=True, padx=14, pady=10)
        self._label(c, "Gitarre → Overdrive → Distortion → Amp → Chorus → Delay → Reverb",
                    muted=False).pack(anchor="w")
        self._label(c, "Fußschalter anklicken = an/aus. Doppelklick auf einen Regler setzt ihn zurück.").pack(
            anchor="w", pady=(0, 6))
        rows = [tk.Frame(c, bg=PANEL), tk.Frame(c, bg=PANEL)]
        for r in rows:
            r.pack(anchor="w")
        self.pedal_ui = {}
        for idx, (name, title, color, knobs) in enumerate(PEDALS):
            ink = CREAM
            box = tk.Frame(rows[0 if idx < 3 else 1], bg=color,
                           highlightbackground=mix(color, "#000000", 0.45), highlightthickness=2)
            box.pack(side="left", anchor="n", padx=(0, 10), pady=5)
            head = tk.Frame(box, bg=color)
            head.pack(fill="x", padx=8, pady=(6, 0))
            tk.Label(head, text=title, bg=color, fg=ink, font=(DISPLAY, 11, "bold")).pack(side="left")
            led = tk.Canvas(head, width=14, height=14, bg=color, highlightthickness=0)
            led.pack(side="right")
            krow = tk.Frame(box, bg=color)
            krow.pack(padx=6)
            kw = {}
            for key, label, vmin, vmax, default, fmt, log in knobs:
                k = Knob(krow, label, vmin, vmax, default, fmt, log=log, size=32, bg=color, ink=ink, pad=22,
                         command=lambda v, n=name, kk=key: self._on_fx(n, kk, v))
                k.pack(side="left")
                kw[key] = k
            if name == "delay":
                srow = tk.Frame(box, bg=color)
                srow.pack(fill="x", padx=8)
                self.delay_sync_var = tk.BooleanVar(value=False)
                tk.Checkbutton(srow, text="Sync zum Metronom", variable=self.delay_sync_var,
                               command=self._on_delay_sync, bg=color, fg=ink, selectcolor=mix(color, "#000000", 0.4),
                               activebackground=color, activeforeground=ink, font=(BODY, 8),
                               highlightthickness=0, bd=0).pack(side="left")
                self.delay_div_var = tk.StringVar(value="1/4")
                cb = ttk.Combobox(srow, textvariable=self.delay_div_var, values=list(DELAY_DIVISIONS),
                                  width=11, state="readonly")
                cb.pack(side="right")
                cb.bind("<<ComboboxSelected>>", lambda e: self._on_delay_sync())
            sw = tk.Canvas(box, width=44, height=44, bg=color, highlightthickness=0, cursor="hand2")
            sw.pack(pady=(4, 8))
            sw.create_oval(4, 4, 40, 40, fill="#9a9a9e", outline="#5a5a5e", width=2)
            sw.create_oval(11, 11, 33, 33, fill="#c8c8cc", outline="#7a7a7e")
            sw.bind("<Button-1>", lambda e, n=name: self._toggle_pedal(n))
            self.pedal_ui[name] = dict(led=led, knobs=kw, color=color)
        self._sync_pedals()

    def _fx(self):
        return core.fx_params(self.params)

    def _set_fx(self, fx):
        self.params = dict(self.params, fx=fx)
        self._schedule_update()

    def _toggle_pedal(self, name):
        fx = self._fx()
        fx[name]["on"] = not fx[name]["on"]
        self._set_fx(fx)
        self._sync_pedals()

    def _on_fx(self, name, key, v):
        fx = self._fx()
        fx[name][key] = v
        if name == "delay" and key == "time_ms" and fx["delay"].get("sync"):
            fx["delay"]["sync"] = False          # Zeit von Hand gedreht -> Sync aus
            self.delay_sync_var.set(False)
        self._set_fx(fx)

    def _on_delay_sync(self):
        fx = self._fx()
        fx["delay"]["sync"] = bool(self.delay_sync_var.get())
        fx["delay"]["division"] = self.delay_div_var.get()
        self._set_fx(fx)
        self._apply_delay_sync()

    def _apply_delay_sync(self):
        fx = self._fx()
        d = fx["delay"]
        if not d.get("sync"):
            return
        ms = 60000.0 / self.engine.bpm * DELAY_DIVISIONS.get(d.get("division", "1/4"), 1.0)
        while ms > 1500:
            ms /= 2
        d["time_ms"] = max(ms, 30.0)
        self.pedal_ui["delay"]["knobs"]["time_ms"].set(d["time_ms"], notify=False)
        self._set_fx(fx)

    def _sync_pedals(self):
        if not hasattr(self, "pedal_ui"):
            return
        fx = self._fx()
        for name, ui in self.pedal_ui.items():
            for key, k in ui["knobs"].items():
                k.set(fx[name][key], notify=False)
            led = ui["led"]
            led.delete("all")
            on = fx[name]["on"]
            if on:
                led.create_oval(0, 0, 14, 14, fill=mix(JEWEL_ON, ui["color"], 0.5), outline="")
            led.create_oval(3, 3, 11, 11, fill=JEWEL_ON if on else "#3a1712", outline="#000")
        self.delay_sync_var.set(bool(fx["delay"].get("sync")))
        self.delay_div_var.set(fx["delay"].get("division", "1/4"))

    def _build_tuner(self, tab):
        c = tk.Frame(tab, bg=PANEL)
        c.pack(fill="both", expand=True, padx=14, pady=12)
        self.tuner = TunerView(c)
        self.tuner.pack(fill="both", expand=True)
        strings = tk.Frame(c, bg=PANEL)
        strings.pack(pady=(10, 0))
        self.string_labels = []
        for name, midi in core.GUITAR_STRINGS:
            lab = tk.Label(strings, text=name, width=4, bg=FIELD, fg=CREAM, font=(DISPLAY, 13, "bold"), pady=4)
            lab.pack(side="left", padx=3)
            self.string_labels.append(lab)
        row = tk.Frame(c, bg=PANEL)
        row.pack(fill="x", pady=(10, 0))
        self._label(row, "Kammerton a").pack(side="left")
        self.a4_var = tk.StringVar(value="440")
        ttk.Spinbox(row, from_=430, to=450, increment=1, textvariable=self.a4_var, width=5).pack(side="left", padx=6)
        self._label(row, "Hz").pack(side="left")
        self.tuner_mute_var = tk.BooleanVar(value=True)
        self._check(row, "Amp stumm, solange das Stimmgerät offen ist",
                    self.tuner_mute_var, self._on_tab).pack(side="right")

    def _build_recorder(self, tab):
        c = tk.Frame(tab, bg=PANEL)
        c.pack(fill="both", expand=True, padx=14, pady=12)
        top = tk.Frame(c, bg=PANEL)
        top.pack(fill="x")
        self.rec_btn = tk.Canvas(top, width=64, height=64, bg=PANEL, highlightthickness=0, cursor="hand2")
        self.rec_btn.pack(side="left")
        self.rec_btn.bind("<Button-1>", lambda e: self._toggle_recording())
        self._draw_rec_button(False)
        info = tk.Frame(top, bg=PANEL)
        info.pack(side="left", padx=14)
        self.rec_time_var = tk.StringVar(value="0:00")
        tk.Label(info, textvariable=self.rec_time_var, bg=PANEL, fg=CREAM,
                 font=(DISPLAY, 24, "bold"), anchor="w").pack(anchor="w")
        self.rec_info_var = tk.StringVar(value="Nimmt dein Signal mit Amp und trocken (DI) auf.")
        self._label(info, "", textvariable=self.rec_info_var, wraplength=260, justify="left").pack(anchor="w")

        back = tk.Frame(top, bg=PANEL)
        back.pack(side="right")
        self._label(back, "Song als Backing-Track", muted=False).pack(anchor="w")
        brow = tk.Frame(back, bg=PANEL)
        brow.pack(anchor="w", pady=(4, 0))
        self.backing_btn = self._button(brow, "Abspielen", self._toggle_backing)
        self.backing_btn.pack(side="left")
        self.player_vol = self._scale(brow, 60, self._apply_player_gain, length=100)
        self.player_vol.pack(side="left", padx=(10, 0))
        self.no_guitar_var = tk.BooleanVar(value=True)
        self._check(back, "ohne Gitarre (wenn isoliert)", self.no_guitar_var, lambda: None).pack(anchor="w", pady=(4, 0))

        head = tk.Frame(c, bg=PANEL)
        head.pack(fill="x", pady=(14, 4))
        self._label(head, "Aufnahmen", muted=False).pack(side="left")
        self.play_var = tk.StringVar(value="")
        self._label(head, "", textvariable=self.play_var).pack(side="right")
        self.take_list = tk.Listbox(c, bg=FIELD, fg=CREAM, selectbackground=AMBER, selectforeground=INK,
                                    relief="flat", highlightthickness=0, font=(BODY, 10), height=5,
                                    activestyle="none")
        self.take_list.pack(fill="both", expand=True)
        self.take_list.bind("<Double-Button-1>", lambda e: self._play_take())
        row = tk.Frame(c, bg=PANEL)
        row.pack(fill="x", pady=(8, 0))
        for txt, cmd in (("Abspielen", self._play_take), ("Stopp", self._stop_playback),
                         ("Re-Amp mit aktuellem Tone", self._reamp), ("Ordner öffnen", self._open_rec_folder),
                         ("Löschen", self._delete_take)):
            self._button(row, txt, cmd).pack(side="left", padx=(0, 4))

    # ---------- Metronom ----------

    def _apply_metro(self):
        try:
            bpm = float(self.bpm_var.get().replace(",", "."))
        except ValueError:
            bpm = self.engine.bpm
        bpm = min(max(bpm, 30), 300)
        self.bpm_var.set(f"{bpm:g}")
        self.engine.set_metronome(bpm=bpm, beats=int(self.sig_var.get().split("/")[0]),
                                  accent=self.accent_var.get(), gain=self.metro_vol.get() / 100 * 0.9)
        if hasattr(self, "pedal_ui"):
            self._apply_delay_sync()

    def _bpm_step(self, d):
        try:
            self.bpm_var.set(f"{round(float(self.bpm_var.get().replace(',', '.'))) + d:g}")
        except ValueError:
            pass
        self._apply_metro()

    def _tap(self):
        now = time.time()
        if self.taps and now - self.taps[-1] > 2.0:
            self.taps = []
        self.taps = (self.taps + [now])[-6:]
        if len(self.taps) >= 2:
            self.bpm_var.set(f"{round(60 / np.mean(np.diff(self.taps)))}")
            self._apply_metro()

    def _toggle_metro(self):
        if not self.engine.running:
            messagebox.showinfo("Audio aus", "Starte zuerst das Audio, dann läuft auch das Metronom.")
            return
        self._apply_metro()
        on = not self.engine.metro_on
        self.engine.set_metronome(on=on)
        self.metro_btn.configure(text="Stopp" if on else "Start")

    # ---------- Stimmgerät ----------

    def _on_tab(self):
        tuner_open = self.tabs.index(self.tabs.select()) == 1
        self.engine.mute = tuner_open and self.tuner_mute_var.get()

    def _update_tuner(self):
        res = core.detect_pitch(self.engine.tuner_buffer(), self.engine.sr)
        now = time.time()
        try:
            a4 = float(self.a4_var.get())
        except ValueError:
            a4 = 440.0
        if res is not None and res[1] > 0.8:
            self.pitch_hist = (self.pitch_hist + [core.freq_to_midi(res[0], a4)])[-5:]
            self.pitch_last = now
        elif now - self.pitch_last > 0.7:
            self.pitch_hist = []
        if not self.pitch_hist:
            self.tuner.set(None)
            nearest, ok = None, False
        else:
            m = float(np.median(self.pitch_hist))
            n = int(round(m))
            cents = (m - n) * 100
            self.tuner.set((core.midi_name(n), cents, a4 * 2 ** ((m - 69) / 12)))
            dist = [abs(m - s) for _, s in core.GUITAR_STRINGS]
            i = int(np.argmin(dist))
            nearest = i if dist[i] < 2.5 else None
            ok = abs(cents) < 3 and n == core.GUITAR_STRINGS[i][1]
        for i, lab in enumerate(self.string_labels):
            if i == nearest:
                lab.configure(bg=TEAL if ok else AMBER, fg=INK)
            else:
                lab.configure(bg=FIELD, fg=CREAM)

    # ---------- Aufnahme ----------

    def _draw_rec_button(self, recording, blink=True):
        cv = self.rec_btn
        cv.delete("all")
        cv.create_oval(2, 2, 62, 62, fill=FIELD, outline=LINE, width=2)
        if recording:
            col = JEWEL_ON if blink else mix(JEWEL_ON, FIELD, 0.5)
            cv.create_rectangle(22, 22, 42, 42, fill=col, outline="")
        else:
            cv.create_oval(16, 16, 48, 48, fill=JEWEL_ON, outline="")

    def _toggle_recording(self):
        if self.engine.recorder is not None:
            paths = self.engine.stop_recording()
            self._draw_rec_button(False)
            self.rec_info_var.set(f"Gespeichert: {paths[0].name} und {paths[1].name}")
            self.status_var.set("Aufnahme gespeichert.")
            self._refresh_takes()
            return
        if not self.engine.running:
            messagebox.showinfo("Audio aus", "Starte zuerst das Audio.")
            return
        try:
            self.engine.start_recording(RECORDINGS_DIR)
        except Exception as e:
            messagebox.showerror("Aufnahme fehlgeschlagen", str(e))
            return
        self.rec_info_var.set("Aufnahme läuft. Metronom und Backing-Track werden nicht mit aufgenommen.")
        self.status_var.set("Aufnahme läuft …")

    @staticmethod
    def _take_label(p):
        parts = p.stem.split("_")
        if len(parts) >= 4 and parts[0] == "take":
            d, t = parts[1], parts[2]
            kind = {"wet": "mit Amp", "dry": "trocken (DI)"}.get(parts[3], "Re-Amp")
            return f"{d[6:8]}.{d[4:6]}.{d[:4]}  {t[:2]}:{t[2:4]}:{t[4:6]}   {kind}"
        return p.stem

    def _refresh_takes(self):
        RECORDINGS_DIR.mkdir(exist_ok=True)
        self.take_files = sorted(RECORDINGS_DIR.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
        self.take_list.delete(0, "end")
        for p in self.take_files:
            self.take_list.insert("end", self._take_label(p))
        if not self.take_files:
            self.take_list.insert("end", "Noch keine Aufnahmen. Starte das Audio und drück den roten Knopf.")
            self.take_list.itemconfig(0, fg=MUTED)

    def _selected_take(self):
        sel = self.take_list.curselection()
        if not sel or sel[0] >= len(self.take_files):
            messagebox.showinfo("Keine Aufnahme gewählt", "Wähle zuerst eine Aufnahme in der Liste.")
            return None
        return self.take_files[sel[0]]

    def _start_player(self, path_or_buf, mode, label):
        if not self.engine.running:
            messagebox.showinfo("Audio aus", "Starte zuerst das Audio.")
            return
        sr = self.engine.sr

        def work():
            try:
                if isinstance(path_or_buf, np.ndarray):
                    buf = path_or_buf
                else:
                    buf = core.load_audio(path_or_buf, sr, min_seconds=0.05)
                self.q.put(("play", buf, mode, label))
            except Exception as e:
                self.q.put(("error", "Wiedergabe fehlgeschlagen", str(e)))
        self.status_var.set("Lade Audio …")
        threading.Thread(target=work, daemon=True).start()

    def _play_stem(self):
        if not self.stems:
            messagebox.showinfo("Keine isolierte Gitarre",
                                "Lade einen Song mit aktivierter Option „Gitarre aus dem Song isolieren“.")
            return
        self._start_player(self.stems[0], "stem", "Isolierte Gitarre")

    def _play_take(self):
        p = self._selected_take()
        if p:
            self._start_player(p, "take", self._take_label(p))

    def _toggle_backing(self):
        if self.player_mode == "backing":
            self._stop_playback()
            return
        if not self.song_path:
            messagebox.showinfo("Kein Song", "Lade zuerst einen Song über „Song laden …“.")
            return
        src = self.stems[1] if (self.stems and self.no_guitar_var.get()) else self.song_path
        key = (str(src), self.engine.sr)
        if self.backing_cache and self.backing_cache[0] == key:
            self._start_player(self.backing_cache[1], "backing", "Backing-Track")
        else:
            sr, path = self.engine.sr, src

            def work():
                try:
                    buf = core.load_audio(path, sr, min_seconds=0.5)
                    self.backing_cache = (key, buf)
                    self.q.put(("play", buf, "backing", "Backing-Track"))
                except Exception as e:
                    self.q.put(("error", "Backing-Track fehlgeschlagen", str(e)))
            if not self.engine.running:
                messagebox.showinfo("Audio aus", "Starte zuerst das Audio.")
                return
            self.status_var.set("Lade Backing-Track …")
            threading.Thread(target=work, daemon=True).start()

    def _apply_player_gain(self):
        self.engine.player_gain = self.player_vol.get() / 100

    def _stop_playback(self):
        self.engine.stop_playback()

    def _reamp(self):
        p = self._selected_take()
        if not p:
            return
        base = "_".join(p.stem.split("_")[:3])
        dry = p.parent / f"{base}_dry.wav"
        if not dry.is_file():
            messagebox.showerror("Keine DI-Spur", "Zu dieser Aufnahme gibt es keine trockene Spur (…_dry.wav).")
            return
        out = p.parent / f"{base}_reamp_{time.strftime('%H%M%S')}.wav"
        params = dict(self.params)
        self.status_var.set("Re-Amp läuft …")

        def work():
            try:
                core.reamp_file(dry, params, out)
                self.q.put(("reamped", out))
            except Exception as e:
                self.q.put(("error", "Re-Amp fehlgeschlagen", str(e)))
        threading.Thread(target=work, daemon=True).start()

    def _open_rec_folder(self):
        RECORDINGS_DIR.mkdir(exist_ok=True)
        if sys.platform.startswith("win"):
            os.startfile(RECORDINGS_DIR)
        else:
            subprocess.Popen(["xdg-open", str(RECORDINGS_DIR)])

    def _delete_take(self):
        p = self._selected_take()
        if p and messagebox.askyesno("Aufnahme löschen", f"„{p.name}“ endgültig löschen?"):
            if self.player_mode == "take":
                self._stop_playback()
            p.unlink(missing_ok=True)
            self._refresh_takes()

    # ---------- Parameter ----------

    def _on_knob(self, key, v):
        self.params[key] = v
        self._schedule_update()

    def _on_match_toggle(self):
        self.params["match_on"] = bool(self.match_var.get())
        self._schedule_update()

    def _schedule_update(self):
        if not self._pending:
            self._pending = True
            self.after(25, self._apply_params)

    def _apply_params(self):
        self._pending = False
        try:
            self.engine.update(self.params)
        except Exception as e:
            self.status_var.set(f"Fehler beim Anwenden: {e}")
        self._update_match_curve()

    def _update_match_curve(self):
        md = self.params.get("match_db")
        if md is not None and self.params.get("match_on"):
            self.spectrum.match = np.asarray(md) * self.params["match_strength"]
        else:
            self.spectrum.match = None
        self.spectrum.redraw()

    def _sync_ui(self):
        for key, knob in self.knobs.items():
            knob.set(self.params[key], notify=False)
            self.params[key] = knob.value
        self.match_var.set(bool(self.params.get("match_on")))
        self._sync_pedals()
        self.spectrum.ref = self.ref_log
        self.song_var.set(self.song_name or "Noch kein Song geladen")
        f = self.features
        self.char_bar.set(f["dist_score"] if f else None)
        if f:
            src = "isolierte Gitarre" if f.get("isolated") else "kompletter Mix"
            dyn = f.get("dyn_db", f["crest_db"])
            self.feat_var.set(f"Referenz: {src}\nDynamik {dyn:.1f} dB, Klangschwerpunkt {f['centroid']:.0f} Hz")
        else:
            self.feat_var.set("")
        self._apply_params()

    # ---------- Audio ----------

    def _refresh_devices(self):
        try:
            self.devices = core.list_asio_devices()
        except Exception as e:
            self.devices = []
            self.status_var.set(str(e))
        if self.devices:
            self.dev_combo["values"] = [d["name"] for d in self.devices]
            self.dev_combo.current(0)
            self._on_device()
        else:
            self.dev_combo["values"] = ["Kein ASIO-Gerät gefunden"]
            self.dev_combo.current(0)
        try:
            self.out_devices = core.list_output_devices()
        except Exception:
            self.out_devices = []
        self.outdev_combo["values"] = ["Über das Interface (ASIO)"] + [
            f"{d['name']}  ·  {d['hostapi'].replace('Windows ', '')}" for d in self.out_devices]
        self.outdev_combo.current(0)
        self._on_outdev()

    def _on_outdev(self):
        separate = self.outdev_combo.current() > 0
        self.out_combo.configure(state="disabled" if separate else "readonly")

    def _on_device(self):
        if not self.devices:
            return
        d = self.devices[self.dev_combo.current()]
        self.in_spin.configure(to=max(d["inputs"], 1))
        outs = [f"{i}/{i + 1}" for i in range(1, d["outputs"], 2)] or ["1"]
        self.out_combo["values"] = outs
        self.out_var.set(outs[0])

    def _toggle_engine(self):
        if self.engine.running:
            self.engine.stop()
            self.engine.set_metronome(on=False)
            self.metro_btn.configure(text="Start")
            self._draw_rec_button(False)
            self._refresh_takes()
            self.engine_btn.configure(text="Audio starten")
            self._draw_jewel(False)
            self.status_var.set("Audio gestoppt.")
            return
        if not self.devices:
            messagebox.showerror("Kein Audio-Interface",
                                 "Es wurde kein ASIO-Gerät gefunden. Installiere den ASIO-Treiber "
                                 "deines Interfaces oder ASIO4ALL und starte ToneMatch neu.")
            return
        d = self.devices[self.dev_combo.current()]
        oi = self.outdev_combo.current()
        out_dev = self.out_devices[oi - 1] if oi > 0 else None
        try:
            self.engine.start(d["index"], int(self.sr_var.get()), int(self.block_var.get()),
                              int(self.in_var.get()) - 1,
                              [int(c) - 1 for c in self.out_var.get().split("/")], self.params,
                              out_device=out_dev["index"] if out_dev else None)
        except Exception as e:
            hint = ("Prüfe, ob Samplerate und Puffer zu den Einstellungen im ASIO-Treiber passen "
                    "und kein anderes Programm das Interface belegt.")
            if out_dev:
                hint += ("\n\nFalls es am Ausgabegerät liegt: probier dasselbe Gerät mit einem anderen "
                         "Treibertyp (WASAPI / DirectSound / MME) in der Ausgabe-Liste.")
            messagebox.showerror("Audio konnte nicht starten", f"{e}\n\n{hint}")
            return
        self._apply_metro()
        self._apply_player_gain()
        self._on_tab()
        self.engine_btn.configure(text="Audio stoppen")
        self._draw_jewel(True)
        lat = self.engine.total_latency_ms()
        target = out_dev["name"] if out_dev else d["name"]
        self.status_var.set(f"Audio läuft – Eingang {d['name']}, Ausgabe {target}, ca. {lat:.0f} ms Latenz.")

    # ---------- Song ----------

    @staticmethod
    def _num(s):
        try:
            return float(s.replace(",", ".")) if s.strip() else None
        except ValueError:
            return None

    def _choose_song(self):
        path = filedialog.askopenfilename(
            title="Referenz-Song wählen",
            filetypes=[("Audio", "*.mp3 *.wav *.flac *.m4a *.ogg *.aac *.webm"), ("Alle Dateien", "*.*")])
        if path:
            self.song_path = path
            self.from_var.set("")
            self.to_var.set("")
            self._analyze()

    def _analyze(self):
        if not self.song_path:
            messagebox.showinfo("Kein Song", "Lade zuerst einen Song.")
            return
        if self.busy:
            return
        start, end = self._num(self.from_var.get()), self._num(self.to_var.get())
        sr = self.engine.sr if self.engine.running else int(self.sr_var.get())
        path = self.song_path
        isolate = self.isolate_var.get() and core.demucs_available()
        self._set_busy(True, "Song wird analysiert …")

        def work():
            try:
                stems = None
                if isolate:
                    stems = core.isolate_guitar(
                        path, STEMS_DIR, progress=lambda f, m: self.q.put(("progress", 100 * f, m)))
                    self.q.put(("progress", 100, "Analysiere die isolierte Gitarre …"))
                x = core.load_audio(stems[0] if stems else path, sr, start, end)
                feat = core.analyze_reference(x, sr, isolated=bool(stems))
                self.q.put(("analyzed", feat, core.reference_log_spectrum(x, sr),
                            core.heuristic_preset(feat), stems))
            except Exception as e:
                self.q.put(("error", "Analyse fehlgeschlagen", str(e)))
        threading.Thread(target=work, daemon=True).start()

    # ---------- Tone Match ----------

    def _tone_match(self):
        if self.busy:
            return
        if self.ref_log is None:
            messagebox.showinfo("Kein Song", "Lade zuerst einen Referenz-Song.")
            return
        if not self.engine.running:
            messagebox.showinfo("Audio aus", "Starte zuerst das Audio, damit deine Gitarre aufgenommen werden kann.")
            return
        try:
            secs = max(10, int(self.secs_var.get()))
        except ValueError:
            secs = 30
        params, ref_log, sr = dict(self.params), self.ref_log.copy(), self.engine.sr
        f = self.features or {}
        target = f.get("dyn_db") if (f.get("isolated") and params.get("ref_isolated")) else None
        self._set_busy(True, "Spiel jetzt los …")

        def work():
            try:
                self.engine.rec = []
                t0 = time.time()
                while (el := time.time() - t0) < secs:
                    self.q.put(("progress", 100 * el / secs, f"Aufnahme läuft – noch {secs - el:.0f} s"))
                    time.sleep(0.1)
                rec, self.engine.rec = self.engine.rec, None
                if not rec:
                    raise RuntimeError("Es kamen keine Audiodaten vom Interface an.")
                self.q.put(("progress", 100, "Angleichung wird berechnet …"))
                new_params, _ = core.learn_from_recording(np.concatenate(rec), ref_log, params, sr,
                                                          target_dyn=target)
                self.q.put(("learned", new_params))
            except Exception as e:
                self.engine.rec = None
                self.q.put(("error", "Tone Match fehlgeschlagen", str(e)))
        threading.Thread(target=work, daemon=True).start()

    def _set_busy(self, busy, msg=None):
        self.busy = busy
        self.match_btn.configure(state="disabled" if busy else "normal")
        if msg:
            self.status_var.set(msg)
        if not busy:
            self.progress["value"] = 0

    # ---------- Bibliothek ----------

    def _refresh_tones(self):
        TONES_DIR.mkdir(exist_ok=True)
        self.tone_files = sorted(TONES_DIR.glob("*.json"), key=lambda p: p.stem.lower())
        self.tone_list.delete(0, "end")
        for p in self.tone_files:
            self.tone_list.insert("end", p.stem)

    def _save_tone(self):
        name = simpledialog.askstring("Tone speichern", "Name für diesen Tone:",
                                      initialvalue=self.song_name or "Mein Tone", parent=self)
        if not name:
            return
        safe = "".join(c for c in name if c not in '\\/:*?"<>|').strip() or "Tone"
        data = dict(name=name, song=self.song_name, params=self.params, features=self.features,
                    ref_log_db=None if self.ref_log is None else [round(float(v), 2) for v in self.ref_log])
        (TONES_DIR / f"{safe}.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
        self._refresh_tones()
        self.status_var.set(f"Tone „{safe}“ gespeichert.")

    def _export_reaper(self):
        name = simpledialog.askstring("Als REAPER-Plugin exportieren", "Name des Plugins:",
                                      initialvalue=self.song_name or "Mein Tone", parent=self)
        if not name:
            return
        safe = "".join(c for c in name if c not in '\\/:*?"<>|').strip() or "Tone"
        effects = core.reaper_effects_dir()
        if effects is not None:
            target = effects / "ToneMatch" / f"{safe}.jsfx"
            if target.exists() and not messagebox.askyesno(
                    "Überschreiben?", f"„{safe}“ gibt es in REAPER schon. Ersetzen?"):
                return
        else:
            path = filedialog.asksaveasfilename(
                title="REAPER-Effects-Ordner nicht gefunden – wohin speichern?",
                defaultextension=".jsfx", initialfile=f"{safe}.jsfx",
                filetypes=[("REAPER JSFX", "*.jsfx")])
            if not path:
                return
            target = Path(path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(core.export_jsfx(self.params, name), encoding="utf-8")
        except Exception as e:
            messagebox.showerror("Export fehlgeschlagen", str(e))
            return
        self.status_var.set(f"REAPER-Plugin exportiert: {target.name}")
        messagebox.showinfo(
            "Exportiert",
            f"Gespeichert unter:\n{target}\n\n"
            "In REAPER: Spur auswählen → FX → im FX-Browser F5 drücken (Liste neu laden) → "
            f"nach „ToneMatch - {name}“ suchen.")

    def _open_tone(self):
        path = filedialog.askopenfilename(title="Tone öffnen", filetypes=[("Tone", "*.json")])
        if path:
            self._load_tone(Path(path))

    def _on_tone_select(self, _):
        sel = self.tone_list.curselection()
        if sel:
            self._load_tone(self.tone_files[sel[0]])

    def _load_tone(self, path):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            messagebox.showerror("Tone nicht lesbar", str(e))
            return
        src = data.get("params", data)
        params = dict(core.DEFAULT_PARAMS)
        params.update({k: v for k, v in src.items() if k in params})
        self.params = params
        self.ref_log = np.asarray(data["ref_log_db"]) if data.get("ref_log_db") else None
        self.features = data.get("features")
        self.song_name = data.get("song") or data.get("name") or path.stem
        self.stems = None
        self._sync_ui()
        self.status_var.set(f"Tone „{path.stem}“ geladen.")

    def _delete_tone(self):
        sel = self.tone_list.curselection()
        if not sel:
            return
        p = self.tone_files[sel[0]]
        if messagebox.askyesno("Tone löschen", f"„{p.stem}“ endgültig löschen?"):
            p.unlink(missing_ok=True)
            self._refresh_tones()

    # ---------- Schleife ----------

    def _poll(self):
        try:
            while True:
                msg = self.q.get_nowait()
                kind = msg[0]
                if kind == "progress":
                    self.progress["value"] = msg[1]
                    self.status_var.set(msg[2])
                elif kind == "analyzed":
                    _, feat, ref_log, preset, self.stems = msg
                    self.backing_cache = None
                    preset["input_trim_db"] = self.params["input_trim_db"]
                    preset["gate_db"] = self.params["gate_db"]
                    preset["fx"] = self._fx()
                    self.features, self.ref_log, self.params = feat, ref_log, preset
                    self.song_name = Path(self.song_path).stem
                    self._sync_ui()
                    self._set_busy(False, "Analyse fertig ("
                                   + ("isolierte Gitarre" if self.stems else "kompletter Mix")
                                   + "). Jetzt Tone Match starten.")
                elif kind == "learned":
                    self.params = msg[1]
                    self._sync_ui()
                    self._set_busy(False, "Tone Match fertig – Gain, Klang und Lautstärke angeglichen."
                                   if self.params.get("ref_isolated") else
                                   "Tone Match fertig. Speichere den Tone, um ihn zu behalten.")
                elif kind == "play":
                    _, buf, mode, label = msg
                    if self.engine.running:
                        self.engine.play(buf, self.player_vol.get() / 100)
                        self.player_mode = mode
                        self.player_label = label
                        self.player_len = len(buf)
                        self.status_var.set(f"Wiedergabe: {label}")
                elif kind == "reamped":
                    self._refresh_takes()
                    self.status_var.set(f"Re-Amp gespeichert: {msg[1].name}")
                elif kind == "error":
                    self._set_busy(False, msg[1])
                    messagebox.showerror(msg[1], msg[2])
        except queue.Empty:
            pass

        if self.engine.running:
            self.meter_in.set(self.engine.in_peak)
            self.meter_out.set(self.engine.out_peak)
            self.engine.in_peak = self.engine.out_peak = 0.0
            self._tick += 1
            if self._tick % 3 == 0:
                live = self.engine.live_spectrum()
                if live is not None:
                    self.live_db = live if self.live_db is None else 0.7 * self.live_db + 0.3 * live
                    self.spectrum.live = self.live_db
                    self.spectrum.redraw()
                x = self.engine.xruns
                u = self.engine.out_underruns if self.engine.separate_output else 0
                err = self.engine.last_error
                parts = []
                if x:
                    parts.append(f"{x} Eingangs-Aussetzer")
                if u:
                    parts.append(f"{u} Ausgabe-Aussetzer (Puffer passt sich an)")
                if err:
                    parts.append(f"Fehler: {err}")
                self.xrun_var.set(" · ".join(parts))
            # Stimmgerät
            if self.tabs.index(self.tabs.select()) == 1:
                self._update_tuner()
            # Metronom-Anzeige
            beats = self.engine.beats
            if self.engine.metro_on and self.engine.metro_beat > 0:
                self.beat_dots.set(beats, (self.engine.metro_beat - 1) % beats)
            else:
                self.beat_dots.set(beats, None)
            # Aufnahme
            r = self.engine.recorder
            if r is not None:
                secs = int(r.seconds)
                self.rec_time_var.set(f"{secs // 60}:{secs % 60:02d}")
                self._draw_rec_button(True, blink=(self._tick // 10) % 2 == 0)
                if r.error:
                    self.engine.stop_recording()
                    messagebox.showerror("Aufnahme fehlgeschlagen", str(r.error))
            # Wiedergabe
            if self.engine.player_buf is not None and self.player_mode:
                pos = min(self.engine.player_pos, self.player_len) // self.engine.sr
                tot = self.player_len // self.engine.sr
                self.play_var.set(f"{self.player_label}  {pos // 60}:{pos % 60:02d} / {tot // 60}:{tot % 60:02d}")
            elif self.player_mode:
                self.player_mode = None
                self.play_var.set("")
            self.backing_btn.configure(text="Stoppen" if self.player_mode == "backing" else "Abspielen")
        else:
            self.meter_in.set(0)
            self.meter_out.set(0)
            self.beat_dots.set(self.engine.beats, None)
            if self.player_mode:
                self.player_mode = None
                self.play_var.set("")
                self.backing_btn.configure(text="Abspielen")
        self.after(50, self._poll)

    def _on_close(self):
        self.engine.rec = None
        self.engine.stop_recording()
        self.engine.stop()
        self.destroy()


if __name__ == "__main__":
    App().mainloop()
