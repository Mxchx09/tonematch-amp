"""
tonematch_core.py – Analyse, Amp-DSP und ASIO-Engine für ToneMatch.
Wird von tonematch_gui.py verwendet.
"""
import os
os.environ.setdefault("SD_ENABLE_ASIO", "1")  # vor dem Import von sounddevice setzen

import hashlib
import importlib.util
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import sounddevice as sd
from scipy import signal

N_FFT = 8192
TARGET_IN_DBFS = -20.0
TARGET_OUT_DBFS = -18.0
LOG_F = 20.0 * 2 ** (np.arange(120) / 12)   # 1/12-Oktav-Raster, 20 Hz .. ~19 kHz
FIR_TAPS = 513

FX_DEFAULTS = {
    "overdrive": dict(on=False, drive=0.5, tone=0.5, level=0.0),
    "distortion": dict(on=False, gain=0.5, tone=0.5, level=0.0),
    "chorus": dict(on=False, rate=0.8, depth=0.5, mix=0.5),
    "delay": dict(on=False, time_ms=380.0, feedback=0.35, mix=0.3, tone=0.6, sync=False, division="1/4"),
    "reverb": dict(on=False, size=0.6, tone=0.5, mix=0.25),
}


def fx_params(p):
    """Vollständige Effekt-Einstellungen (Standardwerte + gespeicherte Werte), immer als neue Kopie."""
    src = p.get("fx") or {}
    return {k: {**v, **src.get(k, {})} for k, v in FX_DEFAULTS.items()}


DEFAULT_PARAMS = dict(
    input_trim_db=0.0, gate_db=-90.0, gain_db=20.0, bias=0.15,
    tight_hz=80.0, mid_push_db=2.0, bass_db=0.0, mid_db=0.0, treble_db=0.0,
    cab_lp_hz=5500.0, master_db=-12.0,
    match_strength=0.8, match_on=True, match_db=None, ref_isolated=False,
    fx=FX_DEFAULTS,
)


# ---------------------------------------------------------------------------
# Laden
# ---------------------------------------------------------------------------

def sniff_format(path):
    with open(path, "rb") as fh:
        head = fh.read(64)
    if head[4:8] == b"ftyp":
        return "M4A/MP4 (AAC)"
    if head.lstrip()[:1] == b"<":
        return "HTML/Text (keine Audiodatei)"
    if head[:4] == b"RIFF":
        return "WAV"
    if head[:4] == b"fLaC":
        return "FLAC"
    if head[:4] == b"OggS":
        return "OGG"
    if head[:4] == b"\x1aE\xdf\xa3":
        return "WebM/MKV"
    if head[:3] == b"ID3" or (len(head) > 1 and head[0] == 0xFF and head[1] & 0xE0 == 0xE0):
        return "MP3"
    return "unbekannt"


def _load_with_ffmpeg(path):
    if shutil.which("ffmpeg") is None:
        return None
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "2", "-ar", "48000", "-"]
    res = subprocess.run(cmd, capture_output=True,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if res.returncode != 0 or not res.stdout:
        raise RuntimeError("ffmpeg konnte die Datei nicht lesen:\n" + res.stderr.decode(errors="replace")[:500])
    return np.frombuffer(res.stdout, dtype=np.float32).astype(np.float64).reshape(-1, 2), 48000


def load_raw(path):
    """Gibt (Samples x Kanäle, Samplerate) zurück."""
    path = Path(path)
    if not path.is_file():
        raise RuntimeError(f"Datei nicht gefunden: {path}")
    kind = sniff_format(path)
    if kind.startswith("HTML"):
        raise RuntimeError("Die Datei ist keine Audiodatei (HTML/Text). Den Song bitte neu herunterladen.")
    try:
        x, fs = sf.read(str(path), always_2d=True)
    except Exception:
        loaded = _load_with_ffmpeg(path)
        if loaded is None:
            raise RuntimeError(
                f"Datei konnte nicht gelesen werden (erkannt: {kind}).\n"
                "ffmpeg installieren (winget install ffmpeg, danach neu starten) "
                "oder die Datei als WAV exportieren.")
        x, fs = loaded
    return x, int(fs)


def load_audio(path, sr, start=None, end=None, min_seconds=3.0):
    """Lädt eine Audiodatei als Mono-float64 mit Samplerate sr, optional nur einen Ausschnitt (Sekunden)."""
    x, fs = load_raw(path)
    x = x.mean(axis=1)
    s = int((start or 0) * fs)
    e = int(end * fs) if end else len(x)
    x = x[s:e]
    if len(x) < min_seconds * fs:
        raise RuntimeError(f"Der gewählte Ausschnitt ist kürzer als {min_seconds:g} Sekunden.")
    if fs != sr:
        g = np.gcd(fs, sr)
        x = signal.resample_poly(x, sr // g, fs // g)
    return x.astype(np.float64)


# ---------------------------------------------------------------------------
# Gitarre aus dem Song isolieren (Demucs, Modell htdemucs_6s)
# ---------------------------------------------------------------------------

def demucs_available():
    return importlib.util.find_spec("demucs") is not None and importlib.util.find_spec("torch") is not None


def stem_paths(path, cache_dir):
    path = Path(path)
    st = path.stat()
    key = hashlib.sha1(f"{path.resolve()}|{st.st_size}|{int(st.st_mtime)}".encode()).hexdigest()[:10]
    base = Path(cache_dir) / f"{path.stem[:40]}_{key}"
    return base.with_name(base.name + "_guitar.wav"), base.with_name(base.name + "_ohne_gitarre.wav")


def isolate_guitar(path, cache_dir, progress=None):
    """Trennt die Gitarre vom Rest. Ergebnis wird zwischengespeichert.
    Gibt (Pfad Gitarre, Pfad Song ohne Gitarre) zurück."""
    g_path, b_path = stem_paths(path, cache_dir)
    if g_path.is_file() and b_path.is_file():
        return g_path, b_path
    if not demucs_available():
        raise RuntimeError("Demucs ist nicht installiert. Im Terminal:  pip install demucs")
    import torch
    from demucs.pretrained import get_model
    from demucs.apply import apply_model

    if progress:
        progress(0.0, "Lade Trennungs-Modell (beim ersten Mal wird es heruntergeladen) …")
    model = get_model("htdemucs_6s")
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    msr = int(model.samplerate)
    gi = list(model.sources).index("guitar")

    x, fs = load_raw(path)
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    x = x[:, :2]
    if fs != msr:
        g = np.gcd(fs, msr)
        x = signal.resample_poly(x, msr // g, fs // g, axis=0)
    wav = x.T.astype(np.float32)                          # (2, n)
    mono = wav.mean(axis=0)
    mean, std = float(mono.mean()), float(mono.std()) + 1e-8
    wav = (wav - mean) / std

    n = wav.shape[1]
    chunk, ctx = 30 * msr, 3 * msr                       # 30-s-Stücke mit 3 s Kontext je Seite
    guitar = np.zeros(n, dtype=np.float64)
    starts = list(range(0, n, chunk))
    for i, s0 in enumerate(starts):
        a, b = max(0, s0 - ctx), min(n, s0 + chunk + ctx)
        seg = torch.from_numpy(np.ascontiguousarray(wav[:, a:b]))[None].to(device)
        with torch.no_grad():
            src = apply_model(model, seg, split=True, overlap=0.25, progress=False, device=device)[0]
        g = src[gi].mean(dim=0).cpu().numpy().astype(np.float64) * std
        e = min(n, s0 + chunk)
        guitar[s0:e] = g[s0 - a:e - a]
        if progress:
            progress((i + 1) / len(starts), f"Gitarre wird isoliert … {100 * (i + 1) / len(starts):.0f} %")

    rest = mono * std + mean - guitar
    g_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(g_path), guitar.astype(np.float32), msr, subtype="PCM_24")
    sf.write(str(b_path), np.clip(rest, -1, 1).astype(np.float32), msr, subtype="PCM_24")
    return g_path, b_path


# ---------------------------------------------------------------------------
# Analyse
# ---------------------------------------------------------------------------

def gated_ltas(x, sr, nfft=N_FFT, gate_db=-30.0):
    hop = nfft // 2
    starts = np.arange(0, len(x) - nfft, hop)
    if len(starts) == 0:
        raise RuntimeError("Audio zu kurz für die Analyse.")
    rms = np.array([np.sqrt(np.mean(x[s:s + nfft] ** 2)) for s in starts]) + 1e-12
    keep = starts[rms > rms.max() * 10 ** (gate_db / 20)]
    win = np.hanning(nfft)
    acc = np.zeros(nfft // 2 + 1)
    for s in keep:
        acc += np.abs(np.fft.rfft(x[s:s + nfft] * win)) ** 2
    return np.fft.rfftfreq(nfft, 1 / sr), acc / len(keep)


def gated_rms(x, sr, gate_db=-30.0):
    hop = int(0.05 * sr)
    n = len(x) // hop
    if n == 0:
        return 0.0
    r = np.sqrt((x[:n * hop].reshape(n, hop) ** 2).mean(axis=1)) + 1e-12
    loud = r[r > r.max() * 10 ** (gate_db / 20)]
    return float(np.sqrt(np.mean(loud ** 2)))


def smooth_octave(f, p, frac=3):
    c = np.concatenate([[0.0], np.cumsum(p)])
    lo = np.searchsorted(f, f * 2 ** (-1 / (2 * frac)))
    hi = np.searchsorted(f, f * 2 ** (1 / (2 * frac)), side="right")
    hi = np.maximum(hi, lo + 1)
    return (c[hi] - c[lo]) / (hi - lo)


def spectrum_to_log(f, p):
    """Leistungsspektrum -> geglättete dB-Kurve auf LOG_F, normiert auf 100–5000 Hz."""
    sm = smooth_octave(f, p, 3)
    db = 10 * np.log10(np.interp(LOG_F, f, sm) + 1e-20)
    band = (LOG_F >= 100) & (LOG_F <= 5000)
    return db - db[band].mean()


def reference_log_spectrum(x, sr):
    f, p = gated_ltas(x, sr)
    return spectrum_to_log(f, p)


def analyze_reference(x, sr, isolated=False):
    sos = signal.butter(4, [80, 6000], btype="band", fs=sr, output="sos")
    xb = signal.sosfilt(sos, x)
    f, p = gated_ltas(xb, sr)
    band = (f >= 80) & (f <= 6000)
    pb, fb = p[band], f[band]
    centroid = float(np.sum(fb * pb) / np.sum(pb))

    def energy(a, b):
        return p[(f >= a) & (f < b)].sum() + 1e-20
    hi_ratio = float(energy(2000, 5000) / energy(200, 1000))

    hop = int(0.05 * sr)
    n = len(xb) // hop
    fr = xb[:n * hop].reshape(n, hop)
    rms = np.sqrt((fr ** 2).mean(axis=1)) + 1e-12
    peak = np.abs(fr).max(axis=1)
    loud = rms > rms.max() * 0.1
    crest_db = float(20 * np.log10(np.median(peak[loud] / rms[loud])))

    dyn_db = guitar_dynamics_db(x, sr)
    if isolated:
        dist = float(np.clip((8 - dyn_db) / 7.5, 0, 1))
    else:
        dist = (0.6 * np.clip((14 - crest_db) / 7, 0, 1)
                + 0.4 * np.clip((np.log10(hi_ratio) + 1.5) / 1.5, 0, 1))

    f2, p2 = gated_ltas(x, sr)
    bands = [(100, 400), (400, 1600), (1600, 6400)]
    e = np.array([p2[(f2 >= a) & (f2 < b)].sum() for a, b in bands]) + 1e-20
    tone = np.clip(0.5 * 10 * np.log10(e / e.mean()), -8, 8)

    return dict(centroid=centroid, hi_ratio=hi_ratio, crest_db=crest_db, dyn_db=dyn_db, isolated=bool(isolated),
                dist_score=float(dist), tone_low_db=float(tone[0]),
                tone_mid_db=float(tone[1]), tone_high_db=float(tone[2]))


def guitar_dynamics_db(x, sr):
    """Wie stark schwankt die Lautstärke innerhalb von 1-s-Fenstern (95. Perzentil minus Median)?
    Verzerrung komprimiert und erzeugt Sustain -> kleiner Wert. Kaum abhängig von EQ/Cabinet."""
    sos = signal.butter(4, [100, 5000], btype="band", fs=sr, output="sos")
    xb = signal.sosfilt(sos, x)
    hop = int(0.02 * sr)
    n = len(xb) // hop
    if n < 60:
        return 10.0
    db = 20 * np.log10(np.sqrt((xb[:n * hop].reshape(n, hop) ** 2).mean(axis=1)) + 1e-9)
    active = db > db.max() - 35
    w = 50
    vals = []
    for i in range(0, n - w, w // 2):
        a = active[i:i + w]
        if a.sum() < w * 0.5:
            continue
        seg = db[i:i + w][a]
        vals.append(np.percentile(seg, 95) - np.median(seg))
    return float(np.median(vals)) if vals else 10.0


def heuristic_preset(feat):
    d = feat["dist_score"]
    iso = feat.get("isolated", False)
    p = dict(DEFAULT_PARAMS)
    p.update(
        ref_isolated=bool(iso),
        gain_db=round((-4 + 30 * d) if iso else (-6 + 50 * d), 1),
        bias=round(0.1 + 0.15 * d, 3),
        tight_hz=round(60 + 80 * d, 1),
        mid_push_db=round(4 * d, 1),
        bass_db=round(feat["tone_low_db"], 1),
        mid_db=round(feat["tone_mid_db"], 1),
        treble_db=round(feat["tone_high_db"], 1),
        cab_lp_hz=float(round(np.clip(feat["centroid"] * 3, 3000 if iso else 3500, 7000 if iso else 7500))),
    )
    return {k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in p.items()}


# ---------------------------------------------------------------------------
# DSP-Bausteine
# ---------------------------------------------------------------------------

def peaking(f0, gain_db, q, fs):
    A = 10 ** (gain_db / 40)
    w = 2 * np.pi * f0 / fs
    al, c = np.sin(w) / (2 * q), np.cos(w)
    b = [1 + al * A, -2 * c, 1 - al * A]
    a = [1 + al / A, -2 * c, 1 - al / A]
    return np.array([*b, *a]) / a[0]


def shelf(f0, gain_db, fs, high=False, S=0.7):
    A = 10 ** (gain_db / 40)
    w = 2 * np.pi * f0 / fs
    c, s = np.cos(w), np.sin(w)
    al = s / 2 * np.sqrt((A + 1 / A) * (1 / S - 1) + 2)
    sq = 2 * np.sqrt(A) * al
    if not high:
        b0 = A * ((A + 1) - (A - 1) * c + sq)
        b1 = 2 * A * ((A - 1) - (A + 1) * c)
        b2 = A * ((A + 1) - (A - 1) * c - sq)
        a0 = (A + 1) + (A - 1) * c + sq
        a1 = -2 * ((A - 1) + (A + 1) * c)
        a2 = (A + 1) + (A - 1) * c - sq
    else:
        b0 = A * ((A + 1) + (A - 1) * c + sq)
        b1 = -2 * A * ((A - 1) + (A + 1) * c)
        b2 = A * ((A + 1) + (A - 1) * c - sq)
        a0 = (A + 1) - (A - 1) * c + sq
        a1 = 2 * ((A - 1) - (A + 1) * c)
        a2 = (A + 1) - (A - 1) * c - sq
    return np.array([b0, b1, b2, a0, a1, a2]) / a0


class SOSFilter:
    def __init__(self, sos):
        self.sos = np.atleast_2d(sos)
        self.zi = np.zeros((self.sos.shape[0], 2))

    def __call__(self, x):
        y, self.zi = signal.sosfilt(self.sos, x, zi=self.zi)
        return y


class Oversampler:
    def __init__(self, factor=4, taps=128):
        self.L = factor
        self.h = signal.firwin(taps, 0.9 / factor)
        self.up_zi = np.zeros(taps - 1)
        self.dn_zi = np.zeros(taps - 1)

    def up(self, x):
        u = np.zeros(len(x) * self.L)
        u[::self.L] = x * self.L
        y, self.up_zi = signal.lfilter(self.h, 1.0, u, zi=self.up_zi)
        return y

    def down(self, y):
        z, self.dn_zi = signal.lfilter(self.h, 1.0, y, zi=self.dn_zi)
        return z[::self.L]


def unit_fir():
    h = np.zeros(FIR_TAPS)
    h[0] = 1.0
    return h


def boost_cap_db(f, isolated=False):
    """Maximal erlaubte Anhebung je Frequenz. Absenkungen sind bis -12 dB erlaubt.
    Tiefen (Bass/Kick im Mix) und Höhen (Becken/Gesang) werden kaum angehoben."""
    caps = [4, 8, 8, 5, 2] if isolated else [2, 7, 7, 2, 0]
    return np.interp(np.log(np.maximum(f, 1.0)), np.log([60, 150, 2000, 5000, 8000]), caps)


def compute_match_curve(ref_log_db, out_log_db, isolated=False):
    diff = ref_log_db - out_log_db
    if not isolated:                            # Original-Verhalten
        return np.clip(diff, -12, 12)
    k = 7                                       # ~1/2-Oktav-Glättung (Raster ist 1/12 Oktave)
    diff = np.convolve(np.pad(diff, k // 2, mode="edge"), np.ones(k) / k, mode="valid")
    return np.clip(np.minimum(diff, boost_cap_db(LOG_F, isolated)), -12, 12)


def design_fir_from_curve(curve_db, strength, fs, isolated=False):
    """Minimalphasiger FIR aus einer Korrekturkurve auf LOG_F."""
    curve_db = np.asarray(curve_db, dtype=np.float64)
    top = 12000
    if isolated:
        curve_db = np.minimum(curve_db, boost_cap_db(LOG_F, True))
        top = 10000
    fl = np.linspace(0, fs / 2, 2049)
    g = np.interp(np.log(np.maximum(fl, 1.0)), np.log(LOG_F), curve_db)
    w = np.clip((fl - 40) / 20, 0, 1) * np.clip((top - fl) / 2000, 0, 1)
    mag = 10 ** (g * w * strength / 20)
    h = signal.firwin2(2 * FIR_TAPS - 1, fl / (fs / 2), mag ** 2)  # min.-phase halbiert den Betrag in dB
    h = signal.minimum_phase(h, method="homomorphic")
    out = np.zeros(FIR_TAPS)
    out[:min(len(h), FIR_TAPS)] = h[:FIR_TAPS]
    return out


# ---------------------------------------------------------------------------
# Effekte (Pedale)
# ---------------------------------------------------------------------------

class Ring:
    """Ringpuffer für Delay-basierte Effekte."""

    def __init__(self, size):
        self.h = np.zeros(size)
        self.L = size
        self.w = 0

    def clear(self):
        self.h[:] = 0.0

    def write(self, x):
        idx = (self.w + np.arange(len(x))) % self.L
        self.h[idx] = x
        self.w = (self.w + len(x)) % self.L

    def read_int(self, D, n):
        """Liest n Samples, die D Samples zurückliegen (D >= n, vor dem Schreiben aufrufen)."""
        return self.h[(self.w - D + np.arange(n)) % self.L]

    def read_frac(self, pos):
        i0 = np.floor(pos).astype(np.int64)
        f = pos - i0
        return self.h[i0 % self.L] * (1 - f) + self.h[(i0 + 1) % self.L] * f


class DrivePedals:
    """Overdrive (Tube-Screamer-Prinzip) und Distortion (RAT-Prinzip), 4x überabgetastet."""

    def __init__(self, fs):
        self.fs = fs
        self.fso = 4 * fs
        self.os = Oversampler(4)
        self.od_hp = self.od_tone = self.ds_hp = self.ds_tone = None
        self.was_on = False

    def update(self, fx):
        od, ds = fx["overdrive"], fx["distortion"]
        fso = self.fso
        self.od_on, self.ds_on = bool(od["on"]), bool(ds["on"])
        self.od_g = 10 ** ((6 + 34 * od["drive"]) / 20)
        self.od_level = 10 ** (od["level"] / 20)
        self.ds_g = 10 ** ((6 + 42 * ds["gain"]) / 20)
        self.ds_level = 10 ** (ds["level"] / 20)
        filters = dict(
            od_hp=signal.butter(1, 720, "high", fs=fso, output="sos"),       # Mitten-Buckel
            od_tone=signal.butter(2, 500 * 10 ** (1.1 * od["tone"]), "low", fs=fso, output="sos"),
            ds_hp=signal.butter(1, 120, "high", fs=fso, output="sos"),
            ds_tone=signal.butter(2, 600 * 10 ** (1.12 * ds["tone"]), "low", fs=fso, output="sos"),
        )
        for name, sos in filters.items():
            flt = getattr(self, name)
            if flt is None:
                setattr(self, name, SOSFilter(sos))
            else:
                flt.sos = sos

    def process(self, x):
        on = self.od_on or self.ds_on
        if not on:
            self.was_on = False
            return x
        if not self.was_on:                     # frisch eingeschaltet: alte Zustände verwerfen
            self.was_on = True
            self.os = Oversampler(4)
            for f in (self.od_hp, self.od_tone, self.ds_hp, self.ds_tone):
                f.zi[:] = 0.0
        u = self.os.up(x)
        if self.od_on:
            clip = np.tanh(self.od_g * self.od_hp(u))
            u = self.od_tone(0.5 * u + 0.25 * clip) * self.od_level
        if self.ds_on:
            z = self.ds_g * self.ds_hp(u)
            clip = z / (1 + np.abs(z) ** 3) ** (1 / 3)                 # harte Dioden-Kennlinie
            u = self.ds_tone(clip) * 0.3 * self.ds_level
        return self.os.down(u)


class Chorus:
    def __init__(self, fs):
        self.fs = fs
        self.ring = Ring(int(0.05 * fs) + 8192)
        self.phase = 0.0

    def reset(self):
        self.ring.clear()

    def process(self, x, c):
        n, fs = len(x), self.fs
        base = self.ring.w
        self.ring.write(x)
        w = 2 * np.pi * c["rate"] / fs
        t = self.phase + w * np.arange(n)
        self.phase = (self.phase + w * n) % (2 * np.pi)
        d = (7.0 + 5.0 * c["depth"] * np.sin(t)) * 1e-3 * fs          # 2–12 ms
        wet = self.ring.read_frac(base + np.arange(n) - d)
        return x * (1 - 0.3 * c["mix"]) + wet * 0.7 * c["mix"]


class Delay:
    def __init__(self, fs):
        self.fs = fs
        self.ring = Ring(int(1.6 * fs) + 8192)
        self.d_cur = None
        self.lp = SOSFilter(signal.butter(1, 5000, "low", fs=fs, output="sos"))
        self._tone = None

    def reset(self):
        self.ring.clear()
        self.lp.zi[:] = 0.0
        self.d_cur = None

    def process(self, x, c):
        n, fs = len(x), self.fs
        if c["tone"] != self._tone:
            self._tone = c["tone"]
            self.lp.sos = signal.butter(1, min(1200 * 10 ** (1.1 * c["tone"]), 0.45 * fs),
                                        "low", fs=fs, output="sos")
        target = max(c["time_ms"] * 1e-3 * fs, n + 2)
        if self.d_cur is None:
            self.d_cur = target
        d_new = self.d_cur + (target - self.d_cur) * 0.15               # sanftes Gleiten bei Zeitänderung
        d = np.maximum(np.linspace(self.d_cur, d_new, n, endpoint=False), n + 2)
        self.d_cur = d_new
        tap = self.lp(self.ring.read_frac(self.ring.w + np.arange(n) - d))  # analog-artig dunkler werdend
        self.ring.write(x + min(c["feedback"], 0.95) * tap)
        return x + c["mix"] * tap


class Reverb:
    """Freeverb-Prinzip: 8 Kammfilter mit Dämpfung + 4 Allpässe."""
    COMBS = [1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617]
    APS = [556, 441, 341, 225]

    def __init__(self, fs):
        s = fs / 44100
        self.cd = [int(c * s) for c in self.COMBS]
        self.ad = [int(a * s) for a in self.APS]
        self.cr = [Ring(d + 4096) for d in self.cd]
        self.ar = [Ring(d + 4096) for d in self.ad]
        self.czi = [np.zeros(1) for _ in self.cd]
        self.chunk = min(self.ad)

    def reset(self):
        for r in self.cr + self.ar:
            r.clear()
        for z in self.czi:
            z[:] = 0.0

    def _chunk(self, x, fb, damp):
        inp = x * 0.015
        acc = np.zeros(len(x))
        for k, (D, r) in enumerate(zip(self.cd, self.cr)):
            rd = r.read_int(D, len(x))
            l, self.czi[k] = signal.lfilter([1 - damp], [1, -damp], rd, zi=self.czi[k])
            r.write(inp + fb * l)
            acc += rd
        for D, r in zip(self.ad, self.ar):
            bo = r.read_int(D, len(acc))
            out = -acc + bo
            r.write(acc + 0.5 * bo)
            acc = out
        return acc * 3.0

    def process(self, x, c):
        fb = min(0.7 + 0.28 * c["size"], 0.97)
        damp = (1 - c["tone"]) * 0.4
        wet = np.concatenate([self._chunk(x[i:i + self.chunk], fb, damp)
                              for i in range(0, len(x), self.chunk)])
        return x * (1 - 0.5 * c["mix"]) + c["mix"] * wet


class Amp:
    """Amp-Simulation. Parameter lassen sich live ändern, ohne Knackser durch Filter-Reset."""

    def __init__(self, fs, params):
        self.fs = fs
        self.pre = self.tone = self.cab = None
        self.os = Oversampler(4)
        self.dc = SOSFilter(signal.butter(1, 20, "high", fs=fs, output="sos"))
        self.fir = unit_fir()
        self.fir_zi = np.zeros(FIR_TAPS - 1)
        self._match_key = None
        self.g_gate = 1.0
        self.pedals = DrivePedals(fs)
        self.chorus, self.delay, self.reverb = Chorus(fs), Delay(fs), Reverb(fs)
        self._post_on = {"chorus": False, "delay": False, "reverb": False}
        self.update(params, ramp=False)

    def update(self, p, ramp=True):
        fs = self.fs
        pre = np.vstack([signal.butter(2, p["tight_hz"], "high", fs=fs, output="sos"),
                         peaking(720, p["mid_push_db"], 0.7, fs)])
        tone = np.vstack([shelf(120, p["bass_db"], fs),
                          peaking(800, p["mid_db"], 0.8, fs),
                          shelf(3000, p["treble_db"], fs, high=True)])
        cab = np.vstack([signal.butter(2, 75, "high", fs=fs, output="sos"),
                         peaking(110, 4.0, 1.2, fs),
                         peaking(2500, 3.0, 1.0, fs),
                         signal.butter(4, min(p["cab_lp_hz"], 0.45 * fs), "low", fs=fs, output="sos")])
        for name, sos in (("pre", pre), ("tone", tone), ("cab", cab)):
            flt = getattr(self, name)
            if flt is None:
                setattr(self, name, SOSFilter(sos))
            else:
                flt.sos = sos

        self.bias = float(p["bias"])
        self.gate_thr = float(p["gate_db"])
        self.t_in = 10 ** (p["input_trim_db"] / 20)
        self.t_drive = 10 ** (p["gain_db"] / 20)
        self.t_master = 10 ** (p["master_db"] / 20)
        if not ramp:
            self.c_in, self.c_drive, self.c_master = self.t_in, self.t_drive, self.t_master

        md = p.get("match_db")
        key = (tuple(md) if md is not None else None,
               round(float(p["match_strength"]), 3), bool(p["match_on"]), bool(p.get("ref_isolated", False)))
        if key != self._match_key:
            self._match_key = key
            if key[0] is not None and key[2] and key[1] > 0:
                self.fir = design_fir_from_curve(np.asarray(key[0]), key[1], fs, key[3])
            else:
                self.fir = unit_fir()

        self.fx = fx_params(p)
        self.pedals.update(self.fx)

    def _ramp(self, name, target, n):
        cur = getattr(self, name)
        setattr(self, name, target)
        return np.linspace(cur, target, n, endpoint=False)

    def process(self, x):
        n = len(x)
        x = x * self._ramp("c_in", self.t_in, n)

        # Noise-Gate (mit Hysterese)
        if self.gate_thr > -89.5:
            lvl = 20 * np.log10(np.sqrt(np.mean(x * x)) + 1e-12)
            thr = self.gate_thr - (6.0 if self.g_gate > 0.5 else 0.0)
            tgt = 1.0 if lvl > thr else 0.0
            coef = 0.7 if tgt > self.g_gate else 0.08
            g_new = self.g_gate + (tgt - self.g_gate) * coef
        else:
            g_new = 1.0
        x = x * np.linspace(self.g_gate, g_new, n, endpoint=False)
        self.g_gate = g_new

        x = self.pedals.process(x)
        x = self.pre(x)
        u = self.os.up(x * self._ramp("c_drive", self.t_drive, n))
        b = self.bias
        u = np.tanh(u + b) - np.tanh(b)
        u = np.tanh(1.8 * u) / np.tanh(1.8)
        x = self.os.down(u)
        x = self.dc(x)
        x = self.tone(x)
        x = np.tanh(1.3 * x) / np.tanh(1.3)
        x = self.cab(x)
        x, self.fir_zi = signal.lfilter(self.fir, 1.0, x, zi=self.fir_zi)
        fx = self.fx
        for name, unit in (("chorus", self.chorus), ("delay", self.delay), ("reverb", self.reverb)):
            on = bool(fx[name]["on"])
            if on and not self._post_on[name]:
                unit.reset()                    # beim Einschalten keine alten Reste abspielen
            self._post_on[name] = on
            if on:
                x = unit.process(x, fx[name])
        return np.clip(x * self._ramp("c_master", self.t_master, n), -1.0, 1.0)

    def process_offline(self, x, block=1024):
        pad = (-len(x)) % block
        xp = np.concatenate([x, np.zeros(pad)])
        out = np.concatenate([self.process(xp[i:i + block]) for i in range(0, len(xp), block)])
        return out[:len(x)]


def _loudest_excerpt(x, sr, seconds=10.0):
    n = int(seconds * sr)
    if len(x) <= n:
        return x
    e = np.concatenate([[0.0], np.cumsum(x * x)])
    i = int(np.argmax(e[n:] - e[:-n]))
    return x[i:i + n]


def fit_gain(dry, params, sr, target_dyn):
    """Sucht den Gain, bei dem dein Signal so stark verzerrt ist wie die Gitarre im Song."""
    ex = _loudest_excerpt(dry, sr)
    lo, hi = -10.0, 50.0
    for _ in range(7):
        mid = (lo + hi) / 2
        p = dict(without_time_fx(params), gain_db=mid, match_on=False)
        if guitar_dynamics_db(Amp(sr, p).process_offline(ex), sr) > target_dyn:
            lo = mid          # noch zu wenig Zerre
        else:
            hi = mid
    return round((lo + hi) / 2, 1)


def without_time_fx(p):
    fx = fx_params(p)
    for k in ("chorus", "delay", "reverb"):
        fx[k]["on"] = False
    return dict(p, fx=fx)


def learn_from_recording(dry, ref_log_db, params, sr, target_dyn=None):
    """Berechnet Input-Trim, Gain (falls Ziel bekannt), Match-EQ und Master aus einer DI-Aufnahme."""
    rms = gated_rms(dry, sr)
    if rms < 1e-4:
        raise RuntimeError("Kein brauchbares Gitarrensignal aufgenommen. Eingangskanal und Pegel prüfen.")
    p = dict(params)
    p["input_trim_db"] = round(float(TARGET_IN_DBFS - 20 * np.log10(rms)), 2)
    if target_dyn is not None:
        p["gain_db"] = fit_gain(dry, p, sr, target_dyn)
    p["match_on"] = False
    out = Amp(sr, without_time_fx(p)).process_offline(dry)
    f, po = gated_ltas(out, sr)
    out_log = spectrum_to_log(f, po)
    p["match_db"] = [round(float(v), 2) for v in
                     compute_match_curve(ref_log_db, out_log, p.get("ref_isolated", False))]
    p["match_on"] = True
    out2 = Amp(sr, without_time_fx(p)).process_offline(dry)
    master = p["master_db"] + TARGET_OUT_DBFS - 20 * np.log10(gated_rms(out2, sr) + 1e-12)
    p["master_db"] = round(float(np.clip(master, -40, 12)), 2)
    return p, out_log


# ---------------------------------------------------------------------------
# ASIO-Engine
# ---------------------------------------------------------------------------

def list_asio_devices():
    apis = sd.query_hostapis()
    asio = [i for i, a in enumerate(apis) if "asio" in a["name"].lower()]
    if not asio:
        raise RuntimeError("Keine ASIO-Schnittstelle gefunden. sounddevice >= 0.5 und einen "
                           "ASIO-Treiber (z. B. vom Interface oder ASIO4ALL) installieren.")
    return [dict(index=i, name=d["name"], inputs=d["max_input_channels"],
                 outputs=d["max_output_channels"])
            for i, d in enumerate(sd.query_devices()) if d["hostapi"] in asio]


# ---------------------------------------------------------------------------
# Stimmgerät
# ---------------------------------------------------------------------------

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "H"]
GUITAR_STRINGS = [("E", 40), ("A", 45), ("D", 50), ("G", 55), ("H", 59), ("e", 64)]  # MIDI-Noten
_TUNER_SOS = {}


def detect_pitch(x, sr, fmin=55.0, fmax=1100.0, threshold=0.15):
    """YIN-Tonhöhenerkennung. Gibt (Frequenz, Sicherheit 0..1) oder None zurück."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    if np.sqrt(np.mean(x * x)) < 2e-3:
        return None
    sos = _TUNER_SOS.get(sr)
    if sos is None:
        sos = _TUNER_SOS[sr] = signal.butter(2, 1500, "low", fs=sr, output="sos")
    x = signal.sosfiltfilt(sos, x)

    tau_max = min(int(sr / fmin) + 2, len(x) // 2)
    tau_min = max(int(sr / fmax), 2)
    W = len(x) - tau_max
    size = 1 << int(np.ceil(np.log2(len(x) + W)))
    corr = np.fft.irfft(np.fft.rfft(x, size) * np.conj(np.fft.rfft(x[:W], size)), size)[:tau_max]
    cs = np.concatenate([[0.0], np.cumsum(x * x)])
    taus = np.arange(tau_max)
    d = cs[W] + (cs[taus + W] - cs[taus]) - 2 * corr
    d[0] = 0.0
    dn = np.ones(tau_max)
    dn[1:] = d[1:] * taus[1:] / np.maximum(np.cumsum(d[1:]), 1e-12)

    cand = np.where(dn[tau_min:] < threshold)[0]
    if len(cand):
        t = int(cand[0]) + tau_min
        while t + 1 < tau_max and dn[t + 1] < dn[t]:
            t += 1
    else:
        t = int(np.argmin(dn[tau_min:])) + tau_min
        if dn[t] > 0.35:
            return None
    tf = float(t)
    if 0 < t < tau_max - 1:
        a, b, c = dn[t - 1], dn[t], dn[t + 1]
        den = a - 2 * b + c
        if den != 0:
            tf += 0.5 * (a - c) / den
    return sr / tf, float(1 - dn[t])


def freq_to_midi(f, a4=440.0):
    return 69 + 12 * np.log2(f / a4)


def midi_name(n):
    return NOTE_NAMES[n % 12] + str(n // 12 - 1)


# ---------------------------------------------------------------------------
# Aufnahme
# ---------------------------------------------------------------------------

class Recorder:
    """Schreibt wet (mit Amp) und dry (DI) parallel als WAV, ohne den Audio-Thread zu blockieren."""

    def __init__(self, folder, sr):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        self.name = "take_" + time.strftime("%Y%m%d_%H%M%S")
        self.paths = (folder / f"{self.name}_wet.wav", folder / f"{self.name}_dry.wav")
        self.sr = sr
        self.frames = 0
        self.error = None
        self.q = queue.Queue()
        self._files = [sf.SoundFile(str(p), "w", sr, 1, "PCM_24") for p in self.paths]
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    @property
    def seconds(self):
        return self.frames / self.sr

    def push(self, dry, wet):
        self.q.put((dry, wet))

    def _run(self):
        try:
            while True:
                item = self.q.get()
                if item is None:
                    break
                dry, wet = item
                self._files[0].write(wet.astype(np.float32))
                self._files[1].write(dry.astype(np.float32))
                self.frames += len(dry)
        except Exception as e:
            self.error = e
        finally:
            for f in self._files:
                f.close()

    def stop(self):
        self.q.put(None)
        self._thread.join(timeout=10)
        return self.paths


def reamp_file(dry_path, params, out_path):
    """Verarbeitet eine trockene Aufnahme mit einem (neuen) Tone."""
    x, fs = sf.read(str(dry_path), always_2d=True)
    y = Amp(int(fs), params).process_offline(x.mean(axis=1).astype(np.float64))
    sf.write(str(out_path), y.astype(np.float32), int(fs), subtype="PCM_24")
    return out_path


# ---------------------------------------------------------------------------
# ASIO-Engine
# ---------------------------------------------------------------------------

def list_output_devices():
    """Alle Ausgabegeräte außer ASIO (WASAPI zuerst) – z. B. USB-Kopfhörer am Laptop."""
    apis = sd.query_hostapis()
    order = {"Windows WASAPI": 0, "Windows DirectSound": 1, "MME": 2}
    out = []
    for i, d in enumerate(sd.query_devices()):
        api = apis[d["hostapi"]]["name"]
        if d["max_output_channels"] > 0 and "asio" not in api.lower() and "wdm-ks" not in api.lower():
            out.append(dict(index=i, name=d["name"], hostapi=api, outputs=d["max_output_channels"]))
    return sorted(out, key=lambda d: (order.get(d["hostapi"], 9), d["name"].lower()))


class SampleFifo:
    """Puffer zwischen zwei Audio-Threads (Eingabe- und Ausgabegerät mit eigener Clock)."""

    def __init__(self, size):
        self.buf = np.zeros(size)
        self.size = size
        self.r = self.w = self.count = 0
        self.lock = threading.Lock()

    def write(self, x):
        with self.lock:
            n = len(x)
            free = self.size - self.count
            if n > free:                       # Überlauf: Älteste Samples verwerfen
                drop = n - free
                self.r = (self.r + drop) % self.size
                self.count -= drop
            idx = (self.w + np.arange(n)) % self.size
            self.buf[idx] = x
            self.w = (self.w + n) % self.size
            self.count += n

    def read(self, n):
        with self.lock:
            k = min(n, self.count)
            idx = (self.r + np.arange(k)) % self.size
            out = self.buf[idx].copy()
            self.r = (self.r + k) % self.size
            self.count -= k
        return out

    def discard(self, n):
        with self.lock:
            k = min(n, self.count)
            self.r = (self.r + k) % self.size
            self.count -= k


def _make_click(sr, freq, dur=0.03):
    t = np.arange(int(sr * dur)) / sr
    env = np.exp(-t * 140) * np.minimum(t * sr / 24, 1)
    return np.sin(2 * np.pi * freq * t) * env


def _ring_write(ring, pos, data):
    n, size = len(data), len(ring)
    if pos + n <= size:
        ring[pos:pos + n] = data
    else:
        k = size - pos
        ring[pos:] = data[:k]
        ring[:n - k] = data[k:]
    return (pos + n) % size


class Engine:
    RING = 8192

    def __init__(self):
        self.amp = None
        self.streams = []
        self.fifo = None
        self.block = 128
        self._primed = False
        self.out_underruns = 0
        self._target = 0
        self._fill_avg = 0.0
        self._phase = 0.0
        self._stage = np.zeros(0)
        self._last = 0.0
        self.sr = 48000
        self.rec = None              # Liste für Tone Match
        self.recorder = None         # WAV-Aufnahme
        self.mute = False            # Amp-Ausgang stumm (Stimmgerät)
        self.ring = np.zeros(self.RING)
        self.ring_pos = 0
        self.dry_ring = np.zeros(self.RING)
        self.dry_pos = 0
        self.in_peak = 0.0
        self.out_peak = 0.0
        self.xruns = 0
        self.last_error = None
        # Player (Backing-Track / Takes)
        self.player_buf = None
        self.player_pos = 0
        self.player_gain = 0.6
        # Metronom
        self.metro_on = False
        self.bpm = 120.0
        self.beats = 4
        self.accent = True
        self.metro_gain = 0.45
        self.metro_beat = 0
        self._next_beat = 0.0
        self._tail = None
        self._click_acc = self._click_norm = None

    @property
    def running(self):
        return bool(self.streams) and all(s.active for s in self.streams)

    @property
    def separate_output(self):
        return self.fifo is not None

    def start(self, device, sr, block, in_ch, out_chs, params, out_device=None):
        """device = ASIO-Eingang. out_device = None -> Ausgabe übers selbe ASIO-Gerät,
        sonst Index eines anderen Ausgabegeräts (z. B. USB-Kopfhörer über WASAPI)."""
        self.stop()
        if block % 4:
            raise ValueError("Die Puffergröße muss durch 4 teilbar sein.")
        self.sr, self.block = sr, block
        self.amp = Amp(sr, params)
        self.xruns = 0
        self._click_acc = _make_click(sr, 1600)
        self._click_norm = _make_click(sr, 1000) * 0.7
        try:
            if out_device is None:
                self.fifo = None
                st = sd.Stream(
                    device=(device, device), samplerate=sr, blocksize=block, dtype="float32",
                    channels=(1, len(out_chs)), latency="low",
                    extra_settings=(sd.AsioSettings(channel_selectors=[in_ch]),
                                    sd.AsioSettings(channel_selectors=out_chs)),
                    callback=self._duplex_callback)
                self.streams = [st]
            else:
                self.fifo = SampleFifo(sr * 2)
                self._primed = False
                self.out_underruns = 0
                self._target = int(2 * block + 0.006 * sr)    # Reserve gegen ungleichmäßige Pakete
                self._fill_avg = float(self._target)
                self._phase = 0.0
                self._stage = np.zeros(0)
                self._last = 0.0
                info = sd.query_devices(out_device)
                api = sd.query_hostapis(info["hostapi"])["name"]
                extra = sd.WasapiSettings(auto_convert=True) if "WASAPI" in api else None
                ist = sd.InputStream(
                    device=device, samplerate=sr, blocksize=block, dtype="float32", channels=1,
                    latency="low", extra_settings=sd.AsioSettings(channel_selectors=[in_ch]),
                    callback=self._in_callback)
                ost = sd.OutputStream(
                    device=out_device, samplerate=sr, dtype="float32",
                    channels=min(2, info["max_output_channels"]), latency=0.02,
                    extra_settings=extra, callback=self._out_callback)
                self.streams = [ist, ost]
            for st in self.streams:
                st.start()
        except Exception:
            self.stop()
            raise

    def total_latency_ms(self):
        lat = 0.0
        for st in self.streams:
            l = st.latency
            lat += sum(l) if isinstance(l, (tuple, list)) else l
        if self.fifo is not None:
            lat += self._target / self.sr
        return 1000 * lat

    def stop(self):
        self.stop_recording()
        self.player_buf = None
        streams, self.streams = self.streams, []
        for st in streams:
            try:
                st.stop()
                st.close()
            except Exception:
                pass
        self.fifo = None

    def update(self, params):
        if self.amp is not None:
            self.amp.update(params)

    # ----- Metronom -----
    def set_metronome(self, on=None, bpm=None, beats=None, accent=None, gain=None):
        if bpm is not None:
            self.bpm = float(min(max(bpm, 20), 400))
        if beats is not None:
            self.beats = max(int(beats), 1)
        if accent is not None:
            self.accent = bool(accent)
        if gain is not None:
            self.metro_gain = float(gain)
        if on is not None and bool(on) != self.metro_on:
            self._next_beat, self._tail, self.metro_beat = 0.0, None, 0
            self.metro_on = bool(on)

    def _metro(self, n):
        if not self.metro_on or self._click_acc is None:
            return None
        out = np.zeros(n)
        tail = self._tail
        if tail is not None:
            k = min(n, len(tail))
            out[:k] += tail[:k]
            self._tail = tail[k:] if len(tail) > k else None
        period = self.sr * 60.0 / self.bpm
        self._next_beat = min(self._next_beat, period)
        while self._next_beat < n:
            off = int(self._next_beat)
            first = self.metro_beat % self.beats == 0
            click = self._click_acc if (self.accent and first) else self._click_norm
            k = min(n - off, len(click))
            out[off:off + k] += click[:k]
            if len(click) > k:
                self._tail = click[k:]
            self.metro_beat += 1
            self._next_beat += period
        self._next_beat -= n
        return out * self.metro_gain

    # ----- Player -----
    def play(self, buf, gain=None):
        self.player_buf = None
        self.player_pos = 0
        if gain is not None:
            self.player_gain = gain
        self.player_buf = np.asarray(buf, dtype=np.float64)

    def stop_playback(self):
        self.player_buf = None

    def _player(self, n):
        buf = self.player_buf
        if buf is None:
            return None
        p = self.player_pos
        chunk = buf[p:p + n]
        out = np.zeros(n)
        out[:len(chunk)] = chunk * self.player_gain
        self.player_pos = p + n
        if self.player_pos >= len(buf):
            self.player_buf = None
        return out

    # ----- Aufnahme -----
    def start_recording(self, folder):
        if not self.running:
            raise RuntimeError("Starte zuerst das Audio.")
        self.recorder = Recorder(folder, self.sr)
        return self.recorder

    def stop_recording(self):
        r, self.recorder = self.recorder, None
        return r.stop() if r is not None else None

    # ----- Stimmgerät -----
    def tuner_buffer(self, n=4096):
        return np.roll(self.dry_ring, -self.dry_pos)[-n:]

    # ----- Audio-Callbacks -----
    def _duplex_callback(self, indata, outdata, frames, t, status):
        if status:
            self.xruns += 1
        outdata[:] = self._process(indata[:, 0], frames)[:, None].astype(np.float32)

    def _in_callback(self, indata, frames, t, status):
        if status:
            self.xruns += 1
        fifo = self.fifo
        out = self._process(indata[:, 0], frames)
        if fifo is not None:
            fifo.write(out)

    def _out_callback(self, outdata, frames, t, status):
        """Liest aus dem Puffer und resampelt stufenlos (Faktor ~1 ± 0,2 %), damit der Füllstand
        trotz unterschiedlicher Geräte-Clocks konstant bleibt. Kein Hin- und Herschalten, keine Sprünge."""
        if status.output_underflow:
            self.out_underruns += 1
        fifo = self.fifo
        if fifo is None:
            outdata.fill(0)
            return
        sr = self.sr
        # Ziel-Füllstand an die tatsächliche Paketgröße des Ausgabegeräts anpassen
        self._target = max(self._target, int(frames + 2 * self.block + 0.006 * sr))
        target = self._target
        if not self._primed:
            if fifo.count < target + frames:
                outdata.fill(0)
                return
            self._primed = True
            self._stage = np.zeros(0)
            self._phase = 0.0
            self._fill_avg = float(fifo.count)

        fill = fifo.count + len(self._stage)
        if fill > target + int(0.06 * sr):              # weit zu voll -> auf Ziel kürzen
            fifo.discard(fill - target)
            fill = fifo.count + len(self._stage)
        self._fill_avg += 0.03 * (fill - self._fill_avg)
        ratio = 1.0 + float(np.clip((self._fill_avg - target) / (0.5 * sr), -0.002, 0.002))

        pos = self._phase + ratio * np.arange(frames)
        need = int(pos[-1]) + 2
        if len(self._stage) < need:
            self._stage = np.concatenate([self._stage, fifo.read(need - len(self._stage))])
        stage = self._stage
        if len(stage) < need:
            # Unterlauf: vorhandene Samples ausgeben, dann weich ausblenden statt hart abzuschneiden
            self.out_underruns += 1
            self._target = min(int(self._target + self.block), int(0.08 * sr))
            self._primed = False
            stage = np.concatenate([stage if len(stage) else np.array([self._last]),
                                    np.zeros(need + 1)])
            i0 = pos.astype(np.int64)
            fr = pos - i0
            y = stage[i0] * (1 - fr) + stage[i0 + 1] * fr
            y *= np.linspace(1.0, 0.0, frames)
            self._last = 0.0
            outdata[:] = y[:, None].astype(np.float32)
            return
        i0 = pos.astype(np.int64)
        fr = pos - i0
        y = stage[i0] * (1 - fr) + stage[i0 + 1] * fr
        end = self._phase + ratio * frames
        used = int(end)
        self._phase = end - used
        self._stage = stage[used:]
        self._last = float(y[-1])
        outdata[:] = y[:, None].astype(np.float32)

    def _process(self, x_in, frames):
        x = x_in.astype(np.float64)
        pk = float(np.abs(x).max())
        if pk > self.in_peak:
            self.in_peak = pk
        self.dry_pos = _ring_write(self.dry_ring, self.dry_pos, x)
        rec = self.rec
        if rec is not None:
            rec.append(x.copy())
        try:
            y = self.amp.process(x)
        except Exception as e:
            self.last_error = e
            y = np.zeros(frames)
        pk = float(np.abs(y).max())
        if pk > self.out_peak:
            self.out_peak = pk
        recorder = self.recorder
        if recorder is not None:
            recorder.push(x.copy(), y.copy())
        self.ring_pos = _ring_write(self.ring, self.ring_pos, y)

        out = np.zeros(frames) if self.mute else y.copy()
        m = self._metro(frames)
        if m is not None:
            out += m
        pb = self._player(frames)
        if pb is not None:
            out += pb
        return np.clip(out, -1.0, 1.0)

    def live_spectrum(self):
        buf = np.roll(self.ring, -self.ring_pos)
        if np.sqrt(np.mean(buf ** 2)) < 1e-4:
            return None
        p = np.abs(np.fft.rfft(buf * np.hanning(self.RING))) ** 2
        return spectrum_to_log(np.fft.rfftfreq(self.RING, 1 / self.sr), p)


# ---------------------------------------------------------------------------
# Export als REAPER-Plugin (JSFX)
# ---------------------------------------------------------------------------

JSFX_RATES = [44100, 48000, 88200, 96000, 176400, 192000]
DELAY_SYNC = ["Aus", "1/2", "1/4", "1/8 punktiert", "1/8", "1/16"]
DELAY_SYNC_MULT = [0, 2.0, 1.0, 0.75, 0.5, 0.25]

_JSFX_BODY = r'''
@init
ext_tail_size = -1;
h_os = 0; fir = 200; firh = 800; firtab = 3000;
osa_x = 1900; osa_v = 1940; osp_x = 2100; osp_v = 2140;
rv_cb = 2400; rv_cl = 2410; rv_ci = 2420; rv_cs = 2430;
rv_ab = 2440; rv_al = 2450; rv_ai = 2460;
@@TABLES@@

function tanh_(x) local(e) (
  x > 20 ? 1 : (x < -20 ? -1 : (e = exp(2*x); (e - 1)/(e + 1)));
);
function bq(x) instance(b0,b1,b2,a1,a2,z1,z2) local(y) (
  y = b0*x + z1;
  z1 = b1*x - a1*y + z2;
  z2 = b2*x - a2*y;
  y;
);
function bq_reset() instance(z1,z2) ( z1 = 0; z2 = 0; );
function bq_peak(f0,g,q,fs) instance(b0,b1,b2,a1,a2) local(A,w,al,c,a0) (
  A = pow(10, g/40); w = 2*$pi*f0/fs; al = sin(w)/(2*q); c = cos(w); a0 = 1 + al/A;
  b0 = (1 + al*A)/a0; b1 = -2*c/a0; b2 = (1 - al*A)/a0; a1 = -2*c/a0; a2 = (1 - al/A)/a0;
);
function bq_lp(f0,q,fs) instance(b0,b1,b2,a1,a2) local(w,al,c,a0) (
  w = 2*$pi*f0/fs; al = sin(w)/(2*q); c = cos(w); a0 = 1 + al;
  b0 = (1 - c)/2/a0; b1 = (1 - c)/a0; b2 = b0; a1 = -2*c/a0; a2 = (1 - al)/a0;
);
function bq_hp(f0,q,fs) instance(b0,b1,b2,a1,a2) local(w,al,c,a0) (
  w = 2*$pi*f0/fs; al = sin(w)/(2*q); c = cos(w); a0 = 1 + al;
  b0 = (1 + c)/2/a0; b1 = -(1 + c)/a0; b2 = b0; a1 = -2*c/a0; a2 = (1 - al)/a0;
);
function bq_lp1(f0,fs) instance(b0,b1,b2,a1,a2) local(k) (
  k = tan($pi*f0/fs); b0 = k/(k + 1); b1 = b0; b2 = 0; a1 = (k - 1)/(k + 1); a2 = 0;
);
function bq_hp1(f0,fs) instance(b0,b1,b2,a1,a2) local(k) (
  k = tan($pi*f0/fs); b0 = 1/(k + 1); b1 = -b0; b2 = 0; a1 = (k - 1)/(k + 1); a2 = 0;
);
function bq_shelf(f0,g,fs,high) instance(b0,b1,b2,a1,a2) local(A,w,c,s,al,sq,a0) (
  A = pow(10, g/40); w = 2*$pi*f0/fs; c = cos(w); s = sin(w);
  al = s/2*sqrt((A + 1/A)*(1/0.7 - 1) + 2); sq = 2*sqrt(A)*al;
  high ? (
    a0 = (A+1) - (A-1)*c + sq;
    b0 = A*((A+1) + (A-1)*c + sq)/a0; b1 = -2*A*((A-1) + (A+1)*c)/a0; b2 = A*((A+1) + (A-1)*c - sq)/a0;
    a1 = 2*((A-1) - (A+1)*c)/a0; a2 = ((A+1) - (A-1)*c - sq)/a0;
  ) : (
    a0 = (A+1) + (A-1)*c + sq;
    b0 = A*((A+1) - (A-1)*c + sq)/a0; b1 = 2*A*((A-1) - (A+1)*c)/a0; b2 = A*((A+1) - (A-1)*c - sq)/a0;
    a1 = -2*((A-1) + (A+1)*c)/a0; a2 = ((A+1) + (A-1)*c - sq)/a0;
  );
);

// 4x-Oversampling (identisch zur Python-Version: Nullen einfügen + 128-Tap-FIR)
function os_init(xbase, vbase) instance(xb, vb, xp, vp) (
  xb = xbase; vb = vbase; xp = 0; vp = 0; memset(xb, 0, 32); memset(vb, 0, 128);
);
function os_push(x) instance(xb, xp) ( xp = (xp + 1) & 31; xb[xp] = x; );
function os_phase(j) instance(xb, xp) local(acc, q) (
  acc = 0; q = 0; loop(32, acc += h_os[j + 4*q] * xb[(xp - q) & 31]; q += 1; ); acc*4;
);
function os_vpush(v) instance(vb, vp) ( vp = (vp + 1) & 127; vb[vp] = v; );
function os_down() instance(vb, vp) local(acc, m) (
  acc = 0; m = 0; loop(128, acc += h_os[m] * vb[(vp - m) & 127]; m += 1; ); acc;
);

// Match-EQ passend zur Projekt-Samplerate wählen
fir_n = 0;
fir_count > 0 ? (
  best = 0; bd = 1000000000; k = 0;
  loop(fir_count, abs(fir_rates[k] - srate) < bd ? (bd = abs(fir_rates[k] - srate); best = k;); k += 1;);
  memcpy(fir, firtab + best*@@FIRTAPS@@, @@FIRTAPS@@); fir_n = @@FIRTAPS@@;
);
memset(firh, 0, 1024); fp = 0;

osa.os_init(osa_x, osa_v); osp.os_init(osp_x, osp_v);
dc.bq_hp1(20, srate);
cab_hp.bq_hp(75, 0.70710678, srate);
cab_p1.bq_peak(110, 4, 1.2, srate);
cab_p2.bq_peak(2500, 3, 1.0, srate);
tb = tanh_(bias); it18 = 1/tanh_(1.8); it13 = 1/tanh_(1.3);
ks = 1 - exp(-1/(0.01*srate));
g_avg = 1/128; g_att = 1 - pow(0.3, 1/128); g_rel = 1 - pow(0.92, 1/128); gg = 1; gms = 0;

ch_buf = 20000; ch_len = ceil(0.05*srate) + 16; ch_w = 0; ch_ph = 0;
dl_buf = ch_buf + ch_len + 16; dl_len = ceil(1.6*srate) + 16; dl_w = 0; dl_d = -1;
dl_k = 1 - pow(0.85, 1/128);
rv_next = dl_buf + dl_len + 16; rs = srate/44100;
k = 0; loop(8, rv_cl[k] = floor(rv_combs[k]*rs); rv_cb[k] = rv_next; rv_next += rv_cl[k] + 4; rv_ci[k] = 0; rv_cs[k] = 0; k += 1;);
k = 0; loop(4, rv_al[k] = floor(rv_aps[k]*rs); rv_ab[k] = rv_next; rv_next += rv_al[k] + 4; rv_ai[k] = 0; k += 1;);
memset(ch_buf, 0, rv_next - ch_buf);
p_od = p_ds = p_ch = p_dl = p_rv = 0;
first = 1;

@slider
trim_t = pow(10, slider1/20);
gate_thr = slider2;
drive_t = pow(10, slider3/20);
pre_hp.bq_hp(slider4, 0.70710678, srate);
pre_pk.bq_peak(720, slider5, 0.7, srate);
t_lo.bq_shelf(120, slider6, srate, 0);
t_mid.bq_peak(800, slider7, 0.8, srate);
t_hi.bq_shelf(3000, slider8, srate, 1);
cab_lp1.bq_lp(min(slider9, 0.45*srate), 0.54119610, srate);
cab_lp2.bq_lp(min(slider9, 0.45*srate), 1.30656296, srate);
match_on = slider10 && fir_n > 0;
master_t = pow(10, slider11/20);

fso = 4*srate;
od_on = slider12; od_g = pow(10, (6 + 34*slider13/100)/20); od_lv = pow(10, slider15/20);
od_hp.bq_hp1(720, fso); od_tn.bq_lp(500*pow(10, 1.1*slider14/100), 0.70710678, fso);
ds_on = slider16; ds_g = pow(10, (6 + 42*slider17/100)/20); ds_lv = pow(10, slider19/20);
ds_hp.bq_hp1(120, fso); ds_tn.bq_lp(600*pow(10, 1.12*slider18/100), 0.70710678, fso);
(od_on || ds_on) && !(p_od || p_ds) ? (
  osp.os_init(osp_x, osp_v); od_hp.bq_reset(); od_tn.bq_reset(); ds_hp.bq_reset(); ds_tn.bq_reset();
);
pdc_delay = (od_on || ds_on) ? 64 : 32; pdc_bot_ch = 0; pdc_top_ch = 2;

ch_on = slider20; ch_wst = 2*$pi*slider21/srate; ch_depth = slider22/100; ch_mix = slider23/100;
ch_on && !p_ch ? ( memset(ch_buf, 0, ch_len); );
dl_on = slider24; dl_fb = min(slider27/100, 0.95); dl_mix = slider28/100;
dl_lp.bq_lp1(min(1200*pow(10, 1.1*slider29/100), 0.45*srate), srate);
dl_on && !p_dl ? ( memset(dl_buf, 0, dl_len); dl_lp.bq_reset(); dl_d = -1; );
rv_on = slider30; rv_fb = min(0.7 + 0.28*slider31/100, 0.97); rv_damp = (1 - slider32/100)*0.4; rv_mix = slider33/100;
rv_on && !p_rv ? (
  k = 0; loop(8, memset(rv_cb[k], 0, rv_cl[k]); rv_cs[k] = 0; k += 1;);
  k = 0; loop(4, memset(rv_ab[k], 0, rv_al[k]); k += 1;);
);
p_od = od_on; p_ds = ds_on; p_ch = ch_on; p_dl = dl_on; p_rv = rv_on;
first ? ( trim_c = trim_t; drive_c = drive_t; master_c = master_t; first = 0; );

@block
dl_ms = slider26 > 0 ? 60000/max(tempo, 1)*sync_mult[slider26] : slider25;
while (dl_ms > 1500) ( dl_ms *= 0.5; );
dl_target = max(dl_ms*0.001*srate, 2);
dl_d < 0 ? (dl_d = dl_target;);

@sample
x = slider34 == 1 ? spl1 : (slider34 == 2 ? (spl0 + spl1)*0.5 : spl0);
trim_c += (trim_t - trim_c)*ks; drive_c += (drive_t - drive_c)*ks; master_c += (master_t - master_c)*ks;
x *= trim_c;

// Noise-Gate
gate_thr > -89.5 ? (
  gms += (x*x - gms)*g_avg;
  tgt = 10*log10(gms + 0.000000000000000000000001) > gate_thr - (gg > 0.5 ? 6 : 0) ? 1 : 0;
  gg += (tgt - gg)*(tgt > gg ? g_att : g_rel);
) : ( gg = 1; );
x *= gg;

// Overdrive / Distortion (4x überabgetastet)
(od_on || ds_on) ? (
  osp.os_push(x);
  j = 0;
  loop(4,
    u = osp.os_phase(j);
    od_on ? ( c = tanh_(od_g*od_hp.bq(u)); u = od_tn.bq(0.5*u + 0.25*c)*od_lv; );
    ds_on ? ( z = ds_g*ds_hp.bq(u); az = abs(z); c = z/pow(1 + az*az*az, 1/3); u = ds_tn.bq(c)*0.3*ds_lv; );
    osp.os_vpush(u);
    j == 0 ? ( x = osp.os_down(); );
    j += 1;
  );
);

// Amp
x = pre_pk.bq(pre_hp.bq(x));
osa.os_push(x*drive_c);
j = 0;
loop(4,
  u = osa.os_phase(j);
  u = tanh_(u + bias) - tb;
  u = tanh_(1.8*u)*it18;
  osa.os_vpush(u);
  j == 0 ? ( y = osa.os_down(); );
  j += 1;
);
x = dc.bq(y);
x = t_hi.bq(t_mid.bq(t_lo.bq(x)));
x = tanh_(1.3*x)*it13;
x = cab_lp2.bq(cab_lp1.bq(cab_p2.bq(cab_p1.bq(cab_hp.bq(x)))));

// Match-EQ
fp = (fp + 1) & 1023; firh[fp] = x;
match_on ? (
  acc = 0; m = 0; loop(fir_n, acc += fir[m]*firh[(fp - m) & 1023]; m += 1;); x = acc;
);

// Chorus
ch_on ? (
  ch_buf[ch_w] = x;
  d = (7 + 5*ch_depth*sin(ch_ph))*0.001*srate;
  ch_ph += ch_wst; ch_ph > 2*$pi ? (ch_ph -= 2*$pi;);
  pos = ch_w - d; pos < 0 ? (pos += ch_len;);
  i0 = floor(pos); fr = pos - i0; i1 = i0 + 1; i1 >= ch_len ? (i1 -= ch_len;);
  wet = ch_buf[i0]*(1 - fr) + ch_buf[i1]*fr;
  ch_w += 1; ch_w >= ch_len ? (ch_w = 0;);
  x = x*(1 - 0.3*ch_mix) + wet*0.7*ch_mix;
);

// Delay
dl_on ? (
  dl_d += (dl_target - dl_d)*dl_k;
  pos = dl_w - max(dl_d, 2); pos < 0 ? (pos += dl_len;);
  i0 = floor(pos); fr = pos - i0; i1 = i0 + 1; i1 >= dl_len ? (i1 -= dl_len;);
  tap = dl_lp.bq(dl_buf[i0]*(1 - fr) + dl_buf[i1]*fr);
  dl_buf[dl_w] = x + dl_fb*tap;
  dl_w += 1; dl_w >= dl_len ? (dl_w = 0;);
  x += dl_mix*tap;
);

// Reverb
rv_on ? (
  inp = x*0.015; acc = 0; k = 0;
  loop(8,
    b = rv_cb[k]; i = rv_ci[k]; rd = b[i];
    rv_cs[k] = (1 - rv_damp)*rd + rv_damp*rv_cs[k];
    b[i] = inp + rv_fb*rv_cs[k];
    i += 1; i >= rv_cl[k] ? (i = 0;); rv_ci[k] = i;
    acc += rd; k += 1;
  );
  k = 0;
  loop(4,
    b = rv_ab[k]; i = rv_ai[k]; bo = b[i];
    o = bo - acc; b[i] = acc + 0.5*bo;
    i += 1; i >= rv_al[k] ? (i = 0;); rv_ai[k] = i;
    acc = o; k += 1;
  );
  x = x*(1 - 0.5*rv_mix) + rv_mix*acc*3;
);

spl0 = spl1 = max(-1, min(1, x*master_c));
'''


def _num(v):
    """Zahl ohne Exponent-Schreibweise (EEL2 versteht kein 1e-5)."""
    v = float(v)
    if abs(v) < 1e-12:
        return "0"
    return np.format_float_positional(v, precision=12, unique=False, fractional=False, trim="-")


def _table(name, values, per_line=8):
    lines = [f"{name}_t = {name};"]
    vals = list(values)
    for i in range(0, len(vals), per_line):
        lines.append(" ".join(f"{name}_t[{i + k}]={_num(v)};" for k, v in enumerate(vals[i:i + per_line])))
    return "\n".join(lines)


def export_jsfx(params, title):
    """Erzeugt den Quelltext eines REAPER-JSFX-Plugins mit diesem Tone (gelernte Angleichung fest eingebaut)."""
    p = dict(DEFAULT_PARAMS)
    p.update(params)
    fx = fx_params(p)

    def cl(v, lo, hi):
        return min(max(float(v), lo), hi)

    od, ds, ch, dl, rv = (fx[k] for k in ("overdrive", "distortion", "chorus", "delay", "reverb"))
    sync = DELAY_SYNC.index(dl.get("division", "1/4")) if dl.get("sync") and dl.get("division") in DELAY_SYNC else 0
    sliders = [
        (cl(p["input_trim_db"], -24, 36), "<-24,36,0.1>Trim (dB)"),
        (cl(p["gate_db"], -90, -30), "<-90,-30,1>Gate (dB, -90 = aus)"),
        (cl(p["gain_db"], -10, 50), "<-10,50,0.1>Gain (dB)"),
        (cl(p["tight_hz"], 20, 250), "<20,250,1>Tight (Hz)"),
        (cl(p["mid_push_db"], 0, 12), "<0,12,0.1>Push (dB)"),
        (cl(p["bass_db"], -12, 12), "<-12,12,0.1>Bass (dB)"),
        (cl(p["mid_db"], -12, 12), "<-12,12,0.1>Middle (dB)"),
        (cl(p["treble_db"], -12, 12), "<-12,12,0.1>Treble (dB)"),
        (cl(p["cab_lp_hz"], 2500, 9000), "<2500,9000,10>Presence (Hz)"),
        (1 if (p.get("match_on") and p.get("match_db") is not None) else 0, "<0,1,1{Aus,An}>Angleichung an den Song"),
        (cl(p["master_db"], -40, 12), "<-40,12,0.1>Master (dB)"),
        (1 if od["on"] else 0, "<0,1,1{Aus,An}>Overdrive"),
        (cl(od["drive"] * 100, 0, 100), "<0,100,1>Overdrive: Drive (%)"),
        (cl(od["tone"] * 100, 0, 100), "<0,100,1>Overdrive: Tone (%)"),
        (cl(od["level"], -12, 12), "<-12,12,0.1>Overdrive: Level (dB)"),
        (1 if ds["on"] else 0, "<0,1,1{Aus,An}>Distortion"),
        (cl(ds["gain"] * 100, 0, 100), "<0,100,1>Distortion: Gain (%)"),
        (cl(ds["tone"] * 100, 0, 100), "<0,100,1>Distortion: Tone (%)"),
        (cl(ds["level"], -12, 12), "<-12,12,0.1>Distortion: Level (dB)"),
        (1 if ch["on"] else 0, "<0,1,1{Aus,An}>Chorus"),
        (cl(ch["rate"], 0.1, 4), "<0.1,4,0.01>Chorus: Rate (Hz)"),
        (cl(ch["depth"] * 100, 0, 100), "<0,100,1>Chorus: Depth (%)"),
        (cl(ch["mix"] * 100, 0, 100), "<0,100,1>Chorus: Mix (%)"),
        (1 if dl["on"] else 0, "<0,1,1{Aus,An}>Delay"),
        (cl(dl["time_ms"], 30, 1500), "<30,1500,1>Delay: Time (ms)"),
        (sync, "<0,5,1{" + ",".join(DELAY_SYNC) + "}>Delay: Sync zum Projekttempo"),
        (cl(dl["feedback"] * 100, 0, 90), "<0,90,1>Delay: Repeats (%)"),
        (cl(dl["mix"] * 100, 0, 100), "<0,100,1>Delay: Mix (%)"),
        (cl(dl["tone"] * 100, 0, 100), "<0,100,1>Delay: Tone (%)"),
        (1 if rv["on"] else 0, "<0,1,1{Aus,An}>Reverb"),
        (cl(rv["size"] * 100, 0, 100), "<0,100,1>Reverb: Size (%)"),
        (cl(rv["tone"] * 100, 0, 100), "<0,100,1>Reverb: Tone (%)"),
        (cl(rv["mix"] * 100, 0, 100), "<0,100,1>Reverb: Mix (%)"),
        (0, "<0,2,1{Links,Rechts,Summe}>Eingang"),
    ]
    safe_title = " ".join(str(title).split()) or "Tone"
    head = [f"desc:ToneMatch - {safe_title}",
            "//tags: guitar amp distortion delay reverb",
            "//author: ToneMatch (Export)",
            "in_pin:Gitarre L", "in_pin:Gitarre R", "out_pin:Left", "out_pin:Right", ""]
    head += [f"slider{i + 1}:{_num(v)}{spec}" for i, (v, spec) in enumerate(sliders)]

    md = p.get("match_db")
    tables = [_table("h_os", Oversampler(4).h),
              f"bias = {_num(p['bias'])};",
              "sync_mult = 2600; " + " ".join(f"sync_mult[{i}]={_num(m)};" for i, m in enumerate(DELAY_SYNC_MULT)),
              "rv_combs = 2700; " + " ".join(f"rv_combs[{i}]={c};" for i, c in enumerate(Reverb.COMBS)),
              "rv_aps = 2720; " + " ".join(f"rv_aps[{i}]={a};" for i, a in enumerate(Reverb.APS)),
              "fir_rates = 2740;"]
    if md is not None:
        firs = [design_fir_from_curve(np.asarray(md), float(p["match_strength"]), fs,
                                      bool(p.get("ref_isolated", False))) for fs in JSFX_RATES]
        tables.append(" ".join(f"fir_rates[{i}]={fs};" for i, fs in enumerate(JSFX_RATES)))
        tables.append(f"fir_count = {len(JSFX_RATES)};")
        tables.append(_table("firtab", np.concatenate(firs)))
    else:
        tables.append("fir_count = 0;")
    body = _JSFX_BODY.replace("@@TABLES@@", "\n".join(tables)).replace("@@FIRTAPS@@", str(FIR_TAPS))
    return "\n".join(head) + "\n" + body


def reaper_effects_dir():
    """Standard-Ordner für JSFX in REAPER (Windows: %APPDATA%\\REAPER\\Effects)."""
    if os.name == "nt" and os.environ.get("APPDATA"):
        base = Path(os.environ["APPDATA"]) / "REAPER" / "Effects"
    else:
        base = Path.home() / ".config" / "REAPER" / "Effects"
    return base if base.parent.is_dir() else None
