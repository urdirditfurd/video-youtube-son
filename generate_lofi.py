#!/usr/bin/env python3
"""Generate an original continuous lo-fi hip-hop study mix.

The piece is synthesized (no samples of existing records): Rhodes-style
electric piano, chorused guitar, rounded bass, brushed drums, pads, vinyl
crackle, and occasional rain or café ambience. Tempo stays at 80 BPM, inside
the requested 75–85 range, so the five-hour form lands on an exact bar grid.
"""

from __future__ import annotations

import argparse
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from scipy.signal import lfilter

SR = 32_000
BPM = 80.0
BEAT_S = 60.0 / BPM
BAR_BEATS = 4
BAR_S = BEAT_S * BAR_BEATS
BAR_SAMPLES = int(round(BAR_S * SR))
TAIL_S = 6.0
TAIL = int(TAIL_S * SR)
SWING = 0.56

# Tuned so a full scene sits around -16 LUFS with the peaks left intact.
GAIN_KICK = 0.48
GAIN_SNARE = 0.42
GAIN_HAT = 0.15
GAIN_BASS = 0.28
GAIN_RHODES = 0.27
GAIN_GUITAR = 0.18
GAIN_PAD = 0.145
GAIN_RAIN = 0.048
GAIN_CAFE = 0.034
GAIN_HISS = 0.0042
GAIN_CRACKLE = 0.07
MASTER = 0.84

PENTATONIC = {
    "minor": (0, 3, 5, 7, 10),
    "major": (0, 2, 4, 7, 9),
}

SCENE_BARS = 600  # 30 minutes at 80 BPM
PARAM_FADE_BARS = 40  # 2 minutes of texture crossfade
TRIGGER_UTC = (2026, 10, 18, 4, 47, 24)


@dataclass(frozen=True)
class Scene:
    name: str
    chords: tuple[tuple[int, ...], ...]
    bass_roots: tuple[int, ...]
    guitar_root: int
    pentatonic: str
    rain: float
    cafe: float
    guitar: float
    drums: float
    pad: float


SCENES: tuple[Scene, ...] = (
    Scene(
        "fenetre-pluie",
        (
            (62, 65, 72, 76),  # Dm9
            (59, 65, 69, 76),  # G13
            (60, 64, 67, 71),  # Cmaj7
            (57, 60, 64, 67),  # Am7
        ),
        (38, 43, 36, 45),
        62,
        "minor",
        rain=0.90,
        cafe=0.02,
        guitar=0.08,
        drums=0.78,
        pad=0.80,
    ),
    Scene(
        "lampe-douce",
        (
            (62, 65, 69, 72),  # Dm7
            (59, 65, 69, 74),  # G
            (60, 64, 67, 71),  # Cmaj7
            (57, 64, 67, 72),  # Am7
        ),
        (38, 43, 36, 33),
        62,
        "minor",
        rain=0.42,
        cafe=0.12,
        guitar=0.40,
        drums=0.88,
        pad=0.74,
    ),
    Scene(
        "nuit-la-mineur",
        (
            (57, 60, 64, 67),  # Am7
            (53, 57, 60, 64),  # Fmaj7
            (60, 64, 67, 71),  # Cmaj7
            (64, 67, 71, 74),  # Em7
        ),
        (45, 41, 36, 40),
        57,
        "minor",
        rain=0.16,
        cafe=0.48,
        guitar=0.58,
        drums=0.96,
        pad=0.62,
    ),
    Scene(
        "pages-tournees",
        (
            (57, 60, 64, 67),  # Am7
            (62, 65, 69, 72),  # Dm7
            (55, 59, 62, 65),  # G7
            (60, 64, 67, 71),  # Cmaj7
        ),
        (33, 38, 43, 36),
        57,
        "minor",
        rain=0.55,
        cafe=0.36,
        guitar=0.70,
        drums=0.90,
        pad=0.70,
    ),
    Scene(
        "lumiere-chaude",
        (
            (60, 64, 67, 71),  # Cmaj7
            (57, 60, 64, 67),  # Am7
            (62, 65, 69, 72),  # Dm7
            (55, 59, 62, 65),  # G7
        ),
        (36, 33, 38, 43),
        60,
        "major",
        rain=0.10,
        cafe=0.58,
        guitar=0.50,
        drums=0.84,
        pad=0.88,
    ),
    Scene(
        "souvenir",
        (
            (53, 57, 60, 64),  # Fmaj7
            (62, 65, 69, 72),  # Dm7
            (55, 58, 62, 65),  # Gm7
            (60, 64, 67, 70),  # C7
        ),
        (41, 38, 43, 36),
        53,
        "major",
        rain=0.06,
        cafe=0.28,
        guitar=0.42,
        drums=0.72,
        pad=1.00,
    ),
    Scene(
        "fin-de-jour",
        (
            (58, 62, 65, 69),  # Bbmaj7
            (55, 58, 62, 65),  # Gm7
            (63, 67, 70, 74),  # Ebmaj7
            (53, 57, 60, 63),  # F7
        ),
        (46, 43, 39, 41),
        58,
        "major",
        rain=0.22,
        cafe=0.16,
        guitar=0.60,
        drums=0.80,
        pad=0.84,
    ),
    Scene(
        "melancolie",
        (
            (55, 58, 62, 65),  # Gm7
            (63, 67, 70, 74),  # Ebmaj7
            (58, 62, 65, 69),  # Bbmaj7
            (53, 57, 60, 63),  # F7
        ),
        (43, 39, 46, 41),
        55,
        "minor",
        rain=0.72,
        cafe=0.06,
        guitar=0.32,
        drums=0.74,
        pad=0.92,
    ),
    Scene(
        "calme-clair",
        (
            (64, 67, 71, 74),  # Em7
            (60, 64, 67, 71),  # Cmaj7
            (55, 59, 62, 66),  # Gmaj7
            (62, 66, 69, 72),  # D7
        ),
        (40, 36, 43, 38),
        64,
        "minor",
        rain=0.34,
        cafe=0.22,
        guitar=0.48,
        drums=0.86,
        pad=0.68,
    ),
    Scene(
        "retour",
        (
            (62, 65, 69, 72),  # Dm7
            (58, 62, 65, 69),  # Bbmaj7
            (53, 57, 60, 64),  # Fmaj7
            (60, 64, 67, 71),  # Cmaj7
        ),
        (38, 46, 41, 36),
        62,
        "minor",
        rain=0.60,
        cafe=0.10,
        guitar=0.22,
        drums=0.70,
        pad=0.86,
    ),
)

# Décale le parcours harmonique selon l'heure de Paris, pour que chaque
# déclenchement parte d'une autre scène tout en gardant les mêmes fondus.
_PARIS_HOUR = (
    datetime(*TRIGGER_UTC, tzinfo=timezone.utc)
    .astimezone(ZoneInfo("Europe/Paris"))
    .hour
)
SCENE_ROTATION = _PARIS_HOUR % len(SCENES)


def midi_to_hz(note: float) -> float:
    return 440.0 * (2.0 ** ((note - 69.0) / 12.0))


def smoothstep(value: float) -> float:
    x = min(1.0, max(0.0, value))
    return x * x * (3.0 - 2.0 * x)


def scene_index(bar: int) -> int:
    raw = min(bar // SCENE_BARS, len(SCENES) - 1)
    return (raw + SCENE_ROTATION) % len(SCENES)


def mix_param(bar: float, name: str) -> float:
    """Texture parameter with a long crossfade into the next scene."""
    raw = min(int(bar) // SCENE_BARS, len(SCENES) - 1)
    idx = (raw + SCENE_ROTATION) % len(SCENES)
    current = float(getattr(SCENES[idx], name))
    if raw >= len(SCENES) - 1:
        return current
    local = int(bar) - raw * SCENE_BARS
    if local < SCENE_BARS - PARAM_FADE_BARS:
        return current
    nxt = float(getattr(SCENES[(idx + 1) % len(SCENES)], name))
    fade = smoothstep((local - (SCENE_BARS - PARAM_FADE_BARS)) / PARAM_FADE_BARS)
    return current * (1.0 - fade) + nxt * fade


def chord_at(bar: int) -> tuple[int, tuple[int, ...], int]:
    idx = scene_index(bar)
    scene = SCENES[idx]
    slot = (bar % 8) // 2
    return idx, scene.chords[slot], scene.bass_roots[slot]


def guitar_scale(scene: Scene) -> list[int]:
    intervals = PENTATONIC[scene.pentatonic]
    notes: list[int] = []
    for octave in range(3):
        for interval in intervals:
            note = scene.guitar_root + octave * 12 + interval
            if 67 <= note <= 83:
                notes.append(note)
    return notes


def one_pole_lowpass(
    samples: np.ndarray, cutoff: float, zi: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    coeff = float(np.exp(-2.0 * np.pi * cutoff / SR))
    numerator = np.array([1.0 - coeff])
    denominator = np.array([1.0, -coeff])
    if zi is None:
        zi = np.zeros(1, dtype=np.float64)
    filtered, next_zi = lfilter(numerator, denominator, samples, zi=zi)
    return filtered, next_zi


def one_pole_highpass(
    samples: np.ndarray, cutoff: float, zi: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    low, next_zi = one_pole_lowpass(samples, cutoff, zi)
    return samples - low, next_zi


def place(buffer: np.ndarray, start: int, signal: np.ndarray) -> None:
    if signal.size == 0 or start >= buffer.shape[0]:
        return
    if start < 0:
        signal = signal[-start:]
        start = 0
    if signal.size == 0:
        return
    end = start + signal.shape[0]
    if end > buffer.shape[0]:
        signal = signal[: buffer.shape[0] - start]
    buffer[start : start + signal.shape[0]] += signal


def pan(mono: np.ndarray, position: float) -> tuple[np.ndarray, np.ndarray]:
    angle = (position + 1.0) * 0.25 * np.pi
    return np.cos(angle) * mono, np.sin(angle) * mono


def attack_decay(n: int, attack_s: float, decay_s: float) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / SR
    attack = 1.0 - np.exp(-t / max(attack_s, 1e-4))
    return attack * np.exp(-t / max(decay_s, 1e-4))


def synth_kick(velocity: float, rng: np.random.Generator) -> np.ndarray:
    n = int(0.42 * SR)
    t = np.arange(n, dtype=np.float64) / SR
    freq = 50.0 + 95.0 * np.exp(-t * 20.0)
    phase = 2.0 * np.pi * np.cumsum(freq) / SR
    body = np.sin(phase) * np.exp(-t * 6.2) * (1.0 - np.exp(-t / 0.004))
    click = rng.standard_normal(n) * np.exp(-t * 160.0) * 0.07
    signal = np.tanh((body + click) * 1.6) * velocity * GAIN_KICK
    return signal


def synth_snare(velocity: float, rng: np.random.Generator) -> np.ndarray:
    n = int(0.55 * SR)
    t = np.arange(n, dtype=np.float64) / SR
    noise = rng.standard_normal(n)
    brush = noise * np.exp(-t * 7.5)
    body = np.sin(2.0 * np.pi * 188.0 * t) * np.exp(-t * 16.0) * 0.45
    signal = (brush * 0.62 + body) * velocity * GAIN_SNARE
    return np.tanh(signal * 1.8)


def synth_hat(velocity: float, open_hat: bool, rng: np.random.Generator) -> np.ndarray:
    n = int((0.16 if open_hat else 0.05) * SR)
    t = np.arange(n, dtype=np.float64) / SR
    noise = rng.standard_normal(n)
    decay = 11.0 if open_hat else 48.0
    return noise * np.exp(-t * decay) * velocity * GAIN_HAT


def synth_bass(freq: float, velocity: float) -> np.ndarray:
    n = int(1.05 * SR)
    t = np.arange(n, dtype=np.float64) / SR
    signal = (
        np.sin(2.0 * np.pi * freq * t)
        + 0.28 * np.sin(2.0 * np.pi * freq * 2.0 * t)
        + 0.06 * np.sin(2.0 * np.pi * freq * 3.0 * t)
    )
    signal *= attack_decay(n, 0.012, 0.75)
    return np.tanh(signal * 1.35) * velocity * GAIN_BASS


def synth_rhodes(freq: float, velocity: float, rng: np.random.Generator) -> np.ndarray:
    n = int(5.2 * SR)
    t = np.arange(n, dtype=np.float64) / SR
    mod_env = np.exp(-t * 5.2)
    modulator = np.sin(2.0 * np.pi * freq * t) * mod_env * (1.6 + velocity)
    carrier = np.sin(2.0 * np.pi * freq * t + modulator)
    tine = np.sin(2.0 * np.pi * freq * 2.0 * t) * np.exp(-t * 3.4) * 0.18
    bell = np.sin(2.0 * np.pi * freq * 3.97 * t) * np.exp(-t * 7.5) * 0.05
    hammer_n = min(n, int(0.03 * SR))
    hammer = np.zeros(n, dtype=np.float64)
    hammer[:hammer_n] = rng.standard_normal(hammer_n) * np.exp(
        -np.arange(hammer_n) / (SR * 0.008)
    )
    signal = (carrier * 0.82 + tine + bell + hammer * 0.04) * attack_decay(
        n, 0.014, 1.15 + (1.0 - velocity) * 0.35
    )
    return np.tanh(signal * 1.15) * (0.55 + 0.45 * velocity) * GAIN_RHODES


def synth_guitar(freq: float, velocity: float, rng: np.random.Generator) -> np.ndarray:
    n = int(2.4 * SR)
    t = np.arange(n, dtype=np.float64) / SR
    signal = np.zeros(n, dtype=np.float64)
    partials = (1.0, 0.42, 0.2, 0.1, 0.045, 0.02)
    for index, amp in enumerate(partials, start=1):
        decay = np.exp(-t * (1.6 + index * 1.15))
        signal += amp * np.sin(2.0 * np.pi * freq * index * t) * decay
    signal *= 1.0 - np.exp(-t / 0.006)
    pick_n = min(n, int(0.02 * SR))
    signal[:pick_n] += rng.standard_normal(pick_n) * np.exp(
        -np.arange(pick_n) / (SR * 0.004)
    ) * 0.12
    return signal * velocity * GAIN_GUITAR


def apply_taps(
    samples: np.ndarray,
    history: np.ndarray,
    taps: tuple[tuple[int, float], ...],
) -> tuple[np.ndarray, np.ndarray]:
    max_delay = history.shape[0]
    full = np.concatenate([history, samples])
    wet = np.zeros_like(samples)
    for delay, gain in taps:
        wet += gain * full[max_delay - delay : max_delay - delay + samples.shape[0]]
    return wet, full[-max_delay:]


def modulated_delay(
    samples: np.ndarray,
    history: np.ndarray,
    t0: float,
    base: float,
    depth: float,
    rate: float,
    phase: float,
) -> tuple[np.ndarray, np.ndarray]:
    max_delay = history.shape[0]
    full = np.concatenate([history, samples])
    n = samples.shape[0]
    times = t0 + np.arange(n, dtype=np.float64) / SR
    delay = base + depth * np.sin(2.0 * np.pi * rate * times + phase)
    delay = np.clip(delay, 1.0, max_delay - 2.0)
    pos = max_delay + np.arange(n) - delay
    i0 = np.floor(pos).astype(np.int32)
    frac = pos - i0
    i0 = np.clip(i0, 0, full.shape[0] - 2)
    wet = full[i0] * (1.0 - frac) + full[i0 + 1] * frac
    return wet, full[-max_delay:]


@dataclass
class RenderState:
    rng: np.random.Generator
    pad_freqs: np.ndarray = field(default_factory=lambda: np.zeros(8))
    pad_phases: np.ndarray = field(default_factory=lambda: np.zeros(8))
    pad_old_freqs: np.ndarray = field(default_factory=lambda: np.zeros(8))
    pad_old_phases: np.ndarray = field(default_factory=lambda: np.zeros(8))
    pad_fade: float = 0.0
    pad_chord: tuple[int, ...] | None = None
    guitar_carry: np.ndarray = field(default_factory=lambda: np.zeros(TAIL))
    rhodes_carry: np.ndarray = field(default_factory=lambda: np.zeros(TAIL))
    bass_carry: np.ndarray = field(default_factory=lambda: np.zeros(TAIL))
    kick_carry: np.ndarray = field(default_factory=lambda: np.zeros(TAIL))
    snare_carry: np.ndarray = field(default_factory=lambda: np.zeros(TAIL))
    hat_carry: np.ndarray = field(default_factory=lambda: np.zeros(TAIL))
    guitar_hist_l: np.ndarray = field(default_factory=lambda: np.zeros(int(0.05 * SR)))
    guitar_hist_r: np.ndarray = field(default_factory=lambda: np.zeros(int(0.05 * SR)))
    rhodes_hist: np.ndarray = field(default_factory=lambda: np.zeros(int(0.32 * SR)))
    wow_hist_l: np.ndarray = field(default_factory=lambda: np.zeros(int(0.02 * SR)))
    wow_hist_r: np.ndarray = field(default_factory=lambda: np.zeros(int(0.02 * SR)))
    zi_hat: np.ndarray | None = None
    zi_snare: np.ndarray | None = None
    zi_music_lp: list[np.ndarray | None] = field(default_factory=lambda: [None, None])
    zi_music_lp_r: list[np.ndarray | None] = field(default_factory=lambda: [None, None])
    zi_dc_l: np.ndarray | None = None
    zi_dc_r: np.ndarray | None = None
    zi_low_m: np.ndarray | None = None
    zi_low_s: np.ndarray | None = None
    zi_rain_hp: np.ndarray | None = None
    zi_rain_lp: np.ndarray | None = None
    zi_rain_hp_r: np.ndarray | None = None
    zi_rain_lp_r: np.ndarray | None = None
    zi_cafe_lp: np.ndarray | None = None
    zi_cafe_hp: np.ndarray | None = None
    zi_hiss: np.ndarray | None = None
    zi_hiss_r: np.ndarray | None = None
    zi_rhodes_rev: np.ndarray | None = None
    pad_l: np.ndarray = field(default_factory=lambda: np.zeros(0))
    pad_r: np.ndarray = field(default_factory=lambda: np.zeros(0))
    rain_level: float = 0.0
    cafe_level: float = 0.0
    guitar_level: float = 0.0
    drum_level: float = 0.75
    pad_level: float = 0.8


def set_pad_chord(state: RenderState, notes: tuple[int, ...]) -> None:
    if notes == state.pad_chord:
        return
    folded: list[int] = []
    for note in notes:
        while note > 67:
            note -= 12
        while note < 48:
            note += 12
        if note not in folded:
            folded.append(note)
    while len(folded) < 4:
        folded.append(folded[0])
    freqs: list[float] = []
    for note in folded[:4]:
        base = midi_to_hz(note)
        freqs.append(base * (2.0 ** (-6.0 / 1200.0)))
        freqs.append(base * (2.0 ** (6.0 / 1200.0)))
    state.pad_old_freqs = state.pad_freqs.copy()
    state.pad_old_phases = state.pad_phases.copy()
    state.pad_fade = 1.0 if state.pad_chord is not None else 0.0
    state.pad_freqs = np.asarray(freqs, dtype=np.float64)
    state.pad_phases = state.rng.random(8) * 2.0 * np.pi
    state.pad_chord = notes


def render_pad(state: RenderState, n: int, level: float) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(n, dtype=np.float64)
    old_env = state.pad_fade * np.exp(-t / (SR * 1.5))
    new_env = 1.0 - old_env

    def layer(freqs: np.ndarray, phases: np.ndarray, env: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not np.any(freqs):
            return np.zeros(n), np.zeros(n), phases
        inc = 2.0 * np.pi * freqs / SR
        ph = phases + inc * t[:, None]
        raw_l = np.sin(ph[:, 0::2]).sum(axis=1) / 4.0
        raw_r = np.sin(ph[:, 1::2]).sum(axis=1) / 4.0
        left = (raw_l * 0.86 + raw_r * 0.14) * env
        right = (raw_r * 0.86 + raw_l * 0.14) * env
        next_phases = np.mod(phases + inc * n, 2.0 * np.pi)
        return left, right, next_phases

    old_l, old_r, state.pad_old_phases = layer(state.pad_old_freqs, state.pad_old_phases, old_env)
    new_l, new_r, state.pad_phases = layer(state.pad_freqs, state.pad_phases, new_env)
    state.pad_fade = float(old_env[-1]) if n else state.pad_fade
    breath_t = t / SR
    breath = 0.84 + 0.16 * np.sin(2.0 * np.pi * breath_t / 28.0)
    amp = level * GAIN_PAD * breath
    return (old_l + new_l) * amp, (old_r + new_r) * amp


def swung(beat: float) -> float:
    whole = np.floor(beat)
    frac = beat - whole
    if frac >= 0.5:
        local = frac - 0.5
        return float(whole + SWING + local * ((1.0 - SWING) / 0.5))
    return float(whole + frac * (SWING / 0.5))


def beat_sample(bar_offset: int, beat: float) -> int:
    return int(round((bar_offset * BAR_BEATS + beat) * BEAT_S * SR))


KICK_PATTERNS = (
    (0.0, 2.0),
    (0.0, 2.5),
    (0.0, 1.75, 2.5),
    (0.0, 2.25),
)


def add_drums(
    state: RenderState,
    kick_buf: np.ndarray,
    snare_buf: np.ndarray,
    hat_buf: np.ndarray,
    bar_offset: int,
    bar: int,
    drum_gain: float,
) -> None:
    rng = state.rng
    pattern = KICK_PATTERNS[(bar // 8) % len(KICK_PATTERNS)]
    for beat in pattern:
        jitter = float(rng.uniform(-0.012, 0.018))
        vel = float(rng.uniform(0.72, 0.95)) * drum_gain
        if beat == 0.0:
            vel *= 1.05
        place(kick_buf, beat_sample(bar_offset, swung(beat) + jitter), synth_kick(vel, rng))
    for beat in (1.0, 3.0):
        jitter = float(rng.uniform(0.0, 0.02))
        vel = float(rng.uniform(0.62, 0.84)) * drum_gain
        place(snare_buf, beat_sample(bar_offset, beat + jitter), synth_snare(vel, rng))
    if rng.random() < 0.65:
        ghost_beat = float(rng.choice([0.75, 2.75, 3.5]))
        place(
            snare_buf,
            beat_sample(bar_offset, ghost_beat),
            synth_snare(0.22 * drum_gain, rng),
        )
    for eighth in range(8):
        if eighth % 2 == 1 and rng.random() < 0.08:
            continue
        beat = swung(eighth * 0.5)
        if eighth % 4 == 0:
            vel = 0.62
        elif eighth % 2 == 0:
            vel = 0.48
        else:
            vel = 0.30
        vel *= float(rng.uniform(0.85, 1.05)) * drum_gain
        open_hat = eighth == 6 and rng.random() < 0.25
        place(hat_buf, beat_sample(bar_offset, beat), synth_hat(vel, open_hat, rng))


def add_bass(
    state: RenderState,
    buffer: np.ndarray,
    bar_offset: int,
    root: int,
) -> None:
    rng = state.rng
    fifth = root + 7
    if fifth > 52:
        fifth -= 12
    octave = root + 12 if root < 42 else root - 12
    events = (
        (0.0, root, 0.92),
        (2.0, fifth, 0.62),
        (4.0, root, 0.84),
        (6.0, octave if rng.random() < 0.5 else fifth, 0.58),
    )
    for beat, note, velocity in events:
        jitter = float(rng.uniform(-0.01, 0.02))
        cents = float(rng.uniform(-3.0, 3.0))
        freq = midi_to_hz(note) * (2.0 ** (cents / 1200.0))
        place(
            buffer,
            beat_sample(bar_offset, beat + jitter),
            synth_bass(freq, velocity * float(rng.uniform(0.92, 1.0))),
        )


def add_rhodes(
    state: RenderState,
    buffer: np.ndarray,
    bar_offset: int,
    notes: tuple[int, ...],
    bar: int,
) -> None:
    rng = state.rng
    if bar % 2 == 1 and rng.random() < 0.22:
        return
    base_vel = 0.78 if bar % 2 == 0 else 0.58
    strum = 0.0
    for note in notes:
        strum += float(rng.uniform(0.010, 0.024))
        jitter = float(rng.uniform(-0.006, 0.012))
        cents = float(rng.uniform(-5.0, 5.0))
        freq = midi_to_hz(note) * (2.0 ** (cents / 1200.0))
        vel = base_vel * float(rng.uniform(0.88, 1.02))
        place(buffer, beat_sample(bar_offset, jitter) + int(strum * SR), synth_rhodes(freq, vel, rng))
    if rng.random() < 0.18:
        soft = notes[-2:]
        late = beat_sample(bar_offset, 2.5 + float(rng.uniform(0.0, 0.08)))
        for index, note in enumerate(soft):
            freq = midi_to_hz(note) * (2.0 ** (float(rng.uniform(-4.0, 4.0)) / 1200.0))
            place(buffer, late + index * int(0.018 * SR), synth_rhodes(freq, 0.34, rng))


def add_guitar(
    state: RenderState,
    buffer: np.ndarray,
    bar_offset: int,
    bar: int,
    amount: float,
) -> None:
    if bar % 2 != 0:
        return
    intro = min(1.0, bar / 64.0)
    probability = amount * intro * 0.62
    rng = state.rng
    if rng.random() > probability:
        return
    scene = SCENES[scene_index(bar)]
    scale = guitar_scale(scene)
    if not scale:
        return
    degree = int(rng.integers(0, len(scale)))
    beat = float(rng.choice([0.0, 0.5, 1.0, 2.0, 2.5]))
    note_count = int(rng.integers(3, 6))
    for _ in range(note_count):
        step = int(rng.choice([-2, -1, 1, 1, 2]))
        degree = min(len(scale) - 1, max(0, degree + step))
        dur_beats = float(rng.choice([0.5, 0.5, 1.0, 1.5]))
        vel = float(rng.uniform(0.40, 0.75)) * (0.45 + 0.55 * amount)
        cents = float(rng.uniform(-4.0, 4.0))
        freq = midi_to_hz(scale[degree]) * (2.0 ** (cents / 1200.0))
        place(buffer, beat_sample(bar_offset, swung(beat)), synth_guitar(freq, vel, rng))
        beat += dur_beats + float(rng.choice([0.0, 0.5, 0.5, 1.0]))
        if beat > 7.2:
            break


def finish_bus(
    buffer: np.ndarray, carry: np.ndarray, chunk: int
) -> tuple[np.ndarray, np.ndarray]:
    buffer[: carry.shape[0]] += carry
    new_carry = buffer[chunk:].copy()
    return buffer[:chunk].copy(), new_carry


def color_hats(state: RenderState, hats: np.ndarray) -> np.ndarray:
    high, state.zi_hat = one_pole_highpass(hats, 2800.0, state.zi_hat)
    return high


def color_snare(state: RenderState, snare: np.ndarray) -> np.ndarray:
    low, state.zi_snare = one_pole_lowpass(snare, 4200.0, state.zi_snare)
    return low


def chorus_guitar(
    state: RenderState, dry: np.ndarray, t0: float
) -> tuple[np.ndarray, np.ndarray]:
    left_wet, state.guitar_hist_l = modulated_delay(
        dry, state.guitar_hist_l, t0, base=18.0 * SR / 1000.0, depth=4.0 * SR / 1000.0, rate=0.27, phase=0.4
    )
    right_wet, state.guitar_hist_r = modulated_delay(
        dry, state.guitar_hist_r, t0, base=23.0 * SR / 1000.0, depth=5.0 * SR / 1000.0, rate=0.31, phase=2.1
    )
    left = dry * 0.62 + left_wet * 0.48
    right = dry * 0.62 + right_wet * 0.48
    return left, right


RHODES_TAPS = (
    (int(0.037 * SR), 0.30),
    (int(0.061 * SR), 0.24),
    (int(0.089 * SR), 0.18),
    (int(0.127 * SR), 0.13),
    (int(0.173 * SR), 0.09),
)


def render_chunk(state: RenderState, start_bar: int, n_bars: int, t0: float) -> np.ndarray:
    chunk = n_bars * BAR_SAMPLES
    length = chunk + TAIL
    kick_buf = np.zeros(length, dtype=np.float64)
    snare_buf = np.zeros(length, dtype=np.float64)
    hat_buf = np.zeros(length, dtype=np.float64)
    bass_buf = np.zeros(length, dtype=np.float64)
    rhodes_buf = np.zeros(length, dtype=np.float64)
    guitar_buf = np.zeros(length, dtype=np.float64)

    for offset in range(n_bars):
        bar = start_bar + offset
        _, notes, root = chord_at(bar)
        drums = mix_param(bar, "drums")
        guitar_amt = mix_param(bar, "guitar")
        add_drums(state, kick_buf, snare_buf, hat_buf, offset, bar, drums)
        add_bass(state, bass_buf, offset, root)
        add_rhodes(state, rhodes_buf, offset, notes, bar)
        add_guitar(state, guitar_buf, offset, bar, guitar_amt)

    kick, state.kick_carry = finish_bus(kick_buf, state.kick_carry, chunk)
    snare, state.snare_carry = finish_bus(snare_buf, state.snare_carry, chunk)
    hats, state.hat_carry = finish_bus(hat_buf, state.hat_carry, chunk)
    bass, state.bass_carry = finish_bus(bass_buf, state.bass_carry, chunk)
    rhodes, state.rhodes_carry = finish_bus(rhodes_buf, state.rhodes_carry, chunk)
    guitar, state.guitar_carry = finish_bus(guitar_buf, state.guitar_carry, chunk)

    hats = color_hats(state, hats)
    snare = color_snare(state, snare)
    hat_l, hat_r = pan(hats, 0.42)
    snare_l, snare_r = pan(snare, -0.1)
    drums_l = kick + snare_l + hat_l
    drums_r = kick + snare_r + hat_r

    rhodes_wet, state.rhodes_hist = apply_taps(rhodes, state.rhodes_hist, RHODES_TAPS)
    rhodes_wet, state.zi_rhodes_rev = one_pole_lowpass(rhodes_wet, 2400.0, state.zi_rhodes_rev)
    dry_l, dry_r = pan(rhodes, -0.08)
    wet_l, wet_r = pan(rhodes_wet, 0.48)
    rhodes_l = dry_l + wet_l
    rhodes_r = dry_r + wet_r
    guitar_l, guitar_r = chorus_guitar(state, guitar, t0)

    left = drums_l + bass + rhodes_l + guitar_l + state.pad_l
    right = drums_r + bass + rhodes_r + guitar_r + state.pad_r
    left, state.wow_hist_l = modulated_delay(
        left, state.wow_hist_l, t0, base=48.0, depth=28.0, rate=0.16, phase=0.2
    )
    right, state.wow_hist_r = modulated_delay(
        right, state.wow_hist_r, t0, base=48.0, depth=28.0, rate=0.16, phase=0.2
    )
    return _master(state, start_bar, n_bars, t0, left, right)


def _master(
    state: RenderState,
    start_bar: int,
    n_bars: int,
    t0: float,
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    mid = 0.5 * (left + right)
    side = 0.5 * (left - right)
    low, state.zi_low_m = one_pole_lowpass(mid, 140.0, state.zi_low_m)
    side_low, state.zi_low_s = one_pole_lowpass(side, 140.0, state.zi_low_s)
    high = mid - low
    side = side - side_low
    left = low + high + side
    right = low + high - side

    for index in range(2):
        left, state.zi_music_lp[index] = one_pole_lowpass(left, 5600.0, state.zi_music_lp[index])
        right, state.zi_music_lp_r[index] = one_pole_lowpass(
            right, 5600.0, state.zi_music_lp_r[index]
        )

    left = np.tanh(left * 1.12) / np.tanh(1.12)
    right = np.tanh(right * 1.12) / np.tanh(1.12)

    n = left.shape[0]
    ambience_l, ambience_r = render_ambience(state, n, t0, start_bar, n_bars)
    left = left + ambience_l
    right = right + ambience_r
    left, state.zi_dc_l = one_pole_highpass(left, 22.0, state.zi_dc_l)
    right, state.zi_dc_r = one_pole_highpass(right, 22.0, state.zi_dc_r)

    stereo = np.column_stack((left, right)) * MASTER
    np.tanh(stereo, out=stereo)
    return stereo.astype(np.float32)


def render_ambience(
    state: RenderState,
    n: int,
    t0: float,
    start_bar: int,
    n_bars: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = state.rng
    end_bar = start_bar + n_bars - 1
    rain_target = mix_param(end_bar, "rain")
    cafe_target = mix_param(end_bar, "cafe")
    rain_env = np.linspace(state.rain_level, rain_target, n)
    cafe_env = np.linspace(state.cafe_level, cafe_target, n)
    state.rain_level = rain_target
    state.cafe_level = cafe_target

    rain_noise = rng.standard_normal(n)
    rain_noise_r = rng.standard_normal(n)
    rain_hp, state.zi_rain_hp = one_pole_highpass(rain_noise, 700.0, state.zi_rain_hp)
    rain_bp, state.zi_rain_lp = one_pole_lowpass(rain_hp, 7000.0, state.zi_rain_lp)
    rain_hp_r, state.zi_rain_hp_r = one_pole_highpass(rain_noise_r, 700.0, state.zi_rain_hp_r)
    rain_bp_r, state.zi_rain_lp_r = one_pole_lowpass(rain_hp_r, 7000.0, state.zi_rain_lp_r)
    drops = np.zeros(n, dtype=np.float64)
    drop_count = int(rng.poisson(max(rain_target, 0.05) * 7.0 * n / SR))
    if drop_count:
        positions = rng.integers(0, n, drop_count)
        amps = rng.uniform(0.15, 1.0, drop_count) ** 2
        for pos, amp in zip(positions, amps):
            length = int(rng.uniform(0.008, 0.03) * SR)
            env = np.exp(-np.arange(length) / (SR * 0.006))
            burst = rng.standard_normal(length) * env * amp * 0.7
            place(drops, int(pos), burst)
    rain_l = (rain_bp * 0.55 + drops) * rain_env * GAIN_RAIN
    rain_r = (rain_bp_r * 0.55 + drops * 0.65) * rain_env * GAIN_RAIN

    cafe_noise = rng.standard_normal(n)
    cafe_lp, state.zi_cafe_lp = one_pole_lowpass(cafe_noise, 900.0, state.zi_cafe_lp)
    cafe_bp, state.zi_cafe_hp = one_pole_highpass(cafe_lp, 180.0, state.zi_cafe_hp)
    times = t0 + np.arange(n) / SR
    murmur = 0.65 + 0.35 * np.sin(2.0 * np.pi * 0.07 * times) * np.sin(
        2.0 * np.pi * 0.13 * times + 0.8
    )
    cafe = cafe_bp * murmur * cafe_env * GAIN_CAFE

    hiss = rng.standard_normal(n)
    hiss, state.zi_hiss = one_pole_highpass(hiss, 1800.0, state.zi_hiss)
    hiss *= GAIN_HISS
    hiss_r = rng.standard_normal(n)
    hiss_r, state.zi_hiss_r = one_pole_highpass(hiss_r, 1800.0, state.zi_hiss_r)
    hiss_r *= GAIN_HISS

    crackle = np.zeros(n, dtype=np.float64)
    pop_count = int(rng.poisson(4.5 * n / SR))
    if pop_count:
        positions = rng.integers(0, n, pop_count)
        amps = rng.uniform(0.2, 1.0, pop_count) ** 2
        for pos, amp in zip(positions, amps):
            length = int(rng.uniform(0.002, 0.01) * SR)
            env = np.exp(-np.arange(length) / (SR * 0.0018))
            burst = rng.standard_normal(length) * env * amp
            place(crackle, int(pos), burst)
    crackle *= GAIN_CRACKLE

    left = rain_l + cafe * 0.8 + hiss + crackle
    right = rain_r + cafe * 1.05 + hiss_r + crackle * 0.85
    return left, right


def apply_fades(stereo: np.ndarray, start_sample: int, total_samples: int) -> None:
    n = stereo.shape[0]
    idx = np.arange(n) + start_sample
    gain = np.ones(n, dtype=np.float32)
    fade_in = 8 * SR
    intro = idx < fade_in
    gain[intro] = (idx[intro] / float(fade_in)).astype(np.float32)
    fade_out = 120 * SR
    if total_samples >= 10 * 60 * SR:
        outro_start = total_samples - fade_out
        outro = idx >= outro_start
        if np.any(outro):
            phase = (idx[outro] - outro_start) / float(fade_out)
            gain[outro] *= (0.5 * (1.0 + np.cos(np.pi * np.clip(phase, 0.0, 1.0)))).astype(np.float32)
    stereo *= gain[:, None]


def render_pad_chunk(state: RenderState, start_bar: int, n_bars: int) -> None:
    parts_l: list[np.ndarray] = []
    parts_r: list[np.ndarray] = []
    for offset in range(n_bars):
        bar = start_bar + offset
        _, notes, _ = chord_at(bar)
        set_pad_chord(state, notes)
        level = mix_param(bar, "pad")
        left, right = render_pad(state, BAR_SAMPLES, level)
        parts_l.append(left)
        parts_r.append(right)
    state.pad_l = np.concatenate(parts_l)
    state.pad_r = np.concatenate(parts_r)


def render_to_ffmpeg(total_bars: int, output: Path, seed: int, start_bar: int = 0) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "f32le",
        "-ar",
        str(SR),
        "-ac",
        "2",
        "-i",
        "pipe:0",
        "-c:a",
        "libmp3lame",
        "-b:a",
        "192k",
        "-ar",
        "44100",
        "-metadata",
        "title=Mix lo-fi pour étudier",
        "-metadata",
        "genre=Lo-fi Hip Hop",
        "-metadata",
        "comment=Mix original continu, piano électrique, guitare, pluie et vinyle",
        str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    state = RenderState(rng=np.random.default_rng(seed))
    total_samples = total_bars * BAR_SAMPLES
    written = 0
    started = time.perf_counter()
    bar = start_bar
    end_bar = start_bar + total_bars
    try:
        while bar < end_bar:
            n_bars = min(4, end_bar - bar)
            t0 = bar * BAR_S
            render_pad_chunk(state, bar, n_bars)
            chunk = render_chunk(state, bar, n_bars, t0)
            apply_fades(chunk, written, total_samples)
            process.stdin.write(chunk.astype("<f4", copy=False).tobytes())
            written += chunk.shape[0]
            bar += n_bars
            done_minutes = written / SR / 60.0
            if bar % 40 == 0 or bar >= end_bar:
                elapsed = time.perf_counter() - started
                speed = (written / SR) / elapsed if elapsed else 0.0
                print(
                    f"{done_minutes:7.1f} min audio  |  {elapsed / 60.0:5.1f} min écoulées  |  {speed:4.1f}×",
                    flush=True,
                )
    finally:
        process.stdin.close()
        code = process.wait()
    if code != 0:
        raise RuntimeError(f"ffmpeg s'est terminé avec le code {code}")


def default_output() -> Path:
    filename_time = datetime(*TRIGGER_UTC, tzinfo=timezone.utc).astimezone(ZoneInfo("Europe/Paris"))
    filename = f"lofi_{filename_time:%Y-%m-%d}_{filename_time:%H}h{filename_time:%M}.mp3"
    windows_mirror = Path("/mnt/c/Users/conta/OneDrive/Documents/Youtube/Son")
    if windows_mirror.is_dir():
        return windows_mirror / filename
    return Path("/workspace/Son") / filename


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Génère un mix lo-fi continu.")
    parser.add_argument("--hours", type=float, default=5.0, help="Durée en heures.")
    parser.add_argument("--seconds", type=float, default=None, help="Durée courte de test.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=202610180647)
    parser.add_argument(
        "--start-bar",
        type=int,
        default=0,
        help="Démarre plus loin dans la forme, pour écouter une scène.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seconds is not None:
        total_bars = max(1, int(round(args.seconds / BAR_S)))
    else:
        total_bars = max(1, int(round(args.hours * 3600.0 / BAR_S)))
    output = args.output or default_output()
    print(
        f"Mix lo-fi : {total_bars * BAR_S / 3600.0:.3f} h, {BPM:.0f} BPM, {output}",
        flush=True,
    )
    render_to_ffmpeg(total_bars, output, args.seed, args.start_bar)
    print(f"Fichier écrit : {output}", flush=True)


if __name__ == "__main__":
    main()
