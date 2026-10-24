#!/usr/bin/env python3
"""Piste lo-fi originale de cette exécution.

Variables tirées au hasard (secrets) avant la synthèse.
La boîte à musique lointaine de 07h48 est écartée. Tirage parmi les cinq autres.
- Instrument : piano électrique Wurlitzer, seul
- Texture : crépitement vinyle très discret, sans autre couche
- Couleur : nostalgique, peaceful study vibes
- Tempo : 76 BPM
- Harmonie : accords de septième mineure, mineur naturel

Prompt :
Genre: Lo-fi hip hop instrumental, chillhop. BPM: 76.
Mood: nostalgique, peaceful study vibes.
Instrumentation: Wurlitzer electric piano, mellow upright bass,
soft brushed drums (SP-404 style, slightly off-beat, snare on beat 3).
Texture: subtle vinyl crackle only.
Harmony: minor seventh chords, jazzy voicings, natural minor.
Characteristics: Slow evolving chord progression, NO sudden changes,
warm analog saturation, low-pass filtered highs, human timing imperfections,
melancholic but comforting.
Interdit: voix, fredonnements, chœurs, nappes, drones, sons alien,
textures science-fiction, foley, pluie, feu, field recordings.

Le master est un tampon circulaire de 4 minutes exactement : la fin
retombe sur le début. Le fichier final enchaîne 75 lectures. Un fondu
linéaire de 3 secondes est calculé entre la tête de boucle et elle-même :
les deux côtés portent le même signal, donc le fondu est transparent et la
jointure reste l'onde circulaire, sans couture audible.
Durée : 75 × 4 min + 3 s = 5 h 00 min 03 s.
"""

from __future__ import annotations

import argparse
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, assert_never
from zoneinfo import ZoneInfo

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, lfilter, sosfilt

TRIGGER_UTC = (2026, 10, 24, 6, 47, 38)
INSTRUMENT = "Piano électrique Wurlitzer"
MOOD = "Nostalgique"
MODE: ModeName = "Mineur naturel"
BPM = 76
SEED = 1_004_654_070
SLUG = "Wurlitzer"
PROMPT = (
    "Genre: Lo-fi hip hop instrumental, chillhop. BPM: 76. "
    "Mood: nostalgique, peaceful study vibes. "
    "Instrumentation: Wurlitzer electric piano, "
    "mellow upright bass, "
    "soft brushed drums (SP-404 style, slightly off-beat, snare on beat 3), "
    "subtle vinyl crackle only. "
    "Harmony: minor seventh chords, jazzy voicings, natural minor. "
    "Characteristics: Slow evolving chord progression, NO sudden changes, "
    "warm analog saturation, low-pass filtered highs, human timing imperfections, "
    "melancholic but comforting. "
    "Instrumental only: no vocals, no humming, no choirs, no synth pads, "
    "no ambient drones, no alien sounds, no sci-fi textures, no background Foley, "
    "no rain, no fire crackle, no field recordings."
)

InstrumentName = Literal[
    "Piano feutré (felt piano)",
    "Guitare jazz clean avec chorus",
    "Kalimba douce",
    "Synthétiseur analogique chaleureux",
    "Piano électrique Wurlitzer",
    "Boîte à musique lointaine",
]
MoodName = Literal[
    "Nostalgique",
    "Cozy",
    "Nocturne",
    "Smooth",
]
ModeName = Literal["Dorien", "Mineur naturel"]
PatternName = Literal["strum", "stab", "arp", "sustain", "sparse"]
SpaceName = Literal["chorus", "tremolo", "room", "wide", "distant"]
HatMode = Literal["eighths", "offbeats", "sparse"]

SR = 32_000
P = 240 * SR
BEAT_S = 60.0 / BPM
BAR_BEATS = 4
N_BARS = BPM
FADE = int(round(3.0 * SR))
LOOPS = 75
PEAK_TARGET = 10 ** (-2.5 / 20.0)
TARGET_RMS = 10 ** (-15.5 / 20.0)

FloatArray = NDArray[np.float64]

QUALITIES: dict[str, tuple[int, ...]] = {
    "m7": (0, 3, 7, 10),
}

NOTE_NAMES = (
    "Do",
    "Do#",
    "Ré",
    "Mib",
    "Mi",
    "Fa",
    "Fa#",
    "Sol",
    "Sol#",
    "La",
    "Sib",
    "Si",
)


@dataclass(frozen=True)
class Voice:
    pattern: PatternName
    space: SpaceName
    chord_low: int
    chord_high: int
    melody_low: int
    melody_high: int
    chord_seconds: float
    chord_decay: float
    melody_seconds: float
    melody_decay: float
    lead_rms: float
    voices: int


@dataclass(frozen=True)
class MoodSpec:
    swing: float
    melody_chance: float
    drum_gain: float
    hat_gain: float
    hat_mode: HatMode
    cutoff: float


VOICES: dict[InstrumentName, Voice] = {
    "Piano feutré (felt piano)": Voice(
        "stab", "wide", 52, 72, 64, 81, 3.1, 0.75, 2.3, 1.05, 0.080, 4
    ),
    "Guitare jazz clean avec chorus": Voice(
        "strum", "chorus", 50, 67, 62, 76, 3.4, 0.78, 2.1, 1.15, 0.078, 4
    ),
    "Kalimba douce": Voice(
        "arp", "room", 72, 88, 74, 91, 1.5, 1.7, 1.2, 2.1, 0.068, 4
    ),
    "Synthétiseur analogique chaleureux": Voice(
        "stab", "wide", 48, 67, 60, 76, 2.2, 1.35, 1.6, 1.55, 0.078, 3
    ),
    "Piano électrique Wurlitzer": Voice(
        "stab", "tremolo", 51, 69, 60, 79, 1.75, 1.55, 1.45, 1.85, 0.082, 3
    ),
    "Boîte à musique lointaine": Voice(
        "sparse", "distant", 79, 96, 81, 96, 1.8, 1.25, 2.0, 0.9, 0.058, 1
    ),
}

# Septièmes mineures uniquement. Dorien : i, ii, v. Mineur naturel : i, iv, v.
PROGRESSIONS: dict[ModeName, tuple[tuple[tuple[int, str], ...], ...]] = {
    "Dorien": (
        ((0, "m7"), (2, "m7"), (7, "m7"), (2, "m7")),
        ((0, "m7"), (2, "m7"), (0, "m7"), (7, "m7")),
    ),
    "Mineur naturel": (
        ((0, "m7"), (5, "m7"), (7, "m7"), (5, "m7")),
        ((0, "m7"), (5, "m7"), (0, "m7"), (7, "m7")),
    ),
}
SCALES: dict[ModeName, tuple[int, ...]] = {
    "Dorien": (0, 2, 3, 5, 7, 9, 10),
    "Mineur naturel": (0, 2, 3, 5, 7, 8, 10),
}

MOODS: dict[MoodName, MoodSpec] = {
    "Nostalgique": MoodSpec(0.62, 0.22, 0.72, 0.48, "offbeats", 4200.0),
    "Cozy": MoodSpec(0.60, 0.20, 0.74, 0.46, "offbeats", 4400.0),
    "Nocturne": MoodSpec(0.66, 0.16, 0.60, 0.36, "sparse", 3800.0),
    "Smooth": MoodSpec(0.61, 0.18, 0.68, 0.42, "offbeats", 4300.0),
}


def midi_to_hz(note: float) -> float:
    return 440.0 * (2.0 ** ((note - 69.0) / 12.0))


def periodic_rate(target_hz: float) -> float:
    duration = P / SR
    cycles = max(1, int(round(target_hz * duration)))
    return cycles / duration


def sample_at(bar: int, beat: float) -> int:
    return int(round((bar * BAR_BEATS + beat) * BEAT_S * SR))


def swung(beat: float, swing: float) -> float:
    whole = np.floor(beat)
    frac = beat - whole
    if frac >= 0.5:
        local = frac - 0.5
        return float(whole + swing + local * ((1.0 - swing) / 0.5))
    return float(whole + frac * (swing / 0.5))


def place_circular(buffer: FloatArray, start: int, signal: FloatArray) -> None:
    if signal.size == 0:
        return
    length = buffer.shape[0]
    cursor = start % length
    view = signal
    while view.size:
        room = length - cursor
        take = min(room, view.size)
        buffer[cursor : cursor + take] += view[:take]
        view = view[take:]
        cursor = 0


def pan(mono: FloatArray, position: float) -> tuple[FloatArray, FloatArray]:
    angle = (position + 1.0) * 0.25 * np.pi
    return np.cos(angle) * mono, np.sin(angle) * mono


def circular_lowpass(samples: FloatArray, cutoff: float, poles: int = 1) -> FloatArray:
    filtered = samples.astype(np.float64, copy=False)
    for _ in range(poles):
        coeff = float(np.exp(-2.0 * np.pi * cutoff / SR))
        warm = min(filtered.shape[0] - 1, int(2.0 * SR))
        warmed = np.concatenate((filtered[-warm:], filtered))
        filtered = np.asarray(
            lfilter(
                np.array([1.0 - coeff], dtype=np.float64),
                np.array([1.0, -coeff], dtype=np.float64),
                warmed,
            )[warm:],
            dtype=np.float64,
        )
    return filtered


def circular_highpass(samples: FloatArray, cutoff: float) -> FloatArray:
    return samples - circular_lowpass(samples, cutoff)


def circular_noise(length: int, rng: np.random.Generator, fade: int) -> FloatArray:
    raw = rng.standard_normal(length + fade)
    mixed = raw[:length].copy()
    theta = np.linspace(0.0, np.pi / 2.0, fade, endpoint=False)
    mixed[:fade] = raw[:fade] * np.sin(theta) + raw[length : length + fade] * np.cos(theta)
    return mixed


def circular_mod_delay(
    signal: FloatArray,
    base: float,
    depth: float,
    rate: float,
    phase: float,
) -> FloatArray:
    count = signal.shape[0]
    times = np.arange(count, dtype=np.float64) / SR
    delay = np.clip(base + depth * np.sin(2.0 * np.pi * rate * times + phase), 1.0, SR * 0.05)
    position = np.mod(np.arange(count, dtype=np.float64) - delay, count)
    index = np.floor(position).astype(np.int32)
    nxt = (index + 1) % count
    fraction = position - index
    return signal[index] * (1.0 - fraction) + signal[nxt] * fraction


def dark_room(signal: FloatArray) -> FloatArray:
    wet = np.zeros_like(signal)
    for delay_s, gain in ((0.029, 0.18), (0.047, 0.12), (0.073, 0.08), (0.113, 0.05)):
        wet += gain * np.roll(signal, int(delay_s * SR))
    return circular_lowpass(wet, 1600.0)


def rms(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(signal), dtype=np.float64)))


def scale_to_rms(signal: np.ndarray, target: float) -> np.ndarray:
    current = rms(signal)
    if current < 1e-9:
        return signal
    return signal * (target / current)


def fade_tail(signal: FloatArray, seconds: float = 0.02) -> FloatArray:
    fade_n = min(signal.size // 2, int(seconds * SR))
    if fade_n > 1:
        signal = signal.copy()
        signal[-fade_n:] *= np.linspace(1.0, 0.0, fade_n, endpoint=True)
    return signal


def synth_felt(
    freq: float,
    velocity: float,
    rng: np.random.Generator,
    seconds: float,
    decay: float,
) -> FloatArray:
    count = int(seconds * SR)
    times = np.arange(count, dtype=np.float64) / SR
    signal = np.zeros(count, dtype=np.float64)
    stiffness = 0.00012
    amplitudes = (1.0, 0.38, 0.18, 0.08, 0.035, 0.014)
    for index, amplitude in enumerate(amplitudes, start=1):
        partial = freq * index * float(np.sqrt(1.0 + stiffness * index * index))
        if partial >= SR * 0.45:
            break
        envelope = np.exp(-times * (decay + (index - 1) * 1.25))
        signal += amplitude * np.sin(2.0 * np.pi * partial * times) * envelope
    signal *= 1.0 - np.exp(-times / 0.007)
    hammer = min(count, int(0.01 * SR))
    signal[:hammer] += rng.standard_normal(hammer) * np.exp(-np.arange(hammer) / (SR * 0.0035)) * 0.01
    return fade_tail(np.tanh(signal) * velocity)


def synth_guitar(
    freq: float,
    velocity: float,
    rng: np.random.Generator,
    seconds: float,
    decay: float,
) -> FloatArray:
    count = int(seconds * SR)
    times = np.arange(count, dtype=np.float64) / SR
    partials = (1.0, 0.20, 0.07, 0.022, 0.008)
    signal = np.zeros(count, dtype=np.float64)
    for index, amplitude in enumerate(partials, start=1):
        partial_hz = freq * index
        if partial_hz >= SR * 0.45:
            break
        detune = 1.0 + float(rng.uniform(-0.00025, 0.00025))
        envelope = np.exp(-times * (decay + index * 0.7))
        signal += amplitude * np.sin(2.0 * np.pi * partial_hz * detune * times) * envelope
    signal *= 1.0 - np.exp(-times / 0.012)
    pick = min(count, int(0.008 * SR))
    signal[:pick] += rng.standard_normal(pick) * np.exp(-np.arange(pick) / (SR * 0.0032)) * 0.022
    return fade_tail(np.tanh(signal * 1.05) * velocity)


def synth_kalimba(
    freq: float,
    velocity: float,
    rng: np.random.Generator,
    seconds: float,
    decay: float,
) -> FloatArray:
    count = int(seconds * SR)
    times = np.arange(count, dtype=np.float64) / SR
    signal = np.zeros(count, dtype=np.float64)
    partials = ((1.0, 1.0, 0.0), (2.71, 0.32, 2.2), (5.16, 0.11, 4.0), (8.04, 0.035, 6.0))
    for ratio, amplitude, extra in partials:
        partial = freq * ratio
        if partial >= SR * 0.45:
            break
        envelope = np.exp(-times * (decay + extra))
        signal += amplitude * np.sin(2.0 * np.pi * partial * times) * envelope
    signal *= 1.0 - np.exp(-times / 0.002)
    tick = min(count, int(0.004 * SR))
    signal[:tick] += rng.standard_normal(tick) * np.exp(-np.arange(tick) / (SR * 0.001)) * 0.04
    return fade_tail(np.tanh(signal * 1.1) * velocity)


def synth_analog(
    freq: float,
    velocity: float,
    rng: np.random.Generator,
    seconds: float,
    decay: float,
) -> FloatArray:
    count = int(seconds * SR)
    times = np.arange(count, dtype=np.float64) / SR
    detune = 2.0 ** (float(rng.uniform(3.0, 7.0)) / 1200.0)
    signal = np.zeros(count, dtype=np.float64)
    for index in range(1, 7):
        envelope = np.exp(-times * (decay + index * 0.28))
        amplitude = envelope / index
        signal += amplitude * np.sin(2.0 * np.pi * freq * index * times)
        signal += 0.62 * amplitude * np.sin(2.0 * np.pi * freq * index * detune * times)
    signal *= 1.0 - np.exp(-times / 0.045)
    return fade_tail(np.tanh(signal * 0.72) * velocity * 0.42)


def synth_wurli(
    freq: float,
    velocity: float,
    rng: np.random.Generator,
    seconds: float,
    decay: float,
) -> FloatArray:
    count = int(seconds * SR)
    times = np.arange(count, dtype=np.float64) / SR
    modulator = np.exp(-times * 7.2) * 1.55 * np.sin(2.0 * np.pi * freq * times)
    carrier = np.sin(2.0 * np.pi * freq * times + modulator)
    second = 0.16 * np.sin(2.0 * np.pi * freq * 2.0 * times) * np.exp(-times * 4.8)
    signal = (carrier + second) * (1.0 - np.exp(-times / 0.004)) * np.exp(-times * decay)
    click = min(count, int(0.005 * SR))
    signal[:click] += rng.standard_normal(click) * np.exp(-np.arange(click) / (SR * 0.0014)) * 0.028
    return fade_tail(np.tanh(signal * 1.12) * velocity)


def synth_box(
    freq: float,
    velocity: float,
    rng: np.random.Generator,
    seconds: float,
    decay: float,
) -> FloatArray:
    count = int(seconds * SR)
    times = np.arange(count, dtype=np.float64) / SR
    signal = np.zeros(count, dtype=np.float64)
    partials = ((1.0, 1.0, 0.4), (2.76, 0.28, 1.8), (5.4, 0.09, 3.2))
    for ratio, amplitude, extra in partials:
        partial = freq * ratio
        if partial >= SR * 0.45:
            break
        signal += amplitude * np.sin(2.0 * np.pi * partial * times) * np.exp(-times * (decay + extra))
    signal *= 1.0 - np.exp(-times / 0.003)
    del rng
    return fade_tail(np.tanh(signal) * velocity * 0.85)


def play_note(
    instrument: InstrumentName,
    freq: float,
    velocity: float,
    rng: np.random.Generator,
    seconds: float,
    decay: float,
) -> FloatArray:
    match instrument:
        case "Piano feutré (felt piano)":
            return synth_felt(freq, velocity, rng, seconds, decay)
        case "Guitare jazz clean avec chorus":
            return synth_guitar(freq, velocity, rng, seconds, decay)
        case "Kalimba douce":
            return synth_kalimba(freq, velocity, rng, seconds, decay)
        case "Synthétiseur analogique chaleureux":
            return synth_analog(freq, velocity, rng, seconds, decay)
        case "Piano électrique Wurlitzer":
            return synth_wurli(freq, velocity, rng, seconds, decay)
        case "Boîte à musique lointaine":
            return synth_box(freq, velocity, rng, seconds, decay)
        case unreachable:
            assert_never(unreachable)


def synth_bass(freq: float, velocity: float, rng: np.random.Generator) -> FloatArray:
    count = int(1.5 * SR)
    times = np.arange(count, dtype=np.float64) / SR
    signal = (
        np.sin(2.0 * np.pi * freq * times)
        + 0.07 * np.sin(2.0 * np.pi * freq * 2.0 * times)
        + 0.012 * np.sin(2.0 * np.pi * freq * 3.0 * times)
    )
    envelope = (1.0 - np.exp(-times / 0.038)) * np.exp(-times / 1.05)
    finger = rng.standard_normal(count) * np.exp(-times * 36.0) * 0.04
    return fade_tail(np.tanh((signal * envelope + finger) * 1.08) * velocity)


def synth_kick(velocity: float, rng: np.random.Generator) -> FloatArray:
    count = int(0.48 * SR)
    times = np.arange(count, dtype=np.float64) / SR
    frequency = 36.0 + 54.0 * np.exp(-times * 14.0)
    phase = 2.0 * np.pi * np.cumsum(frequency) / SR
    body = np.sin(phase) * np.exp(-times * 4.6) * (1.0 - np.exp(-times / 0.009))
    click = rng.standard_normal(count) * np.exp(-times * 140.0) * 0.018
    return fade_tail(np.tanh((body + click) * 1.35) * velocity)


def synth_brush(velocity: float, rng: np.random.Generator) -> FloatArray:
    count = int(0.62 * SR)
    times = np.arange(count, dtype=np.float64) / SR
    noise = rng.standard_normal(count) * np.exp(-times * 6.4)
    body = np.sin(2.0 * np.pi * 132.0 * times) * np.exp(-times * 14.0) * 0.14
    return fade_tail(np.tanh((noise * 0.7 + body) * 1.3) * velocity)


def synth_hat(velocity: float, rng: np.random.Generator, open_hat: bool) -> FloatArray:
    count = int((0.09 if open_hat else 0.030) * SR)
    times = np.arange(count, dtype=np.float64) / SR
    decay = 16.0 if open_hat else 58.0
    return fade_tail(rng.standard_normal(count) * np.exp(-times * decay) * velocity)


def fold_bass(pitch_class: int, anchor: int | None) -> int:
    candidates = [octave * 12 + pitch_class for octave in range(2, 5)]
    window = [note for note in candidates if 34 <= note <= 46]
    pool = window or candidates
    if anchor is None:
        return min(pool, key=lambda note: abs(note - 40))
    return min(pool, key=lambda note: (abs(note - anchor), abs(note - 40)))


def voice_chord(
    pitch_classes: tuple[int, ...],
    previous: tuple[int, ...] | None,
    low: int,
    high: int,
) -> tuple[int, ...]:
    voiced: list[int] = []
    for index, pitch_class in enumerate(pitch_classes):
        candidates = [note for note in range(low, high + 1) if note % 12 == pitch_class]
        if not candidates:
            candidates = [low + (pitch_class - low) % 12]
        if previous is not None and index < len(previous):
            reference = previous[index]
            chosen = min(candidates, key=lambda note: (abs(note - reference), note))
        else:
            center = (low + high) / 2.0
            chosen = min(candidates, key=lambda note: (abs(note - center), note))
        voiced.append(chosen)
    return tuple(voiced)


def bar_chord(bar: int, progression: tuple[tuple[int, str], ...]) -> tuple[int, str]:
    count = len(progression)
    phase = bar / N_BARS
    triangle = 1.0 - abs(2.0 * phase - 1.0)
    index = int(np.round(triangle * (count - 1)))
    index = min(max(index, 0), count - 1)
    return progression[index]


def plan_voicings(
    tonic_pc: int,
    progression: tuple[tuple[int, str], ...],
    low: int,
    high: int,
) -> tuple[tuple[tuple[int, ...], int], ...]:
    cache: dict[tuple[int, str], tuple[int, ...]] = {}
    bass_cache: dict[tuple[int, str], int] = {}
    previous: tuple[int, ...] | None = None
    previous_bass: int | None = None
    raw: list[tuple[tuple[int, ...], int]] = []
    for bar in range(N_BARS):
        chord = bar_chord(bar, progression)
        degree, quality = chord
        root_pc = (tonic_pc + degree) % 12
        if chord not in cache:
            classes = tuple((root_pc + interval) % 12 for interval in QUALITIES[quality])
            cache[chord] = voice_chord(classes, previous, low, high)
            bass_cache[chord] = fold_bass(root_pc, previous_bass)
        raw.append((cache[chord], bass_cache[chord]))
        previous = cache[chord]
        previous_bass = bass_cache[chord]
    return tuple(raw)


def melody_scale(tonic_pc: int, intervals: tuple[int, ...], low: int, high: int) -> list[int]:
    notes: list[int] = []
    for octave in range(4, 9):
        for interval in intervals:
            note = octave * 12 + (tonic_pc + interval) % 12
            if low <= note <= high:
                notes.append(note)
    return sorted(set(notes))


def detuned_freq(note: int, rng: np.random.Generator, cents: float) -> float:
    shift = float(rng.uniform(-cents, cents))
    return midi_to_hz(note) * (2.0 ** (shift / 1200.0))


def add_melody_note(
    buffer: FloatArray,
    instrument: InstrumentName,
    scale: list[int],
    degree: int,
    bar: int,
    beat: float,
    velocity: float,
    rng: np.random.Generator,
    voice: Voice,
    swing: float,
) -> int:
    degree = int(np.clip(degree, 0, len(scale) - 1))
    if rng.random() < 0.18 and degree > 0:
        grace = detuned_freq(scale[degree - 1], rng, 6.0)
        place_circular(
            buffer,
            sample_at(bar, swung(beat, swing)) - int(0.04 * SR),
            play_note(instrument, grace, velocity * 0.28, rng, 0.16, 4.0),
        )
    place_circular(
        buffer,
        sample_at(bar, swung(beat, swing) + float(rng.uniform(-0.006, 0.012))),
        play_note(
            instrument,
            detuned_freq(scale[degree], rng, 7.0),
            velocity,
            rng,
            voice.melody_seconds,
            voice.melody_decay,
        ),
    )
    return degree


def walk_degree(degree: int, scale_len: int, center: float, rng: np.random.Generator) -> int:
    step = int(rng.choice(np.array([-2, -1, -1, 1, 1, 2])))
    degree = int(np.clip(degree + step, 0, scale_len - 1))
    if abs(degree - center) > 2.6:
        degree += int(np.sign(center - degree))
    return int(np.clip(degree, 0, scale_len - 1))


def harmonic_scale(mode: ModeName) -> tuple[int, ...]:
    match mode:
        case "Dorien":
            return SCALES["Dorien"]
        case "Mineur naturel":
            return SCALES["Mineur naturel"]
        case unreachable:
            assert_never(unreachable)


def harmonic_progressions(
    mode: ModeName,
) -> tuple[tuple[tuple[int, str], ...], ...]:
    match mode:
        case "Dorien":
            return PROGRESSIONS["Dorien"]
        case "Mineur naturel":
            return PROGRESSIONS["Mineur naturel"]
        case unreachable:
            assert_never(unreachable)


def select_voices(notes: tuple[int, ...], count: int) -> tuple[int, ...]:
    ordered = tuple(sorted(notes))
    if len(ordered) <= count or count <= 0:
        return ordered
    if count == 1:
        return (ordered[len(ordered) // 2],)
    picked = [ordered[0], ordered[-1]]
    for note in ordered[1:-1]:
        if len(picked) >= count:
            break
        picked.append(note)
    return tuple(sorted(picked[:count]))


def add_lead(
    buffer: FloatArray,
    instrument: InstrumentName,
    voice: Voice,
    mood: MoodSpec,
    tonic_pc: int,
    progression: tuple[tuple[int, str], ...],
    voicings: tuple[tuple[tuple[int, ...], int], ...],
    rng: np.random.Generator,
) -> None:
    scale = melody_scale(tonic_pc, harmonic_scale(MODE), voice.melody_low, voice.melody_high)
    if not scale:
        raise RuntimeError("La tessiture mélodique est vide.")
    degree = len(scale) // 2
    for bar in range(N_BARS):
        breath = 0.78 + 0.22 * (0.5 + 0.5 * np.sin(2.0 * np.pi * bar / N_BARS))
        voices = select_voices(voicings[bar][0], voice.voices)
        match voice.pattern:
            case "strum":
                if bar % 2 == 0:
                    strum = 0.01
                    for index, note in enumerate(voices):
                        strum += float(rng.uniform(0.026, 0.044))
                        velocity = (0.34 + 0.16 * breath) * float(rng.uniform(0.92, 1.0))
                        if index == len(voices) - 1:
                            velocity *= 0.64
                        place_circular(
                            buffer,
                            sample_at(bar, strum + float(rng.uniform(-0.004, 0.006))),
                            play_note(
                                instrument,
                                detuned_freq(note, rng, 4.0),
                                velocity,
                                rng,
                                voice.chord_seconds,
                                voice.chord_decay,
                            ),
                        )
                else:
                    order = list(voices)
                    if rng.random() < 0.5:
                        order.reverse()
                    for index, note in enumerate(order):
                        beat = 0.15 + index * 0.85
                        velocity = (0.28 + 0.12 * breath) * float(rng.uniform(0.9, 1.0))
                        place_circular(
                            buffer,
                            sample_at(bar, beat + float(rng.uniform(-0.008, 0.01))),
                            play_note(
                                instrument,
                                detuned_freq(note, rng, 5.0),
                                velocity,
                                rng,
                                voice.melody_seconds,
                                voice.chord_decay,
                            ),
                        )
            case "stab":
                if bar % 4 != 3 or rng.random() < 0.65:
                    strum = 0.0
                    for note in voices:
                        strum += float(rng.uniform(0.01, 0.024))
                        velocity = (0.32 + 0.14 * breath) * float(rng.uniform(0.9, 1.02))
                        place_circular(
                            buffer,
                            sample_at(bar, strum + float(rng.uniform(-0.005, 0.006))),
                            play_note(
                                instrument,
                                detuned_freq(note, rng, 5.0),
                                velocity,
                                rng,
                                voice.chord_seconds,
                                voice.chord_decay,
                            ),
                        )
                if rng.random() < 0.7:
                    answer = int(voices[int(rng.integers(0, len(voices)))])
                    beat = float(rng.choice(np.array([1.5, 2.5, 3.0])))
                    place_circular(
                        buffer,
                        sample_at(bar, swung(beat, mood.swing) + float(rng.uniform(-0.006, 0.01))),
                        play_note(
                            instrument,
                            detuned_freq(answer, rng, 6.0),
                            (0.22 + 0.1 * breath) * float(rng.uniform(0.9, 1.0)),
                            rng,
                            voice.melody_seconds,
                            voice.melody_decay,
                        ),
                    )
            case "arp":
                order = list(voices)
                if bar % 2:
                    order.reverse()
                for index, note in enumerate(order):
                    beat = index * 0.75 + float(rng.uniform(0.0, 0.04))
                    velocity = (0.30 + 0.12 * breath) * float(rng.uniform(0.88, 1.0))
                    place_circular(
                        buffer,
                        sample_at(bar, swung(beat, mood.swing)),
                        play_note(
                            instrument,
                            detuned_freq(note + 12 if note < voice.melody_low else note, rng, 8.0),
                            velocity,
                            rng,
                            voice.chord_seconds,
                            voice.chord_decay,
                        ),
                    )
            case "sustain":
                if bar % 2 == 0:
                    for note in voices:
                        place_circular(
                            buffer,
                            sample_at(bar, float(rng.uniform(0.0, 0.03))),
                            play_note(
                                instrument,
                                detuned_freq(note, rng, 4.0),
                                (0.30 + 0.08 * breath) * float(rng.uniform(0.94, 1.0)),
                                rng,
                                voice.chord_seconds,
                                voice.chord_decay,
                            ),
                        )
            case "sparse":
                if rng.random() < 0.82:
                    note = int(voices[int(rng.integers(0, len(voices)))])
                    while note < voice.melody_low:
                        note += 12
                    place_circular(
                        buffer,
                        sample_at(bar, float(rng.choice(np.array([0.0, 1.5, 2.75])))),
                        play_note(
                            instrument,
                            detuned_freq(note, rng, 3.0),
                            (0.34 + 0.1 * breath) * float(rng.uniform(0.9, 1.0)),
                            rng,
                            voice.chord_seconds,
                            voice.chord_decay,
                        ),
                    )
            case unreachable:
                assert_never(unreachable)

        if rng.random() > mood.melody_chance or voice.pattern == "sparse":
            continue
        center = (len(scale) * 0.42) + 1.2 * np.sin(2.0 * np.pi * bar / N_BARS + 0.6)
        degree = walk_degree(degree, len(scale), center, rng)
        beat = float(rng.choice(np.array([0.75, 1.75, 2.5, 3.25])))
        velocity = float(rng.uniform(0.24, 0.40)) * (0.8 + 0.2 * breath)
        degree = add_melody_note(
            buffer,
            instrument,
            scale,
            degree,
            bar,
            beat,
            velocity,
            rng,
            voice,
            mood.swing,
        )


def bass_events(mood: MoodName, bar: int, root: int, fifth: int, octave: int) -> tuple[tuple[float, int, float], ...]:
    del bar, octave
    match mood:
        case "Nostalgique":
            return ((0.0, root, 0.66), (2.5, fifth, 0.28))
        case "Cozy":
            return ((0.0, root, 0.70), (2.0, fifth, 0.30))
        case "Nocturne":
            return ((0.0, root, 0.60),)
        case "Smooth":
            return ((0.0, root, 0.68), (2.0, fifth, 0.28))
        case unreachable:
            assert_never(unreachable)


def add_bass(
    buffer: FloatArray,
    mood: MoodName,
    voicings: tuple[tuple[tuple[int, ...], int], ...],
    rng: np.random.Generator,
) -> None:
    for bar in range(N_BARS):
        root = voicings[bar][1]
        fifth = root + 7
        if fifth > 48:
            fifth -= 12
        octave = root + 12
        if octave > 48:
            octave = fifth
        for beat, note, velocity in bass_events(mood, bar, root, fifth, octave):
            freq = midi_to_hz(note) * (2.0 ** (float(rng.uniform(-3.0, 3.0)) / 1200.0))
            place_circular(
                buffer,
                sample_at(bar, beat + float(rng.uniform(-0.006, 0.010))),
                synth_bass(freq, velocity * float(rng.uniform(0.95, 1.0)), rng),
            )


def add_drums(
    kick: FloatArray,
    snare: FloatArray,
    hat: FloatArray,
    voice: Voice,
    mood: MoodSpec,
    rng: np.random.Generator,
) -> None:
    del voice
    for bar in range(N_BARS):
        late = float(rng.uniform(0.012, 0.028))
        place_circular(
            kick,
            sample_at(bar, late + float(rng.uniform(-0.004, 0.006))),
            synth_kick(0.72 * mood.drum_gain * float(rng.uniform(0.94, 1.0)), rng),
        )
        brush = 0.46 * mood.drum_gain * float(rng.uniform(0.92, 1.04))
        place_circular(
            snare,
            sample_at(bar, 2.0 + late + float(rng.uniform(0.004, 0.016))),
            synth_brush(brush, rng),
        )
        if rng.random() < 0.55:
            place_circular(
                snare,
                sample_at(bar, 3.5 + late),
                synth_brush(brush * 0.28, rng),
            )
        for eighth in range(8):
            match mood.hat_mode:
                case "offbeats":
                    play = eighth % 2 == 1
                case "sparse":
                    play = eighth in (1, 5)
                case "eighths":
                    play = eighth % 2 == 1 or eighth in (0, 4)
                case unreachable:
                    assert_never(unreachable)
            if not play:
                continue
            beat = swung(eighth * 0.5, mood.swing) + late * 0.45
            velocity = 0.22 if eighth % 4 == 1 else 0.10
            velocity *= mood.hat_gain * float(rng.uniform(0.88, 1.0))
            place_circular(
                hat,
                sample_at(bar, beat),
                synth_hat(velocity, rng, open_hat=False),
            )


def center_lows(left: FloatArray, right: FloatArray, cutoff: float = 180.0) -> tuple[FloatArray, FloatArray]:
    mid = 0.5 * (left + right)
    low = circular_lowpass(mid, cutoff, poles=2)
    left_out = left - circular_lowpass(left, cutoff, poles=2) + low
    right_out = right - circular_lowpass(right, cutoff, poles=2) + low
    return left_out, right_out


def apply_space(mono: FloatArray, space: SpaceName) -> tuple[FloatArray, FloatArray]:
    match space:
        case "chorus":
            left_wet = circular_mod_delay(mono, 0.018 * SR, 0.0042 * SR, periodic_rate(0.09), 0.4)
            right_wet = circular_mod_delay(mono, 0.027 * SR, 0.0055 * SR, periodic_rate(0.13), 2.4)
            room = dark_room(mono)
            left = mono * 0.54 + left_wet * 0.36 + room * 0.40
            right = mono * 0.30 + right_wet * 0.52 + room * 0.40
        case "tremolo":
            rate = periodic_rate(2.15)
            times = np.arange(P, dtype=np.float64) / SR
            left_mod = 0.86 + 0.14 * np.sin(2.0 * np.pi * rate * times)
            right_mod = 0.86 + 0.14 * np.sin(2.0 * np.pi * rate * times + np.pi * 0.85)
            room = dark_room(mono)
            left = mono * left_mod * 0.84 + room * 0.16
            right = mono * right_mod * 0.84 + room * 0.16
        case "room":
            room = dark_room(mono)
            left = mono * 0.74 + room * 0.26
            right = mono * 0.62 + room * 0.38
        case "wide":
            left_wet = circular_mod_delay(mono, 0.007 * SR, 0.0016 * SR, periodic_rate(0.07), 0.2)
            right_wet = circular_mod_delay(mono, 0.011 * SR, 0.0018 * SR, periodic_rate(0.05), 1.7)
            left = mono * 0.74 + left_wet * 0.26
            right = mono * 0.74 + right_wet * 0.26
        case "distant":
            room = circular_lowpass(dark_room(mono), 2400.0)
            left = mono * 0.40 + room * 0.60
            right = mono * 0.34 + room * 0.66
        case unreachable:
            assert_never(unreachable)
    return center_lows(left, right)



def render_vinyl(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    fade = int(0.05 * SR)
    hiss_l = circular_highpass(circular_noise(P, rng, fade), 2600.0)
    hiss_r = circular_highpass(circular_noise(P, rng, fade), 2600.0)
    crackle_l = np.zeros(P, dtype=np.float64)
    crackle_r = np.zeros(P, dtype=np.float64)
    for position in rng.integers(0, P, int(0.35 * P / SR)):
        length = int(rng.uniform(0.0011, 0.0055) * SR)
        envelope = np.sin(np.linspace(0.0, np.pi, length)) ** 2
        pop = rng.standard_normal(length) * envelope * float(rng.uniform(0.18, 1.0) ** 2)
        pop_l, pop_r = pan(pop, float(rng.uniform(-0.28, 0.28)))
        place_circular(crackle_l, int(position), pop_l)
        place_circular(crackle_r, int(position), pop_r)
    crackle_l = circular_highpass(crackle_l, 1300.0)
    crackle_r = circular_highpass(crackle_r, 1300.0)
    return np.column_stack((hiss_l, hiss_r)), np.column_stack((crackle_l, crackle_r))


def circular_highpass_steep(samples: FloatArray, cutoff: float) -> FloatArray:
    sos = butter(4, cutoff, btype="highpass", fs=SR, output="sos")
    pad = min(samples.shape[0] // 4, SR)
    warmed = np.concatenate((samples[-pad:], samples, samples[:pad]))
    filtered = sosfilt(sos, warmed)
    return np.asarray(filtered[pad:-pad], dtype=np.float64)


def center_bass(stereo: np.ndarray) -> np.ndarray:
    mid = 0.5 * (stereo[:, 0] + stereo[:, 1])
    side = 0.5 * (stereo[:, 0] - stereo[:, 1])
    side = circular_highpass_steep(side, 150.0)
    low = circular_lowpass(mid, 140.0, poles=4)
    high = mid - low
    return np.column_stack((low + high + side, low + high - side))


def finish(stereo: np.ndarray) -> np.ndarray:
    shaped = np.asarray(stereo, dtype=np.float64)
    peak = float(np.max(np.abs(shaped)))
    if peak < 1e-8:
        return shaped
    shaped *= PEAK_TARGET / peak
    for _ in range(8):
        current = rms(shaped)
        if current >= TARGET_RMS * 0.985:
            break
        gain = min(TARGET_RMS / max(current, 1e-8), 1.4)
        shaped = np.tanh(shaped * gain / PEAK_TARGET) * PEAK_TARGET
        peak = float(np.max(np.abs(shaped)))
        if peak > 1e-8:
            shaped *= PEAK_TARGET / peak
    current = rms(shaped)
    if current > TARGET_RMS:
        shaped *= TARGET_RMS / current
    return shaped


def bandpass_circular(signal: FloatArray, low_hz: float, high_hz: float) -> FloatArray:
    sos = butter(4, [low_hz, high_hz], btype="bandpass", fs=SR, output="sos")
    pad = min(signal.shape[0] // 4, SR)
    warmed = np.concatenate((signal[-pad:], signal, signal[:pad]))
    return np.asarray(sosfilt(sos, warmed)[pad:-pad], dtype=np.float64)


def beat_share(signal: FloatArray, low_hz: float, high_hz: float, beat_index: int) -> float:
    band = bandpass_circular(signal, low_hz, high_hz)
    energy = np.zeros(4, dtype=np.float64)
    for bar in range(N_BARS):
        for beat in range(4):
            start = sample_at(bar, float(beat))
            end = sample_at(bar, float(beat + 1))
            energy[beat] += float(np.sum(np.square(band[start:end])))
    total = float(np.sum(energy))
    if total < 1e-12:
        return 0.0
    return float(energy[beat_index] / total)


def onset_lag_ms(signal: FloatArray, low_hz: float, high_hz: float) -> float:
    band = np.square(bandpass_circular(signal, low_hz, high_hz))
    window = int(0.12 * SR)
    lags: list[int] = []
    length = band.shape[0]
    for bar in range(N_BARS):
        start = sample_at(bar, 0.0)
        indexes = (start + np.arange(window)) % length
        lags.append(int(np.argmax(band[indexes])))
    return float(np.mean(lags) / SR * 1000.0)


def render_period(seed: int) -> tuple[np.ndarray, str]:
    instrument: InstrumentName = INSTRUMENT
    mood_name: MoodName = MOOD
    mode: ModeName = MODE
    voice = VOICES[instrument]
    mood = MOODS[mood_name]
    rng = np.random.default_rng(seed)
    pool = harmonic_progressions(mode)
    progression = pool[int(rng.integers(0, len(pool)))]
    tonic_pc = int(rng.integers(0, 12))
    voicings = plan_voicings(tonic_pc, progression, voice.chord_low, voice.chord_high)
    degrees = "-".join(str(degree) for degree, _quality in progression)
    harmony = (
        f"{mode} m7 ({degrees}), "
        f"tonalité {NOTE_NAMES[tonic_pc]}, "
        f"basse {voicings[0][1]}/{voicings[N_BARS // 2][1]}"
    )

    lead = np.zeros(P, dtype=np.float64)
    bass = np.zeros(P, dtype=np.float64)
    kick = np.zeros(P, dtype=np.float64)
    snare = np.zeros(P, dtype=np.float64)
    hat = np.zeros(P, dtype=np.float64)

    add_lead(lead, instrument, voice, mood, tonic_pc, progression, voicings, rng)
    add_bass(bass, mood_name, voicings, rng)
    add_drums(kick, snare, hat, voice, mood, rng)

    hat = circular_highpass(hat, 3400.0)
    snare = circular_lowpass(snare, 2800.0)
    bass = circular_lowpass(bass, 280.0)
    kick_share = beat_share(kick, 30.0, 90.0, 0)
    brush_share = beat_share(snare, 1800.0, 3400.0, 2)
    lag_ms = onset_lag_ms(kick, 30.0, 90.0)
    print(
        f"Balais 1,8–3,4 kHz sur le temps 3 ({100.0 * brush_share:.1f} %), "
        f"kick 30–90 Hz sur le temps 1 ({100.0 * kick_share:.1f} %), "
        f"calage {lag_ms:.0f} ms",
        flush=True,
    )
    if brush_share < 0.70 or kick_share < 0.70:
        raise RuntimeError("La batterie n'est pas calée sur les temps 1 et 3.")
    lead_l, lead_r = apply_space(scale_to_rms(lead, voice.lead_rms), voice.space)
    lead_stereo = scale_to_rms(np.column_stack((lead_l, lead_r)), voice.lead_rms)
    bass = scale_to_rms(bass, 0.068)
    kick = scale_to_rms(kick, 0.050 * mood.drum_gain)
    snare = scale_to_rms(snare, 0.016 * mood.drum_gain)
    hat = scale_to_rms(hat, 0.0048 * mood.hat_gain)

    hat_l, hat_r = pan(hat, 0.18)
    snare_l, snare_r = pan(snare, -0.06)
    left = lead_stereo[:, 0] + bass + kick + snare_l + hat_l
    right = lead_stereo[:, 1] + bass + kick + snare_r + hat_r
    music = center_bass(np.column_stack((left, right)))
    music[:, 0] = circular_lowpass(music[:, 0], mood.cutoff)
    music[:, 1] = circular_lowpass(music[:, 1], mood.cutoff)
    music = np.tanh(music * 1.08) / np.tanh(1.08)
    music_mid = 0.5 * (music[:, 0] + music[:, 1])
    mix_kick = beat_share(music_mid, 30.0, 90.0, 0)
    mix_brush = beat_share(music_mid, 1800.0, 3400.0, 2)
    print(
        f"Mix : balais 1,8–3,4 kHz sur le temps 3 ({100.0 * mix_brush:.1f} %), "
        f"kick 30–90 Hz sur le temps 1 ({100.0 * mix_kick:.1f} %)",
        flush=True,
    )

    music_rms = rms(music)
    hiss, pops = render_vinyl(rng)
    hiss = scale_to_rms(hiss, music_rms * 10.0 ** (-38.0 / 20.0))
    pops = scale_to_rms(pops, music_rms * 10.0 ** (-42.0 / 20.0))
    vinyl = hiss + pops
    vinyl_ratio = rms(vinyl) / max(music_rms, 1e-12)
    print(
        f"Vinyle sous la musique: RMS {20.0 * np.log10(max(vinyl_ratio, 1e-12)):.1f} dB",
        flush=True,
    )
    stereo = music + vinyl
    stereo = center_bass(stereo)
    stereo[:, 0] = circular_highpass(stereo[:, 0], 30.0)
    stereo[:, 1] = circular_highpass(stereo[:, 1], 30.0)
    stereo[:, 0] = circular_lowpass(stereo[:, 0], 6200.0)
    stereo[:, 1] = circular_lowpass(stereo[:, 1], 6200.0)
    wow = periodic_rate(0.08)
    left_wow = circular_mod_delay(stereo[:, 0], 16.0, 3.5, wow, 0.7)
    right_wow = circular_mod_delay(stereo[:, 1], 16.0, 3.5, wow, 0.7)
    stereo = center_bass(np.column_stack((left_wow, right_wow)))
    stereo = center_bass(stereo)
    stereo = finish(stereo)
    return stereo.astype(np.float32), harmony


def seam_jump(stereo: np.ndarray) -> float:
    return float(np.max(np.abs(stereo[0].astype(np.float64) - stereo[-1].astype(np.float64))))


def seam_is_clean(stereo: np.ndarray) -> bool:
    steps = np.abs(np.diff(stereo.astype(np.float64), axis=0))
    boundary = np.abs(stereo[0].astype(np.float64) - stereo[-1].astype(np.float64))
    ceiling = np.percentile(steps, 99.0, axis=0)
    return bool(np.all(boundary <= ceiling))


def linear_crossfade(tail: np.ndarray, head: np.ndarray) -> np.ndarray:
    fade = np.linspace(0.0, 1.0, tail.shape[0], endpoint=False, dtype=np.float32)[:, None]
    return (1.0 - fade) * tail + fade * head


def measure(stereo: np.ndarray) -> dict[str, float]:
    peak = float(np.max(np.abs(stereo)))
    mid = 0.5 * (stereo[:, 0] + stereo[:, 1])
    side = 0.5 * (stereo[:, 0] - stereo[:, 1])
    low_mid = circular_lowpass(mid.astype(np.float64), 140.0, poles=4)
    low_side = circular_lowpass(side.astype(np.float64), 140.0, poles=4)
    band_mid = circular_lowpass(mid.astype(np.float64), 80.0) - circular_lowpass(
        mid.astype(np.float64), 20.0
    )
    band_side = circular_lowpass(side.astype(np.float64), 80.0) - circular_lowpass(
        side.astype(np.float64), 20.0
    )
    body = circular_lowpass(mid.astype(np.float64), 2500.0) - circular_lowpass(
        mid.astype(np.float64), 250.0
    )
    air = circular_lowpass(mid.astype(np.float64), 6000.0) - circular_lowpass(
        mid.astype(np.float64), 2500.0
    )
    high = stereo.astype(np.float64) - np.column_stack(
        (
            circular_lowpass(stereo[:, 0].astype(np.float64), 8000.0),
            circular_lowpass(stereo[:, 1].astype(np.float64), 8000.0),
        )
    )
    return {
        "peak_db": 20.0 * np.log10(max(peak, 1e-9)),
        "rms_db": 20.0 * np.log10(max(rms(stereo), 1e-9)),
        "low_mid_db": 20.0 * np.log10(max(rms(low_mid), 1e-9)),
        "low_side_db": 20.0 * np.log10(max(rms(low_side), 1e-9)),
        "band_mid_db": 20.0 * np.log10(max(rms(band_mid), 1e-9)),
        "band_side_db": 20.0 * np.log10(max(rms(band_side), 1e-9)),
        "body_db": 20.0 * np.log10(max(rms(body), 1e-9)),
        "air_db": 20.0 * np.log10(max(rms(air), 1e-9)),
        "high_db": 20.0 * np.log10(max(rms(high), 1e-9)),
        "jump": seam_jump(stereo),
    }


def output_path() -> Path:
    moment = datetime(*TRIGGER_UTC, tzinfo=timezone.utc).astimezone(ZoneInfo("Europe/Paris"))
    filename = f"lofi_{moment:%Y-%m-%d}_{moment:%H-%M-%S}_{SLUG}.mp3"
    windows = Path("/mnt/c/Users/conta/OneDrive/Documents/Youtube/Son")
    if windows.is_dir():
        return windows / filename
    return Path("/workspace/Son") / filename


def write_mp3(period: np.ndarray, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
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
        f"title=Lo-fi study, {INSTRUMENT}, {BPM} BPM",
        "-metadata",
        "artist=Mix original",
        "-metadata",
        "genre=Lo-fi Hip Hop",
        "-metadata",
        f"comment={PROMPT}",
        str(destination),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    head = np.ascontiguousarray(period[:FADE])
    blend = np.ascontiguousarray(linear_crossfade(head, head))
    if float(np.max(np.abs(blend - head))) > 1e-5:
        process.kill()
        raise RuntimeError("Le fondu de 3 secondes n'est pas transparent.")
    started = time.perf_counter()
    try:
        process.stdin.write(np.ascontiguousarray(period).tobytes())
        body = np.ascontiguousarray(period[FADE:]).tobytes()
        blend_bytes = blend.tobytes()
        for index in range(1, LOOPS):
            process.stdin.write(blend_bytes)
            process.stdin.write(body)
            if index % 5 == 0 or index == LOOPS - 1:
                elapsed = time.perf_counter() - started
                print(f"boucle {index + 1}/{LOOPS}  |  {elapsed / 60.0:.1f} min", flush=True)
        process.stdin.write(head.tobytes())
    finally:
        process.stdin.close()
        code = process.wait()
    if code != 0:
        raise RuntimeError(f"ffmpeg s'est terminé avec le code {code}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Génère la piste lo-fi de 5 heures.")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    drift = abs(sample_at(N_BARS, 0) - P)
    if drift > int(0.005 * SR):
        raise RuntimeError(f"La grille rythmique dérive de {drift} échantillons.")
    print(PROMPT, flush=True)
    print(
        f"Instrument: {INSTRUMENT} | Mode: {MODE} | Humeur: {MOOD} | BPM: {BPM}",
        flush=True,
    )
    started = time.perf_counter()
    period, harmony = render_period(SEED)
    stats = measure(period)
    print(f"Harmonie: {harmony}", flush=True)
    print(
        f"Master musical: {P / SR:.3f} s, crête {stats['peak_db']:.2f} dBFS, "
        f"moyenne {stats['rms_db']:.2f} dB, saut de boucle {stats['jump']:.6f}",
        flush=True,
    )
    print(
        f"Graves mid {stats['low_mid_db']:.2f} dB, side {stats['low_side_db']:.2f} dB, "
        f"20–80 Hz mid {stats['band_mid_db']:.2f} dB, side {stats['band_side_db']:.2f} dB, "
        f"corps {stats['body_db']:.2f} dB, air {stats['air_db']:.2f} dB, "
        f"au-dessus de 8 kHz {stats['high_db']:.2f} dB",
        flush=True,
    )
    if not seam_is_clean(period):
        raise RuntimeError("La boucle du master présente un clic.")
    if args.check_only:
        print(f"Synthèse: {time.perf_counter() - started:.1f} s", flush=True)
        return
    destination = args.output or output_path()
    write_mp3(period, destination)
    print(f"Fichier écrit : {destination}", flush=True)
    print(f"Durée totale: {time.perf_counter() - started:.1f} s", flush=True)


if __name__ == "__main__":
    main()
