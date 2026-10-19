#!/usr/bin/env python3
"""Piste lo-fi originale de cette exécution.

Variables tirées avant la synthèse :
- Instrument : piano électrique Wurlitzer
- Foley : vent doux dans les feuilles d'automne
- Couleur : mystérieux et nocturne (midnight coding)
- Tempo : 74 BPM

Le master est un tampon circulaire de 4 minutes exactement : la fin
retombe sur le début. Le fichier final enchaîne 75 lectures avec un fondu
linéaire de 3 secondes. Comme les deux côtés du fondu portent le même
signal (les 3 premières secondes, recopiées après chaque tour), la jointure
reconstruit l'onde à l'identique. Durée : 75 × 4 min + 3 s = 5 h 00 min 03 s.
"""

from __future__ import annotations

import argparse
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from numpy.typing import NDArray
from scipy.signal import lfilter

TRIGGER_UTC = (2026, 10, 19, 19, 50, 23)
INSTRUMENT = "Piano électrique Wurlitzer"
FOLEY = "Vent doux dans les feuilles d'automne"
MOOD = "Mystérieux et nocturne (midnight coding)"
BPM = 74
SEED = 298_022_780
SLUG = "Wurlitzer"
PROMPT = (
    "Genre: Lo-fi hip hop, chillhop. BPM: 74. "
    "Mood: Mystérieux et nocturne (midnight coding), peaceful study vibes. "
    "Instrumentation: Piano électrique Wurlitzer, mellow bass lines, "
    "soft brushed drums (SP-404 style), subtle vinyl crackle. "
    "Background: Vent doux dans les feuilles d'automne. "
    "Characteristics: Slow evolving progression, NO sudden changes, "
    "consistent warm atmosphere, slight human imperfections, "
    "melancholic but comforting. "
    "Technical: Warm analog saturation, low-pass filtered highs, seamless transitions."
)

SR = 32_000
BEAT_S = 60.0 / BPM
BAR_BEATS = 4
N_BARS = BPM
P = int(round(N_BARS * BAR_BEATS * BEAT_S * SR))
FADE = int(round(3.0 * SR))
LOOPS = 75
SWING = 0.58
PEAK_TARGET = 10 ** (-2.5 / 20.0)

FloatArray = NDArray[np.float64]

QUALITIES: dict[str, tuple[int, ...]] = {
    "m7": (0, 3, 7, 10),
    "m9": (0, 3, 7, 10),
    "maj7": (0, 4, 7, 11),
    "7": (0, 4, 7, 10),
}


@dataclass(frozen=True)
class Vamp:
    name: str
    degrees: tuple[tuple[int, str], ...]
    scale: tuple[int, ...]


VAMPS: tuple[Vamp, ...] = (
    Vamp("dorien i–IV", ((0, "m7"), (5, "maj7")), (0, 2, 3, 5, 7, 9, 10)),
    Vamp("mineur i–bVI", ((0, "m9"), (8, "maj7")), (0, 2, 3, 5, 7, 8, 10)),
    Vamp("mineur i–iv", ((0, "m7"), (5, "m7")), (0, 2, 3, 5, 7, 8, 10)),
    Vamp("dorien i–bVII", ((0, "m7"), (10, "7")), (0, 2, 3, 5, 7, 9, 10)),
    Vamp("phrygien i–bII", ((0, "m7"), (1, "maj7")), (0, 1, 3, 5, 7, 8, 10)),
)

NOTE_NAMES = ("Do", "Do#", "Ré", "Mib", "Mi", "Fa", "Fa#", "Sol", "Sol#", "La", "Sib", "Si")


def midi_to_hz(note: float) -> float:
    return 440.0 * (2.0 ** ((note - 69.0) / 12.0))


def periodic_rate(target_hz: float) -> float:
    duration = P / SR
    cycles = max(1, int(round(target_hz * duration)))
    return cycles / duration


def sample_at(bar: int, beat: float) -> int:
    return int(round((bar * BAR_BEATS + beat) * BEAT_S * SR))


def swung(beat: float) -> float:
    whole = np.floor(beat)
    frac = beat - whole
    if frac >= 0.5:
        local = frac - 0.5
        return float(whole + SWING + local * ((1.0 - SWING) / 0.5))
    return float(whole + frac * (SWING / 0.5))


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
    for delay_s, gain in ((0.023, 0.22), (0.037, 0.14), (0.061, 0.09), (0.089, 0.05)):
        wet += gain * np.roll(signal, int(delay_s * SR))
    return circular_lowpass(wet, 1800.0)


def rms(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(signal), dtype=np.float64)))


def scale_to_rms(signal: np.ndarray, target: float) -> np.ndarray:
    current = rms(signal)
    if current < 1e-9:
        return signal
    return signal * (target / current)


def synth_wurli(freq: float, velocity: float, rng: np.random.Generator, seconds: float) -> FloatArray:
    count = int(seconds * SR)
    times = np.arange(count, dtype=np.float64) / SR
    partials = (
        (1.0, 1.00, 1.35),
        (2.0, 0.22, 2.40),
        (3.0, 0.48, 2.05),
        (4.0, 0.16, 3.30),
        (5.0, 0.20, 3.10),
        (6.0, 0.07, 4.40),
        (7.0, 0.08, 4.80),
    )
    signal = np.zeros(count, dtype=np.float64)
    stiffness = 0.00012
    for index, (ratio, amplitude, decay) in enumerate(partials, start=1):
        partial_hz = freq * ratio * np.sqrt(1.0 + stiffness * index * index)
        if partial_hz >= SR * 0.45:
            break
        phase = float(rng.uniform(0.0, 0.15))
        signal += amplitude * np.sin(2.0 * np.pi * partial_hz * times + phase) * np.exp(-times * decay)
    attack = 1.0 - np.exp(-times / 0.0035)
    signal *= attack
    bark = min(count, int(0.018 * SR))
    burst = rng.standard_normal(bark) * np.exp(-np.arange(bark) / (SR * 0.0028))
    signal[:bark] += burst * 0.11 * (0.45 + velocity)
    thump = min(count, int(0.05 * SR))
    thump_env = np.exp(-np.arange(thump) / (SR * 0.012))
    signal[:thump] += np.sin(2.0 * np.pi * 92.0 * times[:thump]) * thump_env * 0.08
    faded = np.tanh(signal * 1.25) * velocity
    return fade_tail(faded)


def synth_bass(freq: float, velocity: float) -> FloatArray:
    count = int(1.25 * SR)
    times = np.arange(count, dtype=np.float64) / SR
    signal = (
        np.sin(2.0 * np.pi * freq * times)
        + 0.18 * np.sin(2.0 * np.pi * freq * 2.0 * times)
        + 0.03 * np.sin(2.0 * np.pi * freq * 3.0 * times)
    )
    signal *= (1.0 - np.exp(-times / 0.02)) * np.exp(-times / 0.78)
    return fade_tail(np.tanh(signal * 1.2) * velocity)


def synth_kick(velocity: float, rng: np.random.Generator) -> FloatArray:
    count = int(0.46 * SR)
    times = np.arange(count, dtype=np.float64) / SR
    frequency = 40.0 + 70.0 * np.exp(-times * 18.0)
    phase = 2.0 * np.pi * np.cumsum(frequency) / SR
    body = np.sin(phase) * np.exp(-times * 5.6) * (1.0 - np.exp(-times / 0.007))
    click = rng.standard_normal(count) * np.exp(-times * 170.0) * 0.035
    return fade_tail(np.tanh((body + click) * 1.45) * velocity)


def synth_brush(velocity: float, rng: np.random.Generator) -> FloatArray:
    count = int(0.58 * SR)
    times = np.arange(count, dtype=np.float64) / SR
    noise = rng.standard_normal(count) * np.exp(-times * 6.8)
    body = np.sin(2.0 * np.pi * 155.0 * times) * np.exp(-times * 15.0) * 0.24
    return fade_tail(np.tanh((noise * 0.72 + body) * 1.5) * velocity)


def synth_hat(velocity: float, rng: np.random.Generator, open_hat: bool) -> FloatArray:
    count = int((0.12 if open_hat else 0.04) * SR)
    times = np.arange(count, dtype=np.float64) / SR
    decay = 12.0 if open_hat else 48.0
    return fade_tail(rng.standard_normal(count) * np.exp(-times * decay) * velocity)


def synth_pad(freq: float) -> FloatArray:
    count = int(3.2 * SR)
    times = np.arange(count, dtype=np.float64) / SR
    envelope = (1.0 - np.exp(-times / 0.18)) * np.exp(-times / 1.6)
    signal = np.sin(2.0 * np.pi * freq * times) + 0.04 * np.sin(2.0 * np.pi * freq * 2.0 * times)
    return fade_tail(signal * envelope, 0.04)


def fade_tail(signal: FloatArray, seconds: float = 0.02) -> FloatArray:
    fade_n = min(signal.size // 2, int(seconds * SR))
    if fade_n > 1:
        signal = signal.copy()
        signal[-fade_n:] *= np.linspace(1.0, 0.0, fade_n, endpoint=True)
    return signal


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


def plan_voicings(tonic_pc: int, vamp: Vamp) -> tuple[tuple[tuple[int, ...], int], ...]:
    raw: list[tuple[tuple[int, ...], int]] = []
    previous_voices: tuple[int, ...] | None = None
    previous_bass: int | None = None
    for _ in range(4):
        raw = []
        for _bar in range(N_BARS):
            degree, quality = vamp.degrees[_bar % len(vamp.degrees)]
            root_pc = (tonic_pc + degree) % 12
            classes = tuple((root_pc + interval) % 12 for interval in QUALITIES[quality])
            voices = voice_chord(classes, previous_voices, 52, 70)
            bass = fold_bass(root_pc, previous_bass)
            raw.append((voices, bass))
            previous_voices = voices
            previous_bass = bass
    return tuple(raw)


def melody_scale(tonic_pc: int, intervals: tuple[int, ...]) -> list[int]:
    notes: list[int] = []
    for octave in range(4, 7):
        for interval in intervals:
            note = octave * 12 + (tonic_pc + interval) % 12
            if 62 <= note <= 79:
                notes.append(note)
    return sorted(set(notes))


def add_wurli(
    left: FloatArray,
    right: FloatArray,
    melody: FloatArray,
    vamp: Vamp,
    tonic_pc: int,
    voicings: tuple[tuple[tuple[int, ...], int], ...],
    rng: np.random.Generator,
) -> None:
    scale = melody_scale(tonic_pc, vamp.scale)
    degree = len(scale) // 2
    for bar in range(N_BARS):
        breath = 0.5 + 0.5 * np.sin(2.0 * np.pi * bar / N_BARS)
        voices, _bass = voicings[bar]
        roll = 0.0
        for index, note in enumerate(voices):
            roll += float(rng.uniform(0.012, 0.022))
            cents = float(rng.uniform(-3.0, 3.0))
            freq = midi_to_hz(note) * (2.0 ** (cents / 1200.0))
            velocity = (0.46 + 0.08 * breath) * float(rng.uniform(0.92, 1.0))
            position = -0.12 if index % 2 == 0 else 0.10
            tone = synth_wurli(freq, velocity, rng, 2.7)
            tone_l, tone_r = pan(tone, position)
            start = sample_at(bar, roll + float(rng.uniform(-0.004, 0.006)))
            place_circular(left, start, tone_l)
            place_circular(right, start, tone_r)
        ninth = min(voices[-1] + 2, 76)
        ninth_velocity = 0.08 + 0.12 * breath
        freq = midi_to_hz(ninth) * (2.0 ** (float(rng.uniform(-4.0, 4.0)) / 1200.0))
        color = synth_wurli(freq, ninth_velocity, rng, 2.2)
        color_l, color_r = pan(color, 0.22)
        place_circular(left, sample_at(bar, 0.08), color_l)
        place_circular(right, sample_at(bar, 0.08), color_r)

        presence = 0.70 + 0.16 * np.sin(2.0 * np.pi * bar / N_BARS + 0.6)
        if rng.random() > presence:
            continue
        center = (len(scale) * 0.48) + 1.6 * np.sin(2.0 * np.pi * bar / N_BARS)
        degree = int(np.clip(degree + int(rng.choice([-1, 1, 1])), 0, len(scale) - 1))
        if abs(degree - center) > 2.2:
            degree += int(np.sign(center - degree))
            degree = int(np.clip(degree, 0, len(scale) - 1))
        beat = float(rng.choice([0.5, 1.0, 1.5, 2.5]))
        notes_in_bar = 1 if rng.random() < 0.62 else 2
        for _ in range(notes_in_bar):
            degree = int(np.clip(degree + int(rng.choice([-2, -1, 1, 1, 2])), 0, len(scale) - 1))
            cents = float(rng.uniform(-5.0, 5.0))
            freq = midi_to_hz(scale[degree]) * (2.0 ** (cents / 1200.0))
            velocity = float(rng.uniform(0.34, 0.52)) * (0.85 + 0.15 * breath)
            start = sample_at(bar, swung(beat) + float(rng.uniform(-0.006, 0.01)))
            place_circular(melody, start, synth_wurli(freq, velocity, rng, 1.6))
            beat += float(rng.choice([1.0, 1.5, 2.0]))
            if beat > 3.3:
                break


def add_bass(
    buffer: FloatArray,
    voicings: tuple[tuple[tuple[int, ...], int], ...],
    rng: np.random.Generator,
) -> None:
    for bar in range(N_BARS):
        root = voicings[bar][1]
        fifth = root + 7
        if fifth > 50:
            fifth -= 12
        events = ((0.0, root, 0.90), (2.0, fifth if bar % 2 == 0 else root, 0.52))
        for beat, note, velocity in events:
            cents = float(rng.uniform(-2.0, 2.0))
            freq = midi_to_hz(note) * (2.0 ** (cents / 1200.0))
            place_circular(
                buffer,
                sample_at(bar, beat + float(rng.uniform(-0.006, 0.01))),
                synth_bass(freq, velocity * float(rng.uniform(0.95, 1.0))),
            )


def add_drums(
    kick: FloatArray,
    snare: FloatArray,
    hat: FloatArray,
    rng: np.random.Generator,
) -> None:
    for bar in range(N_BARS):
        for beat in (0.0, 2.0):
            velocity = 0.86 if beat == 0.0 else 0.72
            velocity *= float(rng.uniform(0.94, 1.0))
            place_circular(
                kick,
                sample_at(bar, swung(beat) + float(rng.uniform(-0.004, 0.008))),
                synth_kick(velocity, rng),
            )
        for beat in (1.0, 3.0):
            place_circular(
                snare,
                sample_at(bar, beat + float(rng.uniform(0.0, 0.012))),
                synth_brush(float(rng.uniform(0.62, 0.78)), rng),
            )
        if bar % 4 == 3:
            place_circular(snare, sample_at(bar, 2.75), synth_brush(0.22, rng))
        for eighth in range(8):
            beat = swung(eighth * 0.5)
            velocity = 0.48 if eighth % 4 == 0 else 0.34 if eighth % 2 == 0 else 0.20
            velocity *= float(rng.uniform(0.9, 1.02))
            place_circular(
                hat,
                sample_at(bar, beat),
                synth_hat(velocity, rng, open_hat=eighth == 6),
            )


def add_pad(
    left: FloatArray,
    right: FloatArray,
    voicings: tuple[tuple[tuple[int, ...], int], ...],
) -> None:
    for bar in range(N_BARS):
        start = sample_at(bar, 0.0)
        for note in voicings[bar][0][:3]:
            while note > 64:
                note -= 12
            freq = midi_to_hz(note)
            place_circular(left, start, synth_pad(freq * (2.0 ** (-4.0 / 1200.0))))
            place_circular(right, start, synth_pad(freq * (2.0 ** (4.0 / 1200.0))))


def shape_wurli(left: FloatArray, right: FloatArray, melody: FloatArray) -> tuple[FloatArray, FloatArray]:
    times = np.arange(P, dtype=np.float64) / SR
    rate = periodic_rate(4.6)
    tremolo_l = 0.90 + 0.10 * np.sin(2.0 * np.pi * rate * times)
    tremolo_r = 0.90 + 0.10 * np.sin(2.0 * np.pi * rate * times + 0.45)
    melody_l, melody_r = pan(melody, 0.14)
    dry_l = (left + melody_l) * tremolo_l
    dry_r = (right + melody_r) * tremolo_r
    wet_l = dark_room(dry_l)
    wet_r = dark_room(dry_r)
    return dry_l * 0.78 + wet_l * 0.42, dry_r * 0.78 + wet_r * 0.42


def render_wind(rng: np.random.Generator) -> np.ndarray:
    fade = int(0.1 * SR)
    body_noise = circular_noise(P, rng, fade)
    leaf_l = circular_noise(P, rng, fade)
    leaf_r = circular_noise(P, rng, fade)
    whoosh = circular_highpass(circular_lowpass(body_noise, 850.0), 140.0)
    leaves_l = circular_highpass(circular_lowpass(leaf_l, 5400.0), 2400.0)
    leaves_r = circular_highpass(circular_lowpass(leaf_r, 5400.0), 2400.0)
    flutter_l = circular_lowpass(np.abs(circular_noise(P, rng, fade)), 14.0)
    flutter_r = circular_lowpass(np.abs(circular_noise(P, rng, fade)), 14.0)
    flutter_l = flutter_l / max(rms(flutter_l), 1e-9)
    flutter_r = flutter_r / max(rms(flutter_r), 1e-9)

    times = np.arange(P, dtype=np.float64) / SR
    gust = 0.58 + 0.28 * np.sin(2.0 * np.pi * periodic_rate(0.018) * times)
    gust *= 0.82 + 0.18 * np.sin(2.0 * np.pi * periodic_rate(0.043) * times + 0.8)
    gust *= 0.90 + 0.10 * np.sin(2.0 * np.pi * periodic_rate(0.09) * times + 1.7)
    bumps = np.zeros(P, dtype=np.float64)
    gust_count = 8
    for index in range(gust_count):
        center = int((index + 0.5) * P / gust_count)
        center += int(rng.integers(-int(0.35 * SR), int(0.35 * SR)))
        length = int(float(rng.uniform(6.0, 9.0)) * SR)
        bump = np.sin(np.linspace(0.0, np.pi, length)) ** 2
        place_circular(bumps, center - length // 2, bump * float(rng.uniform(0.28, 0.55)))
    gust = np.clip(gust + bumps, 0.0, 1.8)
    left = whoosh * gust + leaves_l * flutter_l * gust * 0.85
    right = whoosh * gust + leaves_r * flutter_r * gust * 0.85
    wind = np.column_stack((left, right))
    for _ in range(4):
        wind[:, 0] = circular_highpass(wind[:, 0], 180.0)
        wind[:, 1] = circular_highpass(wind[:, 1], 180.0)
    wind[:, 0] = circular_lowpass(wind[:, 0], 5000.0)
    wind[:, 1] = circular_lowpass(wind[:, 1], 5000.0)
    return wind


def render_vinyl(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    fade = int(0.05 * SR)
    hiss_l = circular_highpass(circular_noise(P, rng, fade), 2400.0)
    hiss_r = circular_highpass(circular_noise(P, rng, fade), 2400.0)
    crackle_l = np.zeros(P, dtype=np.float64)
    crackle_r = np.zeros(P, dtype=np.float64)
    pops = int(3.4 * P / SR)
    for position in rng.integers(0, P, pops):
        length = int(rng.uniform(0.0015, 0.007) * SR)
        envelope = np.sin(np.linspace(0.0, np.pi, length)) ** 2
        burst = rng.standard_normal(length) * envelope * float(rng.uniform(0.25, 1.0) ** 2)
        burst_l, burst_r = pan(burst, float(rng.uniform(-0.35, 0.35)))
        place_circular(crackle_l, int(position), burst_l)
        place_circular(crackle_r, int(position), burst_r)
    crackle_l = circular_highpass(crackle_l, 1000.0)
    crackle_r = circular_highpass(crackle_r, 1000.0)
    return np.column_stack((hiss_l, hiss_r)), np.column_stack((crackle_l, crackle_r))


def center_bass(stereo: np.ndarray) -> np.ndarray:
    mid = 0.5 * (stereo[:, 0] + stereo[:, 1])
    side = 0.5 * (stereo[:, 0] - stereo[:, 1])
    low = circular_lowpass(mid, 140.0, poles=4)
    side_low = circular_lowpass(side, 140.0, poles=4)
    high = mid - low
    side = side - side_low
    left = low + high + side
    right = low + high - side
    return np.column_stack((left, right))


def render_period(seed: int) -> tuple[np.ndarray, str]:
    rng = np.random.default_rng(seed)
    vamp = VAMPS[int(rng.integers(0, len(VAMPS)))]
    tonic_pc = int(rng.integers(0, 12))
    voicings = plan_voicings(tonic_pc, vamp)
    harmony = (
        f"{vamp.name}, tonalité {NOTE_NAMES[tonic_pc]}, "
        f"basse {voicings[0][1]}/{voicings[1][1]}"
    )

    chords_l = np.zeros(P, dtype=np.float64)
    chords_r = np.zeros(P, dtype=np.float64)
    melody = np.zeros(P, dtype=np.float64)
    bass = np.zeros(P, dtype=np.float64)
    kick = np.zeros(P, dtype=np.float64)
    snare = np.zeros(P, dtype=np.float64)
    hat = np.zeros(P, dtype=np.float64)
    pad_l = np.zeros(P, dtype=np.float64)
    pad_r = np.zeros(P, dtype=np.float64)

    add_wurli(chords_l, chords_r, melody, vamp, tonic_pc, voicings, rng)
    add_bass(bass, voicings, rng)
    add_drums(kick, snare, hat, rng)
    add_pad(pad_l, pad_r, voicings)

    hat = circular_highpass(hat, 3000.0)
    snare = circular_lowpass(snare, 3600.0)
    bass = circular_lowpass(bass, 380.0)
    wurli_l, wurli_r = shape_wurli(chords_l, chords_r, melody)
    wurli = scale_to_rms(np.column_stack((wurli_l, wurli_r)), 0.085)
    bass = scale_to_rms(bass, 0.070)
    kick = scale_to_rms(kick, 0.056)
    snare = scale_to_rms(snare, 0.026)
    hat = scale_to_rms(hat, 0.010)
    pad = scale_to_rms(np.column_stack((pad_l, pad_r)), 0.016)

    hat_l, hat_r = pan(hat, 0.42)
    snare_l, snare_r = pan(snare, -0.06)
    left = wurli[:, 0] + bass + kick + snare_l + hat_l + pad[:, 0]
    right = wurli[:, 1] + bass + kick + snare_r + hat_r + pad[:, 1]
    music = np.column_stack((left, right))
    music = center_bass(music)
    for _ in range(2):
        music[:, 0] = circular_lowpass(music[:, 0], 4800.0)
        music[:, 1] = circular_lowpass(music[:, 1], 4800.0)
    music = np.tanh(music * 1.12) / np.tanh(1.12)

    wind = scale_to_rms(render_wind(rng), rms(music) * 0.26)
    hiss, pops = render_vinyl(rng)
    music_rms = rms(music)
    hiss = scale_to_rms(hiss, music_rms * 0.035)
    pops = scale_to_rms(pops, music_rms * 0.055)
    stereo = music + wind + hiss + pops
    stereo = center_bass(stereo)
    stereo[:, 0] = circular_highpass(stereo[:, 0], 28.0)
    stereo[:, 1] = circular_highpass(stereo[:, 1], 28.0)
    stereo[:, 0] = circular_lowpass(stereo[:, 0], 6800.0)
    stereo[:, 1] = circular_lowpass(stereo[:, 1], 6800.0)
    limit = max(rms(stereo) * 4.5, 1e-6)
    stereo = np.tanh(stereo / limit) * limit
    wow_rate = periodic_rate(0.12)
    left = circular_mod_delay(stereo[:, 0], 28.0, 8.0, wow_rate, 0.3)
    right = circular_mod_delay(stereo[:, 1], 28.0, 8.0, wow_rate, 0.3)
    stereo = np.column_stack((left, right))
    peak = float(np.max(np.abs(stereo)))
    if peak > 0.0:
        stereo *= PEAK_TARGET / peak
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
        "title=Lo-fi nocturne, Wurlitzer et vent d'automne",
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
    if P != 240 * SR:
        raise RuntimeError(f"Le master doit durer 4 minutes, pas {P / SR:.3f} s.")
    print(PROMPT, flush=True)
    print(
        f"Instrument: {INSTRUMENT} | Foley: {FOLEY} | Humeur: {MOOD} | BPM: {BPM}",
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
