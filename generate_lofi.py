#!/usr/bin/env python3
"""Piste lo-fi originale de cette exécution.

Variables tirées avant la synthèse :
- Instrument : guitare jazz clean avec chorus
- Foley : bruit de café lointain et tasses
- Couleur : mystérieux et nocturne (midnight coding)
- Tempo : 82 BPM

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

TRIGGER_UTC = (2026, 10, 19, 20, 50, 40)
INSTRUMENT = "Guitare jazz clean avec chorus"
FOLEY = "Bruit de café lointain et tasses"
MOOD = "Mystérieux et nocturne (midnight coding)"
BPM = 82
SEED = 883_314_101
SLUG = "Guitare_Jazz"
PROMPT = (
    "Genre: Lo-fi hip hop, chillhop. BPM: 82. "
    "Mood: Mystérieux et nocturne (midnight coding), peaceful study vibes. "
    "Instrumentation: Guitare jazz clean avec chorus, mellow bass lines, "
    "soft brushed drums (SP-404 style), subtle vinyl crackle. "
    "Background: Bruit de café lointain et tasses. "
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
SWING = 0.55
PEAK_TARGET = 10 ** (-2.5 / 20.0)
CHORD_LOW = 55
CHORD_HIGH = 70

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


# Cycles de 2 accords : 82 est pair, la boucle retombe sur le même degré.
VAMPS: tuple[Vamp, ...] = (
    Vamp("phrygien i–bII", ((0, "m7"), (1, "maj7")), (0, 1, 3, 5, 7, 8, 10)),
    Vamp("mineur i–bVI", ((0, "m9"), (8, "maj7")), (0, 2, 3, 5, 7, 8, 10)),
    Vamp("mineur i–iv", ((0, "m7"), (5, "m7")), (0, 2, 3, 5, 7, 8, 10)),
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
    for delay_s, gain in ((0.031, 0.18), (0.047, 0.12), (0.073, 0.08), (0.109, 0.05)):
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


def synth_guitar(
    freq: float,
    velocity: float,
    rng: np.random.Generator,
    seconds: float,
    decay: float,
) -> FloatArray:
    count = int(seconds * SR)
    times = np.arange(count, dtype=np.float64) / SR
    partials = (1.0, 0.34, 0.12, 0.045, 0.016)
    signal = np.zeros(count, dtype=np.float64)
    for index, amplitude in enumerate(partials, start=1):
        partial_hz = freq * index
        if partial_hz >= SR * 0.45:
            break
        detune = 1.0 + float(rng.uniform(-0.0004, 0.0004))
        envelope = np.exp(-times * (decay + index * 0.85))
        signal += amplitude * np.sin(2.0 * np.pi * partial_hz * detune * times) * envelope
    signal *= 1.0 - np.exp(-times / 0.007)
    pick = min(count, int(0.010 * SR))
    signal[:pick] += rng.standard_normal(pick) * np.exp(-np.arange(pick) / (SR * 0.0026)) * 0.045
    return fade_tail(np.tanh(signal * 1.2) * velocity)


def synth_bass(freq: float, velocity: float) -> FloatArray:
    count = int(1.2 * SR)
    times = np.arange(count, dtype=np.float64) / SR
    signal = (
        np.sin(2.0 * np.pi * freq * times)
        + 0.16 * np.sin(2.0 * np.pi * freq * 2.0 * times)
        + 0.025 * np.sin(2.0 * np.pi * freq * 3.0 * times)
    )
    signal *= (1.0 - np.exp(-times / 0.022)) * np.exp(-times / 0.82)
    return fade_tail(np.tanh(signal * 1.15) * velocity)


def synth_kick(velocity: float, rng: np.random.Generator) -> FloatArray:
    count = int(0.44 * SR)
    times = np.arange(count, dtype=np.float64) / SR
    frequency = 38.0 + 62.0 * np.exp(-times * 16.0)
    phase = 2.0 * np.pi * np.cumsum(frequency) / SR
    body = np.sin(phase) * np.exp(-times * 5.2) * (1.0 - np.exp(-times / 0.008))
    click = rng.standard_normal(count) * np.exp(-times * 160.0) * 0.028
    return fade_tail(np.tanh((body + click) * 1.4) * velocity)


def synth_brush(velocity: float, rng: np.random.Generator) -> FloatArray:
    count = int(0.55 * SR)
    times = np.arange(count, dtype=np.float64) / SR
    noise = rng.standard_normal(count) * np.exp(-times * 7.2)
    body = np.sin(2.0 * np.pi * 148.0 * times) * np.exp(-times * 16.0) * 0.18
    return fade_tail(np.tanh((noise * 0.66 + body) * 1.45) * velocity)


def synth_hat(velocity: float, rng: np.random.Generator, open_hat: bool) -> FloatArray:
    count = int((0.10 if open_hat else 0.035) * SR)
    times = np.arange(count, dtype=np.float64) / SR
    decay = 14.0 if open_hat else 52.0
    return fade_tail(rng.standard_normal(count) * np.exp(-times * decay) * velocity)


def synth_pad(freq: float) -> FloatArray:
    count = int(3.4 * SR)
    times = np.arange(count, dtype=np.float64) / SR
    envelope = (1.0 - np.exp(-times / 0.22)) * np.exp(-times / 1.8)
    signal = np.sin(2.0 * np.pi * freq * times) + 0.03 * np.sin(2.0 * np.pi * freq * 2.0 * times)
    return fade_tail(signal * envelope, 0.05)


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
    previous_voices: tuple[int, ...] | None = None
    previous_bass: int | None = None
    raw: list[tuple[tuple[int, ...], int]] = []
    for _ in range(8):
        raw = []
        for bar in range(N_BARS):
            degree, quality = vamp.degrees[bar % len(vamp.degrees)]
            root_pc = (tonic_pc + degree) % 12
            classes = tuple((root_pc + interval) % 12 for interval in QUALITIES[quality])
            voices = voice_chord(classes, previous_voices, CHORD_LOW, CHORD_HIGH)
            bass = fold_bass(root_pc, previous_bass)
            raw.append((voices, bass))
            previous_voices = voices
            previous_bass = bass
    degree, quality = vamp.degrees[0]
    root_pc = (tonic_pc + degree) % 12
    classes = tuple((root_pc + interval) % 12 for interval in QUALITIES[quality])
    next_voices = voice_chord(classes, previous_voices, CHORD_LOW, CHORD_HIGH)
    next_bass = fold_bass(root_pc, previous_bass)
    if next_voices != raw[0][0] or next_bass != raw[0][1]:
        raise RuntimeError("Le voicing de guitare ne se referme pas sur la boucle.")
    return tuple(raw)


def melody_scale(tonic_pc: int, intervals: tuple[int, ...]) -> list[int]:
    notes: list[int] = []
    for octave in range(4, 7):
        for interval in intervals:
            note = octave * 12 + (tonic_pc + interval) % 12
            if 64 <= note <= 79:
                notes.append(note)
    return sorted(set(notes))


def add_guitar(
    chords: FloatArray,
    melody: FloatArray,
    vamp: Vamp,
    tonic_pc: int,
    voicings: tuple[tuple[tuple[int, ...], int], ...],
    rng: np.random.Generator,
) -> None:
    scale = melody_scale(tonic_pc, vamp.scale)
    degree = len(scale) // 2
    for bar in range(N_BARS):
        breath = 0.45 + 0.55 * np.sin(2.0 * np.pi * bar / N_BARS)
        voices = tuple(sorted(voicings[bar][0]))
        strum = 0.0
        for index, note in enumerate(voices):
            strum += float(rng.uniform(0.011, 0.020))
            cents = float(rng.uniform(-3.5, 3.5))
            freq = midi_to_hz(note) * (2.0 ** (cents / 1200.0))
            velocity = (0.42 + 0.10 * breath) * float(rng.uniform(0.90, 1.0))
            if index == len(voices) - 1:
                velocity *= 0.72
            tone = synth_guitar(freq, velocity, rng, 3.3, 1.05)
            place_circular(chords, sample_at(bar, strum + float(rng.uniform(-0.003, 0.005))), tone)
        if bar % 2 == 1 and breath > 0.55:
            ghost = 0.0
            for note in voices[:3]:
                ghost += float(rng.uniform(0.008, 0.014))
                freq = midi_to_hz(note) * (2.0 ** (float(rng.uniform(-2.0, 2.0)) / 1200.0))
                place_circular(
                    chords,
                    sample_at(bar, 2.0 + ghost),
                    synth_guitar(freq, 0.16 + 0.06 * breath, rng, 1.6, 1.8),
                )

        if rng.random() > 0.48 + 0.22 * breath:
            continue
        center = (len(scale) * 0.42) + 1.3 * np.sin(2.0 * np.pi * bar / N_BARS + 0.4)
        step = int(rng.choice([-2, -1, 1, 1, 2]))
        degree = int(np.clip(degree + step, 0, len(scale) - 1))
        if abs(degree - center) > 2.4:
            degree += int(np.sign(center - degree))
            degree = int(np.clip(degree, 0, len(scale) - 1))
        beat = float(rng.choice([0.5, 1.5, 2.5]))
        notes_in_bar = 1 if rng.random() < 0.72 else 2
        for _ in range(notes_in_bar):
            degree = int(np.clip(degree + int(rng.choice([-2, -1, 1, 1, 2])), 0, len(scale) - 1))
            cents = float(rng.uniform(-6.0, 6.0))
            freq = midi_to_hz(scale[degree]) * (2.0 ** (cents / 1200.0))
            velocity = float(rng.uniform(0.32, 0.48)) * (0.82 + 0.18 * breath)
            start = sample_at(bar, swung(beat) + float(rng.uniform(-0.005, 0.012)))
            if rng.random() < 0.18 and degree > 0:
                grace = midi_to_hz(scale[degree - 1])
                place_circular(
                    melody,
                    start - int(0.04 * SR),
                    synth_guitar(grace, velocity * 0.32, rng, 0.22, 3.4),
                )
            place_circular(melody, start, synth_guitar(freq, velocity, rng, 1.8, 1.55))
            beat += float(rng.choice([1.0, 1.5]))
            if beat > 3.2:
                break


def add_bass(
    buffer: FloatArray,
    voicings: tuple[tuple[tuple[int, ...], int], ...],
    rng: np.random.Generator,
) -> None:
    for bar in range(N_BARS):
        root = voicings[bar][1]
        fifth = root + 7
        if fifth > 48:
            fifth -= 12
        events = ((0.0, root, 0.86), (2.0, fifth if bar % 2 == 0 else root, 0.48))
        for beat, note, velocity in events:
            cents = float(rng.uniform(-2.0, 2.0))
            freq = midi_to_hz(note) * (2.0 ** (cents / 1200.0))
            place_circular(
                buffer,
                sample_at(bar, beat + float(rng.uniform(-0.005, 0.008))),
                synth_bass(freq, velocity * float(rng.uniform(0.96, 1.0))),
            )


def add_drums(
    kick: FloatArray,
    snare: FloatArray,
    hat: FloatArray,
    rng: np.random.Generator,
) -> None:
    for bar in range(N_BARS):
        for beat in (0.0, 2.0):
            velocity = 0.82 if beat == 0.0 else 0.66
            velocity *= float(rng.uniform(0.95, 1.0))
            place_circular(
                kick,
                sample_at(bar, swung(beat) + float(rng.uniform(-0.003, 0.006))),
                synth_kick(velocity, rng),
            )
        for beat in (1.0, 3.0):
            place_circular(
                snare,
                sample_at(bar, beat + float(rng.uniform(0.0, 0.010))),
                synth_brush(float(rng.uniform(0.55, 0.70)), rng),
            )
        if bar % 8 == 7:
            place_circular(snare, sample_at(bar, 2.75), synth_brush(0.16, rng))
        for eighth in range(8):
            beat = swung(eighth * 0.5)
            velocity = 0.40 if eighth % 4 == 0 else 0.26 if eighth % 2 == 0 else 0.15
            velocity *= float(rng.uniform(0.92, 1.0))
            place_circular(
                hat,
                sample_at(bar, beat),
                synth_hat(velocity, rng, open_hat=eighth == 6 and bar % 4 == 3),
            )


def add_pad(
    left: FloatArray,
    right: FloatArray,
    voicings: tuple[tuple[tuple[int, ...], int], ...],
) -> None:
    for bar in range(N_BARS):
        start = sample_at(bar, 0.0)
        for note in sorted(voicings[bar][0])[:3]:
            while note > 62:
                note -= 12
            freq = midi_to_hz(note)
            place_circular(left, start, synth_pad(freq * (2.0 ** (-3.0 / 1200.0))))
            place_circular(right, start, synth_pad(freq * (2.0 ** (3.0 / 1200.0))))


def chorus_guitar(dry: FloatArray) -> tuple[FloatArray, FloatArray]:
    rate_left = periodic_rate(0.16)
    rate_right = periodic_rate(0.22)
    left_wet = circular_mod_delay(dry, 0.016 * SR, 0.0038 * SR, rate_left, 0.2)
    right_wet = circular_mod_delay(dry, 0.021 * SR, 0.0046 * SR, rate_right, 2.1)
    room = dark_room(dry)
    left = dry * 0.62 + left_wet * 0.40 + room * 0.48
    right = dry * 0.38 + right_wet * 0.55 + room * 0.48
    low = circular_lowpass(0.5 * (left + right), 200.0, poles=2)
    left = left - circular_lowpass(left, 200.0, poles=2) + low
    right = right - circular_lowpass(right, 200.0, poles=2) + low
    return left, right


def synth_cup(rng: np.random.Generator, bright: bool) -> FloatArray:
    count = int(float(rng.uniform(0.10, 0.28 if bright else 0.20)) * SR)
    times = np.arange(count, dtype=np.float64) / SR
    base = float(rng.uniform(1700.0, 2600.0) if bright else rng.uniform(680.0, 1100.0))
    ratios = (1.0, 1.53, 2.21, 2.86, 3.62) if bright else (1.0, 1.41, 2.02, 2.64)
    signal = np.zeros(count, dtype=np.float64)
    for index, ratio in enumerate(ratios):
        partial = base * ratio
        if partial >= SR * 0.42:
            break
        decay = float(rng.uniform(9.0, 16.0)) * (1.0 + 0.28 * index)
        signal += (0.62 / (1.0 + index)) * np.sin(2.0 * np.pi * partial * times) * np.exp(-times * decay)
    signal *= 1.0 - np.exp(-times / 0.0015)
    tick = min(count, int(0.008 * SR))
    signal[:tick] += rng.standard_normal(tick) * np.exp(-np.arange(tick) / (SR * 0.0011)) * 0.35
    return fade_tail(signal, 0.012)


def render_cafe(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    fade = int(0.12 * SR)
    murmur = circular_lowpass(circular_noise(P, rng, fade), 1100.0, poles=2)
    for _ in range(4):
        murmur = circular_highpass(murmur, 260.0)
    times = np.arange(P, dtype=np.float64) / SR
    swell = 0.70 + 0.18 * np.sin(2.0 * np.pi * periodic_rate(0.013) * times)
    swell *= 0.86 + 0.14 * np.sin(2.0 * np.pi * periodic_rate(0.041) * times + 1.3)
    voiced = murmur * swell
    murmur = np.column_stack((voiced, voiced))

    clink_l = np.zeros(P, dtype=np.float64)
    clink_r = np.zeros(P, dtype=np.float64)
    cup_count = int(P / SR / 11.0)
    for _ in range(cup_count):
        burst = synth_cup(rng, bright=True) * float(rng.uniform(0.35, 1.0))
        burst_l, burst_r = pan(burst, float(rng.uniform(-0.75, 0.75)))
        start = int(rng.integers(0, P))
        place_circular(clink_l, start, burst_l)
        place_circular(clink_r, start, burst_r)
    saucer_count = int(P / SR / 23.0)
    for _ in range(saucer_count):
        burst = synth_cup(rng, bright=False) * float(rng.uniform(0.25, 0.7))
        burst_l, burst_r = pan(burst, float(rng.uniform(-0.4, 0.4)))
        start = int(rng.integers(0, P))
        place_circular(clink_l, start, burst_l)
        place_circular(clink_r, start, burst_r)
    clink_l = circular_highpass(circular_lowpass(clink_l, 5400.0), 850.0)
    clink_r = circular_highpass(circular_lowpass(clink_r, 5400.0), 850.0)
    clink_l = clink_l + 0.22 * dark_room(clink_l)
    clink_r = clink_r + 0.22 * dark_room(clink_r)
    return murmur, np.column_stack((clink_l, clink_r))


def render_vinyl(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    fade = int(0.05 * SR)
    hiss_l = circular_highpass(circular_noise(P, rng, fade), 2500.0)
    hiss_r = circular_highpass(circular_noise(P, rng, fade), 2500.0)
    crackle_l = np.zeros(P, dtype=np.float64)
    crackle_r = np.zeros(P, dtype=np.float64)
    pops = int(2.6 * P / SR)
    for position in rng.integers(0, P, pops):
        length = int(rng.uniform(0.0012, 0.006) * SR)
        envelope = np.sin(np.linspace(0.0, np.pi, length)) ** 2
        burst = rng.standard_normal(length) * envelope * float(rng.uniform(0.2, 1.0) ** 2)
        burst_l, burst_r = pan(burst, float(rng.uniform(-0.3, 0.3)))
        place_circular(crackle_l, int(position), burst_l)
        place_circular(crackle_r, int(position), burst_r)
    crackle_l = circular_highpass(crackle_l, 1200.0)
    crackle_r = circular_highpass(crackle_r, 1200.0)
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
    candidates = [pc for pc in range(12) if pc not in (4, 5)]
    tonic_pc = candidates[int(rng.integers(0, len(candidates)))]
    voicings = plan_voicings(tonic_pc, vamp)
    harmony = (
        f"{vamp.name}, tonalité {NOTE_NAMES[tonic_pc]}, "
        f"basse {voicings[0][1]}/{voicings[1][1]}"
    )

    chords = np.zeros(P, dtype=np.float64)
    melody = np.zeros(P, dtype=np.float64)
    bass = np.zeros(P, dtype=np.float64)
    kick = np.zeros(P, dtype=np.float64)
    snare = np.zeros(P, dtype=np.float64)
    hat = np.zeros(P, dtype=np.float64)
    pad_l = np.zeros(P, dtype=np.float64)
    pad_r = np.zeros(P, dtype=np.float64)

    add_guitar(chords, melody, vamp, tonic_pc, voicings, rng)
    add_bass(bass, voicings, rng)
    add_drums(kick, snare, hat, rng)
    add_pad(pad_l, pad_r, voicings)

    hat = circular_highpass(hat, 3200.0)
    snare = circular_lowpass(snare, 3400.0)
    bass = circular_lowpass(bass, 360.0)
    chords = scale_to_rms(chords, 0.078)
    melody = scale_to_rms(melody, 0.042)
    guitar_l, guitar_r = chorus_guitar(chords + melody)
    guitar = scale_to_rms(np.column_stack((guitar_l, guitar_r)), 0.086)
    bass = scale_to_rms(bass, 0.068)
    kick = scale_to_rms(kick, 0.050)
    snare = scale_to_rms(snare, 0.022)
    hat = scale_to_rms(hat, 0.008)
    pad = scale_to_rms(np.column_stack((pad_l, pad_r)), 0.018)

    hat_l, hat_r = pan(hat, 0.36)
    snare_l, snare_r = pan(snare, -0.04)
    left = guitar[:, 0] + bass + kick + snare_l + hat_l + pad[:, 0]
    right = guitar[:, 1] + bass + kick + snare_r + hat_r + pad[:, 1]
    music = center_bass(np.column_stack((left, right)))
    for _ in range(2):
        music[:, 0] = circular_lowpass(music[:, 0], 4600.0)
        music[:, 1] = circular_lowpass(music[:, 1], 4600.0)
    music = np.tanh(music * 1.08) / np.tanh(1.08)

    murmur, clinks = render_cafe(rng)
    hiss, pops = render_vinyl(rng)
    music_rms = rms(music)
    murmur = scale_to_rms(murmur, music_rms * 0.18)
    clink_peak = float(np.max(np.abs(clinks)))
    if clink_peak > 0.0:
        clinks *= (music_rms * 2.4) / clink_peak
    hiss = scale_to_rms(hiss, music_rms * 0.030)
    pops = scale_to_rms(pops, music_rms * 0.040)
    stereo = music + murmur + clinks + hiss + pops
    stereo = center_bass(stereo)
    stereo[:, 0] = circular_highpass(stereo[:, 0], 30.0)
    stereo[:, 1] = circular_highpass(stereo[:, 1], 30.0)
    stereo[:, 0] = circular_lowpass(stereo[:, 0], 6400.0)
    stereo[:, 1] = circular_lowpass(stereo[:, 1], 6400.0)
    limit = max(rms(stereo) * 4.2, 1e-6)
    stereo = np.tanh(stereo / limit) * limit
    stereo = center_bass(stereo)
    wow_rate = periodic_rate(0.11)
    left = circular_mod_delay(stereo[:, 0], 24.0, 6.0, wow_rate, 0.5)
    right = circular_mod_delay(stereo[:, 1], 24.0, 6.0, wow_rate, 0.5)
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
        "title=Lo-fi nocturne, guitare jazz et café lointain",
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
