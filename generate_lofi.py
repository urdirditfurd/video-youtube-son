#!/usr/bin/env python3
"""Piste lo-fi originale de cette exécution.

Variables tirées avant la synthèse :
- Instrument : guitare jazz clean avec chorus
- Foley : crépitement de cheminée
- Couleur : mystérieux et nocturne (midnight coding)
- Tempo : 78 BPM

Le master musical dure exactement 4 minutes (78 mesures). Trois secondes
du début sont recopiées à la fin pour qu'un fondu linéaire de 3 secondes
reconstruise le signal à l'identique. Soixante-quinze lectures donnent
75 × 4 min + 3 s = 5 h 00 min 03 s, sans couture audible.
"""

from __future__ import annotations

import argparse
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from scipy.signal import lfilter

TRIGGER_UTC = (2026, 10, 19, 18, 46, 7)
INSTRUMENT = "Guitare jazz clean avec chorus"
FOLEY = "Crépitement de cheminée"
MOOD = "Mystérieux et nocturne (midnight coding)"
BPM = 78
SEED = 1_389_239_843
SLUG = "Guitare_Jazz"
PROMPT = (
    "Genre: Lo-fi hip hop, chillhop. BPM: 78. "
    "Mood: Mystérieux et nocturne (midnight coding), peaceful study vibes. "
    "Instrumentation: Guitare jazz clean avec chorus, mellow bass lines, "
    "soft brushed drums (SP-404 style), subtle vinyl crackle. "
    "Background: Crépitement de cheminée. "
    "Characteristics: Slow evolving progression, NO sudden changes, "
    "consistent warm atmosphere, slight human imperfections, "
    "melancholic but comforting. "
    "Technical: Warm analog saturation, low-pass filtered highs, seamless transitions."
)

SR = 32_000
BEAT_S = 60.0 / BPM
BAR_BEATS = 4
N_BARS = 78
P = int(round(N_BARS * BAR_BEATS * BEAT_S * SR))
FADE = int(round(3.0 * SR))
LOOPS = 75
SWING = 0.60
PEAK_TARGET = 10 ** (-2.5 / 20.0)

# Cycle de 6 mesures, répété 13 fois : le B7 retombe sur Em7 à la boucle.
CHORDS = (
    (64, 67, 71, 74),  # Em7
    (59, 64, 67, 71),  # Em
    (57, 64, 67, 72),  # Am7
    (60, 64, 67, 69),  # Am7
    (60, 64, 67, 71),  # Cmaj7
    (59, 63, 66, 71),  # B7
)
BASS_ROOTS = (40, 40, 45, 43, 36, 35)
MELODY_INTERVALS = (0, 2, 3, 5, 7, 8, 10)
MELODY_ROOT = 64

GAIN_CHORD = 0.20
GAIN_MELODY = 0.15
GAIN_BASS = 0.36
GAIN_KICK = 0.36
GAIN_SNARE = 0.24
GAIN_HAT = 0.075
GAIN_PAD = 0.045
DRUM_LEVEL = 0.68


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


def place_circular(buffer: np.ndarray, start: int, signal: np.ndarray) -> None:
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


def pan(mono: np.ndarray, position: float) -> tuple[np.ndarray, np.ndarray]:
    angle = (position + 1.0) * 0.25 * np.pi
    return np.cos(angle) * mono, np.sin(angle) * mono


def circular_lowpass(samples: np.ndarray, cutoff: float, poles: int = 1) -> np.ndarray:
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


def circular_highpass(samples: np.ndarray, cutoff: float) -> np.ndarray:
    return samples - circular_lowpass(samples, cutoff)


def circular_noise(length: int, rng: np.random.Generator, fade: int) -> np.ndarray:
    raw = rng.standard_normal(length + fade)
    mixed = raw[:length].copy()
    theta = np.linspace(0.0, np.pi / 2.0, fade, endpoint=False)
    mixed[:fade] = raw[:fade] * np.sin(theta) + raw[length : length + fade] * np.cos(theta)
    return mixed


def circular_mod_delay(
    signal: np.ndarray,
    base: float,
    depth: float,
    rate: float,
    phase: float,
) -> np.ndarray:
    count = signal.shape[0]
    times = np.arange(count, dtype=np.float64) / SR
    delay = np.clip(base + depth * np.sin(2.0 * np.pi * rate * times + phase), 1.0, SR * 0.05)
    position = np.mod(np.arange(count, dtype=np.float64) - delay, count)
    index = np.floor(position).astype(np.int32)
    fraction = position - index
    nxt = (index + 1) % count
    return signal[index] * (1.0 - fraction) + signal[nxt] * fraction


def dark_room(signal: np.ndarray) -> np.ndarray:
    wet = np.zeros_like(signal)
    for delay_s, gain in ((0.029, 0.20), (0.043, 0.15), (0.067, 0.10), (0.097, 0.06)):
        wet += gain * np.roll(signal, int(delay_s * SR))
    return circular_lowpass(wet, 2200.0)


def synth_guitar(
    freq: float,
    velocity: float,
    rng: np.random.Generator,
    seconds: float,
    decay: float,
) -> np.ndarray:
    count = int(seconds * SR)
    times = np.arange(count, dtype=np.float64) / SR
    partials = (1.0, 0.46, 0.20, 0.09, 0.04, 0.016)
    signal = np.zeros(count, dtype=np.float64)
    for index, amplitude in enumerate(partials, start=1):
        detune = 1.0 + float(rng.uniform(-0.0006, 0.0006))
        envelope = np.exp(-times * (decay + index * 0.95))
        signal += amplitude * np.sin(2.0 * np.pi * freq * index * detune * times) * envelope
    signal *= 1.0 - np.exp(-times / 0.006)
    pick = min(count, int(0.012 * SR))
    signal[:pick] += rng.standard_normal(pick) * np.exp(-np.arange(pick) / (SR * 0.0035)) * 0.07
    return np.tanh(signal * 1.35) * velocity


def synth_bass(freq: float, velocity: float) -> np.ndarray:
    count = int(1.15 * SR)
    times = np.arange(count, dtype=np.float64) / SR
    signal = (
        np.sin(2.0 * np.pi * freq * times)
        + 0.22 * np.sin(2.0 * np.pi * freq * 2.0 * times)
        + 0.04 * np.sin(2.0 * np.pi * freq * 3.0 * times)
    )
    signal *= (1.0 - np.exp(-times / 0.018)) * np.exp(-times / 0.72)
    return np.tanh(signal * 1.25) * velocity * GAIN_BASS


def synth_kick(velocity: float, rng: np.random.Generator) -> np.ndarray:
    count = int(0.48 * SR)
    times = np.arange(count, dtype=np.float64) / SR
    frequency = 42.0 + 78.0 * np.exp(-times * 18.0)
    phase = 2.0 * np.pi * np.cumsum(frequency) / SR
    body = np.sin(phase) * np.exp(-times * 5.4) * (1.0 - np.exp(-times / 0.006))
    click = rng.standard_normal(count) * np.exp(-times * 180.0) * 0.04
    return np.tanh((body + click) * 1.5) * velocity * GAIN_KICK


def synth_brush(velocity: float, rng: np.random.Generator) -> np.ndarray:
    count = int(0.62 * SR)
    times = np.arange(count, dtype=np.float64) / SR
    noise = rng.standard_normal(count) * np.exp(-times * 6.4)
    body = np.sin(2.0 * np.pi * 168.0 * times) * np.exp(-times * 14.0) * 0.28
    return np.tanh((noise * 0.78 + body) * 1.6) * velocity * GAIN_SNARE


def synth_hat(velocity: float, rng: np.random.Generator, open_hat: bool) -> np.ndarray:
    count = int((0.14 if open_hat else 0.045) * SR)
    times = np.arange(count, dtype=np.float64) / SR
    decay = 10.0 if open_hat else 46.0
    return rng.standard_normal(count) * np.exp(-times * decay) * velocity * GAIN_HAT


def synth_pad(freq: float) -> np.ndarray:
    count = int(2.4 * SR)
    times = np.arange(count, dtype=np.float64) / SR
    envelope = (1.0 - np.exp(-times / 0.12)) * np.exp(-times / 1.35)
    signal = np.sin(2.0 * np.pi * freq * times) + 0.06 * np.sin(2.0 * np.pi * freq * 2.0 * times)
    return signal * envelope * GAIN_PAD


def melody_scale() -> list[int]:
    notes: list[int] = []
    for octave in range(0, 2):
        for interval in MELODY_INTERVALS:
            note = MELODY_ROOT + octave * 12 + interval
            if 64 <= note <= 81:
                notes.append(note)
    return notes


def add_guitar(
    chords: np.ndarray,
    melody: np.ndarray,
    rng: np.random.Generator,
) -> None:
    scale = melody_scale()
    degree = len(scale) // 2
    for bar in range(N_BARS):
        slot = bar % len(CHORDS)
        strum = 0.0
        for note in CHORDS[slot]:
            strum += float(rng.uniform(0.008, 0.016))
            cents = float(rng.uniform(-4.0, 4.0))
            freq = midi_to_hz(note) * (2.0 ** (cents / 1200.0))
            velocity = float(rng.uniform(0.62, 0.84))
            place_circular(
                chords,
                sample_at(bar, strum + float(rng.uniform(-0.004, 0.008))),
                synth_guitar(freq, velocity * GAIN_CHORD, rng, 3.6, 1.15),
            )
        if rng.random() < 0.30:
            continue
        center = 4.0 + 1.4 * np.sin(2.0 * np.pi * bar / N_BARS)
        degree = int(np.clip(degree + int(rng.choice([-2, -1, 1, 1, 2])), 0, len(scale) - 1))
        if abs(degree - center) > 2.5:
            degree += int(np.sign(center - degree))
            degree = int(np.clip(degree, 0, len(scale) - 1))
        beat = float(rng.choice([0.5, 1.0, 1.5, 2.0]))
        notes_in_bar = int(rng.choice([2, 2, 3]))
        for _ in range(notes_in_bar):
            degree = int(np.clip(degree + int(rng.choice([-2, -1, 1, 1, 2])), 0, len(scale) - 1))
            cents = float(rng.uniform(-5.0, 5.0))
            freq = midi_to_hz(scale[degree]) * (2.0 ** (cents / 1200.0))
            velocity = float(rng.uniform(0.48, 0.74))
            start = sample_at(bar, swung(beat) + float(rng.uniform(-0.008, 0.012)))
            if rng.random() < 0.14:
                grace = midi_to_hz(scale[max(0, degree - 1)])
                place_circular(
                    melody,
                    start - int(0.045 * SR),
                    synth_guitar(grace, velocity * 0.35 * GAIN_MELODY, rng, 0.28, 3.2),
                )
            place_circular(melody, start, synth_guitar(freq, velocity * GAIN_MELODY, rng, 2.1, 1.7))
            beat += float(rng.choice([1.0, 1.5, 2.0]))
            if beat > 3.4:
                break


def add_bass(buffer: np.ndarray, rng: np.random.Generator) -> None:
    for bar in range(N_BARS):
        root = BASS_ROOTS[bar % len(BASS_ROOTS)]
        fifth = root + 7
        if fifth > 50:
            fifth -= 12
        events = ((0.0, root, 0.92), (2.0, fifth if rng.random() < 0.75 else root, 0.58))
        for beat, note, velocity in events:
            cents = float(rng.uniform(-3.0, 3.0))
            freq = midi_to_hz(note) * (2.0 ** (cents / 1200.0))
            place_circular(
                buffer,
                sample_at(bar, beat + float(rng.uniform(-0.008, 0.014))),
                synth_bass(freq, velocity * float(rng.uniform(0.94, 1.0))),
            )


def add_drums(
    kick: np.ndarray,
    snare: np.ndarray,
    hat: np.ndarray,
    rng: np.random.Generator,
) -> None:
    for bar in range(N_BARS):
        kicks = (0.0, 2.0) if bar % 8 != 4 else (0.0, 2.0, 2.5)
        for beat in kicks:
            velocity = float(rng.uniform(0.78, 0.96)) * DRUM_LEVEL
            if beat == 0.0:
                velocity *= 1.06
            place_circular(
                kick,
                sample_at(bar, swung(beat) + float(rng.uniform(-0.008, 0.012))),
                synth_kick(velocity, rng),
            )
        for beat in (1.0, 3.0):
            place_circular(
                snare,
                sample_at(bar, beat + float(rng.uniform(0.0, 0.016))),
                synth_brush(float(rng.uniform(0.58, 0.80)) * DRUM_LEVEL, rng),
            )
        if rng.random() < 0.55:
            ghost = float(rng.choice([0.75, 2.75, 3.5]))
            place_circular(snare, sample_at(bar, ghost), synth_brush(0.18 * DRUM_LEVEL, rng))
        for eighth in range(8):
            if eighth % 2 == 1 and rng.random() < 0.10:
                continue
            beat = swung(eighth * 0.5)
            velocity = 0.55 if eighth % 4 == 0 else 0.40 if eighth % 2 == 0 else 0.24
            velocity *= float(rng.uniform(0.86, 1.04)) * DRUM_LEVEL
            place_circular(
                hat,
                sample_at(bar, beat),
                synth_hat(velocity, rng, open_hat=eighth == 6 and rng.random() < 0.18),
            )


def add_pad(left: np.ndarray, right: np.ndarray) -> None:
    for bar in range(N_BARS):
        start = sample_at(bar, 0.0)
        for index, note in enumerate(CHORDS[bar % len(CHORDS)]):
            while note > 67:
                note -= 12
            freq = midi_to_hz(note)
            place_circular(left, start, synth_pad(freq * (2.0 ** (-5.0 / 1200.0))))
            place_circular(right, start, synth_pad(freq * (2.0 ** (5.0 / 1200.0))))
            if index == 3:
                break


def chorus_guitar(dry: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rate_left = periodic_rate(0.23)
    rate_right = periodic_rate(0.31)
    left_wet = circular_mod_delay(dry, 0.018 * SR, 0.0045 * SR, rate_left, 0.4)
    right_wet = circular_mod_delay(dry, 0.024 * SR, 0.0055 * SR, rate_right, 1.8)
    room = dark_room(dry)
    left = dry * 0.56 + left_wet * 0.46 + room * 0.55
    right = dry * 0.40 + right_wet * 0.58 + room * 0.55
    return left, right


def render_fire(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    fade = int(0.08 * SR)
    rumble = circular_lowpass(circular_noise(P, rng, fade), 160.0)
    rumble = circular_highpass(rumble, 35.0)
    times = np.arange(P, dtype=np.float64) / SR
    breath = 0.78 + 0.22 * np.sin(2.0 * np.pi * periodic_rate(0.0125) * times)
    breath *= 0.90 + 0.10 * np.sin(2.0 * np.pi * periodic_rate(0.07) * times + 0.6)
    rumble *= breath

    crackle_l = np.zeros(P, dtype=np.float64)
    crackle_r = np.zeros(P, dtype=np.float64)
    sparks = int(11 * P / SR)
    positions = rng.integers(0, P, sparks)
    for position in positions:
        amp = float(rng.uniform(0.08, 1.0) ** 3)
        length = int(rng.uniform(0.002, 0.018) * SR)
        envelope = np.exp(-np.arange(length) / (SR * float(rng.uniform(0.0012, 0.004))))
        burst = rng.standard_normal(length) * envelope * amp
        burst_l, burst_r = pan(burst, float(rng.uniform(-0.85, 0.85)))
        place_circular(crackle_l, int(position), burst_l)
        place_circular(crackle_r, int(position), burst_r)
    clusters = int(P / SR / 3.2)
    for _ in range(clusters):
        center = int(rng.integers(0, P))
        for _spark in range(int(rng.integers(3, 7))):
            offset = int(rng.uniform(0.0, 0.07) * SR)
            length = int(rng.uniform(0.004, 0.03) * SR)
            envelope = np.exp(-np.arange(length) / (SR * 0.004))
            burst = rng.standard_normal(length) * envelope * float(rng.uniform(0.35, 1.0))
            burst_l, burst_r = pan(burst, float(rng.uniform(-0.5, 0.5)))
            place_circular(crackle_l, center + offset, burst_l)
            place_circular(crackle_r, center + offset, burst_r)
    crackle_l = circular_highpass(circular_lowpass(crackle_l, 6200.0), 700.0)
    crackle_r = circular_highpass(circular_lowpass(crackle_r, 6200.0), 700.0)
    crackle_l = np.tanh(crackle_l * 2.4)
    crackle_r = np.tanh(crackle_r * 2.4)
    return rumble, np.column_stack((crackle_l, crackle_r))


def render_vinyl(rng: np.random.Generator) -> np.ndarray:
    fade = int(0.05 * SR)
    hiss_l = circular_highpass(circular_noise(P, rng, fade), 2200.0)
    hiss_r = circular_highpass(circular_noise(P, rng, fade), 2200.0)
    crackle_l = np.zeros(P, dtype=np.float64)
    crackle_r = np.zeros(P, dtype=np.float64)
    pops = int(4.2 * P / SR)
    for position in rng.integers(0, P, pops):
        length = int(rng.uniform(0.0015, 0.008) * SR)
        envelope = np.exp(-np.arange(length) / (SR * 0.0016))
        burst = rng.standard_normal(length) * envelope * float(rng.uniform(0.2, 1.0) ** 2)
        burst_l, burst_r = pan(burst, float(rng.uniform(-0.4, 0.4)))
        place_circular(crackle_l, int(position), burst_l)
        place_circular(crackle_r, int(position), burst_r)
    crackle_l = circular_highpass(crackle_l, 900.0)
    crackle_r = circular_highpass(crackle_r, 900.0)
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


def rms(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(signal), dtype=np.float64)))


def scale_to_rms(signal: np.ndarray, target: float) -> np.ndarray:
    current = rms(signal)
    if current < 1e-9:
        return signal
    return signal * (target / current)


def render_period(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    chords = np.zeros(P, dtype=np.float64)
    melody = np.zeros(P, dtype=np.float64)
    bass = np.zeros(P, dtype=np.float64)
    kick = np.zeros(P, dtype=np.float64)
    snare = np.zeros(P, dtype=np.float64)
    hat = np.zeros(P, dtype=np.float64)
    pad_l = np.zeros(P, dtype=np.float64)
    pad_r = np.zeros(P, dtype=np.float64)

    add_guitar(chords, melody, rng)
    add_bass(bass, rng)
    add_drums(kick, snare, hat, rng)
    add_pad(pad_l, pad_r)

    hat = circular_highpass(hat, 2800.0)
    snare = circular_lowpass(snare, 3800.0)
    bass = circular_lowpass(bass, 420.0)
    guitar_l, guitar_r = chorus_guitar(chords + melody)
    times = np.arange(P, dtype=np.float64) / SR
    breath = 1.0 + 0.07 * np.sin(2.0 * np.pi * times / (P / SR))
    guitar_l *= breath
    guitar_r *= breath

    hat_l, hat_r = pan(hat, 0.46)
    snare_l, snare_r = pan(snare, -0.08)
    left = guitar_l + bass + kick + snare_l + hat_l + pad_l
    right = guitar_r + bass + kick + snare_r + hat_r + pad_r
    music = np.column_stack((left, right))
    music = center_bass(music)
    for cutoff in (5000.0, 5000.0):
        music[:, 0] = circular_lowpass(music[:, 0], cutoff)
        music[:, 1] = circular_lowpass(music[:, 1], cutoff)
    music = np.tanh(music * 1.15) / np.tanh(1.15)

    rumble, crackle = render_fire(rng)
    hiss, pops = render_vinyl(rng)
    music_rms = rms(music)
    fire = np.column_stack((crackle[:, 0] + rumble, crackle[:, 1] + rumble))
    fire = scale_to_rms(fire, music_rms * 0.30)
    fire = np.tanh(fire / max(rms(fire) * 3.2, 1e-6)) * max(rms(fire) * 3.2, 1e-6)
    hiss = scale_to_rms(hiss, music_rms * 0.04)
    pops = scale_to_rms(pops, music_rms * 0.08)
    stereo = music + fire + hiss + pops
    stereo = center_bass(stereo)
    stereo[:, 0] = circular_highpass(stereo[:, 0], 24.0)
    stereo[:, 1] = circular_highpass(stereo[:, 1], 24.0)
    stereo[:, 0] = circular_lowpass(stereo[:, 0], 7200.0)
    stereo[:, 1] = circular_lowpass(stereo[:, 1], 7200.0)
    limit = max(rms(stereo) * 4.45, 1e-6)
    stereo = np.tanh(stereo / limit) * limit
    wow_rate = periodic_rate(0.14)
    left = circular_mod_delay(stereo[:, 0], 36.0, 10.0, wow_rate, 0.2)
    right = circular_mod_delay(stereo[:, 1], 36.0, 10.0, wow_rate, 0.2)
    stereo = np.column_stack((left, right))
    peak = float(np.max(np.abs(stereo)))
    if peak > 0.0:
        stereo *= PEAK_TARGET / peak
    return stereo.astype(np.float32)


def seam_jump(stereo: np.ndarray) -> float:
    return float(np.max(np.abs(stereo[0] - stereo[-1])))


def seam_is_clean(stereo: np.ndarray) -> bool:
    steps = np.abs(np.diff(stereo, axis=0))
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
        "title=Lo-fi nocturne, guitare jazz et cheminée",
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
    tail = np.ascontiguousarray(period[:FADE])
    blend = np.ascontiguousarray(linear_crossfade(tail, head))
    if float(np.max(np.abs(blend - head))) > 1e-5:
        process.kill()
        raise RuntimeError("Le fondu de 3 secondes n'est pas transparent.")
    started = time.perf_counter()
    try:
        process.stdin.write(np.ascontiguousarray(period).tobytes())
        for index in range(1, LOOPS):
            process.stdin.write(blend.tobytes())
            process.stdin.write(np.ascontiguousarray(period[FADE:]).tobytes())
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
    print(PROMPT, flush=True)
    print(
        f"Instrument: {INSTRUMENT} | Foley: {FOLEY} | Humeur: {MOOD} | BPM: {BPM}",
        flush=True,
    )
    started = time.perf_counter()
    period = render_period(SEED)
    stats = measure(period)
    duration = P / SR
    print(
        f"Master musical: {duration:.3f} s, crête {stats['peak_db']:.2f} dBFS, "
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
