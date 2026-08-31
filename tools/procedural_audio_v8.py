"""Project-authored procedural audio for the v8 robustness candidate.

This module intentionally has no network, model, corpus, text, MIDI, or sample
input.  Every waveform is derived from numeric seeds with NumPy/SciPy.  The
outputs are synthetic and therefore must never be labelled as real audio.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, lfilter, resample_poly


SAMPLE_RATE = 16_000
GENERATOR_VERSION = "procedural-v8.1"
VOICE_TRAIN_FAMILIES = ("additive_formant", "pulse_lpc", "codec_phase")
VOICE_VALID_FAMILY = "phase_locked_unseen"
MUSIC_TRAIN_FAMILIES = ("additive_band", "fm_percussive", "subtractive_sequence")
MUSIC_VALID_FAMILY = "spectral_block_unseen"


@dataclass(frozen=True)
class AudioStats:
    duration_sec: float
    peak: float
    rms: float
    dc_offset: float
    clipping_fraction: float


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed) & 0xFFFFFFFF)


def _stable_hash(payload: object) -> str:
    packed = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(packed).hexdigest()


def content_hash(kind: str, seed: int, duration_sec: float) -> str:
    """Hash the abstract content, independent of renderer/augmentation."""
    return _stable_hash({
        "kind": kind,
        "content_seed": int(seed),
        "duration_ms": int(round(duration_sec * 1000)),
        "version": GENERATOR_VERSION,
    })


def _normalize(wave: np.ndarray, peak: float = 0.88) -> np.ndarray:
    wave = np.asarray(wave, dtype=np.float64)
    wave -= float(np.mean(wave))
    current = float(np.max(np.abs(wave))) if wave.size else 0.0
    if current < 1e-8:
        raise ValueError("procedural generator produced silence")
    wave *= float(peak) / current
    # A short fade prevents file-boundary clicks without hiding synthetic artifacts.
    fade = min(len(wave) // 8, int(0.012 * SAMPLE_RATE))
    if fade:
        ramp = np.sin(np.linspace(0.0, math.pi / 2.0, fade)) ** 2
        wave[:fade] *= ramp
        wave[-fade:] *= ramp[::-1]
    return np.clip(wave, -0.98, 0.98).astype(np.float32)


def _one_pole_preemphasis(wave: np.ndarray, coefficient: float) -> np.ndarray:
    return lfilter([1.0, -float(coefficient)], [1.0], wave)


def _resonator(wave: np.ndarray, frequency: float, bandwidth: float) -> np.ndarray:
    # Stable second-order resonator expressed through pole radius and angle.
    radius = math.exp(-math.pi * max(20.0, bandwidth) / SAMPLE_RATE)
    theta = 2.0 * math.pi * min(frequency, SAMPLE_RATE * 0.47) / SAMPLE_RATE
    return lfilter([1.0 - radius], [1.0, -2.0 * radius * math.cos(theta), radius * radius], wave)


def _lowpass(wave: np.ndarray, cutoff_hz: float, order: int = 4) -> np.ndarray:
    cutoff = float(np.clip(cutoff_hz, 120.0, SAMPLE_RATE * 0.47))
    b, a = butter(order, cutoff / (SAMPLE_RATE / 2.0), btype="low")
    return lfilter(b, a, wave)


def _bandpass(wave: np.ndarray, low_hz: float, high_hz: float) -> np.ndarray:
    low = max(30.0, float(low_hz)) / (SAMPLE_RATE / 2.0)
    high = min(SAMPLE_RATE * 0.48, float(high_hz)) / (SAMPLE_RATE / 2.0)
    b, a = butter(3, [low, high], btype="band")
    return lfilter(b, a, wave)


def _voice_plan(seed: int, duration_sec: float) -> tuple[np.ndarray, list[dict[str, float]]]:
    rng = _rng(seed)
    frames = max(1, int(round(duration_sec / 0.32)))
    raw = rng.uniform(0.17, 0.62, size=frames)
    raw *= duration_sec / raw.sum()
    # Abstract phonemes, not text or recordings. Values cover varied vocal tracts.
    vowels = np.asarray([
        (320, 820, 2450), (430, 1120, 2550), (570, 1580, 2420),
        (690, 1190, 2580), (790, 1840, 2680), (360, 720, 2200),
    ], dtype=np.float64)
    plan: list[dict[str, float]] = []
    for index, part_duration in enumerate(raw):
        formants = vowels[int(rng.integers(0, len(vowels)))].copy()
        tract_scale = rng.uniform(0.86, 1.17)
        formants *= tract_scale
        plan.append({
            "duration": float(part_duration),
            "f0": float(rng.uniform(82.0, 245.0)),
            "f1": float(formants[0]), "f2": float(formants[1]), "f3": float(formants[2]),
            "voiced": float(rng.random() > 0.17),
            "stress": float(rng.uniform(0.55, 1.0)),
            "index": float(index),
        })
    return raw, plan


def synthesize_voice(content_seed: int, renderer: str, duration_sec: float) -> np.ndarray:
    if renderer not in (*VOICE_TRAIN_FAMILIES, VOICE_VALID_FAMILY):
        raise ValueError(f"unknown voice renderer: {renderer}")
    _, plan = _voice_plan(content_seed, duration_sec)
    rng = _rng(content_seed ^ int(_stable_hash(renderer)[:8], 16))
    chunks: list[np.ndarray] = []
    phase = rng.uniform(0.0, 2.0 * math.pi)
    for phone in plan:
        count = max(64, int(round(phone["duration"] * SAMPLE_RATE)))
        t = np.arange(count, dtype=np.float64) / SAMPLE_RATE
        vibrato_rate = rng.uniform(3.1, 6.7)
        f0 = phone["f0"] * (1.0 + 0.018 * np.sin(2 * math.pi * vibrato_rate * t + phase))
        f0 *= 1.0 + 0.025 * np.linspace(-1.0, 1.0, count)
        instant_phase = phase + 2.0 * math.pi * np.cumsum(f0) / SAMPLE_RATE
        phase = float(instant_phase[-1] % (2.0 * math.pi))
        if phone["voiced"] > 0.5:
            source = np.zeros(count, dtype=np.float64)
            harmonic_count = 20 if renderer != "pulse_lpc" else 28
            for harmonic in range(1, harmonic_count + 1):
                tilt = harmonic ** (-1.15 if renderer == "additive_formant" else -0.92)
                source += tilt * np.sin(harmonic * instant_phase + 0.03 * harmonic * harmonic)
            source += rng.normal(0.0, 0.018, count)
        else:
            source = _bandpass(rng.normal(0.0, 1.0, count), 1400.0, 6900.0)
        if renderer == "phase_locked_unseen":
            # Unseen family: deterministic harmonic locking and quantized F0 contour.
            source = 0.78 * source + 0.22 * np.sin(2.0 * np.round(instant_phase / 2.0))
        filtered = np.zeros_like(source)
        for formant, bandwidth, weight in (
            (phone["f1"], 75.0, 1.0), (phone["f2"], 115.0, 0.72), (phone["f3"], 170.0, 0.42)
        ):
            filtered += weight * _resonator(source, formant, bandwidth)
        attack = min(count // 3, int(0.035 * SAMPLE_RATE))
        release = min(count // 3, int(0.055 * SAMPLE_RATE))
        envelope = np.ones(count, dtype=np.float64) * phone["stress"]
        if attack:
            envelope[:attack] *= np.linspace(0.02, 1.0, attack)
        if release:
            envelope[-release:] *= np.linspace(1.0, 0.02, release)
        chunks.append(filtered * envelope)
    wave = np.concatenate(chunks)
    target_len = int(round(duration_sec * SAMPLE_RATE))
    wave = np.pad(wave[:target_len], (0, max(0, target_len - len(wave))))
    if renderer == "pulse_lpc":
        wave = np.tanh(1.8 * _one_pole_preemphasis(wave, 0.72))
    elif renderer == "codec_phase":
        down = resample_poly(wave, 1, 2)
        down = np.round(np.clip(down, -2.0, 2.0) * 128.0) / 128.0
        wave = resample_poly(down, 2, 1)[:target_len]
        wave = _lowpass(wave, 3650.0)
    elif renderer == "phase_locked_unseen":
        wave = np.tanh(2.35 * wave)
        wave = _lowpass(wave, 6100.0)
    return _normalize(wave, peak=float(rng.uniform(0.72, 0.92)))


def _music_plan(seed: int, duration_sec: float) -> dict[str, object]:
    rng = _rng(seed)
    roots = np.asarray([43, 45, 48, 50, 52, 55, 57], dtype=int)
    root = int(rng.choice(roots))
    scales = ((0, 2, 3, 5, 7, 9, 10), (0, 2, 4, 7, 9), (0, 3, 5, 6, 7, 10))
    scale = np.asarray(scales[int(rng.integers(0, len(scales)))])
    bpm = float(rng.uniform(72.0, 148.0))
    step = 60.0 / bpm / 2.0
    count = int(math.ceil(duration_sec / step))
    degrees = rng.integers(0, len(scale), size=count)
    octaves = rng.choice(np.asarray((0, 0, 0, 12, -12)), size=count)
    midi = root + scale[degrees] + octaves
    velocity = rng.uniform(0.45, 1.0, size=count)
    return {"root": root, "scale": scale.tolist(), "bpm": bpm, "step": step,
            "midi": midi.tolist(), "velocity": velocity.tolist()}


def synthesize_music(content_seed: int, renderer: str, duration_sec: float) -> np.ndarray:
    if renderer not in (*MUSIC_TRAIN_FAMILIES, MUSIC_VALID_FAMILY):
        raise ValueError(f"unknown music renderer: {renderer}")
    plan = _music_plan(content_seed, duration_sec)
    rng = _rng(content_seed ^ int(_stable_hash(renderer)[:8], 16))
    length = int(round(duration_sec * SAMPLE_RATE))
    wave = np.zeros(length, dtype=np.float64)
    step_samples = max(64, int(round(float(plan["step"]) * SAMPLE_RATE)))
    for note_index, (midi, velocity) in enumerate(zip(plan["midi"], plan["velocity"])):
        start = note_index * step_samples
        if start >= length:
            break
        note_len = min(length - start, int(step_samples * rng.uniform(0.72, 1.85)))
        t = np.arange(note_len, dtype=np.float64) / SAMPLE_RATE
        freq = 440.0 * (2.0 ** ((float(midi) - 69.0) / 12.0))
        attack = np.minimum(1.0, t / rng.uniform(0.008, 0.045))
        decay = np.exp(-t / rng.uniform(0.16, 0.8))
        envelope = attack * decay * float(velocity)
        if renderer == "additive_band":
            tone = sum(np.sin(2 * math.pi * freq * h * t + 0.17 * h) / (h ** 1.3) for h in range(1, 9))
        elif renderer == "fm_percussive":
            modulation = 2.2 * np.exp(-4.0 * t) * np.sin(2 * math.pi * freq * 2.01 * t)
            tone = np.sin(2 * math.pi * freq * t + modulation)
            tone += 0.22 * np.sin(2 * math.pi * freq * 0.5 * t)
        elif renderer == "subtractive_sequence":
            phase = 2 * math.pi * freq * t
            tone = 2.0 * ((phase / (2 * math.pi)) % 1.0) - 1.0
            tone = _lowpass(tone, min(5200.0, freq * rng.uniform(5.0, 14.0)))
        else:  # spectral_block_unseen
            bins = np.arange(1, 13, dtype=np.float64)
            phases = rng.uniform(0.0, 2 * math.pi, len(bins))
            tone = np.sum(np.sin(2 * math.pi * freq * bins[:, None] * t + phases[:, None])
                          / np.sqrt(bins[:, None]), axis=0)
            tone *= np.sign(np.sin(2 * math.pi * rng.uniform(7.0, 19.0) * t) + 0.15)
        wave[start:start + note_len] += tone * envelope
    # Original algorithmic percussion; noise is generated here, not sampled.
    beat = max(1, int(round(float(plan["step"]) * 2.0 * SAMPLE_RATE)))
    for start in range(0, length, beat):
        hit_len = min(length - start, int(0.14 * SAMPLE_RATE))
        t = np.arange(hit_len, dtype=np.float64) / SAMPLE_RATE
        noise = rng.normal(0.0, 1.0, hit_len) * np.exp(-32.0 * t)
        kick = np.sin(2 * math.pi * (74.0 * t - 28.0 * t * t)) * np.exp(-19.0 * t)
        wave[start:start + hit_len] += 0.24 * noise + 0.37 * kick
    if renderer == "spectral_block_unseen":
        wave = np.tanh(1.7 * _one_pole_preemphasis(wave, 0.45))
    return _normalize(wave, peak=float(rng.uniform(0.70, 0.91)))


def synthesize_mix(voice_seed: int, music_seed: int, split: str, duration_sec: float,
                   mix_seed: int) -> tuple[np.ndarray, str, str, float]:
    rng = _rng(mix_seed)
    if split == "train":
        voice_renderer = VOICE_TRAIN_FAMILIES[int(rng.integers(0, len(VOICE_TRAIN_FAMILIES)))]
        music_renderer = MUSIC_TRAIN_FAMILIES[int(rng.integers(0, len(MUSIC_TRAIN_FAMILIES)))]
    else:
        voice_renderer = VOICE_VALID_FAMILY
        music_renderer = MUSIC_VALID_FAMILY
    voice = synthesize_voice(voice_seed, voice_renderer, duration_sec)
    music = synthesize_music(music_seed, music_renderer, duration_sec)
    snr_db = float(rng.uniform(-7.0, 7.0))
    voice_rms = float(np.sqrt(np.mean(voice ** 2) + 1e-12))
    music_rms = float(np.sqrt(np.mean(music ** 2) + 1e-12))
    music_gain = voice_rms / max(music_rms, 1e-8) / (10.0 ** (snr_db / 20.0))
    wave = voice.astype(np.float64) + music.astype(np.float64) * music_gain
    return _normalize(wave, peak=float(rng.uniform(0.74, 0.93))), voice_renderer, music_renderer, snr_db


def audio_stats(wave: np.ndarray) -> AudioStats:
    wave = np.asarray(wave, dtype=np.float64)
    return AudioStats(
        duration_sec=float(len(wave) / SAMPLE_RATE),
        peak=float(np.max(np.abs(wave))) if wave.size else 0.0,
        rms=float(np.sqrt(np.mean(wave ** 2))) if wave.size else 0.0,
        dc_offset=float(np.mean(wave)) if wave.size else 0.0,
        clipping_fraction=float(np.mean(np.abs(wave) >= 0.979)) if wave.size else 0.0,
    )


def quality_errors(wave: np.ndarray, expected_duration: float | None = None) -> list[str]:
    stats = audio_stats(wave)
    errors: list[str] = []
    if not np.isfinite(wave).all():
        errors.append("non_finite")
    if stats.duration_sec < 2.5 or stats.duration_sec > 9.0:
        errors.append("duration_out_of_range")
    if expected_duration is not None and abs(stats.duration_sec - expected_duration) > 1.0 / SAMPLE_RATE:
        errors.append("duration_mismatch")
    if stats.rms < 0.025 or stats.rms > 0.55:
        errors.append("rms_out_of_range")
    if stats.peak < 0.55 or stats.peak > 0.981:
        errors.append("peak_out_of_range")
    if abs(stats.dc_offset) > 0.015:
        errors.append("dc_offset")
    if stats.clipping_fraction > 0.002:
        errors.append("clipping")
    return errors
