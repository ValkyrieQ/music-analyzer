#!/usr/bin/env python3
"""Generate a synthetic test track with known tempo, key and chord progression.

Having ground truth matters: on real music you can only eyeball whether the chords "look
about right", which is not a test. Here we know the answer, so `verify.py` can assert on it.

The signal is deliberately not a pure tone stack — it has a bass line, a chord voicing with
some harmonics, and a drum-ish click track, so that beat tracking and separation have
something realistic to work with.

    python make_test_audio.py out.wav --tempo 120 --key C --progression I,V,vi,IV
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 44100
PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NOTE_ALIASES = {"Db": 1, "Eb": 3, "Gb": 6, "Ab": 8, "Bb": 10}

MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]

# Roman numeral -> (scale degree index, quality). Upper case major, lower case minor.
DEGREES = {
    "I": (0, "maj"), "ii": (1, "min"), "iii": (2, "min"), "IV": (3, "maj"),
    "V": (4, "maj"), "vi": (5, "min"), "vii": (6, "dim"),
    "i": (0, "min"), "II": (1, "maj"), "III": (2, "maj"), "iv": (3, "min"),
    "v": (4, "min"), "VI": (5, "maj"), "VII": (6, "maj"),
}

INTERVALS = {"maj": [0, 4, 7], "min": [0, 3, 7], "dim": [0, 3, 6], "7": [0, 4, 7, 10]}


def note_to_pc(name: str) -> int:
    if name in NOTE_ALIASES:
        return NOTE_ALIASES[name]
    if name in PITCH_CLASSES:
        return PITCH_CLASSES.index(name)
    raise ValueError(f"unknown note name: {name}")


def midi_to_hz(midi: float) -> float:
    return 440.0 * 2.0 ** ((midi - 69) / 12.0)


def voice(freq: float, dur: float, amp: float, harmonics: int = 4) -> np.ndarray:
    """A plucked-ish tone: a few harmonics with an exponential decay envelope."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    signal = np.zeros(n)
    for h in range(1, harmonics + 1):
        # 1/h amplitude rolloff approximates a real instrument's spectrum well enough
        # for the CQT to see a convincing pitch.
        signal += np.sin(2 * np.pi * freq * h * t) / h
    envelope = np.exp(-t * 2.2) * (1 - np.exp(-t * 260))
    return signal * envelope * amp


def drum_hit(kind: str, amp: float, seed: int) -> np.ndarray:
    """One drum sound. `kind` is "kick", "snare" or "hat".

    All three carry a broadband transient, not just the snare. That matters: HPSS and
    Demucs route a purely tonal kick into the *harmonic* stem, so a kick with no noise
    component would vanish from the percussive signal that beat tracking runs on.
    """
    dur = {"kick": 0.20, "snare": 0.16, "hat": 0.06}[kind]
    n = int(dur * SR)
    t = np.arange(n) / SR
    noise = np.random.default_rng(seed).standard_normal(n)

    if kind == "kick":
        body = np.sin(2 * np.pi * 58 * t * np.exp(-t * 3)) * np.exp(-t * 24)
        return (body * 0.7 + noise * 0.35 * np.exp(-t * 180)) * amp
    if kind == "snare":
        body = np.sin(2 * np.pi * 190 * t) * np.exp(-t * 40)
        return (noise * np.exp(-t * 42) + body * 0.35) * amp
    return noise * np.exp(-t * 150) * amp  # hat: short, bright


def build(tempo: float, tonic: int, progression: list[str], bars: int, beats_per_bar: int):
    beat_dur = 60.0 / tempo
    bar_dur = beat_dur * beats_per_bar
    total = int(bars * bar_dur * SR) + SR
    mix = np.zeros(total)

    truth = []

    for bar in range(bars):
        numeral = progression[bar % len(progression)]
        degree_idx, quality = DEGREES[numeral]
        root_pc = (tonic + MAJOR_SCALE[degree_idx]) % 12
        start = bar * bar_dur

        truth.append(
            {
                "bar": bar,
                "start": round(start, 4),
                "numeral": numeral,
                "root": root_pc,
                "quality": quality,
                "label": PITCH_CLASSES[root_pc] + ("m" if quality == "min" else ""),
            }
        )

        # Bass: root on beat 1 and 3, an octave below the chord voicing.
        bass_midi = 36 + root_pc
        for beat in (0, 2):
            if beat >= beats_per_bar:
                continue
            offset = int((start + beat * beat_dur) * SR)
            tone = voice(midi_to_hz(bass_midi), beat_dur * 1.9, 0.34, harmonics=3)
            mix[offset:offset + len(tone)] += tone[: max(0, total - offset)]

        # Chord: held triad in the octave above, re-struck each bar.
        for interval in INTERVALS[quality]:
            midi = 60 + (root_pc + interval) % 12
            offset = int(start * SR)
            tone = voice(midi_to_hz(midi), bar_dur * 1.05, 0.15)
            mix[offset:offset + len(tone)] += tone[: max(0, total - offset)]

        # A basic backbeat: kick on 1 and 3, snare on 2 and 4, hat on every beat.
        #
        # Amplitudes are kept close together on purpose. An exaggerated downbeat accent
        # makes the beat tracker lock onto a half-rate grid (it reads the accent pattern as
        # the beat), which is a property of the synthetic signal, not of real drumming.
        # The hat on every beat is what keeps all beats comparably strong; the small
        # accent that remains is still enough for meter/downbeat detection.
        for beat in range(beats_per_bar):
            offset = int((start + beat * beat_dur) * SR)
            layers = [drum_hit("hat", 0.20 if beat == 0 else 0.16, seed=100 + beat)]
            if beat % 2 == 0:
                layers.append(drum_hit("kick", 0.42 if beat == 0 else 0.38, seed=200 + beat))
            else:
                layers.append(drum_hit("snare", 0.34, seed=300 + beat))

            for hit in layers:
                mix[offset:offset + len(hit)] += hit[: max(0, total - offset)]

    peak = np.abs(mix).max()
    if peak > 0:
        mix = mix / peak * 0.85
    return mix.astype(np.float32), truth


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tempo", type=float, default=120.0)
    parser.add_argument("--key", default="C")
    parser.add_argument("--progression", default="I,V,vi,IV")
    parser.add_argument("--bars", type=int, default=16)
    parser.add_argument("--beats-per-bar", type=int, default=4)
    args = parser.parse_args()

    tonic = note_to_pc(args.key)
    progression = [p.strip() for p in args.progression.split(",") if p.strip()]
    for numeral in progression:
        if numeral not in DEGREES:
            print(f"unknown numeral '{numeral}'", file=sys.stderr)
            return 2

    audio, truth = build(args.tempo, tonic, progression, args.bars, args.beats_per_bar)
    sf.write(str(args.output), audio, SR)

    meta = {
        "tempo": args.tempo,
        "key": f"{PITCH_CLASSES[tonic]} major",
        "tonic": tonic,
        "beats_per_bar": args.beats_per_bar,
        "bars": args.bars,
        "progression": progression,
        "chords": truth,
        "duration": round(len(audio) / SR, 3),
    }
    args.output.with_suffix(".truth.json").write_text(json.dumps(meta, indent=2))

    print(f"wrote {args.output} ({meta['duration']}s) and {args.output.with_suffix('.truth.json')}")
    print(f"  tempo={args.tempo} key={meta['key']} progression={'-'.join(progression)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
