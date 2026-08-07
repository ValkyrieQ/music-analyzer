"""Key detection via Krumhansl-Schmuckler style profile correlation.

We correlate the track's average chroma against major/minor tonal profiles at all 12
rotations and take the best fit. The Temperley revision of the original K-S weights is
used because it behaves better on popular music, where the raw Krumhansl weights tend
to over-predict minor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Preferred spellings per key, so we show Bb major rather than A# major.
MAJOR_NAMES = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
MINOR_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "Bb", "B"]

TEMPERLEY_MAJOR = np.array(
    [5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0], dtype=np.float64
)
TEMPERLEY_MINOR = np.array(
    [5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0], dtype=np.float64
)

# Relative-major offsets used to name the scale the UI draws its lanes from.
MAJOR_SCALE = (0, 2, 4, 5, 7, 9, 11)
NATURAL_MINOR_SCALE = (0, 2, 3, 5, 7, 8, 10)


@dataclass
class KeyEstimate:
    tonic: int              # pitch class 0-11
    mode: str               # "major" | "minor"
    name: str               # e.g. "Bb major"
    confidence: float       # 0-1, gap between best and runner-up
    scale_pitches: list[int]
    alternatives: list[dict]


def _correlate(chroma_mean: np.ndarray, profile: np.ndarray) -> np.ndarray:
    """Pearson correlation of the chroma against all 12 rotations of `profile`."""
    scores = np.zeros(12, dtype=np.float64)
    c = chroma_mean - chroma_mean.mean()
    c_norm = np.linalg.norm(c)
    if c_norm < 1e-9:
        return scores

    for tonic in range(12):
        rotated = np.roll(profile, tonic)
        p = rotated - rotated.mean()
        scores[tonic] = float(c @ p / (c_norm * np.linalg.norm(p)))
    return scores


def _pooled_chroma(chroma: np.ndarray) -> np.ndarray:
    """Collapse a chroma matrix to one 12-bin profile for correlation.

    Each frame is L1-normalised before averaging, so every moment of the track gets an
    equal vote regardless of how loud it is. That stops a loud chorus from deciding the
    key on its own.

    Deliberately *not* a median. The median discards pitch classes that appear in only a
    minority of frames — which is exactly what the distinguishing tone of a key usually is.
    In a G major progression like G-C-D-C, F# sounds in a quarter of the bars; taking the
    median erases it and the track reads as C major, whose only difference is F natural.
    """
    if chroma.shape[1] == 0:
        return np.zeros(12)

    sums = chroma.sum(axis=0, keepdims=True)
    # Drop near-silent frames rather than amplifying their noise up to unit sum.
    active = chroma[:, (sums > 1e-6).ravel()]
    if active.shape[1] == 0:
        return np.zeros(12)

    normalised = active / active.sum(axis=0, keepdims=True)
    return normalised.mean(axis=1)


def detect(chroma: np.ndarray) -> KeyEstimate:
    """Estimate the global key from a (12, n_frames) chroma matrix."""
    if chroma.ndim != 2 or chroma.shape[0] != 12:
        raise ValueError(f"expected chroma of shape (12, n), got {chroma.shape}")

    chroma_avg = _pooled_chroma(chroma)

    major_scores = _correlate(chroma_avg, TEMPERLEY_MAJOR)
    minor_scores = _correlate(chroma_avg, TEMPERLEY_MINOR)

    candidates = [
        {"tonic": t, "mode": "major", "score": float(major_scores[t]), "name": f"{MAJOR_NAMES[t]} major"}
        for t in range(12)
    ] + [
        {"tonic": t, "mode": "minor", "score": float(minor_scores[t]), "name": f"{MINOR_NAMES[t]} minor"}
        for t in range(12)
    ]
    candidates.sort(key=lambda c: c["score"], reverse=True)

    best = candidates[0]
    runner_up = candidates[1]
    # Report the margin over the runner-up rather than the raw correlation: a track can
    # correlate 0.9 with two relative keys at once, and that ambiguity is the useful signal.
    confidence = float(np.clip((best["score"] - runner_up["score"]) * 3.0, 0.0, 1.0))

    intervals = MAJOR_SCALE if best["mode"] == "major" else NATURAL_MINOR_SCALE
    scale_pitches = [(best["tonic"] + i) % 12 for i in intervals]

    return KeyEstimate(
        tonic=best["tonic"],
        mode=best["mode"],
        name=best["name"],
        confidence=round(confidence, 3),
        scale_pitches=scale_pitches,
        alternatives=[
            {"name": c["name"], "score": round(c["score"], 4)} for c in candidates[1:5]
        ],
    )


def transpose_key_name(tonic: int, mode: str, semitones: int) -> str:
    """Name the key you land in after shifting by `semitones`."""
    new_tonic = (tonic + semitones) % 12
    names = MAJOR_NAMES if mode == "major" else MINOR_NAMES
    return f"{names[new_tonic]} {mode}"
