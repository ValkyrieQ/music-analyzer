"""Tempo, beat and downbeat estimation.

librosa's beat tracker gives us beat positions and a tempo estimate. On top of that we
infer the meter and which beats are downbeats, because the UI groups chords into bars
and needs to know where each bar starts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import librosa
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class RhythmInfo:
    tempo: float                 # BPM
    beat_times: np.ndarray       # (n_beats,) seconds
    downbeat_indices: list[int]  # indices into beat_times that start a bar
    beats_per_bar: int
    beat_confidence: float


def _fix_tempo_octave(
    onset_env: np.ndarray,
    beat_times: np.ndarray,
    sr: int,
    hop_length: int = 512,
) -> np.ndarray:
    """Halve the beat period when the tracker locked onto every *other* beat.

    Beat trackers routinely settle an octave low — reporting 70 BPM for a 140 BPM track —
    because a half-speed grid still lines up with real onsets. The tell is that the
    *midpoints* between the reported beats also land on strong onsets. When they do, the
    true beat is the subdivision, so we interleave the midpoints back in.

    This matters beyond the BPM readout: a half-rate grid gives the chord decoder half as
    many observations and makes the UI draw half as many cells, so chord changes on the
    missing beats are invisible.
    """
    if beat_times.size < 4:
        return beat_times

    def strength_at(times: np.ndarray, tolerance: int = 2) -> float:
        """Peak onset strength near each time, then the median across times.

        The tolerance window matters: onset peaks are only a frame or two wide, and beat
        times land a little off the exact peak. Sampling single frames reads a real onset
        as near-zero whenever it is a frame out, which is enough to invert this test.
        """
        frames = librosa.time_to_frames(times, sr=sr, hop_length=hop_length)
        frames = np.clip(frames, 0, len(onset_env) - 1)
        if frames.size == 0:
            return 0.0
        peaks = [
            onset_env[max(0, f - tolerance):min(len(onset_env), f + tolerance + 1)].max()
            for f in frames
        ]
        return float(np.median(peaks))

    midpoints = (beat_times[:-1] + beat_times[1:]) / 2.0
    on_beat = strength_at(beat_times)
    off_beat = strength_at(midpoints)

    if on_beat <= 1e-9:
        return beat_times

    # Measured separation is wide: a genuinely half-rate grid scores ~0.5 here, while
    # correctly-tracked material scores ~0.0 (its midpoints fall in the gaps between hits).
    # 0.33 sits in the empty middle of that range.
    if off_beat / on_beat < 0.33:
        return beat_times

    interleaved = np.empty(beat_times.size + midpoints.size, dtype=beat_times.dtype)
    interleaved[0::2] = beat_times
    interleaved[1::2] = midpoints

    log.info(
        "doubling beat grid: midpoint onset strength %.3f vs %.3f on-beat",
        off_beat, on_beat,
    )
    return interleaved


def _estimate_meter(onset_env: np.ndarray, beat_frames: np.ndarray) -> tuple[int, int]:
    """Guess (beats_per_bar, phase) by testing which grouping lands on strong onsets.

    Downbeats carry more onset energy than other beats. For each candidate meter we try
    every phase offset and score the mean onset strength on the implied downbeats; the
    best-scoring combination wins.
    """
    if beat_frames.size < 8:
        return 4, 0

    strengths = onset_env[np.clip(beat_frames, 0, len(onset_env) - 1)]
    if strengths.size == 0 or not np.any(strengths):
        return 4, 0

    best = (4, 0, -np.inf)
    for meter in (4, 3, 6):
        for phase in range(meter):
            downbeat_strength = strengths[phase::meter]
            others = np.delete(strengths, np.arange(phase, strengths.size, meter))
            if downbeat_strength.size < 2 or others.size < 2:
                continue

            # Contrast, not absolute level: a loud song should not read as 6/8 just
            # because it has more energy overall.
            score = float(downbeat_strength.mean() - others.mean())
            # Mild bias toward 4/4, which dominates the material this tool targets.
            if meter == 4:
                score *= 1.12
            if score > best[2]:
                best = (meter, phase, score)

    return best[0], best[1]


def analyse(y_percussive: np.ndarray, sr: int) -> RhythmInfo:
    """Track tempo and beats from the percussive component of a mix.

    Args:
        y_percussive: mono signal — ideally drums-only or HPSS-percussive, which makes
            the onset envelope far cleaner than the full mix.
    """
    onset_env = librosa.onset.onset_strength(y=y_percussive, sr=sr, aggregate=np.median)

    tempo_arr, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env,
        sr=sr,
        trim=False,           # keep beats near the very start/end of the file
        units="frames",
    )
    tempo = float(np.atleast_1d(tempo_arr)[0])
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    # Fall back to a synthetic grid if tracking fails (very sparse or ambient material),
    # so downstream stages always have something to align to.
    if beat_times.size < 2:
        log.warning("beat tracking found %d beats; falling back to a fixed grid", beat_times.size)
        tempo = tempo if tempo > 1 else 120.0
        duration = len(y_percussive) / sr
        beat_times = np.arange(0.0, duration, 60.0 / tempo)
        beat_frames = librosa.time_to_frames(beat_times, sr=sr)

    # Correct an octave-low lock before anything downstream depends on the grid.
    beat_times = _fix_tempo_octave(onset_env, beat_times, sr)
    beat_frames = librosa.time_to_frames(beat_times, sr=sr)

    beats_per_bar, phase = _estimate_meter(onset_env, np.asarray(beat_frames))
    downbeats = list(range(phase, len(beat_times), beats_per_bar))

    # Steadiness of the inter-beat interval, as a proxy for how much to trust the grid.
    intervals = np.diff(beat_times)
    if intervals.size and intervals.mean() > 0:
        jitter = float(intervals.std() / intervals.mean())
        confidence = float(np.clip(1.0 - jitter * 2.5, 0.0, 1.0))
    else:
        confidence = 0.0

    # Prefer the median inter-beat interval over the tracker's reported tempo: it is the
    # tempo actually implied by the beats we are about to draw.
    if intervals.size:
        median_interval = float(np.median(intervals))
        if median_interval > 0:
            tempo = 60.0 / median_interval

    return RhythmInfo(
        tempo=round(tempo, 2),
        beat_times=beat_times,
        downbeat_indices=downbeats,
        beats_per_bar=beats_per_bar,
        beat_confidence=round(confidence, 3),
    )
