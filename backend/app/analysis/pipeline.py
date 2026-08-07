"""The analysis pipeline: audio in, chord timeline out.

Stages, in order:

    load -> separate (Demucs) -> rhythm (tempo/beats/meter)
         -> chroma (CQT on the harmonic stem) -> beat-sync
         -> key detection -> chord decoding -> bar grouping

Every stage reports progress through an optional callback so the API can stream a
percentage to the browser; Demucs dominates the wall clock, so it owns most of the range.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import librosa
import numpy as np

from . import chords as chord_mod
from . import key_detect, rhythm, separate

log = logging.getLogger(__name__)

ProgressFn = Callable[[str, float], None]

# Analysis sample rate. 22050 is plenty for chroma (the CQT tops out around 5 kHz here)
# and halves Demucs/CQT cost versus 44.1k.
ANALYSIS_SR = 22050

# CQT settings: 3 octaves starting at C2 (~65 Hz). Going lower picks up kick drum
# fundamentals; going higher adds melody and cymbal wash that muddies the chroma.
CQT_FMIN = librosa.note_to_hz("C2")
CQT_N_OCTAVES = 4
CQT_BINS_PER_OCTAVE = 36
CQT_HOP = 512


@dataclass
class AnalysisResult:
    tempo: float
    beats_per_bar: int
    beat_confidence: float
    key: dict
    beat_times: list[float]
    downbeat_indices: list[int]
    chords: list[dict]           # merged spans
    beat_chords: list[dict]      # one entry per beat
    bars: list[dict]             # bar-grouped view for the chord sheet
    duration: float
    separation_method: str
    # Which stems the chord analysis actually read, best first (e.g. ["piano", "guitar"]).
    harmonic_sources: list[str] = field(default_factory=list)
    stem_levels: dict = field(default_factory=dict)
    stems: list[str] = field(default_factory=list)   # names available for download
    timings: dict = field(default_factory=dict)


def _noop(stage: str, pct: float) -> None:
    return None


def _beat_sync_chroma(
    chroma: np.ndarray,
    beat_times: np.ndarray,
    sr: int,
    hop: int,
    rms: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average chroma between consecutive beats.

    Returns (beat_chroma, kept_beat_times, beat_loudness). Beats whose window is empty are
    dropped so the returned arrays always agree in length.

    `beat_loudness` is the mean of `rms` over the same window. It has to be sampled here,
    alongside the chroma, because the chroma is peak-normalised per frame upstream and
    therefore carries no loudness information at all — the chord decoder needs a separate
    signal to tell a silent beat from a quiet chord.
    """
    beat_frames = librosa.time_to_frames(beat_times, sr=sr, hop_length=hop)
    beat_frames = np.clip(beat_frames, 0, chroma.shape[1] - 1)

    columns: list[np.ndarray] = []
    kept: list[float] = []
    loudness: list[float] = []

    for i, frame in enumerate(beat_frames):
        end = beat_frames[i + 1] if i + 1 < len(beat_frames) else chroma.shape[1]
        if end <= frame:
            end = min(frame + 1, chroma.shape[1])

        window = chroma[:, frame:end]
        if window.size == 0:
            continue

        level_window = rms[frame:end] if rms is not None else None

        # Trim the first ~15% of each beat: transient attack energy (especially drum
        # bleed) is broadband and pollutes the chroma right after the onset.
        if window.shape[1] >= 4:
            cut = max(1, int(window.shape[1] * 0.15))
            window = window[:, cut:]
            if level_window is not None and level_window.size > cut:
                level_window = level_window[cut:]

        # Median over the beat resists a single loud passing note flipping the chord.
        columns.append(np.median(window, axis=1))
        kept.append(float(beat_times[i]))
        loudness.append(
            float(np.mean(level_window)) if level_window is not None and level_window.size else 0.0
        )

    if not columns:
        return np.zeros((12, 0)), np.zeros(0), np.zeros(0)
    return np.stack(columns, axis=1), np.asarray(kept), np.asarray(loudness)


def _refine_downbeat_phase(
    beat_chords: list[chord_mod.BeatChord],
    beats_per_bar: int,
    fallback_phase: int,
) -> int:
    """Pick which beat starts a bar, using chord changes rather than onset energy.

    `rhythm._estimate_meter` has to guess the phase from onset strength alone, because it
    runs before chords are known. That is the weaker cue by a wide margin: a drum kit with
    an even backbeat — kick on 1 and 3, snare on 2 and 4, hats throughout — carries almost
    no accent on the downbeat, and the estimator then latches onto noise. Measured on the
    verification fixture it chose phase 3 over the true phase 0, which pushes every chord
    change onto the *last* beat of a bar in the chord sheet.

    Harmony is the reliable cue. In the popular material this tool targets, chords change
    on the barline, so the phase that collects the most chord changes is the downbeat.
    """
    if beats_per_bar < 2 or len(beat_chords) < beats_per_bar * 2:
        return fallback_phase

    changes = [
        i for i in range(1, len(beat_chords))
        if beat_chords[i].label != beat_chords[i - 1].label
    ]
    if len(changes) < 2:
        # Static harmony (a drone, a one-chord vamp) tells us nothing about the phase.
        return fallback_phase

    counts = np.bincount(
        [i % beats_per_bar for i in changes], minlength=beats_per_bar
    )
    best = int(counts.argmax())

    # Require the winner to be a genuine peak, not a one-change lead over a tie. Without
    # this, syncopated or anticipated changes could rotate the grid on thin evidence.
    ranked = np.sort(counts)[::-1]
    if ranked[0] < ranked[1] * 1.5 and ranked[0] - ranked[1] < 2:
        return fallback_phase

    if best != fallback_phase:
        log.info(
            "downbeat phase corrected %d -> %d from chord changes (%s per phase)",
            fallback_phase, best, counts.tolist(),
        )
    return best


def _group_into_bars(
    beat_chords: list[chord_mod.BeatChord],
    downbeat_indices: list[int],
    beats_per_bar: int,
) -> list[dict]:
    """Slice the beat list into bars at the detected downbeats."""
    if not beat_chords:
        return []

    boundaries = sorted(set(downbeat_indices) | {0})
    boundaries = [b for b in boundaries if b < len(beat_chords)]
    if not boundaries:
        boundaries = [0]

    bars: list[dict] = []
    for bar_no, start_idx in enumerate(boundaries):
        end_idx = (
            boundaries[bar_no + 1] if bar_no + 1 < len(boundaries) else len(beat_chords)
        )
        slice_ = beat_chords[start_idx:end_idx]
        if not slice_:
            continue

        bars.append(
            {
                "index": bar_no,
                "start": round(slice_[0].start, 4),
                "end": round(slice_[-1].end, 4),
                "beats": [
                    {
                        "label": c.label,
                        "root": c.root,
                        "quality": c.quality,
                        "start": round(c.start, 4),
                        "end": round(c.end, 4),
                        "confidence": c.confidence,
                        "beat_index": c.beat_index,
                    }
                    for c in slice_
                ],
            }
        )
    return bars


def analyse(
    wav_path: Path,
    duration: float,
    enable_demucs: bool = True,
    progress: ProgressFn | None = None,
    stem_dir: Path | None = None,
) -> AnalysisResult:
    """Run the full analysis on a normalised WAV file.

    Args:
        stem_dir: when given (and Demucs is enabled), each separated stem is exported
            here as MP3 so the API can offer them for download.
    """
    report = progress or _noop
    timings: dict[str, float] = {}

    def timed(stage: str, pct: float):
        report(stage, pct)
        return time.perf_counter()

    # --- load -------------------------------------------------------------------------
    t0 = timed("loading audio", 2)
    # Demucs is trained at 44.1 kHz stereo and expects to see the real thing. Loading at
    # ANALYSIS_SR and letting the separator upsample back to 44.1k costs *more* time (80s
    # vs 71s on a 181s track — resampling twice is not free) and hands the model a signal
    # with everything above 11 kHz already discarded. So feed it the native file and let
    # it return the working signals at the analysis rate.
    if enable_demucs:
        y_full, full_sr = librosa.load(str(wav_path), sr=None, mono=False)
        if y_full.size == 0:
            raise ValueError("decoded audio is empty")
        y = librosa.resample(
            separate._to_mono(y_full), orig_sr=full_sr, target_sr=ANALYSIS_SR,
        )
        sr = ANALYSIS_SR
    else:
        y, sr = librosa.load(str(wav_path), sr=ANALYSIS_SR, mono=True)
        if y.size == 0:
            raise ValueError("decoded audio is empty")
        y_full, full_sr = y, sr
    timings["load"] = round(time.perf_counter() - t0, 2)

    # --- separation (the expensive stage) ---------------------------------------------
    t0 = timed("separating stems", 8)
    stems = separate.separate(
        y_full,
        full_sr,
        enable_demucs=enable_demucs,
        export_dir=stem_dir if enable_demucs else None,
        out_sr=sr,
    )
    timings["separate"] = round(time.perf_counter() - t0, 2)

    # --- rhythm ----------------------------------------------------------------------
    t0 = timed("tracking tempo and beats", 68)
    rhythm_info = rhythm.analyse(stems.percussive, sr, y_full=y)
    timings["rhythm"] = round(time.perf_counter() - t0, 2)

    # --- chroma ----------------------------------------------------------------------
    t0 = timed("computing chroma", 78)
    cqt = np.abs(
        librosa.cqt(
            y=stems.harmonic,
            sr=sr,
            hop_length=CQT_HOP,
            fmin=CQT_FMIN,
            n_bins=CQT_N_OCTAVES * CQT_BINS_PER_OCTAVE,
            bins_per_octave=CQT_BINS_PER_OCTAVE,
        )
    )

    # Harmonic-energy weighting (CENS-style folding) maps the CQT onto 12 pitch classes
    # while suppressing timbre, which is exactly what template matching wants.
    chroma = librosa.feature.chroma_cqt(
        C=cqt,
        sr=sr,
        bins_per_octave=CQT_BINS_PER_OCTAVE,
        hop_length=CQT_HOP,
    )

    # Non-local filtering + a horizontal median pass: the standard librosa recipe for
    # smoothing chroma without blurring across genuine chord boundaries.
    chroma = np.minimum(
        chroma,
        librosa.decompose.nn_filter(chroma, aggregate=np.median, metric="cosine"),
    )
    chroma = librosa.util.normalize(chroma, norm=np.inf, axis=0)

    # Per-frame loudness of the harmonic signal, on the same frame grid as the chroma.
    # Captured *before* the normalisation above throws loudness away — the chord decoder
    # needs it to tell a silent bar from a quiet one.
    harmonic_rms = librosa.feature.rms(
        y=stems.harmonic, frame_length=CQT_HOP * 2, hop_length=CQT_HOP,
    )[0]
    if harmonic_rms.size < chroma.shape[1]:
        harmonic_rms = np.pad(harmonic_rms, (0, chroma.shape[1] - harmonic_rms.size))
    timings["chroma"] = round(time.perf_counter() - t0, 2)

    # --- key -------------------------------------------------------------------------
    t0 = timed("detecting key", 86)
    key = key_detect.detect(chroma)
    timings["key"] = round(time.perf_counter() - t0, 2)

    # --- chords ----------------------------------------------------------------------
    t0 = timed("recognising chords", 91)
    beat_chroma, beat_times, beat_loudness = _beat_sync_chroma(
        chroma, rhythm_info.beat_times, sr, CQT_HOP, rms=harmonic_rms,
    )
    # Only steer the decoder toward the detected key when that detection itself is
    # confident. A key guessed on thin evidence (an ambiguous or very short track) is as
    # likely to be wrong as right, and biasing the chords toward a wrong key would make
    # them worse, not better — the whole point of this bias is to add real information.
    key_pitches = key.scale_pitches if key.confidence >= 0.3 else None
    beat_chords = chord_mod.recognise(
        beat_chroma, beat_times, duration, beat_loudness=beat_loudness,
        key_pitches=key_pitches,
    )
    merged = chord_mod.merge_repeats(beat_chords)
    timings["chords"] = round(time.perf_counter() - t0, 2)

    # Dropped beats shift the downbeat indices, so recompute them against the kept grid.
    if beat_times.size != rhythm_info.beat_times.size:
        phase = 0
    else:
        phase = rhythm_info.downbeat_indices[0] if rhythm_info.downbeat_indices else 0

    # Now that chords are known, re-derive the bar phase from where the harmony changes.
    phase = _refine_downbeat_phase(beat_chords, rhythm_info.beats_per_bar, phase)
    downbeats = list(range(phase, len(beat_chords), rhythm_info.beats_per_bar))

    bars = _group_into_bars(beat_chords, downbeats, rhythm_info.beats_per_bar)
    report("done", 100)

    log.info(
        "analysis complete: %.1f BPM, key=%s, %d beats, %d chord spans "
        "(%s via %s) timings=%s",
        rhythm_info.tempo, key.name, len(beat_chords), len(merged), stems.method,
        "+".join(stems.harmonic_sources) or "n/a", timings,
    )

    return AnalysisResult(
        tempo=rhythm_info.tempo,
        beats_per_bar=rhythm_info.beats_per_bar,
        beat_confidence=rhythm_info.beat_confidence,
        key={
            "tonic": key.tonic,
            "mode": key.mode,
            "name": key.name,
            "confidence": key.confidence,
            "scale_pitches": key.scale_pitches,
            "alternatives": key.alternatives,
        },
        beat_times=[round(float(t), 4) for t in beat_times],
        downbeat_indices=downbeats,
        chords=merged,
        beat_chords=[
            {
                "label": c.label,
                "root": c.root,
                "quality": c.quality,
                "start": round(c.start, 4),
                "end": round(c.end, 4),
                "confidence": c.confidence,
                "beat_index": c.beat_index,
            }
            for c in beat_chords
        ],
        bars=bars,
        duration=round(duration, 3),
        separation_method=stems.method,
        harmonic_sources=stems.harmonic_sources,
        stem_levels=stems.stem_levels,
        stems=sorted(stems.stem_paths),
        timings=timings,
    )
