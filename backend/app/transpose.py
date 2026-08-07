"""Pitch-shifted renders so a transposed chart still plays back in tune.

When the user changes the key we must shift the *audio* by the same number of semitones
while keeping the tempo identical — otherwise the beat grid and the playhead drift apart.
ffmpeg's rubberband filter does exactly this (formant-preserving, tempo-locked) and is
much better than the naive asetrate+atempo chain.

Renders are cached on disk per (job, semitones) and produced on demand, so the common
case (no transposition) costs nothing.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)

MIN_SEMITONES = -12
MAX_SEMITONES = 12


class TransposeError(RuntimeError):
    pass


def _has_rubberband() -> bool:
    try:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-filters"],
            capture_output=True, text=True, timeout=30,
        ).stdout
        return "rubberband" in out
    except (OSError, subprocess.SubprocessError):
        return False


_RUBBERBAND = None


def rubberband_available() -> bool:
    global _RUBBERBAND
    if _RUBBERBAND is None:
        _RUBBERBAND = _has_rubberband()
        if not _RUBBERBAND:
            log.warning("ffmpeg has no rubberband filter; using asetrate+atempo fallback")
    return _RUBBERBAND


def _filter_chain(semitones: int) -> str:
    """Build the ffmpeg audio filter for an `n`-semitone shift at constant tempo."""
    ratio = 2.0 ** (semitones / 12.0)

    if rubberband_available():
        return f"rubberband=pitch={ratio:.10f}:pitchq=quality:transients=crisp"

    # Fallback: resample to change pitch (which also changes speed), then correct the
    # speed back with atempo. atempo is only well-behaved in [0.5, 2.0], so chain it.
    stages = [f"asetrate=44100*{ratio:.10f}"]
    remaining = 1.0 / ratio
    while remaining > 2.0:
        stages.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        stages.append("atempo=0.5")
        remaining /= 0.5
    stages.append(f"atempo={remaining:.10f}")
    stages.append("aresample=44100")
    return ",".join(stages)


def render(source_wav: Path, semitones: int, out_dir: Path) -> Path:
    """Return a path to `source_wav` shifted by `semitones`, rendering if not cached.

    Output is MP3 rather than WAV: a 5-minute WAV is ~50 MB and the browser has to
    download it before playback starts, whereas 192 kbps MP3 is ~7 MB and streams.
    """
    if not MIN_SEMITONES <= semitones <= MAX_SEMITONES:
        raise TransposeError(
            f"transposition must be between {MIN_SEMITONES} and {MAX_SEMITONES} semitones"
        )
    if not source_wav.exists():
        raise TransposeError(f"source audio missing: {source_wav}")

    out_dir.mkdir(parents=True, exist_ok=True)
    sign = "p" if semitones >= 0 else "m"
    dest = out_dir / f"audio_{sign}{abs(semitones)}.mp3"

    if dest.exists() and dest.stat().st_size > 0:
        return dest

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source_wav)]
    if semitones != 0:
        cmd += ["-filter:a", _filter_chain(semitones)]
    cmd += ["-c:a", "libmp3lame", "-b:a", "192k", str(dest)]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if result.returncode != 0 or not dest.exists():
        dest.unlink(missing_ok=True)
        raise TransposeError(f"ffmpeg render failed: {result.stderr.strip()[:400]}")

    log.info("rendered %+d semitone version (%.1f MB)", semitones, dest.stat().st_size / 1e6)
    return dest
