"""Source separation with Demucs.

Chord recognition is much cleaner on an isolated harmonic mix. Vocals smear the chroma
with melody and vibrato; drums add broadband noise that the CQT reads as pitch content.
So we split the track and rebuild two working signals:

* **harmonic**   -> feeds chroma / chord / key detection
* **percussive** -> feeds tempo / beat tracking

The harmonic signal is not simply "everything but drums and vocals". `htdemucs_6s` gives
dedicated `piano` and `guitar` stems, and a chord is most legible in whichever instrument
is actually stating it: a piano voicing spells the whole chord, a guitar part spells most
of it, and bass alone gives only the root. `_pick_harmonic` therefore weights the stems by
how much harmonic material each one really contains, rather than summing them blindly —
adding a near-silent piano stem's noise floor to a loud guitar part only dilutes the chroma.

Demucs is optional at runtime: if the model is unavailable (no weights cached, out of
memory) we fall back to librosa's HPSS, which is far cheaper but noticeably worse on
dense mixes. The API reports which path was taken so the UI can say so.
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# htdemucs_6s adds `guitar` and `piano` to the v4 four-stem set. Measured in this image it
# costs the same wall-clock as plain htdemucs (15.8s for a 33s clip, either way) because
# both are a single transformer pass — the extra stems are just more output channels. So
# the two extra stems are effectively free, and they are what makes piano-first chord
# analysis possible at all.
#
# "htdemucs_ft" is a little cleaner but bags 4 sub-models, so ~4x the runtime.
DEMUCS_MODEL = os.environ.get("DEMUCS_MODEL", "htdemucs_6s")

# Chunk overlap for split inference. The default 0.25 spends 25% of the compute on
# re-processing audio to smooth chunk seams; 0.1 is measurably faster (60s vs 71s on a
# 181s track) and the seams are inaudible in a signal we only ever take chroma from.
DEMUCS_OVERLAP = float(os.environ.get("DEMUCS_OVERLAP", "0.1"))

# Priority for the chord-bearing signal, best first. The user's requested order: piano
# states the fullest voicing, guitar next, bass only implies the root.
HARMONIC_PRIORITY = ("piano", "guitar", "other", "bass")

# A stem counts as "present" if its RMS is at least this fraction of the loudest
# harmonic stem's. Demucs never returns true silence for an absent instrument — it
# returns bleed and noise floor, typically 20-40 dB down — so a plain "is it non-zero"
# test would treat every track as having piano.
STEM_PRESENCE_RATIO = 0.12

# Stems offered to the user for download.
DOWNLOADABLE = ("vocals", "drums", "bass", "guitar", "piano", "other")


@dataclass
class Stems:
    harmonic: np.ndarray    # mono, sr
    percussive: np.ndarray  # mono, sr
    sr: int
    method: str             # "demucs" | "hpss"
    # Which stems fed the harmonic signal, loudest first, for the UI to report.
    harmonic_sources: list[str] = field(default_factory=list)
    # Per-stem RMS relative to the loudest harmonic stem; diagnostic, and what the
    # priority decision was made on.
    stem_levels: dict[str, float] = field(default_factory=dict)
    # name -> path of exported audio, populated only when `export_dir` is given.
    stem_paths: dict[str, str] = field(default_factory=dict)


def _to_mono(x: np.ndarray) -> np.ndarray:
    return x if x.ndim == 1 else x.mean(axis=0)


def _rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def _hpss_fallback(y: np.ndarray, sr: int, reason: str) -> Stems:
    import librosa

    log.warning("falling back to HPSS separation: %s", reason)
    y_harm, y_perc = librosa.effects.hpss(_to_mono(y))
    return Stems(
        harmonic=y_harm,
        percussive=y_perc,
        sr=sr,
        method="hpss",
        harmonic_sources=["hpss"],
    )


def _pick_harmonic(mono: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str], dict[str, float]]:
    """Build the chord-bearing signal from whichever harmonic stems are really present.

    Returns (signal, contributing stem names loudest-first, per-stem relative levels).

    The priority order is respected but not exclusive. Taking *only* the piano throws away
    a guitar part that states the same harmony and would have reinforced it; summing
    everything drags in stems that are pure bleed. So each present stem is included, scaled
    down as it goes further down the priority list — the top stem at unity, the rest at a
    discount — which keeps the leading instrument dominant in the chroma while still
    letting a second one confirm it.
    """
    levels = {name: _rms(sig) for name, sig in mono.items()}
    harmonic_peak = max((levels.get(n, 0.0) for n in HARMONIC_PRIORITY), default=0.0)

    relative = {
        name: (level / harmonic_peak if harmonic_peak > 0 else 0.0)
        for name, level in levels.items()
    }

    present = [
        name for name in HARMONIC_PRIORITY
        if name in mono and relative.get(name, 0.0) >= STEM_PRESENCE_RATIO
    ]

    if not present:
        # Nothing cleared the bar (a very quiet or a cappella track). Fall back to the
        # full non-percussive residue rather than returning silence.
        present = [n for n in HARMONIC_PRIORITY if n in mono]

    signal = np.zeros_like(next(iter(mono.values())))
    for rank, name in enumerate(present):
        # 1.0, 0.6, 0.36, 0.216 ... : each step down the priority list is worth less.
        weight = 0.6 ** rank
        stem = mono[name]
        # Normalise each stem to comparable loudness first, or a hot guitar stem
        # overwhelms a correctly-prioritised but quieter piano.
        level = levels.get(name, 0.0)
        if level > 0:
            signal = signal + stem * (weight / level)

    return signal, present, {k: round(v, 4) for k, v in sorted(relative.items())}


def _export_stems(
    stems: dict[str, np.ndarray],
    sr: int,
    export_dir: Path,
    bitrate: str = "192k",
) -> dict[str, str]:
    """Write each stem to MP3 for download. Best-effort: a failure is logged, not raised.

    MP3 rather than WAV because these are downloads over HTTP — a 5-minute stereo WAV is
    ~50 MB per stem and six of those is a 300 MB job directory, versus ~7 MB each.
    """
    import soundfile as sf

    export_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    for name in DOWNLOADABLE:
        if name not in stems:
            continue
        data = stems[name]
        # soundfile wants (samples, channels).
        frames = data.T if data.ndim == 2 else data
        wav_path = export_dir / f"{name}.wav"
        mp3_path = export_dir / f"{name}.mp3"
        try:
            sf.write(wav_path, frames, sr, subtype="PCM_16")
            result = subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", str(wav_path),
                    "-c:a", "libmp3lame", "-b:a", bitrate,
                    str(mp3_path),
                ],
                capture_output=True, text=True, timeout=600,
            )
            if result.returncode != 0 or not mp3_path.exists():
                log.warning("stem export failed for %s: %s", name, result.stderr[:200])
                continue
            paths[name] = str(mp3_path)
        except Exception as exc:  # noqa: BLE001 - a missing download must not fail the job
            log.warning("stem export failed for %s: %s", name, exc)
        finally:
            wav_path.unlink(missing_ok=True)

    log.info("exported %d stem(s) to %s", len(paths), export_dir)
    return paths


def _resample(x: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return x
    import librosa

    return librosa.resample(x, orig_sr=orig_sr, target_sr=target_sr)


def _degraded(y: np.ndarray, sr: int, target_sr: int, reason: str) -> Stems:
    """HPSS fallback, delivered at the caller's requested sample rate."""
    stems = _hpss_fallback(y, sr, reason)
    if target_sr != sr:
        stems.harmonic = _resample(stems.harmonic, sr, target_sr)
        stems.percussive = _resample(stems.percussive, sr, target_sr)
        stems.sr = target_sr
    return stems


def separate(
    y: np.ndarray,
    sr: int,
    enable_demucs: bool = True,
    threads: int | None = None,
    export_dir: Path | None = None,
    out_sr: int | None = None,
) -> Stems:
    """Split `y` into harmonic and percussive working signals.

    Args:
        y: mono or (channels, samples) float32 audio.
        sr: sample rate of `y`.
        enable_demucs: set False to force the cheap HPSS path.
        threads: torch intra-op threads. Only useful as a *reduction*: the effective
            ceiling is set by OMP_NUM_THREADS before the process starts (see
            backend/entrypoint.sh), because OpenMP reads it at library init.
        export_dir: when given, write each stem here as MP3 for download.
        out_sr: sample rate for the returned working signals; defaults to `sr`. The
            caller wants these at its analysis rate, and resampling here saves a
            round trip through the full-rate signal.
    """
    target_sr = out_sr or sr

    if not enable_demucs:
        return _degraded(y, sr, target_sr, "demucs disabled by request")

    try:
        import torch
        from demucs.apply import apply_model
        from demucs.pretrained import get_model
    except ImportError as exc:
        return _degraded(y, sr, target_sr, f"demucs/torch not importable ({exc})")

    try:
        if threads:
            torch.set_num_threads(threads)
        log.info(
            "demucs starting: model=%s threads=%d overlap=%.2f",
            DEMUCS_MODEL, torch.get_num_threads(), DEMUCS_OVERLAP,
        )

        model = get_model(DEMUCS_MODEL)
        model.eval()

        # Demucs expects (batch, channels, samples) at its own sample rate, and is
        # trained on stereo — feeding it mono duplicated across channels is the
        # documented way to handle mono input.
        wav = y if y.ndim == 2 else np.stack([y, y])
        if wav.shape[0] == 1:
            wav = np.repeat(wav, 2, axis=0)

        if sr != model.samplerate:
            import librosa

            wav = np.stack([
                librosa.resample(ch, orig_sr=sr, target_sr=model.samplerate) for ch in wav
            ])
            work_sr = model.samplerate
        else:
            work_sr = sr

        tensor = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32))

        # Demucs is sensitive to input level; the reference implementation normalises by
        # the mixture's std and rescales the stems afterwards.
        ref = tensor.mean(0)
        mean, std = float(ref.mean()), float(ref.std()) or 1.0
        tensor = (tensor - mean) / std

        with torch.no_grad():
            sources = apply_model(
                model,
                tensor[None],
                device="cpu",
                shifts=0,          # shift trick doubles runtime for a marginal gain
                split=True,        # chunked inference keeps peak RAM bounded
                overlap=DEMUCS_OVERLAP,
                progress=False,
            )[0]

        sources = sources * std + mean
        stems = {name: sources[i].numpy() for i, name in enumerate(model.sources)}

        stem_paths: dict[str, str] = {}
        if export_dir is not None:
            stem_paths = _export_stems(stems, work_sr, export_dir)

        mono = {name: _to_mono(data) for name, data in stems.items()}
        harmonic, used, levels = _pick_harmonic(mono)
        percussive = mono.get("drums", np.zeros_like(harmonic))

        harmonic = _resample(harmonic, work_sr, target_sr)
        percussive = _resample(percussive, work_sr, target_sr)

        log.info(
            "demucs separation complete (model=%s harmonic=%s levels=%s)",
            DEMUCS_MODEL, "+".join(used), levels,
        )
        return Stems(
            harmonic=harmonic,
            percussive=percussive,
            sr=target_sr,
            method="demucs",
            harmonic_sources=used,
            stem_levels=levels,
            stem_paths=stem_paths,
        )

    except Exception as exc:  # noqa: BLE001 - degrade instead of failing the whole job
        return _degraded(y, sr, target_sr, f"demucs failed: {type(exc).__name__}: {exc}")
