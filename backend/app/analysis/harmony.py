"""Generate a harmony part from the isolated vocal stem.

Given the vocal Demucs separated out and the chords the pipeline decoded, this builds
backing vocal lines that sing *with* the lead: for every sung note it works out which
chord is sounding, then moves that note to a different tone of the same chord.

Why chord-aware rather than a fixed interval
--------------------------------------------
The cheap way to fake harmony is to duplicate the vocal shifted by a constant interval —
say four semitones for "a third up". That is wrong roughly half the time: over a C major
chord an E lead harmonises up to G (three semitones), but a C harmonises up to E (four).
A constant shift alternates between right and badly wrong, which is heard as sour rather
than as harmony. So each note is snapped to an actual tone of the chord underneath it.

How it works
------------
1.  `librosa.pyin` tracks the lead's fundamental. It is the slow step (~17s on a 3-minute
    stem) but it is the only way to know what note to harmonise.
2.  Contiguous voiced frames become *notes*, each with one median pitch. Harmonising per
    note rather than per frame is deliberate: a per-frame shift chases vibrato and
    pitch-tracker jitter, and the result warbles. A per-note constant shift keeps the
    lead's own expression intact because the whole note moves together.
3.  For each note, the chord sounding at its midpoint gives a set of pitch classes. The
    harmony note is the chord tone nearest the requested interval away, so the voice
    stays inside the harmony while roughly tracking the melody's contour.
4.  Each note is rendered with `librosa.effects.pitch_shift` and written into an output
    buffer at its original position, with short raised-cosine fades at the edges so the
    joins do not click.

Voices are mixed under the lead, not over it — a harmony louder than the melody stops
sounding like a backing vocal.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# pyin search range. A low bass voice sits near 80 Hz; 1000 Hz is above a soprano's
# fundamental, so anything higher found there is an overtone or a tracking error.
F0_MIN = 80.0
F0_MAX = 1000.0
F0_FRAME = 2048
F0_HOP = 256

# Frames whose voicing probability is below this are treated as unvoiced. pyin marks
# breaths and consonants as weakly voiced with nonsense pitch; harmonising those produces
# audible chirps between words.
VOICED_MIN_PROB = 0.5

# Gaps in the confident-voicing mask up to this long are bridged over rather than treated
# as a break between notes. Measured on real vocal: pyin's voicing probability oscillates
# above and below VOICED_MIN_PROB *during a single sustained note* — vibrato and normal
# pitch wobble repeatedly drag it under 0.5 for a couple of frames at a time, even though a
# listener hears one continuous note. Without bridging, that single note was cut into 5-6
# fragments (measured: 656 raw confident runs on one vocal collapse to 123 once gaps up to
# 150ms are bridged), each rendered as its own independent pitch_shift call with its own
# fade. The result is not "wrong notes", it is the same note stuttering and re-attacking
# every 100-150ms — which is exactly what "ฟังไม่รู้เรื่อง" (unintelligible) sounds like.
MAX_GAP_SECONDS = 0.15

# Shortest note worth harmonising, in seconds. Below this a shifted fragment is a click.
MIN_NOTE_SECONDS = 0.08

# A note's pitch has to be stable to be worth harmonising: if the tracked f0 wanders by
# more than this within one segment, it is a glide or a tracking failure, not a note.
MAX_NOTE_SPREAD_SEMITONES = 2.0

# The harmony voices, as (requested interval in semitones, mix gain). Both sit below the
# lead: a third down and a sixth down is the standard two-part backing arrangement, and
# staying below keeps the melody the highest and therefore most audible line.
DEFAULT_VOICES: tuple[tuple[int, float], ...] = ((-4, 0.5), (-9, 0.35))

# How loud the whole harmony bus sits against the lead vocal.
LEAD_GAIN = 1.0
HARMONY_BUS_GAIN = 0.7


@dataclass
class Note:
    """One sung note: sample range plus its median pitch in MIDI numbers."""

    start: int
    end: int
    midi: float


@dataclass
class HarmonyResult:
    path: Path
    voices: list[int]        # intervals actually rendered
    notes: int               # how many notes were harmonised
    seconds: float           # total harmonised duration


@dataclass
class HarmonyBus:
    """The backing-voice signal alone, before it is mixed into anything.

    Kept separate from the final mixdown because there are two legitimate things to mix
    it into: the isolated vocal (for the standalone download) and the full song (so
    playback can be "the track, plus harmony" rather than "just the a cappella vocal").
    Rendering the bus is the slow part (pitch tracking + per-note shifting); building
    either mixdown from it afterwards is nearly free, so both can be produced without
    tracking the vocal's pitch twice.
    """

    audio: np.ndarray
    voices: list[int]
    notes: int
    seconds: float


def _hann_fade(n: int) -> np.ndarray:
    """Half a raised cosine, for fading a rendered note in or out."""
    if n <= 0:
        return np.ones(0, dtype=np.float32)
    return (0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, n))).astype(np.float32)


def _bridge_gaps(confident: np.ndarray, max_gap_frames: int) -> np.ndarray:
    """Fill short False runs that are flanked by True on both sides.

    A gap at the very start or end of the signal is left alone — there is nothing to
    interpolate between, only a trailing edge to guess at, which is exactly the kind of
    guess that produces the tracking errors this function exists to avoid.
    """
    if max_gap_frames <= 0:
        return confident

    bridged = confident.copy()
    n = bridged.size
    i = 0
    while i < n:
        if bridged[i]:
            i += 1
            continue
        j = i
        while j < n and not bridged[j]:
            j += 1
        if i > 0 and j < n and (j - i) <= max_gap_frames:
            bridged[i:j] = True
        i = j
    return bridged


def _detect_notes(vocal: np.ndarray, sr: int) -> list[Note]:
    """Track the lead's pitch and segment it into notes."""
    import librosa

    f0, voiced, prob = librosa.pyin(
        vocal,
        fmin=F0_MIN,
        fmax=F0_MAX,
        sr=sr,
        frame_length=F0_FRAME,
        hop_length=F0_HOP,
    )

    confident = voiced & (prob >= VOICED_MIN_PROB) & np.isfinite(f0)
    if not confident.any():
        return []

    confident = _bridge_gaps(confident, int(MAX_GAP_SECONDS * sr / F0_HOP))

    # f0 is NaN across a bridged gap (pyin had nothing confident to report there), so
    # interpolate through it linearly in log-frequency — a gap is short by construction
    # (<= MAX_GAP_SECONDS) and mid-note, so the true pitch barely moved; holding it flat
    # or leaving NaN would either introduce a fake portamento or break the median below.
    midi = np.full(f0.shape, np.nan)
    voiced_now = voiced & np.isfinite(f0)
    midi[voiced_now] = librosa.hz_to_midi(f0[voiced_now])
    gap_only = confident & ~voiced_now
    if gap_only.any() and voiced_now.any():
        known_idx = np.flatnonzero(voiced_now)
        gap_idx = np.flatnonzero(gap_only)
        midi[gap_idx] = np.interp(gap_idx, known_idx, midi[known_idx])

    notes: list[Note] = []
    min_frames = max(2, int(MIN_NOTE_SECONDS * sr / F0_HOP))

    # Walk the voiced mask, cutting a new note wherever the pitch jumps by a semitone or
    # more. Without that cut, a whole legato phrase reads as one note and the harmony
    # holds a single pitch under a moving melody.
    start = None
    for i in range(len(confident) + 1):
        ongoing = i < len(confident) and confident[i]
        breaks = (
            ongoing
            and start is not None
            and abs(midi[i] - np.nanmedian(midi[start:i])) >= 1.0
        )

        if ongoing and start is None:
            start = i
        elif start is not None and (not ongoing or breaks):
            segment = midi[start:i]
            if i - start >= min_frames and np.isfinite(segment).any():
                spread = float(np.nanmax(segment) - np.nanmin(segment))
                if spread <= MAX_NOTE_SPREAD_SEMITONES:
                    notes.append(
                        Note(
                            start=int(start * F0_HOP),
                            end=int(i * F0_HOP),
                            midi=float(np.nanmedian(segment)),
                        )
                    )
            start = i if breaks else None

    return notes


def _chord_at(chords: list[dict], time: float) -> dict | None:
    """The chord span containing `time`, or None (also None for a no-chord span)."""
    for span in chords:
        if span["start"] <= time < span["end"]:
            return span if span.get("root") is not None else None
    return None


def _chord_pitch_classes(span: dict) -> list[int]:
    from .chords import CHORD_QUALITIES

    root = span["root"]
    intervals = CHORD_QUALITIES.get(span.get("quality") or "maj", (0, 4, 7))
    return sorted({(root + i) % 12 for i in intervals})


def _harmony_midi(lead_midi: float, interval: int, pitch_classes: list[int]) -> float | None:
    """Snap `lead_midi + interval` to the nearest tone of the chord.

    Searching a window of absolute MIDI notes (rather than doing modular arithmetic on
    pitch classes) keeps the result in a sane octave — a pitch-class-only answer can land
    eleven semitones away from where the voice should be.
    """
    if not pitch_classes:
        return None

    target = lead_midi + interval
    candidates = [
        note
        for note in range(int(target) - 6, int(target) + 7)
        if note % 12 in pitch_classes
    ]
    if not candidates:
        return None

    best = min(candidates, key=lambda note: abs(note - target))
    # Refuse a snap that drags the voice more than a whole tone off the requested
    # interval: at that distance it is no longer the harmony that was asked for, and
    # skipping the note is less noticeable than singing the wrong one.
    if abs(best - target) > 2.0:
        return None
    return float(best)


def _render_voice(
    vocal: np.ndarray,
    sr: int,
    notes: list[Note],
    chords: list[dict],
    interval: int,
) -> tuple[np.ndarray, int]:
    """Render one harmony voice. Returns (audio, notes actually rendered)."""
    import librosa

    out = np.zeros_like(vocal)
    rendered = 0
    fade_len = max(1, int(0.012 * sr))

    for note in notes:
        segment = vocal[note.start:note.end]
        if segment.size < fade_len * 2:
            continue

        midpoint = (note.start + note.end) / 2 / sr
        span = _chord_at(chords, midpoint)
        if span is None:
            continue

        target = _harmony_midi(note.midi, interval, _chord_pitch_classes(span))
        if target is None:
            continue

        steps = target - note.midi
        if abs(steps) < 0.5:      # same note; nothing to add
            continue

        shifted = librosa.effects.pitch_shift(segment, sr=sr, n_steps=steps)
        shifted = shifted[:segment.size]

        # Taper both ends so consecutive notes cross-fade instead of stepping.
        window = np.ones(shifted.size, dtype=np.float32)
        window[:fade_len] = _hann_fade(fade_len)
        window[-fade_len:] = _hann_fade(fade_len)[::-1]

        out[note.start:note.start + shifted.size] += shifted * window
        rendered += 1

    return out, rendered


def build_bus(
    vocal: np.ndarray,
    sr: int,
    chords: list[dict],
    voices: tuple[tuple[int, float], ...] = DEFAULT_VOICES,
) -> HarmonyBus | None:
    """Render the backing-voice signal alone — the slow, pitch-tracking part.

    Returns None when there is nothing to harmonise — an instrumental track, or a stem
    whose pitch could not be tracked. Callers should treat that as normal.
    """
    vocal = np.asarray(vocal, dtype=np.float32)
    if vocal.ndim == 2:
        vocal = vocal.mean(axis=0)

    # An instrumental track still yields a vocals stem — it is just bleed 30-40 dB down.
    # Running pyin over that wastes ~17s to find nothing, so check the level first.
    level = float(np.sqrt(np.mean(np.square(vocal, dtype=np.float64)))) if vocal.size else 0.0
    if level < 1e-3:
        log.info("skipping harmony: vocal stem is essentially silent (rms=%.5f)", level)
        return None

    notes = _detect_notes(vocal, sr)
    if not notes:
        log.info("skipping harmony: no stable pitched notes found in the vocal stem")
        return None

    bus = np.zeros_like(vocal)
    used: list[int] = []
    total_notes = 0

    for interval, gain in voices:
        voice, count = _render_voice(vocal, sr, notes, chords, interval)
        if count == 0:
            continue
        bus += voice * gain
        used.append(interval)
        total_notes += count

    if not used:
        log.info("skipping harmony: no note could be mapped onto a detected chord")
        return None

    seconds = sum(n.end - n.start for n in notes) / sr
    log.info("harmony bus rendered: voices=%s notes=%d (%.1fs of vocal)", used, total_notes, seconds)
    return HarmonyBus(audio=bus, voices=used, notes=total_notes, seconds=round(seconds, 1))


def _encode(mix: np.ndarray, sr: int, out_path: Path, bitrate: str = "192k") -> bool:
    """Peak-guard and encode a mixdown to MP3 at `out_path`. Returns success."""
    import soundfile as sf

    # Only attenuate if we actually clipped; scaling unconditionally would quietly make
    # the mix softer than its inputs.
    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak > 0.99:
        mix = mix * (0.99 / peak)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    wav_path = out_path.with_suffix(".wav")

    try:
        sf.write(wav_path, mix, sr, subtype="PCM_16")
        result = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(wav_path),
                "-c:a", "libmp3lame", "-b:a", bitrate,
                str(out_path),
            ],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0 or not out_path.exists():
            log.warning("mix encode failed: %s", result.stderr[:200])
            return False
        return True
    except Exception as exc:  # noqa: BLE001 - harmony is a bonus, never a job failure
        log.warning("mix encode failed: %s", exc)
        return False
    finally:
        wav_path.unlink(missing_ok=True)


def generate(
    vocal: np.ndarray,
    sr: int,
    chords: list[dict],
    out_dir: Path,
    voices: tuple[tuple[int, float], ...] = DEFAULT_VOICES,
    bitrate: str = "192k",
) -> HarmonyResult | None:
    """Build a lead + harmony mixdown from the vocal stem alone (the standalone download)."""
    vocal = np.asarray(vocal, dtype=np.float32)
    if vocal.ndim == 2:
        vocal = vocal.mean(axis=0)

    bus = build_bus(vocal, sr, chords, voices=voices)
    if bus is None:
        return None

    mix = vocal * LEAD_GAIN + bus.audio * HARMONY_BUS_GAIN
    mp3_path = out_dir / "harmony.mp3"
    if not _encode(mix, sr, mp3_path, bitrate=bitrate):
        return None

    return HarmonyResult(path=mp3_path, voices=bus.voices, notes=bus.notes, seconds=bus.seconds)


def generate_with_track(
    vocal: np.ndarray,
    full_mix: np.ndarray,
    sr: int,
    chords: list[dict],
    out_dir: Path,
    voices: tuple[tuple[int, float], ...] = DEFAULT_VOICES,
    bitrate: str = "192k",
) -> HarmonyResult | None:
    """Build the *whole song* plus the harmony bus, for "listen with harmony" playback.

    Unlike `generate`, the lead vocal is not re-added on top — it is already present in
    `full_mix` (this is the same audio the player normally streams). Adding it again would
    double the vocal and drown the instruments; only the backing-voice bus is new here.
    """
    vocal = np.asarray(vocal, dtype=np.float32)
    if vocal.ndim == 2:
        vocal = vocal.mean(axis=0)
    full_mix = np.asarray(full_mix, dtype=np.float32)
    if full_mix.ndim == 2:
        full_mix = full_mix.mean(axis=0)

    bus = build_bus(vocal, sr, chords, voices=voices)
    if bus is None:
        return None

    n = min(full_mix.size, bus.audio.size)
    mix = full_mix[:n] + bus.audio[:n] * HARMONY_BUS_GAIN
    mp3_path = out_dir / "harmony_with_track.mp3"
    if not _encode(mix, sr, mp3_path, bitrate=bitrate):
        return None

    return HarmonyResult(path=mp3_path, voices=bus.voices, notes=bus.notes, seconds=bus.seconds)
