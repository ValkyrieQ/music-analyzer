"""Chord recognition: beat-synchronous chroma -> template matching -> Viterbi decoding.

The pipeline mirrors the classic Mauch/Harte approach that tools like Chordify build on:

1.  Take a log-frequency (CQT) spectrogram of the *harmonic* part of the mix.
2.  Fold it into a 12-bin chroma vector per frame, then average chroma **between
    beats** so that one observation == one beat. Chords change on beats, so this
    both denoises and aligns the output to the grid the UI draws.
3.  Score every beat against a dictionary of chord templates.
4.  Decode the most likely chord *sequence* with Viterbi, using a transition
    matrix that penalises changing chord. Frame-wise argmax flickers badly;
    the smoothing prior is what makes the output look like real chord charts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


# --------------------------------------------------------------------------------------
# Chord dictionary
# --------------------------------------------------------------------------------------
# Semitone offsets from the root for each supported quality. Order matters only for
# readability; `QUALITY_PRIORS` below is what biases the decoder toward common chords.
CHORD_QUALITIES: dict[str, tuple[int, ...]] = {
    "maj": (0, 4, 7),
    "min": (0, 3, 7),
    "7": (0, 4, 7, 10),
    "maj7": (0, 4, 7, 11),
    "min7": (0, 3, 7, 10),
    "dim": (0, 3, 6),
    "aug": (0, 4, 8),
    "sus2": (0, 2, 7),
    "sus4": (0, 5, 7),
    "6": (0, 4, 7, 9),
    "min6": (0, 3, 7, 9),
}

# How chord *labels* are rendered in the UI. "maj" is implicit (C, not Cmaj).
QUALITY_SUFFIX = {
    "maj": "",
    "min": "m",
    "7": "7",
    "maj7": "maj7",
    "min7": "m7",
    "dim": "dim",
    "aug": "aug",
    "sus2": "sus2",
    "sus4": "sus4",
    "6": "6",
    "min6": "m6",
}

# Prior log-probability bonus per quality. Triads dominate real music; without this the
# richer templates (which contain more non-zero bins) win too often and every chord ends
# up a 7th or a sus. Tuned by ear against pop/rock material.
QUALITY_PRIORS = {
    "maj": 0.85,
    "min": 0.80,
    "7": 0.30,
    "min7": 0.30,
    "maj7": 0.18,
    "sus4": 0.05,
    "sus2": 0.00,
    "6": -0.05,
    "min6": -0.20,
    "dim": -0.25,
    "aug": -0.55,
}

NO_CHORD = "N"


@dataclass(frozen=True)
class ChordVocab:
    """The decoder's label set plus the matching template matrix."""

    labels: list[str]           # e.g. ["N", "C", "Cm", "C7", ...]
    roots: list[int | None]     # pitch class per label, None for "N"
    qualities: list[str | None]
    templates: np.ndarray       # (n_labels, 12), L2-normalised
    priors: np.ndarray          # (n_labels,) additive log-prob bonus


def build_vocab() -> ChordVocab:
    labels: list[str] = [NO_CHORD]
    roots: list[int | None] = [None]
    qualities: list[str | None] = [None]
    rows: list[np.ndarray] = []
    priors: list[float] = []

    # "No chord" template: flat. A silent/percussive beat matches this better than any
    # specific triad, which is what lets intros and breaks come out blank.
    rows.append(np.full(12, 1.0 / np.sqrt(12.0)))
    priors.append(0.0)

    for quality, intervals in CHORD_QUALITIES.items():
        for root in range(12):
            vec = np.zeros(12, dtype=np.float64)
            for i, interval in enumerate(intervals):
                # Weight the root and third/fifth more than added tones: extensions are
                # often quiet or absent in a real recording, so treating them as
                # mandatory makes the template too brittle.
                vec[(root + interval) % 12] = 1.0 if i < 3 else 0.55
            vec /= np.linalg.norm(vec)

            labels.append(PITCH_CLASSES[root] + QUALITY_SUFFIX[quality])
            roots.append(root)
            qualities.append(quality)
            rows.append(vec)
            priors.append(QUALITY_PRIORS[quality])

    return ChordVocab(
        labels=labels,
        roots=roots,
        qualities=qualities,
        templates=np.vstack(rows),
        priors=np.asarray(priors, dtype=np.float64),
    )


VOCAB = build_vocab()


# --------------------------------------------------------------------------------------
# Observation likelihoods
# --------------------------------------------------------------------------------------
def _emission_logprob(
    chroma: np.ndarray,
    vocab: ChordVocab,
    loudness: np.ndarray | None = None,
) -> np.ndarray:
    """Score each beat (column of `chroma`) against each chord template.

    Args:
        chroma: (12, n_beats) non-negative beat-synchronous chroma.
        loudness: optional (n_beats,) per-beat RMS of the *audio*, used to decide which
            beats are genuinely silent. Without it, no beat is treated as silent.

    Returns:
        (n_beats, n_labels) log-probabilities, each row normalised via log-softmax.
    """
    if chroma.ndim != 2 or chroma.shape[0] != 12:
        raise ValueError(f"expected chroma of shape (12, n), got {chroma.shape}")

    # L2-normalise each beat so loud and quiet sections are scored on equal footing;
    # cosine similarity against the (also normalised) templates then lives in [0, 1].
    norms = np.linalg.norm(chroma, axis=0, keepdims=True)
    unit = chroma / np.maximum(norms, 1e-9)
    similarity = unit.T @ vocab.templates.T          # (n_beats, n_labels)

    # Sharpen before the softmax: raw cosine scores sit in a narrow band and would give
    # an almost-uniform posterior, letting the transition prior override the audio.
    logits = similarity * 11.0 + vocab.priors

    # Silence detection has to come from the audio's own loudness, not from the chroma.
    #
    # This used to test the chroma column norms against a fraction of their median, which
    # cannot work: the pipeline runs `librosa.util.normalize(norm=inf, axis=0)` over the
    # chroma, forcing every column's peak bin to 1.0 regardless of how loud that beat was.
    # Column norms then sit in a band as narrow as 1.01-1.54 (measured over 362 beats), so a
    # 0.12*median threshold of ~0.13 never fired on any beat — while a silent passage, whose
    # noise floor gets normalised up to full scale like everything else, produced a *flat*
    # chroma that matches the flat "N" template better than any triad. The gate was both
    # dead and inverted: it never suppressed a real chord, and it never had to, because
    # silence was already winning "N" on template shape alone.
    #
    # That is the visible bug: a beat that is quiet-but-harmonic (a sustained pad, a
    # decaying piano chord, a breakdown) normalises into a near-flat chroma too, so it also
    # matched "N" and rendered as an empty cell where the music plainly changes chord.
    #
    # With real loudness available, "N" is reserved for beats that are actually silent and
    # actively discouraged everywhere else.
    if loudness is not None and loudness.size == logits.shape[0] and loudness.size:
        finite = loudness[loudness > 0]
        reference = float(np.percentile(finite, 90)) if finite.size else 0.0
        if reference > 0:
            relative = loudness / reference
            # Below -40 dB relative to the track's loud passages there is nothing to name.
            logits[relative < 0.01, 0] += 6.0
            # Everywhere else, "N" must beat the best triad by a real margin to win.
            logits[relative >= 0.01, 0] -= 4.0
    else:
        # No loudness reference: still refuse to prefer "N" on shape alone, since a flat
        # chroma is as likely to be a dense voicing as an empty bar.
        logits[:, 0] -= 2.0

    logits -= logits.max(axis=1, keepdims=True)
    return logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))


# How strongly a change is discouraged the further its root sits from the previous
# chord on the circle of fifths. Real progressions overwhelmingly move by a fifth or
# stay put; a root change a tritone away is the rarest move in tonal music. Distance is
# normalised to [0, 1] (0 = same root, 1 = a tritone away) before this weight is applied,
# so the number below is directly the log-probability cost of the single worst jump.
#
# Tuned against the synthetic fixtures, not guessed: the first value tried (1.2) pushed a
# vi-IV-I-V progression's opening F#m to be decoded as its relative major D instead (they
# share two of three chord tones, so the emission alone barely favours one), because a
# large circle-of-fifths bonus for "the rest of the track turns out to move by fifths a
# lot" outweighed that thin emission evidence. 0.3 fixed it and cost nothing on the other
# two fixtures — the structural prior should nudge a close call, not overrule the audio.
COF_WEIGHT = 0.3

# Bonus for moving *into* a chord whose root is in the track's detected key. Applied only
# to the destination root, since a transition's plausibility is about where you're going,
# not where you came from — modulating out of the home key briefly is normal, but the
# decoder should need real evidence for it rather than drifting there for free.
DIATONIC_BONUS = 0.15


def _transition_logprob(
    vocab: ChordVocab,
    self_bonus: float = 3.4,
    key_pitches: frozenset[int] | None = None,
) -> np.ndarray:
    """Change penalty shaped by music theory, not just a flat "stay" bonus.

    Real chord progressions are not a uniform random walk over the label set: they move
    by a fifth far more often than a tritone, and they gravitate toward the tonal centre
    the rest of the track establishes. A transition matrix that only rewards staying put
    (the previous version) removes flicker but treats every *change* as equally likely,
    so a wrong chord next door on the circle of fifths and a wrong chord a tritone away
    cost the decoder the same — the acknowledged weak point of this decoder before this
    change. Weighting by circle-of-fifths distance and by the detected key directly
    targets that, using signal (the key) the pipeline already computes for free.

    Args:
        key_pitches: pitch classes of the track's detected key/scale. When given, moving
            to a diatonic root gets a bonus. Omit to fall back to circle-of-fifths
            shaping alone (used by tests that decode chroma with no key context).
    """
    n_labels = len(vocab.labels)
    trans = np.zeros((n_labels, n_labels), dtype=np.float64)

    roots = np.array([r if r is not None else -1 for r in vocab.roots])
    has_root = roots >= 0

    # Steps around the circle of fifths from root i to root j. 7 is invertible mod 12
    # (7*7 = 49 = 1 mod 12), so multiplying the semitone difference by 7 converts "steps
    # of a semitone" into "steps of a fifth" without a lookup table.
    semitone_diff = roots[None, :] - roots[:, None]
    fifths_steps = np.mod(semitone_diff * 7, 12)
    cof_distance = np.minimum(fifths_steps, 12 - fifths_steps)  # 0..6

    both_rooted = has_root[:, None] & has_root[None, :]
    trans = np.where(both_rooted, -COF_WEIGHT * cof_distance / 6.0, 0.0)

    if key_pitches:
        diatonic = np.isin(roots, list(key_pitches))
        trans += np.where(both_rooted & diatonic[None, :], DIATONIC_BONUS, 0.0)

    np.fill_diagonal(trans, self_bonus)
    trans -= np.log(np.exp(trans).sum(axis=1, keepdims=True))
    return trans


def _viterbi(emission: np.ndarray, transition: np.ndarray) -> np.ndarray:
    """Standard Viterbi in log space. Returns the best label index per observation."""
    n_obs, n_labels = emission.shape
    if n_obs == 0:
        return np.empty(0, dtype=int)

    delta = emission[0].copy()
    psi = np.zeros((n_obs, n_labels), dtype=np.int32)

    for t in range(1, n_obs):
        # scores[i, j] = best path ending in i at t-1, then i -> j
        scores = delta[:, None] + transition
        psi[t] = np.argmax(scores, axis=0)
        delta = scores[psi[t], np.arange(n_labels)] + emission[t]

    path = np.zeros(n_obs, dtype=int)
    path[-1] = int(np.argmax(delta))
    for t in range(n_obs - 1, 0, -1):
        path[t - 1] = psi[t, path[t]]
    return path


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------
@dataclass
class BeatChord:
    """One decoded beat. Times are seconds; `root`/`quality` are None for "N"."""

    start: float
    end: float
    label: str
    root: int | None
    quality: str | None
    confidence: float
    beat_index: int


def recognise(
    beat_chroma: np.ndarray,
    beat_times: np.ndarray,
    audio_duration: float,
    beat_loudness: np.ndarray | None = None,
    key_pitches: list[int] | None = None,
) -> list[BeatChord]:
    """Decode a chord label for every beat.

    Args:
        beat_chroma: (12, n_beats) beat-synchronous chroma.
        beat_times:  (n_beats,) beat onsets in seconds, ascending.
        audio_duration: total track length, used to close the final beat's interval.
        beat_loudness: (n_beats,) per-beat RMS of the harmonic signal. Supplying it is
            what lets "no chord" be reserved for genuinely silent beats.
        key_pitches: pitch classes of the track's detected key/scale, from
            `key_detect.detect(...).scale_pitches`. Biases the transition matrix toward
            chords in that key — see `_transition_logprob`.
    """
    n_beats = beat_chroma.shape[1]
    if n_beats == 0 or beat_times.size == 0:
        return []
    if beat_times.size != n_beats:
        raise ValueError(
            f"beat_times ({beat_times.size}) must match beat_chroma columns ({n_beats})"
        )

    emission = _emission_logprob(beat_chroma, VOCAB, loudness=beat_loudness)
    transition = _transition_logprob(
        VOCAB,
        key_pitches=frozenset(key_pitches) if key_pitches else None,
    )
    path = _viterbi(emission, transition)

    # Per-beat confidence for the UI, as the chosen label's share against its closest
    # rival: 0.5 at a dead tie, approaching 1.0 when nothing else comes close.
    #
    # Deliberately *not* the raw posterior. With 133 labels, probability mass spreads over
    # the relatives, inversions-as-other-roots and added-tone variants that any real chord
    # partly matches, so the raw figure tops out very low — measured on a track whose chords
    # were all correct, it ranged 0.03–0.36 with a median of 0.16. Read as a percentage that
    # says "16% certain", which is misleading, and it made the UI's 0.35 threshold mark
    # 99.6% of beats as unsure. A pairwise margin is scale-free and stays interpretable
    # however large the vocabulary grows.
    posterior = np.exp(emission)
    posterior /= posterior.sum(axis=1, keepdims=True)

    # Measured against the *decoded* label, not the frame-wise argmax: Viterbi legitimately
    # overrides the local best to keep the sequence coherent, and scoring the argmax there
    # would report high confidence in a label the output does not contain.
    chosen = posterior[np.arange(n_beats), path]
    rival = posterior.copy()
    rival[np.arange(n_beats), path] = -np.inf
    rival = rival.max(axis=1)

    totals = chosen + rival
    margin = np.divide(chosen, totals, out=np.full(n_beats, 0.5), where=totals > 0)

    out: list[BeatChord] = []
    for i, label_idx in enumerate(path):
        start = float(beat_times[i])
        end = float(beat_times[i + 1]) if i + 1 < n_beats else float(audio_duration)
        if end <= start:  # guard against a duplicate or out-of-range final beat
            end = start + 0.25

        out.append(
            BeatChord(
                start=start,
                end=end,
                label=VOCAB.labels[label_idx],
                root=VOCAB.roots[label_idx],
                quality=VOCAB.qualities[label_idx],
                confidence=round(float(margin[i]), 4),
                beat_index=i,
            )
        )
    return out


def merge_repeats(chords: list[BeatChord]) -> list[dict]:
    """Collapse runs of identical labels into spans, keeping the beat count.

    The UI draws one cell per beat, but a span list is what makes a readable chord
    sheet (and it is much smaller over the wire for long tracks).
    """
    if not chords:
        return []

    spans: list[dict] = []
    run = [chords[0]]

    def flush(run: list[BeatChord]) -> None:
        spans.append(
            {
                "label": run[0].label,
                "root": run[0].root,
                "quality": run[0].quality,
                "start": round(run[0].start, 4),
                "end": round(run[-1].end, 4),
                "beats": len(run),
                "beat_index": run[0].beat_index,
                "confidence": round(sum(c.confidence for c in run) / len(run), 4),
            }
        )

    for chord in chords[1:]:
        if chord.label == run[-1].label:
            run.append(chord)
        else:
            flush(run)
            run = [chord]
    flush(run)

    return spans


def transpose_label(label: str, semitones: int) -> str:
    """Shift a chord label by `semitones`, preserving its quality suffix.

    Used server-side for exports; the UI transposes locally so the grid updates
    instantly when the user drags the key control.
    """
    if label == NO_CHORD:
        return label

    # Longest-prefix match so "C#" is not read as "C" followed by a stray "#".
    for length in (2, 1):
        root = label[:length]
        if root in PITCH_CLASSES:
            new_root = PITCH_CLASSES[(PITCH_CLASSES.index(root) + semitones) % 12]
            return new_root + label[length:]
    return label
