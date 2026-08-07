#!/usr/bin/env python3
"""End-to-end check of the analysis pipeline against synthetic ground truth.

Run inside the api container:

    docker-compose exec api python tools/verify.py

Generates a track with a known tempo, key and progression, analyses it, and asserts the
results match. Exits non-zero on failure so it works as a smoke test.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from app.analysis import chords as chord_mod  # noqa: E402
from app.analysis import pipeline, separate  # noqa: E402
from app.analysis.chords import transpose_label  # noqa: E402

TOOLS = Path(__file__).resolve().parent

CASES = [
    # (tempo, key, progression, bars) — a spread of tempi, keys and qualities.
    (120.0, "C", "I,V,vi,IV", 12),
    (90.0, "G", "I,IV,V,IV", 12),
    (140.0, "A", "vi,IV,I,V", 12),
]

# Tolerances. Tempo is asserted tightly because the grid is synthetic. Chord accuracy is
# asserted loosely: even on ground truth, beat-synchronous template matching mislabels the
# occasional beat near a chord boundary, and demanding 100% would make this a flaky test
# rather than a useful one.
#
# Tempo is checked as a *percentage*, not an absolute BPM window. Beat times are quantised
# to the CQT hop (512 samples at 22050 Hz ≈ 23 ms), so the error in the derived BPM scales
# with tempo: ±23 ms on a 500 ms beat is ±2.7 BPM at 120, but ±5 BPM at 180. A fixed window
# would be too loose at slow tempi and impossible at fast ones. 3% is comfortably inside
# "same tempo" for every practical purpose, and still fails a half/double-rate lock.
TEMPO_TOLERANCE_PERCENT = 3.0
MIN_CHORD_ACCURACY = 0.70
# Fraction of interior bars that must hold exactly one chord. Not 1.0: the bar at a
# boundary can legitimately pick up a neighbouring label when a beat lands within a few
# milliseconds of the barline.
MIN_BAR_ALIGNMENT = 0.80


def tempo_matches(detected: float, truth: float) -> bool:
    """Assert the *actual* tempo, not an octave-equivalent one.

    Accepting half or double would defeat the point: the beat grid drives how many chord
    cells the UI draws, so a half-rate grid silently hides every chord change that falls
    on an omitted beat. `rhythm._fix_tempo_octave` exists to correct that, and this check
    is what holds it honest.
    """
    return abs(detected - truth) / truth * 100.0 <= TEMPO_TOLERANCE_PERCENT


def run_case(tempo: float, key: str, progression: str, bars: int, work: Path) -> dict:
    wav = work / f"test_{key}_{int(tempo)}.wav"

    result = subprocess.run(
        [
            sys.executable, str(TOOLS / "make_test_audio.py"), str(wav),
            "--tempo", str(tempo), "--key", key,
            "--progression", progression, "--bars", str(bars),
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"test audio generation failed: {result.stderr}")

    truth = json.loads(wav.with_suffix(".truth.json").read_text())

    # Demucs is skipped here: on a synthetic signal it adds minutes and nothing else, and
    # this test is about the DSP chain, not the separator. The separator is exercised by
    # the real-audio path in the UI.
    analysis = pipeline.analyse(wav, duration=truth["duration"], enable_demucs=False)

    # --- tempo ---
    tempo_ok = tempo_matches(analysis.tempo, tempo)

    # --- meter ---
    meter_ok = analysis.beats_per_bar == truth["beats_per_bar"]

    # --- key ---
    key_ok = analysis.key["name"].split()[0].rstrip("b#") == key.rstrip("b#") or (
        analysis.key["tonic"] == truth["tonic"]
    )

    # --- chords: score each detected beat against the bar it falls in ---
    hits = misses = 0
    mismatches: list[str] = []
    bar_duration = (60.0 / tempo) * truth["beats_per_bar"]

    for beat in analysis.beat_chords:
        # Compare at the beat's midpoint so a beat straddling a bar line is attributed to
        # the bar it mostly occupies.
        midpoint = (beat["start"] + beat["end"]) / 2
        bar_index = int(midpoint // bar_duration)
        if bar_index >= len(truth["chords"]):
            continue

        expected = truth["chords"][bar_index]
        if beat["root"] == expected["root"] and beat["quality"] in (
            expected["quality"], "maj" if expected["quality"] == "maj" else expected["quality"]
        ):
            hits += 1
        elif beat["root"] == expected["root"]:
            # Right root, wrong quality (e.g. C detected as C7). Half credit: the chart is
            # still playable, which is what matters to a user.
            hits += 0.5
            misses += 0.5
        else:
            misses += 1
            if len(mismatches) < 4:
                mismatches.append(
                    f"{midpoint:5.2f}s bar{bar_index:2d} expected {expected['label']:4s} got {beat['label']}"
                )

    total = hits + misses
    accuracy = hits / total if total else 0.0

    # --- bar alignment ---
    # The progressions here change chord once per bar, so a correctly-phased grid puts each
    # change on the first beat of a bar. Checked separately from chord accuracy because the
    # two fail independently: the labels can be perfect while the grid is rotated, which
    # renders every change on the wrong beat of the chord sheet.
    interior = [b for b in analysis.bars[1:-1] if b["beats"]]
    aligned = sum(
        1 for b in interior
        if len({beat["label"] for beat in b["beats"]}) == 1
    )
    bar_alignment = aligned / len(interior) if interior else 0.0

    return {
        "case": f"{key} {progression} @ {tempo:.0f}",
        "bars": {
            "alignment": round(bar_alignment, 3),
            "counted": len(interior),
            "ok": bar_alignment >= MIN_BAR_ALIGNMENT,
        },
        "tempo": {"expected": tempo, "got": analysis.tempo, "ok": tempo_ok},
        "key": {"expected": f"{key} major", "got": analysis.key["name"], "ok": key_ok},
        "meter": {
            "expected": truth["beats_per_bar"],
            "got": analysis.beats_per_bar,
            "ok": meter_ok,
        },
        "chords": {
            "accuracy": round(accuracy, 3),
            "beats": int(total),
            "ok": accuracy >= MIN_CHORD_ACCURACY,
            "mismatches": mismatches,
        },
        "timings": analysis.timings,
    }


def check_no_chord_gate() -> None:
    """A quiet but harmonic beat must not decode as "no chord".

    This is the regression that produced empty cells in the chord sheet where the music
    plainly changes chord. The old silence test compared *chroma* column norms against
    their own median, but the pipeline peak-normalises every chroma column, so the norms
    sit in a band too narrow for any threshold to separate — the gate never fired, while a
    near-flat chroma (which is what both silence *and* a quiet sustained chord normalise
    to) matched the flat "N" template on shape alone and won.

    So: a clean C major triad at 1/50th of full scale must still come out as C.
    """
    quiet = np.zeros((12, 8))
    for pitch_class in (0, 4, 7):
        quiet[pitch_class, :] = 0.02

    loudness = np.full(8, 0.02)
    decoded = chord_mod.recognise(
        quiet, np.arange(8, dtype=float) * 0.5, 4.0, beat_loudness=loudness,
    )
    labels = {c.label for c in decoded}
    assert labels == {"C"}, f"quiet C major decoded as {labels}, expected all C"

    # ...and true silence must still be "N", or intros and breaks fill with invented chords.
    silent = np.full((12, 8), 1.0 / np.sqrt(12.0))
    decoded = chord_mod.recognise(
        silent, np.arange(8, dtype=float) * 0.5, 4.0, beat_loudness=np.zeros(8),
    )
    labels = {c.label for c in decoded}
    assert labels == {"N"}, f"silence decoded as {labels}, expected all N"


def check_harmonic_priority() -> None:
    """The chord signal must be led by piano, then guitar, then bass.

    Worth asserting because it fails silently: the old build summed `bass + other`, and
    Demucs routes a piano into its own `piano` stem, leaving `other` near-empty. The chord
    instrument was therefore dropped from the analysis entirely and only the bass root
    survived — measured at 44% root accuracy against 98% once the piano is used.
    """
    assert separate.HARMONIC_PRIORITY[:3] == ("piano", "guitar", "other"), \
        separate.HARMONIC_PRIORITY

    sr = 100
    loud = np.ones(sr, dtype=np.float32) * 0.5
    faint = np.ones(sr, dtype=np.float32) * 0.001

    # Piano present alongside a louder bass: piano still has to lead.
    signal, used, _ = separate._pick_harmonic(
        {"piano": loud * 0.6, "guitar": faint, "other": faint, "bass": loud, "drums": loud}
    )
    assert used[0] == "piano", f"expected piano to lead, got {used}"

    # No piano: guitar leads.
    signal, used, _ = separate._pick_harmonic(
        {"piano": faint, "guitar": loud * 0.6, "other": faint, "bass": loud}
    )
    assert used[0] == "guitar", f"expected guitar to lead, got {used}"

    # Neither: fall back to whatever carries harmony, and never return silence.
    signal, used, _ = separate._pick_harmonic(
        {"piano": faint, "guitar": faint, "other": faint, "bass": loud}
    )
    assert used == ["bass"], f"expected bass only, got {used}"
    assert np.any(signal != 0), "harmonic signal is silent"


def main() -> int:
    print("=" * 74)
    print("Music Analyzer — pipeline verification")
    print("=" * 74)

    # transpose_label is pure logic; check it here rather than needing a separate test file.
    assert transpose_label("C", 2) == "D", transpose_label("C", 2)
    assert transpose_label("C#m7", 1) == "Dm7", transpose_label("C#m7", 1)
    assert transpose_label("B", 1) == "C", transpose_label("B", 1)
    assert transpose_label("N", 5) == "N"
    print("chord label transposition: ok")

    check_no_chord_gate()
    print("no-chord gate (quiet chord stays a chord, silence stays N): ok")

    check_harmonic_priority()
    print("harmonic stem priority (piano > guitar > other > bass): ok\n")

    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for tempo, key, progression, bars in CASES:
            report = run_case(tempo, key, progression, bars, work)

            tempo_r, key_r, chord_r = report["tempo"], report["key"], report["chords"]
            meter_r, bar_r = report["meter"], report["bars"]
            ok = (
                tempo_r["ok"] and key_r["ok"] and chord_r["ok"]
                and meter_r["ok"] and bar_r["ok"]
            )
            if not ok:
                failures += 1

            print(f"{'PASS' if ok else 'FAIL'}  {report['case']}")
            drift = (tempo_r["got"] - tempo_r["expected"]) / tempo_r["expected"] * 100.0
            print(
                f"      tempo  {tempo_r['got']:6.1f} BPM  (expected {tempo_r['expected']:.0f}, "
                f"{drift:+.1f}%) {'ok' if tempo_r['ok'] else 'MISMATCH'}"
            )
            print(
                f"      key    {key_r['got']:<12s} (expected {key_r['expected']}) "
                f"{'ok' if key_r['ok'] else 'MISMATCH'}"
            )
            print(
                f"      meter  {meter_r['got']}/4  (expected {meter_r['expected']}/4) "
                f"{'ok' if meter_r['ok'] else 'MISMATCH'}"
            )
            print(
                f"      chords {chord_r['accuracy'] * 100:5.1f}% over {chord_r['beats']} beats "
                f"{'ok' if chord_r['ok'] else f'BELOW {MIN_CHORD_ACCURACY:.0%}'}"
            )
            print(
                f"      bars   {bar_r['alignment'] * 100:5.1f}% of {bar_r['counted']} bars hold one chord "
                f"{'ok' if bar_r['ok'] else f'BELOW {MIN_BAR_ALIGNMENT:.0%}'}"
            )
            for line in chord_r["mismatches"]:
                print(f"             {line}")
            print(f"      timing {report['timings']}")
            print()

    print("=" * 74)
    if failures:
        print(f"{failures} of {len(CASES)} case(s) FAILED")
        return 1
    print(f"all {len(CASES)} cases passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
