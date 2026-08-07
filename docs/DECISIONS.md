# Engineering decisions & environment notes

A record of *why* this project is built the way it is, and the environment constraints that
shaped it. Written for whoever (including future me) has to change this later.

---

## Hard constraints from the user

1. **Everything in Docker.** Nothing is installed on the host — no ffmpeg, no Python
   packages, no Node modules. The goal is that this deploys to a real server unchanged.
2. **Best available chord accuracy**, not the fastest option. That is why Demucs source
   separation is on by default rather than offered as an upsell.

---

## Environment discovered on this machine

| Thing | Value | Consequence |
| --- | --- | --- |
| Docker host | colima, a custom profile sized for Demucs | Needs at least 6 CPU / 10 GiB / aarch64; the `default` profile (2 CPU / 2 GiB) is too small |
| `docker compose` subcommand | **broken on this machine** | `~/.docker/cli-plugins/docker-buildx` was a 9-byte file containing the text `Not Found` — a failed download — which shadowed the working Homebrew plugin |
| Working buildx | `/opt/homebrew/lib/docker/cli-plugins/docker-buildx` (v0.20.1, arm64) | Invoke it directly, or fix the broken file |
| `docker-compose` v2.33.1 standalone | works | Use `docker-compose` (hyphen) instead of `docker compose` if the plugin is broken |
| Corporate TLS-inspecting proxy | present on some networks | See below — affects only building the image on such a network, not running it |
| Docker VM disk | filled to 98% by other projects' layers on one occasion | Builds fail with `no space left on device`; `docker image prune -f` reclaims space |

These are notes about *this developer's* machine, not requirements of the project. A clean
Docker install on any OS/architecture with `docker compose` (or `docker-compose`) builds and
runs this unchanged — see `README.md`.

### A TLS-inspecting proxy breaks download.pytorch.org

`pip install --index-url https://download.pytorch.org/whl/cpu` fails with
`CERTIFICATE_VERIFY_FAILED` on a network that TLS-inspects (Zscaler, Netskope, Blue Coat, a
corporate MITM proxy) — it re-signs TLS and pip does not trust its CA. **PyPI itself works
fine.**

Resolution: install torch from PyPI. On `linux/arm64` the PyPI wheels are already
CPU-only (no aarch64 CUDA builds exist), so the CPU index buys nothing here anyway.
**If this is ever deployed on amd64, add back `--index-url https://download.pytorch.org/whl/cpu`**
or the image gains ~2 GB of unused CUDA libraries.

---

## Why the pipeline is shaped this way

### Demucs before analysis, not after
Vocals contribute melody notes that are frequently *not* in the underlying chord, and
vibrato smears chroma across bins. Drums add broadband energy that a CQT reads as pitch.
Splitting the mix and analysing only the harmony instruments is the largest single accuracy
improvement available. Drums are kept separately and used for beat tracking, where the
clean onset envelope helps just as much.

Demucs is wrapped in a `try/except` that degrades to librosa HPSS rather than failing the
job — a missing model download or an OOM should produce a worse answer, not no answer. The
chosen path is reported in the API response and shown in the UI.

### The chord signal is led by piano, then guitar, then bass
`htdemucs_6s` separates piano and guitar into stems of their own, which means the 4-stem
model's `other` bucket is **not** where the chord instrument lives. The original build summed
`bass + other`; on a piano-led fixture that measured `piano: 0.7911` against `other: 0.0054`, so
the analysis was reading essentially the bass line alone. Root accuracy was 44.4%. Ranking the
stems by `HARMONIC_PRIORITY = ("piano", "guitar", "other", "bass")` — weighted `0.6 ** rank`,
each normalised by its own level so a loud bass cannot drown a quiet piano — took it to 98.4%.

The 6-stem model costs the same wall clock as the 4-stem one, measured: it is one transformer
pass and the extra stems are just output channels. So this was free.

Stems below `STEM_PRESENCE_RATIO` (12% of the loudest) are treated as absent rather than mixed
in at low level, because Demucs leaks a faint ghost of every instrument into every stem and
summing those ghosts is what blurs the chroma.

### Thread count has to be set before the interpreter starts
The Dockerfile pinned `OMP_NUM_THREADS=1` with the comment "torch manages its own pool". That
is false: OpenMP reads the variable once, when the library initialises on `import torch`.
Calling `torch.set_num_threads(5)` afterwards *reports* 5 but recovers nothing — measured 1.43 s
for a benchmark op at one thread, 1.40 s after `set_num_threads(5)`, and 0.34 s when the
variable was 5 from the start. Demucs had been running single-threaded.

The fix therefore cannot live in Python. `backend/entrypoint.sh` resolves the budget and
exports OMP, MKL and OpenBLAS before `exec`ing uvicorn. It reads the cgroup CPU quota
(`/sys/fs/cgroup/cpu.max`, falling back to v1's `cpu.cfs_quota_us`/`cpu.cfs_period_us`) because
`os.cpu_count()` sees host cores and ignores the compose `cpus:` limit. It does *not* use
`nproc`, which honours `OMP_NUM_THREADS` and so returns exactly the value being computed.

### Beat-synchronous chroma
One observation per beat, not per frame. Chords change on beats, so averaging between beat
onsets both denoises and aligns the output to the grid the UI draws. The first ~15% of each
beat is trimmed because transient attack energy is broadband and pollutes the chroma right
after the onset. The median (not mean) is taken over the window so one loud passing note
cannot flip the chord.

### Viterbi, not argmax
Frame-wise argmax flickers between relative chords constantly. A single self-transition bonus
(3.4, tuned by ear) removes nearly all the flicker on its own — that used to be the whole
transition matrix, with a comment saying a circle-of-fifths-weighted version "would do
marginally better and be much harder to reason about." It turned out worth doing: see
"A key-aware, circle-of-fifths transition matrix" below. It is layered on top of the same
self-transition bonus, not a replacement for it.

### A key-aware, circle-of-fifths transition matrix
The flat transition matrix penalised every *change* equally — a new chord a fifth away (the
overwhelmingly common move in tonal music) cost the decoder exactly as much as one a tritone
away (the rarest). That is real information the decoder was throwing away for free: given two
labels that fit the audio about equally well, the one that is a more natural next chord should
win the tie.

`_transition_logprob` now costs a change by how far its root sits from the previous chord on the
circle of fifths (`COF_WEIGHT`, scaled 0 at the same root to the full weight at a tritone away),
and gives a small bonus to landing on a root that is in the track's own detected key
(`DIATONIC_BONUS`, from `key_detect`'s `scale_pitches` — already computed earlier in the
pipeline, so this costs nothing new to obtain). The key bias is skipped when `key.confidence <
0.3`: a key guessed on thin evidence is as likely to steer the chords wrong as right, and this
bias only earns its place when the key itself is trustworthy.

Both weights were tuned down hard from the first guess. `COF_WEIGHT=1.2` sounded reasonable in
isolation but pushed a `vi-IV-I-V` progression's opening `F#m` to decode as its relative major
`D` instead — cosine similarity barely favours one over the other (they share two of three
chord tones), and a strong "the rest of the track moves by fifths a lot" prior was enough to
override that thin emission evidence at the very first beat, before there was any path history
to lean on. `COF_WEIGHT=0.3, DIATONIC_BONUS=0.15` (a quarter of the first guess) fixed that
fixture with no loss on the other two — the lesson being that a structural prior should nudge
a close call, not overrule the audio, and that has to be checked against a real decode, not
reasoned about from the formula.

### Template weighting and quality priors
Richer templates have more non-zero bins and therefore win cosine similarity too often;
without a prior, every chord becomes a 7th or a sus. Two corrections:
- Added tones (beyond the first three) are weighted 0.55 rather than 1.0, since extensions
  are often quiet or absent in a real recording.
- `QUALITY_PRIORS` in `chords.py` adds a log-prob bonus favouring plain triads.

Both are tuned for pop/rock. Jazz material would want the priors flattened.

### The bar phase comes from harmony, not from onset energy
`rhythm._estimate_meter` guesses `beats_per_bar` and the phase from onset strength, because
it runs before chords are known. That is a weak cue for the phase: an even backbeat — kick
on 1 and 3, snare on 2 and 4, hats throughout — puts no particular accent on the downbeat,
and the estimator then latches onto noise. Measured on the verification fixture it preferred
phase 3 over the true phase 0 by a wide margin (score 12.1 vs −10.1).

The symptom is nasty because it is *not* a wrong chord: labels stay perfect, but the grid is
rotated, so every chord change renders on the last beat of a bar instead of the barline, and
bar 1 shows a single beat. It reads as a subtly broken chord sheet.

`pipeline._refine_downbeat_phase` re-derives the phase after chord decoding, from the beat
positions where the chord changes — in the material this tool targets, chords change on the
barline. It only overrides when one phase is a clear winner, so a static vamp or a syncopated
section falls back to the onset-based guess. `verify.py` asserts bar alignment separately
from chord accuracy, because the two fail independently.

### Beat tracking needs the full mix, not just drums, or a quiet intro has no grid at all
`rhythm.analyse` used to run `onset_strength` on the isolated **drums** stem alone — cleaner
onsets, better tempo lock. But a drums-only stem is not quietly weaker where the kit lays out
for an intro or a sparse verse, it is **exactly zero**: `onset_strength` on silence is zero at
every frame, not "uncertain", so `beat_track` has nothing to lock onto there. On a real song
with a 65-second vocal-and-guitar intro before the drums come in, this meant the *entire* beat
grid — and everything downstream that is indexed by it: bars, chord spans, downbeats — started
at 65.3s on a 250s track. The chord sheet visibly began in the middle of the song, and the
generated harmony had nothing to attach to for the first two verses, because
`harmony._chord_at` returns `None` for any note before the first chord span exists.

Fix: `rhythm.analyse` takes an optional `y_full` (the whole mix) and blends its onset envelope
with the percussive one — `np.maximum` of each independently peak-normalised, so the full mix
only fills in where percussive is silent rather than diluting the cleaner signal everywhere.
Measured on the drum-heavy chorus of the same track, beat times from the blended envelope were
identical to drums-only to the millisecond; the difference only shows up where it needs to.

One trap this produced: `_fix_tempo_octave`'s 0.33 threshold was calibrated against how much a
drum pattern's off-beat midpoints leak (~0%) versus a genuinely half-rate lock (~50%). The full
mix's onset envelope is continuous — sustained harmony and vocal consonants land energy on the
midpoints too — which pushed a *correctly* tracked 117 BPM over that threshold and "fixed" it
to 235 BPM, a real regression caught only by re-running `verify.py` and a real download. The
octave check has to keep running against the percussive-only envelope; only `beat_track` itself
runs on the blended one.

### Confidence is a margin, not a posterior
Per-beat confidence is the chosen chord's share against its closest rival — `best / (best +
runner_up)` — so 0.5 means a dead tie and 1.0 means nothing else is close. It is measured
against the label **Viterbi decoded**, not the frame-wise argmax, since Viterbi legitimately
overrides the local best to keep the sequence coherent.

The raw softmax posterior was the original choice and it was wrong. With 133 labels,
probability mass spreads across the relatives and added-tone variants that any real chord
partly matches, so on a track whose chords were *all correct* it ranged 0.03–0.36 with a
median of 0.16 — a number that reads as "16% certain" and made the UI's threshold flag 99.6%
of beats as unsure. A pairwise margin is scale-free and stays interpretable as the vocabulary
grows. Same reasoning as key-detection confidence, which reports the margin over the
runner-up for the same reason.

`UNSURE_BELOW = 0.5` in `ChordGrid.jsx` is therefore not a tuned constant: it is exactly the
tie point. Measured on real audio it marks ~29% of beats, 61% of which fall within half a
second of a chord change.

### The "no chord" gate needs loudness, and chroma has none
Cells appeared in the sheet with no chord at all on beats where the music plainly changes chord.
The gate that produced them compared **chroma column norms** against `0.12 * median`. But the
pipeline peak-normalises every chroma column (`normalize(norm=inf)`), which forces each column's
largest bin to 1.0 — so the norms carry no loudness information whatsoever. Measured, they spanned
1.01–1.54 across 362 beats against a threshold of ~0.13: the gate fired on **zero** beats.

Meanwhile the failure it was meant to prevent was happening in reverse. Silence normalises up to a
near-flat chroma, and so does a quiet sustained chord — and a flat chroma matches the flat `N`
template better than any triad template, on shape alone. The gate was simultaneously dead and
inverted, which is why no amount of threshold tuning could have fixed it.

The fix had to introduce a signal that survives normalisation: `pipeline` takes
`librosa.feature.rms` on the harmonic stem *before* normalising, trims it to the same beat window
as the chroma, and passes it as `beat_loudness`. `chords._emission_logprob` gates `N` on loudness
relative to the track's own 90th percentile, so it adapts to quiet masters. `verify.py` locks both
directions: a C triad at 1/50th full scale must decode as C, and true silence must stay `N`.

The visible symptom had a second, independent cause — see "A UI glyph the font cannot draw" below.

### Chord boundaries are accurate to about one beat
Measured on *Paradise*: chord spans average 1.72 s against a 1.67 s bar, so the harmonic
rhythm is right, but individual boundaries land 5+3 beats instead of 4+4 — off by one beat
either way. That is inherent to beat-synchronous template matching, where a boundary can only
fall on a beat and the chroma either side of it is blended. Consequence: on real music only
about half of bars hold a single chord, even when every label is correct. The synthetic
fixtures hit 100% because their changes are exactly on beats with no reverb tail.

### Median chroma for key detection
The mean is skewed by a loud modulated bridge or a long pedal note — enough to flip the
answer a fifth away. Confidence is reported as the *margin over the runner-up*, not the raw
correlation, because a track can correlate 0.9 with two relative keys at once and that
ambiguity is the useful signal.

---

## Transposition: the part that is easy to get wrong

The chart and the audio must move together.

- **Chart**: shifted client-side. Chords are stored as pitch class + quality, so it is
  integer arithmetic — instant, no round trip. Spelling follows the destination key.
- **Audio**: ffmpeg's `rubberband` filter, which shifts pitch at **constant tempo**.

Verified present: `ffmpeg 7.1.5` in `python:3.11-slim-bookworm` includes the `rubberband`
filter (needs `librubberband2` in the runtime image).

**The naive approach is a trap.** `asetrate` + `atempo` changes speed as well as pitch,
which desynchronises the entire beat grid from the audio — the playhead would drift further
out with every bar. That chain exists in `transpose.py` only as a fallback for an ffmpeg
build without rubberband, and it is worse.

Renders are cached per (job, semitones) and output as 192 kbps MP3, not WAV: a 5-minute WAV
is ~50 MB and the browser must download it before playback, versus ~7 MB that streams.

---

## Frontend decisions

- **`<audio>` element, not Web Audio buffers.** Decoding a 5-minute track into memory on
  every load is wasteful; `<audio>` gives HTTP range requests, streaming start and native
  seeking for free.
- **Playhead driven by `requestAnimationFrame`, not `timeupdate`.** The `timeupdate` event
  fires ~4×/second, which visibly stutters. rAF is per-frame and paint-synchronised. The
  playhead has *no* CSS transition, deliberately — a transition would add lag between the
  sound and the line.
- **Active beat found by binary search.** It runs every animation frame, so it must be
  O(log n), not a scan.
- **Auto-scroll follows the active *row*, not the active cell**, and parks it one third down
  the viewport. Scrolling per beat yanks the view constantly; you need to read ahead.
- **Low-confidence beats are marked, not hidden** (dotted underline). Presenting every
  chord as equally certain would be dishonest. Only the *labelled* beat carries the mark —
  underlining every held beat would smear one uncertain decision across a whole bar.

### A chord label only where the chord changes
Every beat cell used to carry a label. On a 4/4 track that is four identical chord names per
bar, and the sheet becomes a wall of text in which the one thing you need — *where the harmony
moves* — is the hardest thing to see. Continuation cells are now blank, so the pattern of
labels is the harmonic rhythm. This is what a printed chart does, and what chordify's chord
overview does.

The change is computed across bar boundaries, not per bar, so a chord held over a barline
stays blank in the next bar rather than being re-announced.

One case needs care: when the indicator lands on a blank cell, the player looks like it has
lost its place mid-bar. The cell carries its chord in a `data-held` attribute which CSS reads
back through `::before` only in the `.is-active` state — one attribute rather than a second
always-rendered span that would have to be hidden the rest of the time.
`e2e/held-beat-check.mjs` exists solely to assert this, because the synthetic fixtures change
chord on every downbeat and the main run therefore never lands the indicator on a held cell.

### The grid is CSS Grid, not flex-wrap
Bars are `repeat(auto-fill, minmax(300px, 1fr))` and beats within a bar are equal `1fr`
columns. Wrapping fixed-width bars left a ragged gap at the right of every row, which reads as
a rendering fault on a grid that is meant to look ruled; unequal beat widths make the rhythm
itself look uneven. Cells butt together with no gap and share edges, and the bar-line is a
4px `border-left` on `.bar`.

`--barline` is *lighter* than `--border`, which looks backwards until you try it: on a dark
ground a darker rule reads as a gap between cells rather than as a line drawn between
measures.

### Auto-scroll measured against the container, not `offsetTop`
`row.offsetTop` is relative to the nearest **positioned** ancestor. `.bar` sets
`position: relative` for its bar number, so the measurement silently came back in the wrong
coordinate space and the grid scrolled to a place the active row was not — it tracked the beat
but parked the row above the top edge of its own frame, so the one thing you needed to look at
was the one thing you could not see. Now it is
`row.getBoundingClientRect().top - container.getBoundingClientRect().top + container.scrollTop`,
clamped to the scrollable range.

`scroll-behavior: smooth` was also removed from `.grid`: the auto-scroll passes
`behavior: 'smooth'` explicitly, and having it on the element as well makes the user's own
drag lag.

The browser check asserts the active cell is fully inside the frame across 14 samples **and**
that `scrollTop > 0`, after seeking to 55% of the track. Without the seek the first rows fit
without scrolling at all and the check passed vacuously — it did, on the first run.

### A UI glyph the font cannot draw reads as a broken app
The no-chord marker was `𝄽` (U+1D13D MUSICAL SYMBOL QUARTER REST), which is correct musical
notation and which almost no UI font carries. It rendered as a tofu box. The bug report was
"dark cells with a 3-stroke line and no chord at all, I don't know what it is" — the glyph was
*itself* the defect, and it also camouflaged the genuine mis-decode underneath it, sending the
first look at the problem to the wrong layer.

Now a plain `–` (U+2013). `ui-check.mjs` asserts every `.beat.is-empty` cell contains exactly
that character and prints the codepoint on failure, so a regression names itself. General rule
for this codebase: UI markers stay inside Latin-1 / General Punctuation unless a webfont is
actually bundled.

### Selecting a source does not start the job
Analysis with stem separation runs for minutes and cannot be cancelled, and progress sits at
8% during separation so it looks hung. Starting that on a drag-and-drop — an action that is
easy to do by accident and easy to do twice — was wrong. The file is held in local state until
the user presses Analyze. (This bit in practice: two jobs were queued for the same file, and
clearing them meant restarting the container.)

---

## Stem downloads and generated harmony

**Stems are exported during analysis, not on request.** Demucs already holds all six in memory
at that point; regenerating one later would mean a second full separation. They are written as
MP3 (libmp3lame 192k) via a temporary WAV, best-effort — an export failure must not fail the
analysis, since the chord sheet is what the user asked for.

**The stem route takes a whitelist, not sanitisation.** `stem` lands in a filesystem path, and a
whitelist against `separate.DOWNLOADABLE` cannot be talked into resolving somewhere else, which
is not true of any amount of `..` stripping.

**Harmony is chord-aware.** A fixed −4/−9 semitone shift is wrong roughly half the time, because
a third below a note is only in the chord for some scale degrees. `_harmony_midi` searches ±6
semitones around the nominal target for a note whose pitch class is in the chord sounding
underneath, takes the nearest, and returns `None` — emitting nothing — if the nearest is more than
2 semitones away. Silence is better than a wrong note.

**Shifted per note, not per frame.** Per-frame shifting follows the pyin trace, which includes
vibrato and tracker jitter, and the result warbles. Notes are segmented and shifted whole, with
12 ms raised-cosine fades at the joins.

**Note segmentation has to bridge short confidence dropouts, or vibrato itself fragments a note.**
The first version cut a new note the instant pyin's voicing probability dipped below
`VOICED_MIN_PROB`. Measured on a real vocal, that probability oscillates above and below 0.5
*during a single sustained note* — ordinary vibrato and pitch wobble drag it under for a couple
of frames at a time, repeatedly, even though a listener hears one continuous note. Without
bridging, one note was cut into 5-6 fragments (656 raw confident runs collapsed to 123 once gaps
up to 150ms were bridged), each rendered as its own independent `pitch_shift` call with its own
fade — not wrong notes, but the same note stuttering and re-attacking every 100-150ms, which is
exactly what a user reported as "unintelligible." `_bridge_gaps` fills short `False` runs in the
voicing mask when they are flanked by confident voicing on both sides (never at the very start
or end, where there is nothing to interpolate between); the pitch across a bridged gap is filled
by linear interpolation in log-frequency (`np.interp` on MIDI numbers) rather than held flat or
left `NaN`, since a gap this short is by construction mid-note. Effect on the same real vocal:
median note length 0.16s → 0.26s, notes under 200ms (i.e. audible fragments) 62% → 37%.

**Rendered on first request, then cached.** Pitch-tracking the vocal and rendering every note has
measured up to ~90s on a real 3-4 minute song — considerably more than an early "~8s" estimate
based on a short clip suggested. Putting that in the analysis pipeline would work directly against
the goal of making analysis faster. Instrumentals are rejected before pyin runs (vocal level
`< 1e-3`) and return 422, so the endpoint fails fast and legibly instead of returning silence. The
UI ticks an elapsed-seconds counter on the button while it waits, rather than a static "building…"
label — at this length, a label that never changes reads as stuck, which is exactly the complaint
this produced ("ทำไมต้องรอ process อะไรบางอย่าง ไม่ download เลยเหมือน stem อื่น").

That 422 is also why the frontend fetches harmony rather than using `<a download>`: following a
link to an endpoint that legitimately returns JSON would replace the page with raw JSON.

**The harmony bus is rendered once and mixed two ways.** `harmony.build_bus` does the slow part
(pitch tracking + per-note shifting) and returns the backing-voice signal alone.
`harmony.generate` mixes it under the isolated vocal (the standalone download);
`harmony.generate_with_track` mixes the *same* bus under the full song instead, for the
Transport bar's "Play with harmony" toggle — swapping the player's audio source rather than
downloading a file. The two must not both re-add the lead vocal: `generate_with_track` mixes the
bus into `full_mix` as-is, because the lead is already in that signal. Each mix is a separate
cached file (`harmony.mp3` / `harmony_with_track.mp3`) behind its own endpoint
(`/harmony` / `/harmony/with-track`), so downloading one does not force rendering the other, but
neither re-runs pitch tracking if the other was requested first — day-two chords would need
`build_bus` re-run since it takes the decoded chords as an argument, but the render itself does
not depend on which mixdown is wanted.

Transposition is not offered while "Play with harmony" is on: the harmony render has no
pitch-shifted variant, and offering the control would either silently do nothing or need a third
render path for a combination nobody asked for.

---

## Things deliberately not done

- **No Celery/Redis.** One container, one process, a `ThreadPoolExecutor` with one worker.
  `JobStore`'s interface is small on purpose so it can be swapped for a Redis-backed
  implementation if this ever needs to scale horizontally.
- **No key-change detection.** Key detection is global. A modulating track reports whichever
  key dominates.
- **No inversions / slash chords.** A C/E reads as C.
- **`MAX_CONCURRENT_JOBS=1`.** Demucs saturates every core it is given; two concurrent jobs
  make both slower and risk the memory cap.

---

## Verification

Two layers, both run in containers:

```bash
# 1. DSP against synthetic ground truth (tempo, key, meter, chords, bar alignment)
docker-compose exec api python tools/verify.py

# 2. The browser behaviour curl cannot check — rendering, playhead sync, transport, transpose
docker build -t music-analyzer-e2e frontend/e2e
docker run --rm -v "$PWD/frontend/e2e:/work" -v "$PWD/.tmp-shots:/shots" \
  --network music-analyzer_default -e WEB_URL=http://web:80 music-analyzer-e2e
```

The browser runner is built rather than pulled: `ghcr.io/puppeteer/puppeteer` is amd64-only,
and its bundled Chrome cannot start under QEMU on an arm64 host (it times out waiting for the
DevTools WS endpoint). Debian's `chromium` is native on both architectures, so the image uses
`puppeteer-core` pointed at the system binary. `puppeteer-core` is installed at `/node_modules`
because ESM ignores `NODE_PATH` — it resolves bare specifiers by walking up parent
directories, so a global install under `/usr/local/lib` is invisible to a bind-mounted script.

Three things the check had to work around, all worth knowing before editing it:

- **`protocolTimeout` must be raised.** `waitForSelector`'s own timeout is not enough: each
  poll is a CDP call and puppeteer aborts the *connection* after `protocolTimeout` (180s
  default) regardless of the selector timeout. The symptom is a `ProtocolError` naming
  `.grid__bars`, which reads as "the grid never rendered" when in fact Demucs was still
  running.

- **`usePlayer` creates its `<audio>` with `new Audio()` and never attaches it to the DOM.**
  A detached element plays fine and avoids a stray node in the layout, but it means
  `document.querySelector('audio')` finds nothing. The script hooks `window.Audio` via
  `evaluateOnNewDocument` to capture the real instance.
- **Chrome needs `--autoplay-policy=no-user-gesture-required`**, or `el.play()` rejects and
  playback can't be scripted.

The active cell *is* the indicator now that the scale lanes are gone, so sync is asserted by
reading the beat's start time out of its `title` and checking the audio clock falls inside that
beat's window, rather than by comparing a pixel transform.

A fourth trap, found while writing the held-beat check: **`position` is only sampled while
playing.** The rAF loop exists only when `isPlaying`, so setting `currentTime` on a paused
element moves the audio but never re-renders the grid. A probe that seeks while paused reports
the indicator as stuck when nothing is wrong.

A fifth, found while adding the stem-menu checks: **a click that changes React state and a query
for the result cannot share one `page.evaluate` call.** The re-render has not happened when a
synchronous callback returns, so the menu reads as empty. Click, sleep, then query.

---

## Operational notes

- Model weights land in `/data/models` (`TORCH_HOME`) on the named volume, so they survive
  container replacement. First Demucs job downloads ~80 MB.
- The thread budget is resolved by `entrypoint.sh` before Python starts, not in the Dockerfile
  and not in Python. See "Thread count has to be set before the interpreter starts" below.
- The API runs as non-root (uid 10001).
- Job artefacts are pruned after 12 hours, on every job completion.
- Backend logs go to both stdout and `/app/log/api.log`, bind-mounted to `./log/`.
