# Music Analyzer

Detect **tempo**, **key** and **chords** in any song — from an uploaded audio file or a
YouTube link — and play along with a beat-synchronised chord sheet you can transpose to
any key without the audio going out of tune.

Everything runs in Docker. Nothing is installed on the host.

```bash
docker compose up -d --build
open http://localhost:8080
```

---

## What it does

| Feature | Notes |
| --- | --- |
| **Tempo** | Beat tracking on the isolated drum stem, plus meter (4/4, 3/4, 6/8) and downbeat detection |
| **Key** | Krumhansl–Schmuckler profile correlation (Temperley weights), with runner-up candidates |
| **Chords** | 132-chord vocabulary (maj, min, 7, maj7, m7, dim, aug, sus2, sus4, 6, m6 × 12 roots) + no-chord |
| **Chord sheet** | One cell per beat, grouped into bars with heavy bar-lines. A chord name appears only where it changes; the active beat is a filled block |
| **Transpose** | −12…+12 semitones. Chart shifts instantly; audio is pitch-shifted at constant tempo |
| **Playback** | Play / pause / stop, click-to-seek anywhere, 0.5×–1.25× speed with pitch preserved |
| **Input** | mp3, wav, flac, m4a, aac, ogg, opus, wma, aiff, alac, mp4, webm, mkv + YouTube URLs |

Choosing a file or pasting a link does **not** start analysis — you pick the source, set the
accuracy mode, then press **Analyze**. With stem separation a job runs for minutes and cannot
be cancelled, so committing to it on a drag-and-drop was the wrong default.

---

## Architecture

```
browser ──▶ web (nginx :8080) ──┬──▶ static SPA bundle
                                └──▶ /api/* proxied to api:8000
                                          │
                                          ▼
                        api (FastAPI + ffmpeg + Demucs + librosa)
                                          │
                                          ▼
                          analyzer-data volume (jobs, model weights)
```

Only `web` publishes a port. The browser talks to a single origin, so there is no CORS in
production and the same relative URLs work in dev and prod.

### Analysis pipeline

```
audio in (file or YouTube)
   │
   ├─ ffmpeg / yt-dlp ──▶ normalised WAV (44.1 kHz stereo)
   │
   ├─ Demucs htdemucs_6s ──▶ piano ▸ guitar ▸ other ▸ bass ──▶ "harmonic"  (chords, key)
   │                          drums                        ──▶ "percussive" (tempo, beats)
   │                          vocals                        ──▶ download, harmony
   │
   ├─ beat tracking on percussive ──▶ tempo, beat times, meter, downbeats
   │
   ├─ CQT ──▶ chroma on harmonic ──▶ averaged between beats
   │
   ├─ key: profile correlation over the whole track
   │
   └─ chords: template match per beat ──▶ Viterbi smoothing ──▶ beat/bar timeline (JSON)
```

**Why separate the stems first.** Vocals smear the chroma with melody and vibrato, and
drums add broadband energy the CQT reads as pitch. Removing both before analysis is the
single biggest accuracy win available, which is why it is on by default. If Demucs is
unavailable at runtime the pipeline falls back to librosa HPSS and reports which path it
took — the UI says "fast analysis" instead of "stem-separated".

**Which stem the chords come from matters more than anything else.** Demucs gives piano and
guitar their own stems, so the 4-stem model's `other` bucket is not where the chord instrument
lives. Reading `bass + other` on a piano track meant analysing the bass line alone — 44% root
accuracy. Ranking `piano ▸ guitar ▸ other ▸ bass` and skipping stems below 12% of the loudest
took the same fixture to 98%. The UI reports which stems were used.

**Why beat-synchronous chroma.** Chords change on beats. Averaging chroma between beat
onsets denoises the signal *and* aligns the output to the grid the UI draws, so the
playhead and the highlighted chord never drift apart.

**Why Viterbi.** Frame-wise argmax flickers between relative chords on almost every beat.
A transition prior that penalises changing chord is what turns raw template scores into
something that reads like a real chord chart.

---

## Transposition

Two things have to move together, or you would read one key and hear another:

- **The chart** shifts client-side. Each chord is stored as a pitch class (0–11) plus a
  quality, so transposing is integer arithmetic — instant, no server round trip. Note
  spelling follows the destination key (E♭ major shows flats, E major shows sharps).
- **The audio** is re-rendered server-side by ffmpeg's `rubberband` filter, which shifts
  pitch at **constant tempo**. That matters: the naive `asetrate` trick changes speed too,
  which would desynchronise the entire beat grid. Renders are cached per (job, semitones),
  so switching back and forth between keys is instant after the first time.

The player restores the exact playback position across a source swap and resumes if it was
playing, so changing key does not interrupt what you were working on.

---

## Configuration

Set in `docker-compose.yml` under `api.environment`:

| Variable | Default | Meaning |
| --- | --- | --- |
| `ENABLE_DEMUCS` | `1` | `0` forces the fast HPSS path for every job |
| `DEMUCS_MODEL` | `htdemucs_6s` | 6 stems, including piano and guitar. Costs the same as 4-stem `htdemucs` |
| `DEMUCS_OVERLAP` | `0.1` | Chunk overlap. The 0.25 default spends 25% of compute re-processing seams |
| `ANALYZER_THREADS` | auto | Thread budget. Detected from the container's CPU allowance; override to pin it |
| `MAX_UPLOAD_MB` | `80` | Rejected with 413 above this |
| `MAX_CONCURRENT_JOBS` | `1` | Demucs saturates all cores; >1 makes each job slower |
| `WEB_PORT` | `8080` | Host port for the UI (set in your shell or `.env`) |

Do **not** set `OMP_NUM_THREADS` in the Dockerfile. OpenMP latches the value when the library
initialises (on `import torch`), so `torch.set_num_threads()` afterwards reports the new value
but cannot recover the parallelism. `entrypoint.sh` resolves the budget from the cgroup CPU
quota and exports it before Python starts; `ANALYZER_THREADS` is the supported override.

Tracks are capped at 15 minutes (`MAX_DURATION_SECONDS` in `backend/app/analysis/ingest.py`).
Job artefacts are pruned after 12 hours.

---

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Capabilities: Demucs, rubberband, ffmpeg, yt-dlp, job counts |
| `POST` | `/api/analyze/upload` | multipart `file` + `demucs` → `202 {id}` |
| `POST` | `/api/analyze/youtube` | `{url, demucs}` → `202 {id}` |
| `GET` | `/api/jobs/{id}` | Status + progress; includes `analysis` once done |
| `GET` | `/api/jobs/{id}/audio?semitones=N` | Audio, pitch-shifted if `N ≠ 0` |
| `GET` | `/api/jobs/{id}/stems` | Which stems exist, and whether harmony can be built |
| `GET` | `/api/jobs/{id}/stems/{stem}` | One separated stem as MP3 |
| `GET` | `/api/jobs/{id}/harmony` | Generated vocal harmony as MP3; 422 on an instrumental |
| `DELETE` | `/api/jobs/{id}` | Delete a job and its files |

Analysis is asynchronous: `POST` returns immediately with a job id, and the client polls
`GET /api/jobs/{id}` once a second for a stage name and percentage.

Harmony is the one endpoint that does real work on request rather than during analysis: pitch
tracking the vocal costs ~8s, and adding that to every job would work against making analysis
fast. It renders on the first call and is cached afterwards.

---

## Stems and harmony

A High-accuracy analysis keeps all six separated stems on disk, so the **Download stems** menu
in the UI offers vocals, drums, bass, guitar, piano and other instruments as MP3s.

The same vocal stem drives **Vocal harmony**: backing voices a third and a sixth below the lead,
where each note is snapped to a pitch class that is actually in the chord sounding underneath. A
fixed interval would be wrong about half the time. Shifting happens per *note*, not per frame —
per-frame chases vibrato and tracker jitter and warbles. Instrumental tracks are detected and
reported rather than processed.

---

## Performance

Measured on 6 CPU cores (arm64):

| Mode | Wall clock | Accuracy |
| --- | --- | --- |
| Demucs (`ENABLE_DEMUCS=1`) | ~45–60 s per 2 min of audio | Best — recommended |
| HPSS fallback (`demucs: false`) | ~10–20 s | Good on sparse mixes, weaker on dense ones |

Separation used to be ~4× slower. Three changes, in order of size:

| Change | 33 s clip | 181 s clip | Real 121 s job (end to end) |
| --- | --- | --- | --- |
| Before (`OMP_NUM_THREADS=1`) | 44.8 s | ~220 s | ~110 s |
| Threads resolved in the entrypoint | 17.6 s | 88 s | — |
| + native 44.1 kHz input, `overlap=0.1` | — | 60 s | **55 s** |

Moving from `htdemucs` to `htdemucs_6s` cost nothing measurable — it is a single transformer
pass and the extra stems are just output channels — while raising chord accuracy on a
piano-led fixture from 44% to 98%.

The first Demucs job also downloads ~80 MB of model weights, cached on the volume
afterwards. Give the container at least 4 GB of memory; the compose file sets a 6 GB cap.

---

## Verification

Both layers run in containers — nothing is installed on the host.

```bash
# DSP against synthetic ground truth: tempo, key, meter, chord accuracy, bar alignment
docker-compose exec api python tools/verify.py

# Browser behaviour: rendering, playhead sync, transport, transpose, seek
docker build -t music-analyzer-e2e frontend/e2e
docker run --rm -v "$PWD/frontend/e2e:/work" -v "$PWD/.tmp-shots:/shots" \
  --network music-analyzer_default -e WEB_URL=http://web:80 music-analyzer-e2e
```

The browser check drives a real headless Chromium through the whole flow and writes
screenshots to `.tmp-shots/`. Beyond "does it render", it asserts the things that break
silently: that picking a file does **not** start a job, that the highlighted cell is the beat
whose time window contains the audio clock, that the active beat stays inside the scroll
frame across 14 samples while auto-scrolling (with `scrollTop > 0`, so the check cannot pass
by never having scrolled), that continuation cells are blank, that every separated stem is
offered and its link actually serves `audio/mpeg`, and that no-chord cells stay rare and use a
glyph the browser can render.

`held-beat-check.mjs` covers one case the main run cannot reach on a fixture that changes
chord on every downbeat: what a blank continuation cell shows once the indicator lands on it.

`tools/verify.py` also asserts the two things this round fixed, because both fail silently: that
a quiet chord is not decoded as "no chord" while true silence still is, and that the chord signal
is led by piano, then guitar, then bass.

Last full run: 42/42 browser checks and 5/5 held-beat checks passing, 3/3 DSP cases, and a
121 s track analysed end to end in 55 s with all six stems exported and harmony rendered.

---

## Development

The images are production builds. To iterate on the frontend with hot reload while keeping
everything containerised:

```bash
docker compose up -d api
docker run --rm -it -p 5173:5173 \
  -v "$PWD/frontend:/app" -w /app \
  --network music-analyzer_default \
  -e VITE_API_TARGET=http://api:8000 \
  node:22-alpine sh -c "npm install && npm run dev"
```

Vite proxies `/api` to the backend container, so the relative URLs behave exactly as they
do in production.

---

## Limitations

Worth knowing before you trust the output:

- **Chord quality beyond triads and sevenths** is unreliable. Inversions and slash chords
  are not detected at all — a C/E reads as C.
- **Key changes mid-song** are not modelled. Key detection is global, so a track that
  modulates reports whichever key dominates.
- **Rubato and free time** break beat tracking, and everything downstream is aligned to
  the beat grid. Steady-tempo material works far better.
- **Chord boundaries are accurate to about one beat.** Measured on real music, the harmonic
  rhythm comes out right (spans average one bar) but individual changes can land a beat early
  or late, so roughly half of bars hold a single chord even when every label is correct.
- **Low-confidence beats** are marked with a dotted underline in the chord sheet rather
  than hidden, so you can see where the algorithm was unsure. Confidence is the winning
  chord's margin over the runner-up, so 0.5 means two chords fit equally well.
- **No section labels** (Intro / Verse / Chorus). Structure detection is not implemented, so
  the grid runs continuously with bar numbers as the only orientation.
- **Chords are only as good as the stem they came from.** A track with no clear harmony
  instrument leaves the analysis reading the bass line, where a root note alone cannot
  distinguish major from minor. The UI names the stems it used, so a weak result is at least
  explainable.
- **Generated harmony follows the detected chords**, so it inherits their mistakes: a wrong chord
  produces a harmony note that is wrong in the same way. It also needs a clearly separated lead
  vocal — dense or heavily processed vocals confuse the pitch tracker, and instrumentals are
  rejected outright.
- YouTube occasionally blocks downloads with a bot check. Uploading the file works.
  **`yt-dlp` is intentionally unpinned** — YouTube changes its player constantly and a pinned
  version stops working within months. Rebuild the api image when downloads start failing.
