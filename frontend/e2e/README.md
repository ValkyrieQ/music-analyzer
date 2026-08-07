# Browser check

Drives a real headless Chromium through the whole flow — upload, analysis, playback,
transpose, seek — and asserts the things `curl` cannot see. Screenshots land in `.tmp-shots/`.

## Running it

Needs the stack up (`docker-compose up -d`) and a fixture to upload:

```bash
# from the repo root
docker-compose exec -T api python tools/make_test_audio.py /tmp/fixture.wav \
  --tempo 120 --key C --progression I,V,vi,IV --bars 8
docker cp music-analyzer-api-1:/tmp/fixture.wav frontend/e2e/fixture.wav

docker build -t music-analyzer-e2e frontend/e2e
docker run --rm -v "$PWD/frontend/e2e:/work" -v "$PWD/.tmp-shots:/shots" \
  --network music-analyzer_default -e WEB_URL=http://web:80 music-analyzer-e2e
```

Set `AUDIO_FILE` to point at a different track (mount it into the container first).

## Why this image is built rather than pulled

`ghcr.io/puppeteer/puppeteer` is published for amd64 only. On an arm64 host its bundled
Chrome starts under QEMU but never comes up, and puppeteer dies with *"Timed out after
30000 ms while waiting for the WS endpoint URL to appear in stdout"* — an error that names no
architecture and reads like a sandbox problem. Debian's `chromium` package is native on both
architectures, so this image pairs it with `puppeteer-core`.

Two details in the Dockerfile that look odd and are load-bearing:

- `puppeteer-core` is installed to `/node_modules`. ESM ignores `NODE_PATH`; it resolves bare
  specifiers by walking up parent directories, so a `-g` install under `/usr/local/lib` is
  invisible to a script bind-mounted at `/work`.
- It is installed in a scratch directory and moved, because `npm install --prefix /` fails
  with *"Tracker idealTree already exists"* against the root-owned npm tree.

`--network music-analyzer_default` with `WEB_URL=http://web:80` addresses the web service by
name. `--network host` does not reach published host ports on Docker Desktop for Mac.

## Four things the script works around

- **`protocolTimeout: 900000`.** `waitForSelector`'s timeout is not enough on its own — each
  poll is a CDP call, and puppeteer tears down the connection after `protocolTimeout` (180s by
  default) no matter what the selector timeout says. It fails with a `ProtocolError` naming
  `.grid__bars`, which looks like "the grid never rendered" when Demucs was simply still going.
- **`usePlayer` builds its `<audio>` with `new Audio()` and never attaches it to the DOM.**
  A detached element plays fine and keeps a stray node out of the layout, but it means
  `document.querySelector('audio')` finds nothing. The script hooks `window.Audio` via
  `evaluateOnNewDocument` to capture the instance. Six checks silently failed on this before
  it was added — every one of them a false alarm about working behaviour.
- **Chrome needs `--autoplay-policy=no-user-gesture-required`**, or `el.play()` rejects and
  playback cannot be scripted at all.
- **A React state change needs its own round trip.** Clicking `.stems__trigger` and reading
  `.stems__item` in the same `page.evaluate` finds nothing: the re-render has not happened when a
  synchronous callback returns. Click, `sleep`, then query.

## What it actually asserts

Sync is checked against the clock, not just "it moved": the highlighted cell must be the beat
whose time window contains `audio.currentTime`, read out of the cell's own `title`.

These assertions exist because the thing they cover can regress while chord labels stay
perfectly correct — no accuracy metric notices any of them:

- **Bar alignment.** A rotated bar grid renders the right chords on the wrong beats.
- **Confidence flag rate.** A miscalibrated scale marks a correct track as uncertain.
- **The active cell stays inside the scroll frame**, sampled 14 times, *and* `scrollTop > 0`.
  Without the second half the check passes by never having scrolled — it did exactly that on
  the first run, because the opening rows fit in the frame. The script now seeks to 55% of the
  track first.
- **Picking a file does not start a job.** The one behaviour the Analyze button exists for.
- **Every separated stem is downloadable and its link serves `audio/mpeg`.** Checked over the
  wire, because a wrong URL shape renders identically in the menu.
- **No-chord cells stay under 5% of beats and contain exactly `–`.** Two independent defects
  produced the reported "dark cell with a 3-stroke line and no chord": a loudness gate that
  mis-decoded quiet chords, and a `𝄽` glyph no UI font carries, which drew as a tofu box. The
  check prints the codepoint on failure so a regression to an undrawable glyph names itself.
- **The chord source is reported** (`chords from piano + bass`), since which stem the harmony was
  read from is the single biggest factor in whether the chords are right.

## `held-beat-check.mjs`

A second, focused script for the one case the main run cannot reach: labels are drawn only
where the chord changes, so most cells are blank, and the synthetic fixtures change chord on
every downbeat — the indicator therefore never lands on a blank cell during a normal run. This
one seeks onto a continuation cell deliberately and asserts the borrowed `data-held` chord
actually paints.

```bash
docker run --rm -v "$PWD/frontend/e2e:/work" \
  --network music-analyzer_default -e WEB_URL=http://web:80 \
  music-analyzer-e2e node /work/held-beat-check.mjs
```

It analyses with High accuracy **off** — it tests rendering, not chord accuracy, and does not
need to wait minutes for stems.

**`position` is only sampled while playing.** The rAF loop in `usePlayer` exists only when
`isPlaying`, so setting `currentTime` on a paused element moves the audio without re-rendering
the grid. A probe that seeks while paused reports a stuck indicator when nothing is wrong —
this cost two false failures here. Press play, seek, then pause.
