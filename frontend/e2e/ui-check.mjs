/**
 * Browser check for the parts curl cannot verify: that the chart renders, that the active
 * beat tracks playback and stays visible while auto-scrolling, and that transposing updates
 * the chords.
 *
 * Run from the repo root (needs a running stack on WEB_URL):
 *
 *   docker build -t music-analyzer-e2e frontend/e2e
 *   docker run --rm -v "$PWD/frontend/e2e:/work" -v "$PWD/.tmp-shots:/shots" \
 *     --network music-analyzer_default -e WEB_URL=http://web:80 music-analyzer-e2e
 *
 * Joins the compose network and addresses the web service by name rather than using
 * --network host, which does not bridge to the host's ports on Docker Desktop for Mac.
 */

import puppeteer from 'puppeteer-core'

const WEB_URL = process.env.WEB_URL || 'http://web:80'
const AUDIO = process.env.AUDIO_FILE || '/work/fixture.wav'
const SHOTS = process.env.SHOT_DIR || '/shots'

const checks = []
const record = (name, pass, detail = '') => {
  checks.push({ name, pass, detail })
  console.log(`${pass ? 'PASS' : 'FAIL'}  ${name}${detail ? ` — ${detail}` : ''}`)
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

const browser = await puppeteer.launch({
  headless: true,
  executablePath: process.env.CHROME_PATH || '/usr/bin/chromium',
  // Demucs runs for minutes. waitForSelector's own timeout is not enough: each poll is a
  // CDP call, and puppeteer aborts the *connection* after protocolTimeout (180s default)
  // regardless of the selector timeout, killing the whole run mid-analysis.
  protocolTimeout: 900000,
  args: [
    '--no-sandbox',
    '--disable-dev-shm-usage',
    // Let <audio>.play() succeed without a user gesture, so playback can be scripted.
    '--autoplay-policy=no-user-gesture-required',
  ],
})

try {
  const page = await browser.newPage()
  await page.setViewport({ width: 1440, height: 1000 })

  const errors = []
  page.on('console', (m) => m.type() === 'error' && errors.push(m.text()))
  page.on('pageerror', (e) => errors.push(String(e)))

  const audioRequests = []
  page.on('request', (r) => {
    if (/\/audio(\?|$)/.test(r.url())) audioRequests.push(r.url())
  })

  // usePlayer builds its element with `new Audio()` and deliberately never attaches it to
  // the document — a detached element plays fine and avoids a stray node in the layout.
  // So querySelector('audio') finds nothing; hook the constructor instead to get a handle
  // on the real instance. Must be installed before any app script runs.
  await page.evaluateOnNewDocument(() => {
    const Native = window.Audio
    window.__audio = null
    window.Audio = function (...args) {
      const el = new Native(...args)
      window.__audio = el
      return el
    }
    window.Audio.prototype = Native.prototype
  })

  const audioState = () =>
    page.evaluate(() => {
      const el = window.__audio
      return el
        ? {
            currentTime: el.currentTime,
            paused: el.paused,
            duration: el.duration,
            src: el.getAttribute('src'),
            rate: el.playbackRate,
          }
        : null
    })

  // --- load ---
  await page.goto(WEB_URL, { waitUntil: 'networkidle2', timeout: 60000 })
  record('page loads', true, await page.title())
  await page.screenshot({ path: `${SHOTS}/01-landing.png` })

  const hasDropzone = await page.$('.dropzone')
  record('landing UI renders', Boolean(hasDropzone))

  // --- source selection must NOT start the job ---
  const analyzeDisabled = await page.$eval('.btn--analyze', (b) => b.disabled)
  record('analyze is disabled with no source', analyzeDisabled === true)

  const input = await page.$('input[type=file]')
  await input.uploadFile(AUDIO)
  await sleep(400)

  const afterPick = await page.evaluate(() => ({
    working: Boolean(document.querySelector('.panel--working')),
    name: document.querySelector('.dropzone__file')?.textContent || null,
    enabled: document.querySelector('.btn--analyze')?.disabled === false,
  }))
  // The whole point of the Analyze button: picking a file commits to nothing. Analysis with
  // stem separation runs for minutes and cannot be cancelled, so a stray drop must not
  // start it.
  record('picking a file does not start analysis', afterPick.working === false)
  record('chosen file is shown', Boolean(afterPick.name), afterPick.name || 'no name rendered')
  record('analyze becomes enabled', afterPick.enabled === true)

  // --- analyze ---
  await page.click('.btn--analyze')
  await page.waitForSelector('.panel--working', { timeout: 15000 })
  await page.screenshot({ path: `${SHOTS}/02-analyzing.png` })
  record('analyze starts the job', true)

  // Analysis with Demucs takes minutes; the default 30s timeout is far too short.
  await page.waitForSelector('.grid__bars', { timeout: 600000 })
  record('analysis completed', true)
  await page.screenshot({ path: `${SHOTS}/03-result.png`, fullPage: true })

  // --- results content ---
  const summary = await page.evaluate(() => {
    const stats = [...document.querySelectorAll('.stat')].map((s) => ({
      label: s.querySelector('.stat__label')?.textContent,
      value: s.querySelector('.stat__value')?.textContent,
    }))
    return {
      stats,
      beats: document.querySelectorAll('.beat').length,
      bars: document.querySelectorAll('.bar').length,
      labelled: document.querySelectorAll('.beat.is-change').length,
      held: document.querySelectorAll('.beat.is-held').length,
      chords: [...document.querySelectorAll('.beat__root')].slice(0, 8).map((e) => e.textContent),
    }
  })
  record('chord grid rendered', summary.beats > 0, `${summary.beats} beats / ${summary.bars} bars`)
  record('scale view is gone', (await page.$('.panel--lanes')) === null)

  // A chord label belongs only on the beat where the chord changes. Every cell carrying a
  // label means the sheet has regressed to a wall of repeated text, which is what the bar
  // grid exists to avoid.
  const heldBlank = await page.evaluate(() =>
    [...document.querySelectorAll('.beat.is-held')].every((c) => c.textContent.trim() === ''),
  )
  record('held beats are blank', heldBlank, `${summary.labelled} labelled / ${summary.beats} beats`)
  record(
    'labels are sparser than beats',
    summary.labelled > 0 && summary.labelled < summary.beats,
    `${summary.labelled} labels for ${summary.beats} beats`,
  )

  // Bars must hold one chord each on this fixture (it changes chord once per bar), which
  // is what proves the downbeat phase is right. A rotated grid still renders correct chord
  // *labels*, so the only visible symptom is every change landing on the wrong beat.
  // With continuation cells blank, "one chord per bar" means at most one label per interior
  // bar, and it must sit on the first beat.
  const barAlign = await page.evaluate(() => {
    const bars = [...document.querySelectorAll('.bar')].slice(1, -1)
    const onDownbeat = bars.filter((b) => {
      const cells = [...b.querySelectorAll('.beat')]
      const labelled = cells.filter((c) => c.classList.contains('is-change'))
      return labelled.length === 0 || (labelled.length === 1 && labelled[0] === cells[0])
    }).length
    return { total: bars.length, onDownbeat }
  })
  record(
    'chord changes land on the barline',
    barAlign.total > 0 && barAlign.onDownbeat === barAlign.total,
    `${barAlign.onDownbeat}/${barAlign.total} interior bars change only on beat 1`,
  )

  // Guards the confidence calibration. A vocabulary-sized softmax posterior caps out around
  // 0.36, which would flag essentially every beat here as unsure despite all labels being
  // correct — so a high flag rate on a clean synthetic fixture means the scale regressed.
  const unsure = await page.evaluate(
    () => document.querySelectorAll('.beat.is-unsure:not(.is-empty)').length,
  )
  record(
    'confidence marks few beats unsure',
    unsure / Math.max(1, summary.labelled) < 0.25,
    `${unsure}/${summary.labelled} labelled beats flagged`,
  )
  // The reported bug was dark cells carrying a three-stroke mark and no chord, on beats where
  // the music plainly does change chord. Two independent defects produced it: the loudness
  // gate mis-decoded quiet chords as "no chord", and the marker itself was `𝄽`, a glyph no UI
  // font carries, so it drew as a tofu box. This fixture is continuous harmony throughout, so
  // a run of empty cells here means the gate has regressed.
  const empties = await page.evaluate(() => {
    const cells = [...document.querySelectorAll('.beat.is-empty')]
    return {
      count: cells.length,
      // A continuation N beat (is-held, not is-change) is deliberately blank — the label
      // is only drawn on the beat where "no chord" starts. Only check the glyph on cells
      // that actually carry a label, or every held N beat reads as a missing glyph.
      // A tofu box is a single char outside the BMP; the dash is U+2013.
      glyphs: [...new Set(
        cells.filter((c) => c.classList.contains('is-change')).map((c) => c.textContent.trim()),
      )],
    }
  })
  record(
    'harmonic beats are not decoded as no-chord',
    empties.count / Math.max(1, summary.beats) < 0.05,
    `${empties.count}/${summary.beats} beats empty`,
  )
  record(
    'no-chord cells use a renderable glyph',
    empties.glyphs.every((g) => g === '–'),
    empties.glyphs.length ? empties.glyphs.map((g) => `${g} U+${g.codePointAt(0)?.toString(16).toUpperCase()}`).join(', ') : 'no empty cells',
  )

  record(
    'tempo & key shown',
    summary.stats.some((s) => s.label === 'Tempo') && summary.stats.some((s) => s.label === 'Key'),
    summary.stats.map((s) => `${s.label}=${s.value}`).join(' '),
  )

  // --- stem download menu (High accuracy only) ---
  // The fixture is analysed with Demucs on, so all six stems must be offered. Asserted
  // through the real menu rather than the API, because the menu is the only way a user can
  // reach them and it renders from a different field than the chord sheet does.
  const hasTrigger = Boolean(await page.$('.stems__trigger'))
  record('stem download menu is offered', hasTrigger)
  // Opening is a React state change, so the click and the query cannot share one evaluate
  // call — the re-render has not happened when a synchronous callback returns.
  if (hasTrigger) await page.click('.stems__trigger')
  await sleep(300)
  const menu = await page.evaluate(() => ({
    items: [...document.querySelectorAll('.stems__item')].map((el) => ({
      text: el.querySelector('span')?.textContent?.trim(),
      href: el.getAttribute('href'),
      isAction: el.classList.contains('stems__item--action'),
    })),
  }))
  const stemLinks = menu.items.filter((i) => i.href)
  record(
    'every separated stem is downloadable',
    ['Vocals', 'Drums', 'Bass', 'Guitar', 'Piano'].every((name) =>
      stemLinks.some((i) => i.text === name),
    ),
    stemLinks.map((i) => i.text).join(', ') || 'none',
  )
  record(
    'vocal harmony is offered',
    menu.items.some((i) => i.isAction && /harmony/i.test(i.text || '')),
    menu.items.find((i) => i.isAction)?.text || 'no harmony entry',
  )

  // The links have to actually serve audio, not a 404 page. Checked over the wire because a
  // wrong URL shape renders identically in the menu.
  const probeHref = stemLinks.find((i) => i.text === 'Piano')?.href || stemLinks[0]?.href
  const stemFetch = probeHref
    ? await page.evaluate(async (href) => {
        const res = await fetch(href)
        return { status: res.status, type: res.headers.get('content-type') }
      }, probeHref)
    : { status: 0, type: 'no link to probe' }
  record(
    'a stem link serves audio',
    stemFetch.status === 200 && /audio/.test(stemFetch.type || ''),
    `HTTP ${stemFetch.status} ${stemFetch.type}`,
  )

  // Close the menu again so it does not overlay the cells the later checks click.
  await page.keyboard.press('Escape')
  await sleep(200)
  record(
    'menu closes on escape',
    (await page.$('.stems__menu')) === null,
  )

  // Which instrument the chords came from is shown, since it is the main determinant of
  // whether they are right.
  const provenance = await page.$eval('.summary__source', (el) => el.textContent)
  record(
    'chord source is reported',
    /chords from/i.test(provenance),
    provenance.replace(/\s+/g, ' ').trim(),
  )

  // --- "play with harmony" toggle ---
  // The vocals stem is always exported on a Demucs job, so the toggle offers itself
  // regardless of whether the track actually has singable content — this fixture is a
  // synthetic instrumental with no real vocal, so the point here is that turning the
  // toggle on surfaces a clear, contained error rather than hanging or breaking playback,
  // not that harmony actually renders (a real vocal track is exercised separately).
  const harmonyToggleLabel = await page.evaluateHandle(() =>
    [...document.querySelectorAll('.control--toggles .toggle')].find((l) =>
      l.textContent.includes('Play with harmony'),
    ),
  )
  const hasHarmonyToggle = await page.evaluate((el) => Boolean(el), harmonyToggleLabel)
  record('play-with-harmony toggle is offered', hasHarmonyToggle)

  if (hasHarmonyToggle) {
    await page.evaluate((el) => el.querySelector('input').click(), harmonyToggleLabel)
    // Building on an instrumental fixture 422s quickly; a real vocal render can take over
    // a minute, so this waits for either outcome rather than a fixed sleep.
    await page
      .waitForFunction(
        () =>
          document.querySelector('.alert--error') ||
          !document.querySelector('.transport__buffering'),
        { timeout: 120000 },
      )
      .catch(() => {})

    const afterToggle = await page.evaluate(() => ({
      transposeDisabled: document
        .querySelectorAll('.control__transpose .btn--step')[0]
        ?.disabled,
      error: document.querySelector('.alert--error')?.textContent || null,
    }))
    // Whichever way it resolves (harmony built, or the fixture's lack of a real vocal
    // rejected it), the UI must stay coherent: transpose disabled while harmony mode is
    // on, and any failure reported in place rather than silently doing nothing.
    record(
      'harmony mode disables transpose while active',
      afterToggle.transposeDisabled === true,
      `transpose btn disabled=${afterToggle.transposeDisabled}`,
    )

    // Turn it back off so later checks (transpose, seek) run against the normal mix.
    await page.evaluate((el) => el.querySelector('input').click(), harmonyToggleLabel)
    await sleep(500)
    record(
      'turning harmony mode off re-enables transpose',
      (await page.$eval(
        '.control__transpose .btn--step',
        (b) => b.disabled,
      )) === false,
    )
  }

  // --- playback and the active-beat indicator ---
  const view = () =>
    page.evaluate(() => {
      const cells = [...document.querySelectorAll('.beat')]
      const active = cells.find((c) => c.classList.contains('is-active'))
      const grid = document.querySelector('.grid')
      let visible = null
      if (active && grid) {
        const a = active.getBoundingClientRect()
        const g = grid.getBoundingClientRect()
        // Fully inside the scroll frame, not clipped off either edge. Auto-scrolling the
        // active row out of view above the top of the frame was the original bug here.
        visible = a.top >= g.top - 1 && a.bottom <= g.bottom + 1
      }
      return {
        index: active ? cells.indexOf(active) : -1,
        count: cells.filter((c) => c.classList.contains('is-active')).length,
        activeLabel: active?.textContent.trim() || active?.dataset.held || null,
        visible,
        clock: document.querySelector('.transport__time strong')?.textContent,
      }
    })

  await page.click('.btn--play')
  await sleep(1200)
  const a1 = await audioState()
  const v1 = await view()
  await sleep(2000)
  const a2 = await audioState()
  const v2 = await view()

  record(
    'audio plays',
    a1?.paused === false && a2.currentTime > a1.currentTime,
    `t=${a1?.currentTime.toFixed(2)} → ${a2.currentTime.toFixed(2)}s of ${a2.duration?.toFixed(2)}s`,
  )
  record('indicator advances', v2.index > v1.index, `beat ${v1.index} → ${v2.index}`)
  record('exactly one beat is active', v2.count === 1, `${v2.count} cell — ${v2.activeLabel}`)
  record('clock readout ticks', v1.clock !== v2.clock, `${v1.clock} → ${v2.clock}`)

  // The highlighted chord must be the one sounding now, not merely some cell. This replaces
  // the old numeric playhead-transform check: the active cell *is* the indicator now.
  const beatCheck = await page.evaluate((t) => {
    const cells = [...document.querySelectorAll('.beat')]
    const active = cells.findIndex((c) => c.classList.contains('is-active'))
    // Beat start times live in the title attribute: "Beat 7 · 3.01s · C · confidence 62%".
    const startOf = (i) => Number(cells[i]?.getAttribute('title')?.match(/·\s*([\d.]+)s/)?.[1])
    return { active, start: startOf(active), next: startOf(active + 1), audio: t }
  }, a2.currentTime)
  record(
    'highlighted beat is the sounding beat',
    beatCheck.audio >= beatCheck.start - 0.15 &&
      (Number.isNaN(beatCheck.next) || beatCheck.audio <= beatCheck.next + 0.15),
    `audio ${beatCheck.audio.toFixed(2)}s within beat ${beatCheck.active} [${beatCheck.start}s, ${beatCheck.next}s)`,
  )

  await page.screenshot({ path: `${SHOTS}/04-playing.png`, fullPage: true })

  // --- auto-scroll keeps the indicator on screen ---
  // The regression this guards: the grid scrolled with the beat but parked the active row
  // above the top edge of its own scroll frame, so the thing you need to look at was the
  // one thing you could not see.
  //
  // Jump into the middle of the track first. The first few rows fit inside the frame
  // without scrolling at all, so playing from the start exercises none of this — the bug
  // only appears once scrollTop is genuinely non-zero.
  await page.evaluate((d) => {
    window.__audio.currentTime = d * 0.55
  }, a2.duration)
  await sleep(1500)

  const visibility = []
  for (let i = 0; i < 14; i += 1) {
    // eslint-disable-next-line no-await-in-loop
    const v = await view()
    if (v.index >= 0) visibility.push({ index: v.index, visible: v.visible })
    // eslint-disable-next-line no-await-in-loop
    await sleep(700)
  }
  const offscreen = visibility.filter((v) => v.visible === false)
  record(
    'active beat stays inside the scroll frame',
    visibility.length > 3 && offscreen.length === 0,
    `${visibility.length - offscreen.length}/${visibility.length} samples visible` +
      (offscreen.length ? ` — first miss at beat ${offscreen[0].index}` : ''),
  )
  // Asserted strictly, so the visibility check above cannot pass vacuously by never having
  // scrolled in the first place.
  const scrolled = await page.$eval('.grid', (g) => ({
    top: g.scrollTop,
    max: g.scrollHeight - g.clientHeight,
  }))
  record(
    'grid auto-scrolled to follow playback',
    scrolled.top > 0,
    `scrollTop=${Math.round(scrolled.top)} of ${Math.round(scrolled.max)}`,
  )
  await page.screenshot({ path: `${SHOTS}/04b-scrolled.png` })

  // --- pause / stop ---
  await page.click('.btn--play')
  await sleep(500)
  const paused = await audioState()
  const heldAt = paused.currentTime
  await sleep(700)
  const stillHeld = await audioState()
  record('pause works', paused.paused === true, `t=${heldAt.toFixed(2)}s`)
  record(
    'pause holds position',
    Math.abs(stillHeld.currentTime - heldAt) < 0.05,
    `${heldAt.toFixed(2)}s → ${stillHeld.currentTime.toFixed(2)}s after 700ms`,
  )

  await page.click('.btn--icon')
  await sleep(400)
  const stopped = await audioState()
  const atStop = await view()
  record('stop resets to 0', stopped.paused && stopped.currentTime === 0, `t=${stopped.currentTime}`)
  record(
    'indicator returns to the start',
    atStop.index <= 0,
    `active beat index ${atStop.index}`,
  )

  // --- transpose ---
  const before = await page.evaluate(() => ({
    chords: [...document.querySelectorAll('.beat__root')].slice(0, 6).map((e) => e.textContent),
    key: document.querySelector('.stat__value--key')?.textContent,
  }))

  // The +/- steppers are the 2nd and 3rd .btn--step in the transpose control.
  await page.evaluate(() => {
    const steps = document.querySelectorAll('.control__transpose .btn--step')
    steps[steps.length - 1].click() // "+"
  })
  await sleep(900)

  const after = await page.evaluate(() => ({
    chords: [...document.querySelectorAll('.beat__root')].slice(0, 6).map((e) => e.textContent),
    key: document.querySelector('.stat__value--key')?.textContent,
    labelled: document.querySelectorAll('.beat.is-change').length,
  }))
  const afterAudio = await audioState()

  record(
    'transpose changes chords',
    JSON.stringify(before.chords) !== JSON.stringify(after.chords),
    `${before.chords.slice(0, 4).join(',')} → ${after.chords.slice(0, 4).join(',')}`,
  )
  record('transpose changes key readout', before.key !== after.key, `${before.key} → ${after.key}`)
  // Transposition is pitch-class arithmetic, so it must not move a single chord boundary.
  record(
    'transpose keeps the same chord changes',
    after.labelled === summary.labelled,
    `${after.labelled} labels (was ${summary.labelled})`,
  )
  record(
    'transpose re-requests shifted audio',
    (afterAudio?.src || '').includes('semitones=1'),
    afterAudio?.src || 'no src',
  )
  record(
    'shifted audio actually served',
    audioRequests.some((u) => u.includes('semitones=1')),
    audioRequests.map((u) => u.replace(/^https?:\/\/[^/]+/, '')).join(' '),
  )
  await page.screenshot({ path: `${SHOTS}/05-transposed.png`, fullPage: true })

  // --- seek by clicking a beat ---
  const target = await page.evaluate(() => {
    const cell = document.querySelectorAll('.beat')[12]
    cell?.click()
    return Number(cell?.getAttribute('title')?.match(/·\s*([\d.]+)s/)?.[1])
  })
  await sleep(600)
  const seeked = await audioState()
  record(
    'click-to-seek jumps to that beat',
    Math.abs(seeked.currentTime - target) < 0.2,
    `clicked beat at ${target}s, audio at ${seeked.currentTime.toFixed(2)}s`,
  )

  // --- half speed, for learning a part ---
  await page.evaluate(() => {
    const chip = [...document.querySelectorAll('.control__speeds .btn--chip')].find((b) =>
      b.textContent.includes('0.5'),
    )
    chip?.click()
  })
  await sleep(300)
  const slowed = await audioState()
  record('speed control applies', slowed.rate === 0.5, `playbackRate=${slowed.rate}`)

  // The harmony-toggle check above deliberately provokes a 422 on this instrumental
  // fixture (no vocal to harmonise) and the app reports it correctly via the fetch's own
  // rejection — Chrome also logs the underlying network response as a console error,
  // which is App reacting correctly to an expected condition, not a bug.
  const unexpectedErrors = errors.filter((e) => !/422 \(Unprocessable Entity\)/.test(e))
  record(
    'no console errors',
    unexpectedErrors.length === 0,
    unexpectedErrors.slice(0, 2).join(' | ') || 'clean',
  )

  console.log('\n' + '='.repeat(70))
  const failed = checks.filter((c) => !c.pass)
  console.log(`${checks.length - failed.length}/${checks.length} checks passed`)
  if (failed.length) {
    console.log('failed: ' + failed.map((f) => f.name).join(', '))
    process.exit(1)
  }
} finally {
  await browser.close()
}
