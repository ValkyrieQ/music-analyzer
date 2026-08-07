/**
 * Focused check on one detail the main run cannot reach reliably: what a *continuation*
 * cell shows while it is the active beat.
 *
 * Labels are drawn only where the chord changes, which means three of every four cells in a
 * held bar are blank. If the indicator lands on one of those and shows nothing, the player
 * appears to lose its place mid-bar. The blank cell borrows its chord back through
 * `data-held` + a CSS ::before, so this asserts the attribute is present and the rendered
 * text is non-empty once the cell goes active.
 *
 * Run against a stack that already has a result on screen is not possible (each run starts
 * fresh), so this analyses with demucs off — it only needs a grid, not accuracy.
 */

import puppeteer from 'puppeteer-core'

const WEB_URL = process.env.WEB_URL || 'http://web:80'
const AUDIO = process.env.AUDIO_FILE || '/work/fixture.wav'

const browser = await puppeteer.launch({
  headless: true,
  executablePath: '/usr/bin/chromium',
  protocolTimeout: 900000,
  args: ['--no-sandbox', '--disable-dev-shm-usage', '--autoplay-policy=no-user-gesture-required'],
})

let failures = 0
const record = (name, pass, detail = '') => {
  if (!pass) failures += 1
  console.log(`${pass ? 'PASS' : 'FAIL'}  ${name}${detail ? ` — ${detail}` : ''}`)
}

try {
  const page = await browser.newPage()
  await page.setViewport({ width: 1440, height: 1000 })
  await page.evaluateOnNewDocument(() => {
    const Native = window.Audio
    window.__audio = null
    window.Audio = function (...a) {
      const el = new Native(...a)
      window.__audio = el
      return el
    }
    window.Audio.prototype = Native.prototype
  })

  await page.goto(WEB_URL, { waitUntil: 'networkidle2', timeout: 60000 })
  // Fast path: this check is about rendering, not chord accuracy.
  await page.evaluate(() => {
    const box = document.querySelector('.toggle--quality input')
    if (box.checked) box.click()
  })
  await (await page.$('input[type=file]')).uploadFile(AUDIO)
  await page.click('.btn--analyze')
  await page.waitForSelector('.grid__bars', { timeout: 600000 })

  // Pick a held cell that is not the first beat of its bar, then seek exactly to it.
  const held = await page.evaluate(() => {
    const cell = [...document.querySelectorAll('.beat.is-held')].find(
      (c) => c.previousElementSibling && !c.classList.contains('is-empty'),
    )
    if (!cell) return null
    const cells = [...document.querySelectorAll('.beat')]
    return {
      index: cells.indexOf(cell),
      dataHeld: cell.dataset.held || null,
      blank: cell.textContent.trim() === '',
      start: Number(cell.getAttribute('title')?.match(/·\s*([\d.]+)s/)?.[1]),
    }
  })
  record('found a held continuation cell', Boolean(held), held ? `beat ${held.index}` : 'none')
  if (!held) process.exit(1)

  record('held cell is blank in the DOM', held.blank === true)
  record('held cell carries its chord in data-held', Boolean(held.dataHeld), held.dataHeld || 'missing')

  // Seek onto it and confirm the ::before actually paints something.
  //
  // Playback has to be running: `position` is sampled in a requestAnimationFrame loop that
  // only exists while playing, so setting `currentTime` on a paused element moves the audio
  // but never re-renders the grid. Click play, then land on the beat.
  await page.click('.btn--play')
  await new Promise((r) => setTimeout(r, 400))
  await page.evaluate((t) => {
    window.__audio.currentTime = t + 0.1
  }, held.start)
  await new Promise((r) => setTimeout(r, 400))
  await page.click('.btn--play') // pause, so the cell does not advance under the assertions
  await new Promise((r) => setTimeout(r, 200))

  const shown = await page.evaluate(() => {
    const cell = document.querySelector('.beat.is-active')
    if (!cell) return null
    const before = getComputedStyle(cell, '::before')
    return {
      isHeld: cell.classList.contains('is-held'),
      dataHeld: cell.dataset.held || null,
      content: before.content,
      fontSize: before.fontSize,
    }
  })

  record('a held cell is the active one', Boolean(shown?.isHeld), shown ? `data-held=${shown.dataHeld}` : 'no active cell')
  record(
    'active held cell renders its borrowed chord',
    Boolean(shown?.content) && shown.content !== 'none' && shown.content !== 'normal',
    `content=${shown?.content} at ${shown?.fontSize}`,
  )

  console.log(failures ? `\n${failures} check(s) failed` : '\nall held-beat checks passed')
  if (failures) process.exit(1)
} finally {
  await browser.close()
}
