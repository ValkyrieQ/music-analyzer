/**
 * The chord sheet: one cell per beat, grouped into bars, wrapping across lines — the main
 * Chordify-style view.
 *
 * Three things make this feel right:
 *
 * 1. **The active beat is derived from playback position by binary search** over the beat
 *    times, not by scanning. It runs on every animation frame, so it has to be O(log n).
 *
 * 2. **A chord label is drawn only where the chord changes.** A cell repeating the chord
 *    that is already sounding carries no information and turns the sheet into a wall of
 *    text; blank continuation cells make the harmonic rhythm visible at a glance, which is
 *    the whole point of a bar grid.
 *
 * 3. **Auto-scroll follows the active row, not the active cell.** Scrolling per-beat
 *    yanks the view constantly; keeping the current row parked in the upper third of the
 *    viewport lets you read ahead, which is what you actually need when playing along.
 */

import { useEffect, useLayoutEffect, useMemo, useRef } from 'react'
import { chordParts, romanNumeral } from '../lib/theory.js'

/**
 * Below this, a beat gets a dotted underline meaning "we are not sure".
 *
 * The backend reports confidence as the chosen chord's share against its closest rival, so
 * 0.5 is exactly a dead tie — the point where a second chord fits the beat equally well.
 * That makes it the one non-arbitrary threshold available. On a correctly-analysed track it
 * marks ~29% of beats, 61% of which sit within half a second of a chord change, where the
 * ambiguity is real rather than a failure.
 */
const UNSURE_BELOW = 0.5

/** Index of the last beat whose start is <= `time`, or -1 before the first beat. */
function findActiveBeat(beatTimes, time) {
  let lo = 0
  let hi = beatTimes.length - 1
  let found = -1
  while (lo <= hi) {
    const mid = (lo + hi) >> 1
    if (beatTimes[mid] <= time) {
      found = mid
      lo = mid + 1
    } else {
      hi = mid - 1
    }
  }
  return found
}

/** Chord identity for change detection. `null` root means no-chord. */
const chordKey = (beat) =>
  beat.root === null || beat.root === undefined ? 'N' : `${beat.root}:${beat.quality}`

export default function ChordGrid({
  bars,
  beatTimes,
  position,
  semitones,
  useFlats,
  keyInfo,
  showRomanNumerals,
  isPlaying,
  autoScroll,
  onSeek,
}) {
  const containerRef = useRef(null)
  const activeRowRef = useRef(null)
  const lastScrolledRow = useRef(-1)

  const activeBeat = useMemo(() => findActiveBeat(beatTimes, position), [beatTimes, position])

  // Which rendered row holds the active beat. Bars are laid out by CSS flex-wrap, so we
  // read the DOM offsetTop rather than trying to predict the wrap points.
  const activeBarIndex = useMemo(() => {
    if (activeBeat < 0) return -1
    for (let i = 0; i < bars.length; i += 1) {
      const beats = bars[i].beats
      if (beats.length && activeBeat >= beats[0].beat_index && activeBeat <= beats[beats.length - 1].beat_index) {
        return i
      }
    }
    return -1
  }, [bars, activeBeat])

  /**
   * Flat set of beat indices that start a new chord. Computed across bar boundaries so a
   * chord held from the previous bar stays blank, exactly as on a printed chart.
   */
  const changeBeats = useMemo(() => {
    const set = new Set()
    let previous = null
    for (const bar of bars) {
      for (const beat of bar.beats) {
        const key = chordKey(beat)
        if (key !== previous) {
          set.add(beat.beat_index)
          previous = key
        }
      }
    }
    return set
  }, [bars])

  useLayoutEffect(() => {
    if (!autoScroll || !isPlaying) return
    const row = activeRowRef.current
    const container = containerRef.current
    if (!row || !container) return

    // Measure against the scroll container, not `offsetTop`. `offsetTop` is relative to the
    // nearest *positioned* ancestor — `.bar` sets `position: relative`, so a nested measure
    // silently returns an offset in the wrong coordinate space and the view scrolls to a
    // place the active row is not. Rect deltas plus the current scrollTop are unambiguous.
    const rowTop = row.getBoundingClientRect().top - container.getBoundingClientRect().top + container.scrollTop
    if (Math.abs(rowTop - lastScrolledRow.current) < 1) return
    lastScrolledRow.current = rowTop

    // Park the active row one third down the viewport, so upcoming chords stay visible.
    const target = rowTop - container.clientHeight / 3
    const max = container.scrollHeight - container.clientHeight
    container.scrollTo({ top: Math.min(Math.max(0, target), Math.max(0, max)), behavior: 'smooth' })
  }, [activeBarIndex, autoScroll, isPlaying])

  // Reset the scroll memo when the track changes, otherwise the first bar of a new song
  // can be skipped because its offset matches the previous song's remembered row.
  useEffect(() => {
    lastScrolledRow.current = -1
  }, [bars])

  if (!bars.length) {
    return <p className="grid__empty">No chords were detected in this track.</p>
  }

  return (
    <div className="grid" ref={containerRef}>
      <div className="grid__bars">
        {bars.map((bar, barIndex) => {
          const isActiveBar = barIndex === activeBarIndex
          return (
            <div
              key={bar.index}
              ref={isActiveBar ? activeRowRef : null}
              className={`bar ${isActiveBar ? 'is-active' : ''}`}
            >
              <span className="bar__number">{bar.index + 1}</span>
              <div className="bar__beats">
                {bar.beats.map((beat) => {
                  const isActive = beat.beat_index === activeBeat
                  const isPast = beat.beat_index < activeBeat
                  const isChange = changeBeats.has(beat.beat_index)
                  const { root, suffix } = chordParts(beat.root, beat.quality, semitones, useFlats)
                  const isNoChord = beat.root === null || beat.root === undefined
                  const numeral = showRomanNumerals && keyInfo && !isNoChord
                    ? romanNumeral(beat.root, beat.quality, keyInfo.tonic, keyInfo.mode)
                    : null

                  return (
                    <button
                      type="button"
                      key={beat.beat_index}
                      className={[
                        'beat',
                        isActive ? 'is-active' : '',
                        isPast ? 'is-past' : '',
                        isChange ? 'is-change' : 'is-held',
                        isNoChord ? 'is-empty' : '',
                        // Only the labelled cell carries the confidence mark; underlining
                        // every held beat would smear one uncertain decision across a bar.
                        isChange && beat.confidence < UNSURE_BELOW ? 'is-unsure' : '',
                      ].filter(Boolean).join(' ')}
                      onClick={() => onSeek(beat.start)}
                      // A blank continuation cell has to say what is sounding once it is
                      // the active one, or the playhead lands on an empty box. CSS reads
                      // this back through ::before rather than rendering a second span
                      // that would have to be hidden the rest of the time.
                      data-held={isChange || isNoChord ? undefined : `${root}${suffix}`}
                      title={`Beat ${beat.beat_index + 1} · ${beat.start.toFixed(2)}s · ${
                        isNoChord ? 'no chord' : `${root}${suffix}`
                      } · confidence ${(beat.confidence * 100).toFixed(0)}%`}
                    >
                      {isChange &&
                        (isNoChord ? (
                          // A plain glyph on purpose. This was briefly `𝄽` (U+1D13D
                          // MUSICAL SYMBOL QUARTER REST), which almost no UI font carries —
                          // it rendered as a tofu box that read as a rendering fault rather
                          // than as "no chord here".
                          <span className="beat__dash" aria-label="no chord">
                            –
                          </span>
                        ) : (
                          <>
                            <span className="beat__label">
                              <span className="beat__root">{root}</span>
                              {suffix && <span className="beat__suffix">{suffix}</span>}
                            </span>
                            {numeral && <span className="beat__numeral">{numeral}</span>}
                          </>
                        ))}
                    </button>
                  )
                })}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
