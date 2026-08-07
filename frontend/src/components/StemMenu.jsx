/**
 * Download menu for the separated stems, plus the generated vocal harmony.
 *
 * Only shown for a High-accuracy analysis: the fast path never runs Demucs, so there are
 * no stems to offer and the menu would be an empty promise.
 *
 * The stem links are plain <a download> — the files already exist on disk, so the browser
 * can fetch them directly. Harmony is not: it renders on first request, and on a real
 * 3-4 minute song that has measured at up to 90 seconds (pitch-tracking the vocal, then
 * shifting every note), which reads as broken next to instant stem downloads unless the
 * wait is called out up front. It can also legitimately fail on an instrumental track.
 * Following a link there would replace the page with raw JSON, so it goes through fetch
 * and reports the error in place.
 */

import { useEffect, useRef, useState } from 'react'
import * as api from '../lib/api.js'

export default function StemMenu({ jobId, stems, harmonyAvailable, title }) {
  const [open, setOpen] = useState(false)
  const [harmonyState, setHarmonyState] = useState('idle') // idle | working | error
  const [harmonyError, setHarmonyError] = useState(null)
  const [workingSeconds, setWorkingSeconds] = useState(0)
  const wrapperRef = useRef(null)

  // A tick while rendering, not just a static "building…": the render can run past a
  // minute on a real song, and a label that never changes is indistinguishable from one
  // that is stuck. Reported to the user as "unresponsive" before this existed.
  useEffect(() => {
    if (harmonyState !== 'working') return undefined
    setWorkingSeconds(0)
    const id = setInterval(() => setWorkingSeconds((s) => s + 1), 1000)
    return () => clearInterval(id)
  }, [harmonyState])

  // Close on an outside click or Escape — a menu that can only be dismissed by its own
  // button feels stuck.
  useEffect(() => {
    if (!open) return undefined

    const onPointerDown = (event) => {
      if (!wrapperRef.current?.contains(event.target)) setOpen(false)
    }
    const onKey = (event) => {
      if (event.key === 'Escape') setOpen(false)
    }

    document.addEventListener('mousedown', onPointerDown)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onPointerDown)
      document.removeEventListener('keydown', onKey)
    }
  }, [open])

  if (!stems?.length) return null

  const downloadHarmony = async () => {
    setHarmonyState('working')
    setHarmonyError(null)
    try {
      const blob = await api.fetchHarmony(jobId)
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = `${title || 'track'} - Harmony.mp3`
      document.body.appendChild(link)
      link.click()
      link.remove()
      // Revoking immediately can cancel the download in some browsers; a tick is enough.
      setTimeout(() => URL.revokeObjectURL(url), 10000)
      setHarmonyState('idle')
    } catch (err) {
      setHarmonyError(err.message || 'Could not build the harmony.')
      setHarmonyState('error')
    }
  }

  return (
    <div className="stems" ref={wrapperRef}>
      <button
        type="button"
        className="btn btn--ghost stems__trigger"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        aria-haspopup="true"
      >
        <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true">
          <path
            d="M12 3v12m0 0 4-4m-4 4-4-4M4 18v2a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-2"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.8"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
        Download stems
        <span className="stems__count">{stems.length}</span>
      </button>

      {open && (
        <div className="stems__menu" role="menu">
          <p className="stems__heading">Separated tracks</p>
          {stems.map((stem) => (
            <a
              key={stem.name}
              className="stems__item"
              role="menuitem"
              href={api.stemUrl(jobId, stem.name)}
              download
            >
              <span>{stem.label}</span>
              <small>MP3</small>
            </a>
          ))}

          {harmonyAvailable && (
            <>
              <p className="stems__heading">Generated</p>
              <button
                type="button"
                className="stems__item stems__item--action"
                role="menuitem"
                onClick={downloadHarmony}
                disabled={harmonyState === 'working'}
              >
                <span>
                  Vocal harmony
                  <small className="stems__hint">
                    lead + backing thirds, built from the chords — first build can take
                    a minute or two on a long track
                  </small>
                </span>
                <small>
                  {harmonyState === 'working' ? `building… ${workingSeconds}s` : 'MP3'}
                </small>
              </button>
              {harmonyError && <p className="stems__error">{harmonyError}</p>}
            </>
          )}
        </div>
      )}
    </div>
  )
}
