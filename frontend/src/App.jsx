/**
 * Application shell: owns the analysis lifecycle (idle -> analysing -> ready) and the
 * view state that the chart and player both depend on (transposition, speed, toggles).
 */

import { useCallback, useEffect, useMemo, useState } from 'react'
import ChordGrid from './components/ChordGrid.jsx'
import SourcePicker from './components/SourcePicker.jsx'
import StemMenu from './components/StemMenu.jsx'
import Transport from './components/Transport.jsx'
import * as api from './lib/api.js'
import { usePlayer } from './lib/usePlayer.js'
import { keyName, preferFlats } from './lib/theory.js'

export default function App() {
  const [phase, setPhase] = useState('idle') // idle | working | ready
  const [job, setJob] = useState(null)
  const [analysis, setAnalysis] = useState(null)
  const [error, setError] = useState(null)
  const [health, setHealth] = useState(null)

  const [demucs, setDemucs] = useState(true)
  const [semitones, setSemitones] = useState(0)
  const [speed, setSpeed] = useState(1)
  const [autoScroll, setAutoScroll] = useState(true)
  const [showRomanNumerals, setShowRomanNumerals] = useState(false)
  const [harmonyMode, setHarmonyMode] = useState(false)

  const harmonyAvailable = analysis?.stems?.some((s) => s.name === 'vocals') ?? false

  const player = usePlayer({
    audioUrl: analysis ? api.audioUrl(analysis.id) : null,
    semitones,
    harmonyUrl: analysis ? api.harmonyWithTrackUrl(analysis.id) : null,
    harmonyMode,
  })

  useEffect(() => {
    api.getHealth().then(setHealth).catch(() => setHealth(null))
  }, [])

  useEffect(() => {
    player.setRate(speed)
  }, [speed, player])

  const start = useCallback(
    async (starter) => {
      setError(null)
      setAnalysis(null)
      setSemitones(0)
      setHarmonyMode(false)
      setPhase('working')

      try {
        const created = await starter()
        setJob({ ...created, progress: 0, stage: 'queued' })

        const finished = await api.pollJob(created.id, {
          onProgress: (update) => setJob(update),
        })
        setAnalysis(finished.analysis)
        setPhase('ready')
      } catch (err) {
        setError(err.message || 'Something went wrong.')
        setPhase('idle')
        setJob(null)
      }
    },
    [],
  )

  const handleFile = useCallback(
    (file) => start(() => api.analyzeUpload(file, { demucs })),
    [demucs, start],
  )
  const handleYouTube = useCallback(
    (url) => start(() => api.analyzeYouTube(url, { demucs })),
    [demucs, start],
  )

  const reset = () => {
    player.stop()
    setPhase('idle')
    setAnalysis(null)
    setJob(null)
    setError(null)
    setSemitones(0)
    setHarmonyMode(false)
  }

  // Spelling follows the *transposed* key, so a chart moved to Eb shows flats even if the
  // original was in E.
  const useFlats = useMemo(() => {
    if (!analysis) return false
    const tonic = (((analysis.key.tonic + semitones) % 12) + 12) % 12
    return preferFlats(tonic, analysis.key.mode)
  }, [analysis, semitones])

  // Keyboard transport: space to play/pause, arrows to nudge, [ / ] to transpose.
  useEffect(() => {
    if (phase !== 'ready') return undefined

    const onKey = (event) => {
      const tag = event.target?.tagName
      if (tag === 'INPUT' || tag === 'TEXTAREA') return

      switch (event.key) {
        case ' ':
          event.preventDefault()
          player.toggle()
          break
        case 'ArrowLeft':
          player.seek(player.position - (event.shiftKey ? 10 : 3))
          break
        case 'ArrowRight':
          player.seek(player.position + (event.shiftKey ? 10 : 3))
          break
        case '[':
          if (!harmonyMode) setSemitones((s) => Math.max(-12, s - 1))
          break
        case ']':
          if (!harmonyMode) setSemitones((s) => Math.min(12, s + 1))
          break
        case 'Escape':
          player.stop()
          break
        default:
      }
    }

    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [phase, player, harmonyMode])

  return (
    <div className="app">
      <header className="header">
        <div className="header__brand">
          <svg viewBox="0 0 24 24" width="26" height="26" aria-hidden="true">
            <path
              d="M9 18V5l10-2v13M9 18a3 3 0 1 1-6 0 3 3 0 0 1 6 0Zm10-2a3 3 0 1 1-6 0 3 3 0 0 1 6 0Z"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.7"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
          <h1>Music Analyzer</h1>
        </div>
        {analysis && (
          <button type="button" className="btn btn--ghost" onClick={reset}>
            Analyze another track
          </button>
        )}
      </header>

      <main className="main">
        {phase === 'idle' && (
          <section className="panel panel--intro">
            <h2>Find the tempo, key and chords of any song</h2>
            <p className="panel__lead">
              Upload a track or paste a YouTube link. You get a beat-aligned chord sheet you
              can play along with — and transpose to any key without the audio going out of
              tune.
            </p>
            {error && <div className="alert alert--error">{error}</div>}
            <SourcePicker
              onFile={handleFile}
              onYouTube={handleYouTube}
              busy={false}
              demucs={demucs}
              onDemucsChange={setDemucs}
            />
            {health && !health.demucs_enabled && (
              <div className="alert alert--warn">
                Stem separation is disabled on this server — chord accuracy will be lower on
                dense mixes.
              </div>
            )}
          </section>
        )}

        {phase === 'working' && (
          <section className="panel panel--working">
            <div className="spinner" aria-hidden="true" />
            <h2>{job?.title || 'Analyzing…'}</h2>
            <p className="working__stage">{job?.stage || 'starting'}</p>
            <div className="progress">
              <div className="progress__bar" style={{ width: `${job?.progress || 0}%` }} />
            </div>
            <p className="working__hint">
              {demucs
                ? 'Separating the stems for the most accurate chords. On a long track this can take a few minutes.'
                : 'Running fast analysis.'}
            </p>
          </section>
        )}

        {phase === 'ready' && analysis && (
          <>
            <section className="summary">
              <div className="summary__title">
                <h2>{analysis.title}</h2>
                <span className="summary__source">
                  {analysis.source_type === 'youtube' ? 'YouTube' : 'Uploaded file'} ·{' '}
                  {analysis.separation_method === 'demucs' ? 'stem-separated' : 'fast analysis'}
                  {/* Which instrument the chords were read from. Worth showing: it is the
                      single biggest factor in whether they are right, and it explains a
                      weak result on a track with no clear harmony instrument. */}
                  {analysis.harmonic_sources?.length > 0 &&
                    analysis.separation_method === 'demucs' && (
                      <> · chords from {analysis.harmonic_sources.join(' + ')}</>
                    )}
                </span>
                <StemMenu
                  jobId={analysis.id}
                  stems={analysis.stems}
                  harmonyAvailable={analysis.stems?.some((s) => s.name === 'vocals')}
                  title={analysis.title}
                />
              </div>
              <div className="stats">
                <div className="stat">
                  <span className="stat__label">Tempo</span>
                  <span className="stat__value">{Math.round(analysis.tempo)}</span>
                  <span className="stat__unit">BPM</span>
                </div>
                <div className="stat">
                  <span className="stat__label">Key</span>
                  <span className="stat__value stat__value--key">
                    {keyName(analysis.key.tonic, analysis.key.mode, semitones)}
                  </span>
                  {semitones !== 0 && (
                    <span className="stat__unit">
                      from {keyName(analysis.key.tonic, analysis.key.mode, 0)}
                    </span>
                  )}
                </div>
                <div className="stat">
                  <span className="stat__label">Meter</span>
                  <span className="stat__value">{analysis.beats_per_bar}/4</span>
                </div>
                <div className="stat">
                  <span className="stat__label">Chords</span>
                  <span className="stat__value">{analysis.chords.length}</span>
                  <span className="stat__unit">changes</span>
                </div>
              </div>
            </section>

            {player.error && <div className="alert alert--error">{player.error}</div>}

            <Transport
              isPlaying={player.isPlaying}
              position={player.position}
              duration={player.duration || analysis.duration}
              isBuffering={player.isBuffering}
              semitones={semitones}
              onSemitonesChange={setSemitones}
              speed={speed}
              onSpeedChange={setSpeed}
              keyInfo={analysis.key}
              tempo={analysis.tempo}
              onToggle={player.toggle}
              onStop={player.stop}
              onSeek={player.seek}
              autoScroll={autoScroll}
              onAutoScrollChange={setAutoScroll}
              showRomanNumerals={showRomanNumerals}
              onShowRomanNumeralsChange={setShowRomanNumerals}
              harmonyAvailable={harmonyAvailable}
              harmonyMode={harmonyMode}
              onHarmonyModeChange={setHarmonyMode}
            />

            <section className="panel panel--grid">
              <h3 className="panel__heading">
                Chord sheet
                <small>One cell per beat · a chord is shown where it changes · click to jump</small>
              </h3>
              <ChordGrid
                bars={analysis.bars}
                beatTimes={analysis.beat_times}
                position={player.position}
                semitones={semitones}
                useFlats={useFlats}
                keyInfo={analysis.key}
                showRomanNumerals={showRomanNumerals}
                isPlaying={player.isPlaying}
                autoScroll={autoScroll}
                onSeek={player.seek}
              />
            </section>
          </>
        )}
      </main>

      <footer className="footer">
        <span>
          Analysis is automatic and imperfect — treat it as a strong first draft, not gospel.
        </span>
        {phase === 'ready' && (
          <span className="footer__keys">
            space play/pause · ← → seek · [ ] transpose · esc stop
          </span>
        )}
      </footer>
    </div>
  )
}
