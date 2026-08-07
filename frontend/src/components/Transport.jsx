/**
 * Transport bar: play/pause/stop, seek, transpose, speed.
 *
 * The transpose control is the interesting one. Changing it must update the chord chart
 * *and* the audio pitch together, or you would be reading one key while hearing another.
 * The chart shifts instantly (it is integer arithmetic on pitch classes); the audio is
 * re-rendered server-side, which is why `isBuffering` is surfaced next to the control.
 */

import { formatTime, keyName, transposeLabel } from '../lib/theory.js'

const SPEEDS = [0.5, 0.75, 0.9, 1, 1.1, 1.25]

export default function Transport({
  isPlaying,
  position,
  duration,
  isBuffering,
  semitones,
  onSemitonesChange,
  speed,
  onSpeedChange,
  keyInfo,
  tempo,
  onToggle,
  onStop,
  onSeek,
  autoScroll,
  onAutoScrollChange,
  showRomanNumerals,
  onShowRomanNumeralsChange,
}) {
  const progress = duration > 0 ? (position / duration) * 100 : 0

  return (
    <div className="transport">
      <div className="transport__row">
        <button
          type="button"
          className="btn btn--primary btn--play"
          onClick={onToggle}
          aria-label={isPlaying ? 'Pause' : 'Play'}
        >
          {isPlaying ? (
            <svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true">
              <rect x="6" y="5" width="4" height="14" rx="1" fill="currentColor" />
              <rect x="14" y="5" width="4" height="14" rx="1" fill="currentColor" />
            </svg>
          ) : (
            <svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true">
              <path d="M8 5.5v13l11-6.5z" fill="currentColor" />
            </svg>
          )}
        </button>

        <button type="button" className="btn btn--icon" onClick={onStop} aria-label="Stop">
          <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
            <rect x="6" y="6" width="12" height="12" rx="1.5" fill="currentColor" />
          </svg>
        </button>

        <span className="transport__time">
          <strong>{formatTime(position)}</strong> / {formatTime(duration)}
        </span>

        <input
          type="range"
          className="transport__seek"
          min={0}
          max={duration || 0}
          step={0.01}
          value={Math.min(position, duration || 0)}
          onChange={(e) => onSeek(Number(e.target.value))}
          aria-label="Seek"
          style={{ '--progress': `${progress}%` }}
        />

        {isBuffering && <span className="transport__buffering">rendering…</span>}
      </div>

      <div className="transport__row transport__row--controls">
        <div className="control">
          <label className="control__label" htmlFor="transpose">
            Key
          </label>
          <div className="control__transpose">
            <button
              type="button"
              className="btn btn--step"
              onClick={() => onSemitonesChange(Math.max(-12, semitones - 1))}
              disabled={semitones <= -12}
              aria-label="Transpose down one semitone"
            >
              −
            </button>
            <div className="control__keyreadout">
              <strong>{keyInfo ? keyName(keyInfo.tonic, keyInfo.mode, semitones) : '—'}</strong>
              <small>{transposeLabel(semitones)}</small>
            </div>
            <button
              type="button"
              className="btn btn--step"
              onClick={() => onSemitonesChange(Math.min(12, semitones + 1))}
              disabled={semitones >= 12}
              aria-label="Transpose up one semitone"
            >
              +
            </button>
            {semitones !== 0 && (
              <button type="button" className="btn btn--ghost" onClick={() => onSemitonesChange(0)}>
                Reset
              </button>
            )}
          </div>
          <input
            id="transpose"
            type="range"
            className="control__slider"
            min={-12}
            max={12}
            step={1}
            value={semitones}
            onChange={(e) => onSemitonesChange(Number(e.target.value))}
          />
        </div>

        <div className="control">
          <span className="control__label">Speed</span>
          <div className="control__speeds">
            {SPEEDS.map((s) => (
              <button
                type="button"
                key={s}
                className={`btn btn--chip ${speed === s ? 'is-active' : ''}`}
                onClick={() => onSpeedChange(s)}
              >
                {s}×
              </button>
            ))}
          </div>
        </div>

        <div className="control control--toggles">
          <label className="toggle">
            <input
              type="checkbox"
              checked={autoScroll}
              onChange={(e) => onAutoScrollChange(e.target.checked)}
            />
            <span>Auto-scroll</span>
          </label>
          <label className="toggle">
            <input
              type="checkbox"
              checked={showRomanNumerals}
              onChange={(e) => onShowRomanNumeralsChange(e.target.checked)}
            />
            <span>Roman numerals</span>
          </label>
          <span className="transport__tempo">{tempo ? `${Math.round(tempo)} BPM` : ''}</span>
        </div>
      </div>
    </div>
  )
}
