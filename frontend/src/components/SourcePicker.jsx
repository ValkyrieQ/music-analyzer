/**
 * Entry point: pick a file (drop or browse) or paste a YouTube URL, choose the accuracy
 * mode, then press Analyze.
 *
 * Selecting a source deliberately does **not** start the job. Analysis with stem separation
 * takes minutes and cannot be cancelled, so committing to it on a drag-and-drop — an action
 * that is easy to do by accident, and easy to do twice — was the wrong default. The file is
 * held locally until the user presses the button.
 */

import { useRef, useState } from 'react'

export default function SourcePicker({ onFile, onYouTube, busy, demucs, onDemucsChange }) {
  const inputRef = useRef(null)
  const [dragging, setDragging] = useState(false)
  const [file, setFile] = useState(null)
  const [url, setUrl] = useState('')

  // A file wins over a URL: it is the more explicit gesture, and the field may just hold
  // a leftover paste.
  const ready = Boolean(file) || url.trim().length > 0

  const handleDrop = (event) => {
    event.preventDefault()
    setDragging(false)
    const dropped = event.dataTransfer.files?.[0]
    if (dropped) setFile(dropped)
  }

  const submit = (event) => {
    event.preventDefault()
    if (busy) return
    if (file) onFile(file)
    else if (url.trim()) onYouTube(url.trim())
  }

  const clearFile = (event) => {
    event.stopPropagation()
    setFile(null)
  }

  return (
    <form className="source" onSubmit={submit}>
      <div
        className={`dropzone ${dragging ? 'is-dragging' : ''} ${busy ? 'is-busy' : ''} ${
          file ? 'has-file' : ''
        }`}
        onDragOver={(e) => {
          e.preventDefault()
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        onClick={() => !busy && inputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault()
            inputRef.current?.click()
          }
        }}
      >
        {file ? (
          <>
            <svg viewBox="0 0 24 24" width="34" height="34" aria-hidden="true">
              <path
                d="M9 18V7l9-2v11M9 18a3 3 0 1 1-6 0 3 3 0 0 1 6 0Zm9-2a3 3 0 1 1-6 0 3 3 0 0 1 6 0Z"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.7"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
            <p className="dropzone__title dropzone__file">{file.name}</p>
            <p className="dropzone__hint">
              {(file.size / 1048576).toFixed(1)} MB · click to choose a different file
            </p>
            <button type="button" className="btn btn--ghost dropzone__clear" onClick={clearFile}>
              Remove
            </button>
          </>
        ) : (
          <>
            <svg viewBox="0 0 24 24" width="34" height="34" aria-hidden="true">
              <path
                d="M12 16V4m0 0L8 8m4-4 4 4M4 17v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.6"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
            <p className="dropzone__title">Drop an audio file, or click to browse</p>
            <p className="dropzone__hint">
              MP3, WAV, FLAC, M4A, OGG, OPUS, AIFF, MP4 · up to 15 minutes
            </p>
          </>
        )}
        <input
          ref={inputRef}
          type="file"
          accept="audio/*,video/mp4,video/webm,.m4a,.aiff,.aif,.opus,.wma,.flac"
          hidden
          onChange={(e) => {
            const picked = e.target.files?.[0]
            if (picked) setFile(picked)
            e.target.value = '' // let the same file be re-picked after clearing it
          }}
        />
      </div>

      <div className="source__divider">
        <span>or</span>
      </div>

      <div className="urlbar">
        <input
          type="url"
          placeholder="Paste a YouTube link…"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          disabled={busy || Boolean(file)}
          aria-label="YouTube URL"
        />
      </div>

      <label className="toggle toggle--quality">
        <input
          type="checkbox"
          checked={demucs}
          onChange={(e) => onDemucsChange(e.target.checked)}
          disabled={busy}
        />
        <span>
          <strong>High accuracy</strong> — separate the stems first (slower, much better on
          dense mixes)
        </span>
      </label>

      <button type="submit" className="btn btn--primary btn--analyze" disabled={busy || !ready}>
        Analyze
      </button>
      <p className="source__ready">
        {file
          ? `Ready to analyze ${file.name}`
          : url.trim()
            ? 'Ready to analyze the pasted link'
            : 'Choose a file or paste a link to continue'}
      </p>
    </form>
  )
}
