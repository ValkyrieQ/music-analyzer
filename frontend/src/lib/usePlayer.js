/**
 * Playback engine driving the chord grid's playhead.
 *
 * Design notes:
 *
 * - We use a plain `<audio>` element rather than Web Audio buffer playback. A 5-minute
 *   track is far too large to decode into memory on every load, and `<audio>` gives us
 *   HTTP range requests, streaming start, and native seeking for free.
 *
 * - Position is read on `requestAnimationFrame`, not from `timeupdate`. The `timeupdate`
 *   event only fires ~4x/second, which makes the playhead visibly stutter; rAF gives us
 *   per-frame smoothness and stays in step with the browser's paint cycle.
 *
 * - Changing the transposition swaps the audio source (the server renders a pitch-shifted
 *   file). We restore the exact playback position and resume if we were playing, so the
 *   key control does not interrupt the take.
 */

import { useCallback, useEffect, useRef, useState } from 'react'

export function usePlayer({ audioUrl, semitones, harmonyUrl, harmonyMode }) {
  const audioRef = useRef(null)
  const rafRef = useRef(0)
  // Position we want to be at after a source swap. A ref (not state) because the
  // `loadedmetadata` handler reads it outside React's render cycle.
  const pendingSeekRef = useRef(null)
  const wasPlayingRef = useRef(false)

  const [isPlaying, setIsPlaying] = useState(false)
  const [position, setPosition] = useState(0)
  const [duration, setDuration] = useState(0)
  const [isBuffering, setIsBuffering] = useState(false)
  const [error, setError] = useState(null)

  // --- rAF position loop ---------------------------------------------------------------
  useEffect(() => {
    if (!isPlaying) return undefined

    const tick = () => {
      const el = audioRef.current
      if (el) setPosition(el.currentTime)
      rafRef.current = requestAnimationFrame(tick)
    }
    rafRef.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(rafRef.current)
  }, [isPlaying])

  // --- source management ---------------------------------------------------------------
  useEffect(() => {
    if (!audioUrl) return undefined

    const el = audioRef.current ?? new Audio()
    audioRef.current = el
    el.preload = 'auto'

    // Harmony playback is a different render (song + backing voices, no transposition
    // support) rather than a variant of the transposed one, so the two are mutually
    // exclusive on the same element: harmonyMode wins outright over semitones.
    const useHarmony = harmonyMode && harmonyUrl
    const src = useHarmony
      ? harmonyUrl
      : semitones === 0
        ? audioUrl
        : `${audioUrl}?semitones=${semitones}`
    // Guard against re-setting the same src: assigning `src` restarts loading even when
    // the value is unchanged, which would stutter playback on unrelated re-renders.
    if (el.dataset.src !== src) {
      // A non-zero transposition, or the harmony render, happens on demand server-side
      // and can take a while the first time. Surface that instead of looking frozen.
      if (semitones !== 0 || useHarmony) setIsBuffering(true)
      el.dataset.src = src
      el.src = src
      el.load()
    }

    const onLoaded = () => {
      setDuration(el.duration || 0)
      setIsBuffering(false)
      setError(null)

      if (pendingSeekRef.current !== null) {
        // Clamp: a transposed or harmony render can differ by a few milliseconds in length.
        el.currentTime = Math.min(pendingSeekRef.current, (el.duration || 0) - 0.05)
        pendingSeekRef.current = null
        if (wasPlayingRef.current) {
          el.play().catch(() => setIsPlaying(false))
        }
      }
    }
    const onEnded = () => {
      setIsPlaying(false)
      setPosition(el.duration || 0)
    }
    const onWaiting = () => setIsBuffering(true)
    const onPlaying = () => setIsBuffering(false)
    const onError = () => {
      setIsBuffering(false)
      setIsPlaying(false)
      setError(
        useHarmony
          ? 'Could not build the harmony for this track.'
          : semitones === 0
            ? 'Could not load the audio for this track.'
            : `Could not render the ${semitones > 0 ? '+' : ''}${semitones} semitone version.`,
      )
    }

    el.addEventListener('loadedmetadata', onLoaded)
    el.addEventListener('ended', onEnded)
    el.addEventListener('waiting', onWaiting)
    el.addEventListener('playing', onPlaying)
    el.addEventListener('error', onError)

    return () => {
      el.removeEventListener('loadedmetadata', onLoaded)
      el.removeEventListener('ended', onEnded)
      el.removeEventListener('waiting', onWaiting)
      el.removeEventListener('playing', onPlaying)
      el.removeEventListener('error', onError)
    }
  }, [audioUrl, semitones, harmonyUrl, harmonyMode])

  // Before the source swaps, remember where we were so `onLoaded` can restore it.
  useEffect(() => {
    const el = audioRef.current
    if (el && el.dataset.src) {
      pendingSeekRef.current = el.currentTime
      wasPlayingRef.current = !el.paused
    }
  }, [semitones, harmonyMode])

  // Release the element (and its network connection) when the track changes or unmounts.
  useEffect(
    () => () => {
      const el = audioRef.current
      if (el) {
        el.pause()
        el.removeAttribute('src')
        el.load()
      }
      cancelAnimationFrame(rafRef.current)
    },
    [audioUrl],
  )

  // --- controls ------------------------------------------------------------------------
  const play = useCallback(async () => {
    const el = audioRef.current
    if (!el) return
    try {
      await el.play()
      setIsPlaying(true)
      setError(null)
    } catch {
      // Autoplay policy, or the source failed. Either way we are not playing.
      setIsPlaying(false)
    }
  }, [])

  const pause = useCallback(() => {
    audioRef.current?.pause()
    setIsPlaying(false)
  }, [])

  const toggle = useCallback(() => {
    if (isPlaying) pause()
    else play()
  }, [isPlaying, pause, play])

  const stop = useCallback(() => {
    const el = audioRef.current
    if (!el) return
    el.pause()
    el.currentTime = 0
    setIsPlaying(false)
    setPosition(0)
  }, [])

  const seek = useCallback((seconds) => {
    const el = audioRef.current
    if (!el) return
    const target = Math.max(0, Math.min(seconds, el.duration || seconds))
    el.currentTime = target
    // Update immediately rather than waiting for the next rAF frame, so a click on the
    // grid moves the playhead with no perceptible lag.
    setPosition(target)
  }, [])

  const setRate = useCallback((rate) => {
    const el = audioRef.current
    if (!el) return
    el.playbackRate = rate
    // Keep pitch stable when slowing down to learn a part; browsers default to true but
    // are inconsistent about it across versions.
    el.preservesPitch = true
  }, [])

  const setVolume = useCallback((value) => {
    const el = audioRef.current
    if (el) el.volume = Math.max(0, Math.min(1, value))
  }, [])

  return {
    isPlaying,
    position,
    duration,
    isBuffering,
    error,
    play,
    pause,
    toggle,
    stop,
    seek,
    setRate,
    setVolume,
  }
}
