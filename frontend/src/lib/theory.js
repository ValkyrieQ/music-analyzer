/**
 * Music theory helpers for display and transposition.
 *
 * Transposition happens entirely client-side: the analysis returns each chord's root as
 * a pitch class (0-11) plus a quality string, so shifting the whole chart is an integer
 * add. That keeps the key control instant — no round trip to re-analyse.
 */

export const SHARP_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
export const FLAT_NAMES = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']

/** Suffix rendered after the root for each quality the backend can emit. */
export const QUALITY_SUFFIX = {
  maj: '',
  min: 'm',
  7: '7',
  maj7: 'maj7',
  min7: 'm7',
  dim: 'dim',
  aug: 'aug',
  sus2: 'sus2',
  sus4: 'sus4',
  6: '6',
  min6: 'm6',
}

/**
 * Keys conventionally written with flats. Using flats in (say) Eb major avoids showing
 * "D#" where every chart in the world shows "Eb".
 */
const FLAT_MAJOR_TONICS = new Set([1, 3, 5, 8, 10])
const FLAT_MINOR_TONICS = new Set([2, 3, 5, 8, 10])

export function preferFlats(tonic, mode) {
  return mode === 'minor' ? FLAT_MINOR_TONICS.has(tonic) : FLAT_MAJOR_TONICS.has(tonic)
}

export function pitchName(pitchClass, useFlats = false) {
  const names = useFlats ? FLAT_NAMES : SHARP_NAMES
  return names[((pitchClass % 12) + 12) % 12]
}

/** Render a chord from its pitch class + quality, shifted by `semitones`. */
export function chordLabel(root, quality, semitones = 0, useFlats = false) {
  if (root === null || root === undefined) return 'N.C.'
  const shifted = (((root + semitones) % 12) + 12) % 12
  return pitchName(shifted, useFlats) + (QUALITY_SUFFIX[quality] ?? '')
}

/** Split a chord into root and suffix so the UI can size them differently. */
export function chordParts(root, quality, semitones = 0, useFlats = false) {
  if (root === null || root === undefined) return { root: 'N.C.', suffix: '' }
  const shifted = (((root + semitones) % 12) + 12) % 12
  return {
    root: pitchName(shifted, useFlats),
    suffix: QUALITY_SUFFIX[quality] ?? '',
  }
}

export function keyName(tonic, mode, semitones = 0) {
  const shifted = (((tonic + semitones) % 12) + 12) % 12
  return `${pitchName(shifted, preferFlats(shifted, mode))} ${mode}`
}

const MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
const MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]

/** Roman numeral for a chord's function in the key, or null if it's not diatonic. */
export function romanNumeral(chordRoot, quality, tonic, mode) {
  if (chordRoot === null || chordRoot === undefined) return null
  const degree = ((chordRoot - tonic) % 12 + 12) % 12
  const intervals = mode === 'minor' ? MINOR_SCALE : MAJOR_SCALE
  const index = intervals.indexOf(degree)
  if (index === -1) return null

  const numerals = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII']
  const numeral = numerals[index]
  // Minor and diminished qualities are written lower-case by convention.
  const isMinorish = quality === 'min' || quality === 'min7' || quality === 'dim' || quality === 'min6'
  return isMinorish ? numeral.toLowerCase() : numeral
}

export function formatTime(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return '0:00'
  const mins = Math.floor(seconds / 60)
  const secs = Math.floor(seconds % 60)
  return `${mins}:${String(secs).padStart(2, '0')}`
}

/** Human label for a transposition amount, e.g. -2 -> "−2 (2 semitones down)". */
export function transposeLabel(semitones) {
  if (semitones === 0) return 'Original key'
  const dir = semitones > 0 ? 'up' : 'down'
  const n = Math.abs(semitones)
  return `${semitones > 0 ? '+' : '−'}${n} · ${n} semitone${n === 1 ? '' : 's'} ${dir}`
}
