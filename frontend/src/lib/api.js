/**
 * Backend client. All URLs are relative so the same build works behind the nginx proxy
 * in production and behind Vite's dev proxy locally.
 */

const BASE = '/api'

async function parseError(response, fallback) {
  try {
    const body = await response.json()
    return body.detail || body.message || fallback
  } catch {
    return fallback
  }
}

export async function analyzeUpload(file, { demucs = true } = {}) {
  const form = new FormData()
  form.append('file', file)
  form.append('demucs', String(demucs))

  const response = await fetch(`${BASE}/analyze/upload`, { method: 'POST', body: form })
  if (!response.ok) {
    throw new Error(await parseError(response, `Upload failed (${response.status})`))
  }
  return response.json()
}

export async function analyzeYouTube(url, { demucs = true } = {}) {
  const response = await fetch(`${BASE}/analyze/youtube`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url, demucs }),
  })
  if (!response.ok) {
    throw new Error(await parseError(response, `Request failed (${response.status})`))
  }
  return response.json()
}

export async function getJob(jobId) {
  const response = await fetch(`${BASE}/jobs/${jobId}`)
  if (!response.ok) {
    throw new Error(await parseError(response, `Could not fetch job (${response.status})`))
  }
  return response.json()
}

export async function getHealth() {
  const response = await fetch(`${BASE}/health`)
  if (!response.ok) throw new Error('Backend is not reachable')
  return response.json()
}

export function audioUrl(jobId) {
  return `${BASE}/jobs/${jobId}/audio`
}

export function stemUrl(jobId, stem) {
  return `${BASE}/jobs/${jobId}/stems/${stem}`
}

export function harmonyUrl(jobId) {
  return `${BASE}/jobs/${jobId}/harmony`
}

export function harmonyWithTrackUrl(jobId) {
  return `${BASE}/jobs/${jobId}/harmony/with-track`
}

/**
 * Fetch the harmony render, reporting a real error message rather than navigating.
 *
 * A plain <a download> would be simpler, but this endpoint renders on first request
 * (~20s) and can legitimately fail — an instrumental track has no vocal to harmonise.
 * Following the link in that case replaces the page with a raw JSON error, so the blob
 * is fetched here and only handed to the browser once it is known to be audio.
 */
export async function fetchHarmony(jobId) {
  const response = await fetch(harmonyUrl(jobId))
  if (!response.ok) {
    throw new Error(
      await parseError(response, `Could not build the harmony (${response.status})`),
    )
  }
  return response.blob()
}

/**
 * Poll a job until it finishes.
 *
 * Analysis is long and the interval is deliberately modest: each poll is cheap, and a
 * 1s cadence keeps the progress bar feeling live without hammering the API.
 */
export async function pollJob(jobId, { onProgress, intervalMs = 1000, signal } = {}) {
  for (;;) {
    if (signal?.aborted) throw new DOMException('Polling aborted', 'AbortError')

    const job = await getJob(jobId)
    onProgress?.(job)

    if (job.status === 'done') return job
    if (job.status === 'error') throw new Error(job.error || 'Analysis failed')

    await new Promise((resolve) => setTimeout(resolve, intervalMs))
  }
}
