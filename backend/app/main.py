"""FastAPI application: upload/analyse endpoints, job polling, audio streaming."""

from __future__ import annotations

import logging
import os
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from . import transpose
from .analysis import harmony, ingest, pipeline, separate
from .jobs import Job, JobStatus, JobStore

# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
LOG_DIR = Path(os.environ.get("LOG_DIR", "/app/log"))
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "80")) * 1024 * 1024
ENABLE_DEMUCS = os.environ.get("ENABLE_DEMUCS", "1") not in {"0", "false", "False"}
MAX_CONCURRENT_JOBS = int(os.environ.get("MAX_CONCURRENT_JOBS", "1"))
ALLOWED_ORIGINS = [
    o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()
]

LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "api.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("music_analyzer")

store = JobStore(DATA_DIR / "jobs", max_workers=MAX_CONCURRENT_JOBS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info(
        "starting up: data=%s demucs=%s workers=%d rubberband=%s",
        DATA_DIR, ENABLE_DEMUCS, MAX_CONCURRENT_JOBS, transpose.rubberband_available(),
    )
    store.prune()
    yield
    store.shutdown()
    log.info("shut down")


app = FastAPI(
    title="Music Analyzer API",
    description="Tempo, key and chord analysis for audio files and YouTube URLs.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------------------
class YouTubeRequest(BaseModel):
    url: str = Field(..., min_length=8, description="A YouTube watch/share/shorts URL")
    demucs: bool = Field(True, description="Use Demucs separation (slower, more accurate)")


class JobCreated(BaseModel):
    id: str
    status: str
    title: str


# Display names for the downloadable stems.
STEM_LABELS = {
    "vocals": "Vocals",
    "drums": "Drums",
    "bass": "Bass",
    "guitar": "Guitar",
    "piano": "Piano",
    "other": "Other instruments",
}


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _run_analysis(job: Job, source: ingest.AudioSource, use_demucs: bool) -> dict[str, Any]:
    """The body of a job: analyse the normalised WAV and assemble the response payload."""
    result = pipeline.analyse(
        source.wav_path,
        duration=source.duration,
        enable_demucs=use_demucs,
        progress=store.progress_callback(job),
        # Stems are only exported on the high-accuracy path — the fast path never
        # produces them, so there would be nothing to write.
        stem_dir=store.stem_dir(job.id) if use_demucs else None,
    )
    return {
        "id": job.id,
        "title": source.title,
        "source_type": source.source_type,
        "duration": result.duration,
        "tempo": result.tempo,
        "beats_per_bar": result.beats_per_bar,
        "beat_confidence": result.beat_confidence,
        "key": result.key,
        "beat_times": result.beat_times,
        "downbeat_indices": result.downbeat_indices,
        "chords": result.chords,
        "beat_chords": result.beat_chords,
        "bars": result.bars,
        "separation_method": result.separation_method,
        "harmonic_sources": result.harmonic_sources,
        "stem_levels": result.stem_levels,
        "stems": [
            {
                "name": name,
                "label": STEM_LABELS.get(name, name.title()),
                "url": f"/api/jobs/{job.id}/stems/{name}",
            }
            for name in result.stems
        ],
        "timings": result.timings,
        "audio_url": f"/api/jobs/{job.id}/audio",
        "harmony_with_track_url": f"/api/jobs/{job.id}/harmony/with-track",
    }


def _save_upload(upload: UploadFile, dest: Path) -> int:
    """Stream an upload to disk, enforcing the size cap as we go.

    Reading `upload.file` in chunks (rather than `.read()`) keeps a large file from being
    held in memory, and lets us abort as soon as the cap is exceeded.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with dest.open("wb") as fh:
        while chunk := upload.file.read(1024 * 1024):
            written += len(chunk)
            if written > MAX_UPLOAD_BYTES:
                fh.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"file exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit",
                )
            fh.write(chunk)

    if written == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="uploaded file is empty")
    return written


# --------------------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------------------
@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "demucs_enabled": ENABLE_DEMUCS,
        "rubberband": transpose.rubberband_available(),
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "yt_dlp": shutil.which("yt-dlp") is not None,
        "jobs": store.stats(),
        "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
        "supported_formats": sorted(ingest.SUPPORTED_EXTENSIONS),
    }


@app.post("/api/analyze/upload", response_model=JobCreated, status_code=202)
def analyze_upload(
    file: Annotated[UploadFile, File(description="Audio file")],
    demucs: Annotated[bool, Form()] = True,
) -> JobCreated:
    """Accept an audio file and start analysing it."""
    original_name = file.filename or "upload"
    suffix = Path(original_name).suffix.lower()
    if suffix and suffix not in ingest.SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"unsupported file type '{suffix}'. Supported: "
                   + ", ".join(sorted(ingest.SUPPORTED_EXTENSIONS)),
        )

    job = store.create(title=Path(original_name).stem or "Untitled", source_type="upload")
    raw_path = store.job_dir(job.id) / f"raw{suffix or '.bin'}"

    try:
        size = _save_upload(file, raw_path)
    except HTTPException:
        shutil.rmtree(store.job_dir(job.id), ignore_errors=True)
        raise
    finally:
        file.file.close()

    log.info("job %s: received %s (%.1f MB)", job.id, original_name, size / 1e6)

    def work(j: Job) -> dict[str, Any]:
        j.update(stage="decoding audio", progress=2.0)
        source = ingest.from_upload(raw_path, original_name, store.job_dir(j.id))
        j.update(title=source.title)
        # The normalised WAV is what we keep; the original upload is now redundant.
        raw_path.unlink(missing_ok=True)
        return _run_analysis(j, source, use_demucs=demucs and ENABLE_DEMUCS)

    store.submit(job, work)
    return JobCreated(id=job.id, status=job.status.value, title=job.title)


@app.post("/api/analyze/youtube", response_model=JobCreated, status_code=202)
def analyze_youtube(req: YouTubeRequest) -> JobCreated:
    """Download a YouTube URL's audio and start analysing it."""
    if not ingest.is_youtube_url(req.url):
        raise HTTPException(status_code=400, detail="not a recognised YouTube URL")

    job = store.create(title="YouTube track", source_type="youtube")

    def work(j: Job) -> dict[str, Any]:
        j.update(stage="downloading from YouTube", progress=2.0)
        source = ingest.from_youtube(req.url, store.job_dir(j.id))
        j.update(title=source.title)
        return _run_analysis(j, source, use_demucs=req.demucs and ENABLE_DEMUCS)

    store.submit(job, work)
    return JobCreated(id=job.id, status=job.status.value, title=job.title)


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict[str, Any]:
    """Poll a job. Returns status while running, and the full analysis once done."""
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")

    payload = job.public()
    if job.status is JobStatus.DONE:
        result = store.load_result(job_id)
        if result is None:
            raise HTTPException(status_code=500, detail="analysis result missing on disk")
        payload["analysis"] = result
    return payload


@app.get("/api/jobs/{job_id}/audio")
def job_audio(job_id: str, request: Request, semitones: int = 0) -> FileResponse:
    """Stream the job's audio, pitch-shifted by `semitones` if asked.

    `semitones=0` serves the source WAV directly; anything else is rendered once via
    rubberband and cached, so scrubbing back and forth between keys stays instant.
    """
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")

    source = store.source_wav(job_id)
    if not source.exists():
        raise HTTPException(status_code=404, detail="audio not available for this job")

    if semitones == 0:
        return FileResponse(
            source,
            media_type="audio/wav",
            # Let the browser cache and range-request: <audio> seeking depends on it.
            headers={"Accept-Ranges": "bytes", "Cache-Control": "public, max-age=3600"},
        )

    try:
        rendered = transpose.render(source, semitones, store.job_dir(job_id) / "transposed")
    except transpose.TransposeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return FileResponse(
        rendered,
        media_type="audio/mpeg",
        headers={"Accept-Ranges": "bytes", "Cache-Control": "public, max-age=3600"},
    )


@app.get("/api/jobs/{job_id}/stems")
def job_stems(job_id: str) -> dict[str, Any]:
    """List the separated stems available for download.

    Only the high-accuracy (Demucs) path produces these; a fast-path job returns an
    empty list rather than a 404, so the UI can hide the menu without special-casing.
    """
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")

    result = store.load_result(job_id) or {}
    return {
        "id": job_id,
        "separation_method": result.get("separation_method"),
        "harmonic_sources": result.get("harmonic_sources", []),
        "stems": result.get("stems", []),
        # Harmony is rendered on demand from the vocal stem, so it is "available" as soon
        # as that stem exists — not only once it has been generated.
        "harmony_available": (store.stem_dir(job_id) / "vocals.mp3").exists(),
        "harmony_url": f"/api/jobs/{job_id}/harmony",
        "harmony_with_track_url": f"/api/jobs/{job_id}/harmony/with-track",
    }


@app.get("/api/jobs/{job_id}/stems/{stem}")
def job_stem(job_id: str, stem: str) -> FileResponse:
    """Download one separated stem as MP3."""
    if store.get(job_id) is None:
        raise HTTPException(status_code=404, detail="job not found")

    # Validate against the known set rather than sanitising the string. `stem` lands in a
    # filesystem path, and a whitelist cannot be talked into resolving somewhere else.
    if stem not in separate.DOWNLOADABLE:
        raise HTTPException(
            status_code=404,
            detail=f"unknown stem '{stem}'. Available: "
                   + ", ".join(separate.DOWNLOADABLE),
        )

    path = store.stem_dir(job_id) / f"{stem}.mp3"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="that stem is not available — the track was analysed without "
                   "stem separation",
        )

    job = store.get(job_id)
    title = (job.title if job else job_id) or job_id
    # Keep the download filename readable but filesystem-safe in the Content-Disposition.
    safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in title).strip() or "track"

    return FileResponse(
        path,
        media_type="audio/mpeg",
        filename=f"{safe} - {STEM_LABELS.get(stem, stem)}.mp3",
        headers={"Accept-Ranges": "bytes", "Cache-Control": "public, max-age=3600"},
    )


def _require_vocals(job_id: str) -> Path:
    vocals = store.stem_dir(job_id) / "vocals.mp3"
    if not vocals.exists():
        raise HTTPException(
            status_code=404,
            detail="no vocal stem for this job — harmony needs a High accuracy analysis",
        )
    return vocals


def _safe_filename(title: str | None, job_id: str) -> str:
    return "".join(
        c if c.isalnum() or c in " -_" else "_" for c in (title or job_id)
    ).strip() or "track"


@app.get("/api/jobs/{job_id}/harmony")
def job_harmony(job_id: str) -> FileResponse:
    """Lead vocal plus generated backing harmony alone, as MP3 — for downloading.

    Rendered on first request and cached, rather than during analysis. Pitch tracking the
    vocal and rendering every note costs on the order of a minute on a real 3-4 minute
    song (measured 87s on one), and adding that to every job would work directly against
    making analysis faster — most users never ask for the harmony. The cost lands only on
    whoever wants it, once.
    """
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    vocals = _require_vocals(job_id)

    cached = store.stem_dir(job_id) / "harmony.mp3"
    if not cached.exists():
        result = store.load_result(job_id)
        if result is None:
            raise HTTPException(status_code=409, detail="analysis is not finished yet")

        import librosa

        vocal, sr = librosa.load(str(vocals), sr=None, mono=True)
        generated = harmony.generate(
            vocal, sr, result.get("chords", []), store.stem_dir(job_id)
        )
        if generated is None:
            raise HTTPException(
                status_code=422,
                detail="could not build a harmony — no sung notes were found in the "
                       "vocal stem (an instrumental track, most likely)",
            )
        cached = generated.path

    return FileResponse(
        cached,
        media_type="audio/mpeg",
        filename=f"{_safe_filename(job.title, job_id)} - Harmony.mp3",
        headers={"Accept-Ranges": "bytes", "Cache-Control": "public, max-age=3600"},
    )


@app.get("/api/jobs/{job_id}/harmony/with-track")
def job_harmony_with_track(job_id: str) -> FileResponse:
    """The full song with the generated backing harmony mixed under it, as MP3.

    This is what "listen with harmony" plays: the same audio as `/audio`, with only the
    backing-voice bus added — the lead vocal already in the mix is not duplicated. Same
    render-once-and-cache behaviour and cost as `/harmony`; a separate file because it is
    a different mix, not a different quality of the same one.
    """
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    vocals = _require_vocals(job_id)

    cached = store.stem_dir(job_id) / "harmony_with_track.mp3"
    if not cached.exists():
        result = store.load_result(job_id)
        if result is None:
            raise HTTPException(status_code=409, detail="analysis is not finished yet")

        source = store.source_wav(job_id)
        if not source.exists():
            raise HTTPException(status_code=404, detail="audio not available for this job")

        import librosa

        vocal, sr = librosa.load(str(vocals), sr=None, mono=True)
        full_mix, _ = librosa.load(str(source), sr=sr, mono=True)
        generated = harmony.generate_with_track(
            vocal, full_mix, sr, result.get("chords", []), store.stem_dir(job_id)
        )
        if generated is None:
            raise HTTPException(
                status_code=422,
                detail="could not build a harmony — no sung notes were found in the "
                       "vocal stem (an instrumental track, most likely)",
            )
        cached = generated.path

    return FileResponse(
        cached,
        media_type="audio/mpeg",
        filename=f"{_safe_filename(job.title, job_id)} - with Harmony.mp3",
        headers={"Accept-Ranges": "bytes", "Cache-Control": "public, max-age=3600"},
    )


@app.delete("/api/jobs/{job_id}", status_code=204, response_class=Response)
def delete_job(job_id: str) -> Response:
    """Remove a job and everything on disk for it."""
    if store.get(job_id) is None:
        raise HTTPException(status_code=404, detail="job not found")
    store.forget(job_id)
    # 204 must not carry a body, so return an empty Response rather than letting FastAPI
    # try to serialise a JSON payload.
    return Response(status_code=204)


@app.exception_handler(ingest.IngestError)
async def ingest_error_handler(request: Request, exc: ingest.IngestError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})
