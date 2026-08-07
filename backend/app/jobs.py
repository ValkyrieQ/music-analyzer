"""In-process job store for analysis runs.

Analysis takes tens of seconds to minutes, so requests kick off a background job and the
browser polls for status. State lives in memory with the artefacts on disk; a single
worker thread pool serialises the heavy work.

Deliberately not Celery/Redis: one container, one process, bounded concurrency. If this
ever needs to scale horizontally, swap `JobStore` for a Redis-backed implementation —
the interface is small on purpose.
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger(__name__)


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


@dataclass
class Job:
    id: str
    status: JobStatus
    title: str
    source_type: str
    created_at: float
    stage: str = "queued"
    progress: float = 0.0
    error: str | None = None
    result: dict[str, Any] | None = None
    finished_at: float | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def public(self) -> dict[str, Any]:
        """The shape the API returns. Excludes the lock and the bulky result payload."""
        with self._lock:
            return {
                "id": self.id,
                "status": self.status.value,
                "title": self.title,
                "source_type": self.source_type,
                "stage": self.stage,
                "progress": round(self.progress, 1),
                "error": self.error,
                "created_at": self.created_at,
                "finished_at": self.finished_at,
                "elapsed": round((self.finished_at or time.time()) - self.created_at, 1),
            }

    def update(self, **fields: Any) -> None:
        with self._lock:
            for key, value in fields.items():
                setattr(self, key, value)


class JobStore:
    """Thread-safe job registry plus the worker pool that runs them."""

    def __init__(self, data_dir: Path, max_workers: int = 1, retention_hours: float = 12.0):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.retention_seconds = retention_hours * 3600

        # One worker by default: Demucs saturates every core it is given, so running two
        # jobs at once makes both slower and risks OOM in a memory-capped container.
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="analysis")
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    # -- paths ------------------------------------------------------------------------
    def job_dir(self, job_id: str) -> Path:
        return self.data_dir / job_id

    def result_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "analysis.json"

    def source_wav(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "audio.wav"

    def stem_dir(self, job_id: str) -> Path:
        """Where separated stems are exported for download."""
        return self.job_dir(job_id) / "stems"

    # -- lifecycle --------------------------------------------------------------------
    def create(self, title: str, source_type: str) -> Job:
        job_id = uuid.uuid4().hex[:16]
        job = Job(
            id=job_id,
            status=JobStatus.QUEUED,
            title=title,
            source_type=source_type,
            created_at=time.time(),
        )
        with self._lock:
            self._jobs[job_id] = job
        self.job_dir(job_id).mkdir(parents=True, exist_ok=True)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is not None:
            return job

        # Survive a restart: if the artefacts are still on disk, rebuild a completed job
        # record from them rather than 404-ing a link the user already has.
        result_file = self.result_path(job_id)
        if result_file.exists():
            try:
                payload = json.loads(result_file.read_text())
            except (json.JSONDecodeError, OSError) as exc:
                log.warning("could not rehydrate job %s: %s", job_id, exc)
                return None

            job = Job(
                id=job_id,
                status=JobStatus.DONE,
                title=payload.get("title", "Untitled"),
                source_type=payload.get("source_type", "upload"),
                created_at=result_file.stat().st_mtime,
                stage="done",
                progress=100.0,
                result=payload,
                finished_at=result_file.stat().st_mtime,
            )
            with self._lock:
                self._jobs[job_id] = job
            return job
        return None

    def load_result(self, job_id: str) -> dict[str, Any] | None:
        job = self.get(job_id)
        if job is None:
            return None
        if job.result is not None:
            return job.result

        path = self.result_path(job_id)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            log.error("corrupt result for job %s: %s", job_id, exc)
            return None
        job.update(result=payload)
        return payload

    def submit(self, job: Job, work: Callable[[Job], dict[str, Any]]) -> None:
        """Queue `work` for `job`, handling status transitions and error capture."""

        def runner() -> None:
            job.update(status=JobStatus.RUNNING, stage="starting", progress=1.0)
            try:
                payload = work(job)
                self.result_path(job.id).write_text(json.dumps(payload))
                job.update(
                    status=JobStatus.DONE,
                    stage="done",
                    progress=100.0,
                    result=payload,
                    finished_at=time.time(),
                )
                log.info("job %s finished in %.1fs", job.id, time.time() - job.created_at)
            except Exception as exc:  # noqa: BLE001 - any failure must land on the job
                log.exception("job %s failed", job.id)
                job.update(
                    status=JobStatus.ERROR,
                    stage="failed",
                    error=f"{type(exc).__name__}: {exc}"[:500],
                    finished_at=time.time(),
                )
            finally:
                self.prune()

        self._pool.submit(runner)

    def progress_callback(self, job: Job) -> Callable[[str, float], None]:
        def report(stage: str, pct: float) -> None:
            job.update(stage=stage, progress=float(pct))

        return report

    # -- housekeeping -----------------------------------------------------------------
    def prune(self) -> None:
        """Delete jobs (records and files) older than the retention window."""
        cutoff = time.time() - self.retention_seconds
        with self._lock:
            stale = [
                jid for jid, job in self._jobs.items()
                if (job.finished_at or job.created_at) < cutoff
            ]
            for jid in stale:
                self._jobs.pop(jid, None)

        for jid in stale:
            shutil.rmtree(self.job_dir(jid), ignore_errors=True)
        if stale:
            log.info("pruned %d expired job(s)", len(stale))

    def forget(self, job_id: str) -> None:
        """Drop a job's record and delete its files."""
        with self._lock:
            self._jobs.pop(job_id, None)
        shutil.rmtree(self.job_dir(job_id), ignore_errors=True)

    def stats(self) -> dict[str, int]:
        with self._lock:
            jobs = list(self._jobs.values())
        return {
            "total": len(jobs),
            "running": sum(1 for j in jobs if j.status is JobStatus.RUNNING),
            "queued": sum(1 for j in jobs if j.status is JobStatus.QUEUED),
        }

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
