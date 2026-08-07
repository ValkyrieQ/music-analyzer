"""Getting audio in: uploaded files of any format, or a YouTube URL.

Everything is normalised to a single WAV (mono/stereo preserved, fixed sample rate) via
ffmpeg so the rest of the pipeline never has to care about container or codec.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

TARGET_SR = 44100

# Formats we accept on upload. ffmpeg handles far more; this is the advertised set.
SUPPORTED_EXTENSIONS = {
    ".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".oga", ".opus",
    ".wma", ".aiff", ".aif", ".alac", ".mp4", ".webm", ".mkv", ".mov",
}

_YOUTUBE_HOSTS = re.compile(
    r"^(?:https?://)?(?:www\.|m\.|music\.)?"
    r"(?:youtube\.com/(?:watch\?|embed/|shorts/|live/)|youtu\.be/)",
    re.IGNORECASE,
)

MAX_DURATION_SECONDS = 15 * 60


class IngestError(RuntimeError):
    """Raised when audio cannot be obtained or decoded."""


@dataclass
class AudioSource:
    wav_path: Path
    duration: float
    title: str
    source_type: str            # "upload" | "youtube"
    original_name: str
    sample_rate: int = TARGET_SR


def is_youtube_url(value: str) -> bool:
    return bool(_YOUTUBE_HOSTS.match(value.strip()))


def _require(binary: str) -> str:
    path = shutil.which(binary)
    if not path:
        raise IngestError(f"required binary '{binary}' not found in PATH")
    return path


def probe_duration(path: Path) -> float:
    """Read a media file's duration with ffprobe."""
    result = subprocess.run(
        [
            _require("ffprobe"), "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json", str(path),
        ],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        raise IngestError(f"ffprobe failed: {result.stderr.strip()[:300]}")

    try:
        return float(json.loads(result.stdout)["format"]["duration"])
    except (KeyError, ValueError, json.JSONDecodeError) as exc:
        raise IngestError(f"could not read duration: {exc}") from exc


def transcode_to_wav(src: Path, dest: Path, sr: int = TARGET_SR) -> Path:
    """Decode anything ffmpeg understands into 16-bit PCM WAV at `sr`."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            _require("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(src),
            "-vn",                       # drop any video stream
            "-ac", "2",
            "-ar", str(sr),
            "-c:a", "pcm_s16le",
            str(dest),
        ],
        capture_output=True, text=True, timeout=900,
    )
    if result.returncode != 0 or not dest.exists():
        raise IngestError(f"ffmpeg decode failed: {result.stderr.strip()[:400]}")
    return dest


def from_upload(uploaded_path: Path, original_name: str, work_dir: Path) -> AudioSource:
    """Normalise an uploaded file into the job's working directory."""
    suffix = Path(original_name).suffix.lower()
    if suffix and suffix not in SUPPORTED_EXTENSIONS:
        raise IngestError(
            f"unsupported file type '{suffix}'. Supported: "
            + ", ".join(sorted(SUPPORTED_EXTENSIONS))
        )

    wav_path = work_dir / "audio.wav"
    transcode_to_wav(uploaded_path, wav_path)
    duration = probe_duration(wav_path)

    if duration > MAX_DURATION_SECONDS:
        raise IngestError(
            f"track is {duration / 60:.1f} min; the limit is "
            f"{MAX_DURATION_SECONDS // 60} min"
        )
    if duration < 1.0:
        raise IngestError("track is shorter than 1 second")

    return AudioSource(
        wav_path=wav_path,
        duration=duration,
        title=Path(original_name).stem or "Untitled",
        source_type="upload",
        original_name=original_name,
    )


def from_youtube(url: str, work_dir: Path) -> AudioSource:
    """Download a YouTube URL's audio track and normalise it."""
    if not is_youtube_url(url):
        raise IngestError("not a recognised YouTube URL")

    work_dir.mkdir(parents=True, exist_ok=True)
    template = str(work_dir / "download.%(ext)s")

    # yt-dlp's Python API is importable, but the CLI is the interface its own docs treat
    # as stable, and shelling out keeps its (chatty, retry-heavy) logging out of ours.
    cmd = [
        _require("yt-dlp"),
        "--no-playlist",
        "--no-progress",
        # Use the system trust store instead of the bundled certifi one. yt-dlp defaults to
        # certifi, which does not include the corporate TLS-inspection root CA installed in
        # this image, so every download fails with CERTIFICATE_VERIFY_FAILED even though
        # the CA is present and torch.hub (pointed at the system store via REQUESTS_CA_BUNDLE)
        # works fine. Harmless on a normal network.
        "--compat-options", "no-certifi",
        "--extract-audio",
        "--audio-format", "wav",
        "--audio-quality", "0",
        "--match-filter", f"duration < {MAX_DURATION_SECONDS}",
        "--print-json",
        "--no-simulate",
        "-o", template,
        url,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        # Prefer the ERROR lines. yt-dlp emits multi-line WARNINGs first (the JS-runtime
        # deprecation notice is four lines on its own), and truncating raw stderr to 400
        # characters otherwise reports the warning and cuts off the actual cause.
        errors = [ln for ln in stderr.splitlines() if ln.startswith("ERROR:")]
        detail = " ".join(errors) if errors else stderr

        # Surface the common, actionable failures rather than a wall of yt-dlp output.
        if "Private video" in stderr or "members-only" in stderr:
            raise IngestError("that video is private or members-only")
        if "Video unavailable" in stderr:
            raise IngestError("that video is unavailable")
        if "Sign in to confirm" in stderr or "bot" in stderr.lower():
            raise IngestError(
                "YouTube blocked the download (bot check). Try uploading the file instead."
            )
        if "does not pass filter" in stderr:
            raise IngestError(
                f"that video is longer than {MAX_DURATION_SECONDS // 60} minutes"
            )
        raise IngestError(f"YouTube download failed: {detail[:400]}")

    title = "YouTube track"
    try:
        # --print-json emits one JSON object per line; the last is the completed download.
        for line in reversed(result.stdout.strip().splitlines()):
            if line.startswith("{"):
                title = json.loads(line).get("title") or title
                break
    except json.JSONDecodeError:
        log.warning("could not parse yt-dlp metadata; using default title")

    downloaded = next(iter(sorted(work_dir.glob("download.*"))), None)
    if downloaded is None:
        raise IngestError("download produced no file")

    wav_path = work_dir / "audio.wav"
    if downloaded != wav_path:
        transcode_to_wav(downloaded, wav_path)
        downloaded.unlink(missing_ok=True)
    else:
        # yt-dlp already wrote WAV, but not necessarily at our target sample rate.
        tmp = work_dir / "audio.norm.wav"
        transcode_to_wav(wav_path, tmp)
        tmp.replace(wav_path)

    duration = probe_duration(wav_path)
    if duration > MAX_DURATION_SECONDS:
        raise IngestError(
            f"track is {duration / 60:.1f} min; the limit is "
            f"{MAX_DURATION_SECONDS // 60} min"
        )

    return AudioSource(
        wav_path=wav_path,
        duration=duration,
        title=title,
        source_type="youtube",
        original_name=url,
    )
