import os
import shutil
import tempfile
import threading
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import urlparse

import requests
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from scripts.inference import MuseTalkSynthesizer


ALLOWED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".webm", ".png", ".jpg", ".jpeg"}
ALLOWED_AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac"}
MAX_DOWNLOAD_SIZE = int(os.environ.get("MAX_DOWNLOAD_SIZE", str(1024 * 1024 * 1024)))  # 1GB default


@dataclass
class OutputItem:
    job_id: str
    output_path: str


class SynthesizeRequest(BaseModel):
    video_source: str = Field(..., description="Local path or HTTP(S) URL for template video/image")
    audio_source: str = Field(..., description="Local path or HTTP(S) URL for driving audio")
    output_name: Optional[str] = Field(default=None, description="Optional output file name")
    return_mode: str = Field(default="link", description="link or file")

    bbox_shift: int = 0
    extra_margin: int = Field(default=10, ge=0, le=40)
    fps: int = Field(default=25, ge=1, le=120)
    batch_size: int = Field(default=8, ge=1, le=64)
    audio_padding_length_left: int = Field(default=2, ge=0, le=20)
    audio_padding_length_right: int = Field(default=2, ge=0, le=20)
    use_saved_coord: bool = False
    saved_coord: bool = False
    parsing_mode: str = Field(default="jaw", description="jaw or raw")


class SynthesizeResponse(BaseModel):
    job_id: str
    output_path: str
    download_url: str


class AppState:
    def __init__(self) -> None:
        self.synthesizer: Optional[MuseTalkSynthesizer] = None
        self.outputs: Dict[str, OutputItem] = {}
        self.lock = threading.Lock()


state = AppState()


def _is_http_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _suffix_from_url(url: str) -> str:
    parsed = urlparse(url)
    return Path(parsed.path).suffix.lower()


def _ensure_supported_suffix(source: str, allowed_exts: set[str], source_name: str) -> str:
    suffix = _suffix_from_url(source) if _is_http_url(source) else Path(source).suffix.lower()
    if suffix not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported {source_name} format '{suffix}'. Allowed: {sorted(allowed_exts)}",
        )
    return suffix


def _download_to_file(url: str, dest: Path) -> None:
    try:
        with requests.get(url, stream=True, timeout=(10, 180)) as resp:
            resp.raise_for_status()
            total = 0
            with dest.open("wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > MAX_DOWNLOAD_SIZE:
                        raise HTTPException(status_code=413, detail="Downloaded file exceeds MAX_DOWNLOAD_SIZE")
                    f.write(chunk)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to download '{url}': {exc}")


def _prepare_source(source: str, allowed_exts: set[str], source_name: str, work_dir: Path) -> Path:
    suffix = _ensure_supported_suffix(source, allowed_exts, source_name)
    target = work_dir / f"{source_name}{suffix}"

    if _is_http_url(source):
        _download_to_file(source, target)
        return target

    source_path = Path(source).expanduser().resolve()
    if source_name == "video" and source_path.is_dir():
        return source_path

    if not source_path.exists():
        raise HTTPException(status_code=404, detail=f"{source_name} source not found: {source}")

    if source_path.is_file():
        shutil.copy2(source_path, target)
        return target

    raise HTTPException(status_code=400, detail=f"Invalid {source_name} source: {source}")


def _build_output_name(video_path: Path, audio_path: Path, output_name: Optional[str]) -> str:
    if output_name:
        name = Path(output_name).name
        return name if name.lower().endswith(".mp4") else f"{name}.mp4"
    return f"{video_path.stem}_{audio_path.stem}_{uuid.uuid4().hex[:8]}.mp4"


@asynccontextmanager
async def lifespan(app: FastAPI):
    ffmpeg_path = os.environ.get("FFMPEG_PATH", "./ffmpeg-4.4-amd64-static/")
    gpu_id = int(os.environ.get("GPU_ID", "0"))
    use_float16 = os.environ.get("USE_FLOAT16", "false").lower() == "true"
    version = os.environ.get("MODEL_VERSION", "v15")

    state.synthesizer = MuseTalkSynthesizer(
        ffmpeg_path=ffmpeg_path,
        gpu_id=gpu_id,
        vae_type=os.environ.get("VAE_TYPE", "sd-vae"),
        unet_config=os.environ.get("UNET_CONFIG", "models\musetalkV15\musetalk.json"),
        unet_model_path=os.environ.get("UNET_MODEL_PATH", "./models/musetalkV15/unet.pth"),
        whisper_dir=os.environ.get("WHISPER_DIR", "./models/whisper"),
        use_float16=use_float16,
        version=version,
        left_cheek_width=int(os.environ.get("LEFT_CHEEK_WIDTH", "90")),
        right_cheek_width=int(os.environ.get("RIGHT_CHEEK_WIDTH", "90")),
    )
    try:
        yield
    finally:
        if state.synthesizer is not None:
            state.synthesizer.close()
            state.synthesizer = None


app = FastAPI(
    title="MuseTalk Path/URL Service",
    version="1.0.0",
    description="Synthesize talking videos using local paths or URLs as video/audio inputs.",
    lifespan=lifespan,
)


@app.get("/api/v1/health")
def health():
    return {
        "status": "ready" if state.synthesizer else "loading",
        "gpu_available": torch.cuda.is_available(),
        "device": str(state.synthesizer.device) if state.synthesizer else None,
    }


@app.post("/api/v1/synthesize", response_model=SynthesizeResponse)
def synthesize(req: SynthesizeRequest):
    if state.synthesizer is None:
        raise HTTPException(status_code=503, detail="Model is still loading")
    if req.parsing_mode not in {"jaw", "raw"}:
        raise HTTPException(status_code=400, detail="parsing_mode must be 'jaw' or 'raw'")
    if req.return_mode not in {"link", "file"}:
        raise HTTPException(status_code=400, detail="return_mode must be 'link' or 'file'")

    results_root = Path(os.environ.get("RESULT_DIR", "./results")).resolve()
    outputs_root = results_root / "service_outputs"
    outputs_root.mkdir(parents=True, exist_ok=True)

    work_dir = Path(tempfile.mkdtemp(prefix="musetalk_req_", dir=str(results_root)))
    try:
        video_path = _prepare_source(req.video_source, ALLOWED_VIDEO_EXTS, "video", work_dir)
        audio_path = _prepare_source(req.audio_source, ALLOWED_AUDIO_EXTS, "audio", work_dir)

        output_name = _build_output_name(Path(video_path), Path(audio_path), req.output_name)
        output_path = outputs_root / output_name
        if output_path.exists():
            output_path = outputs_root / f"{Path(output_name).stem}_{uuid.uuid4().hex[:6]}.mp4"

        with state.lock:
            final_path = state.synthesizer.synthesize(
                audio_path=str(audio_path),
                video_path=str(video_path),
                output_path=str(output_path),
                result_dir=str(results_root),
                bbox_shift=req.bbox_shift,
                extra_margin=req.extra_margin,
                fps=req.fps,
                batch_size=req.batch_size,
                audio_padding_length_left=req.audio_padding_length_left,
                audio_padding_length_right=req.audio_padding_length_right,
                use_saved_coord=req.use_saved_coord,
                saved_coord=req.saved_coord,
                parsing_mode=req.parsing_mode,
            )

        job_id = uuid.uuid4().hex
        state.outputs[job_id] = OutputItem(job_id=job_id, output_path=final_path)
        download_url = f"/api/v1/files/{job_id}"

        if req.return_mode == "file":
            return FileResponse(
                final_path,
                media_type="video/mp4",
                filename=Path(final_path).name,
            )

        return SynthesizeResponse(job_id=job_id, output_path=final_path, download_url=download_url)
    except HTTPException:
        raise
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {exc}")
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


@app.get("/api/v1/files/{job_id}")
def download_result(job_id: str):
    output = state.outputs.get(job_id)
    if output is None:
        raise HTTPException(status_code=404, detail="job_id not found")
    path = Path(output.output_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="result file not found")
    return FileResponse(path=str(path), media_type="video/mp4", filename=path.name)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
        reload=False,
    )
