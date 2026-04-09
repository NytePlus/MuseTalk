import os
import re
import copy
import glob
import uuid
import shutil
import logging
import subprocess
import threading
import tempfile
from contextlib import asynccontextmanager

import cv2
import torch
import numpy as np
import imageio
from tqdm import tqdm
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from starlette.background import BackgroundTask
from transformers import WhisperModel

from musetalk.utils.blending import get_image
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.utils import get_file_type, get_video_fps, datagen, load_all_model
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder
from musetalk.utils.audio_utils import ensure_wav

logger = logging.getLogger("musetalk-api")
logging.basicConfig(level=logging.INFO)

# Serialize GPU inference to avoid CUDA concurrency issues
_inference_lock = threading.Lock()


class ModelState:
    """Container for loaded model artifacts."""
    def __init__(self):
        self.device = None
        self.vae = None
        self.unet = None
        self.pe = None
        self.timesteps = None
        self.audio_processor = None
        self.whisper = None
        self.weight_dtype = None
        self.is_ready = False


models = ModelState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup: load all models ---
    logger.info("Loading models...")
    use_float16 = os.environ.get("USE_FLOAT16", "false").lower() == "true"

    models.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models.vae, models.unet, models.pe = load_all_model(
        unet_model_path="./models/musetalkV15/unet.pth",
        vae_type="sd-vae",
        unet_config="./models/musetalkV15/musetalk.json",
        device=models.device,
    )

    if use_float16:
        models.pe = models.pe.half()
        models.vae.vae = models.vae.vae.half()
        models.unet.model = models.unet.model.half()
        models.weight_dtype = torch.float16
    else:
        models.weight_dtype = torch.float32

    models.pe = models.pe.to(models.device)
    models.vae.vae = models.vae.vae.to(models.device)
    models.unet.model = models.unet.model.to(models.device)
    models.timesteps = torch.tensor([0], device=models.device)

    models.audio_processor = AudioProcessor(feature_extractor_path="./models/whisper")
    models.whisper = WhisperModel.from_pretrained("./models/whisper")
    models.whisper = models.whisper.to(
        device=models.device, dtype=models.weight_dtype
    ).eval()
    models.whisper.requires_grad_(False)

    models.is_ready = True
    logger.info("All models loaded successfully on %s (dtype=%s)", models.device, models.weight_dtype)

    # Clean up any leftover temp dirs from previous runs
    temp_root = os.path.join(os.getcwd(), "temp")
    if os.path.isdir(temp_root):
        for entry in os.listdir(temp_root):
            path = os.path.join(temp_root, entry)
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)

    yield

    # --- Shutdown ---
    logger.info("Shutting down...")
    models.is_ready = False
    del models.vae, models.unet, models.pe, models.whisper
    torch.cuda.empty_cache()


app = FastAPI(
    title="MuseTalk API",
    description="Audio-driven lip-sync video synthesis. Upload a template video and driving audio to generate a talking-head video.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/api/v1/health")
async def health_check():
    gpu_available = torch.cuda.is_available()
    return {
        "status": "healthy" if models.is_ready else "loading",
        "models_loaded": models.is_ready,
        "gpu_available": gpu_available,
        "gpu_name": torch.cuda.get_device_name(0) if gpu_available else None,
        "device": str(models.device) if models.device else None,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
ALLOWED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv"}
ALLOWED_AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac"}


def _save_upload(upload: UploadFile, dest_path: str):
    with open(dest_path, "wb") as f:
        while chunk := upload.file.read(1024 * 1024):
            f.write(chunk)


def _normalize_video_fps(video_path: str, output_path: str, target_fps: int = 25):
    """Resample a video to the target fps (same logic as app.py:check_video)."""
    reader = imageio.get_reader(video_path)
    fps = reader.get_meta_data()["fps"]
    frames = [im for im in reader]
    reader.close()

    if abs(fps - target_fps) < 0.5:
        # Already close enough — just copy
        shutil.copy2(video_path, output_path)
        return

    L = len(frames)
    L_target = int(L / fps * target_fps)
    original_t = [x / fps for x in range(1, L + 1)]
    t_idx = 0
    target_frames = []
    for target_t in range(1, L_target + 1):
        while target_t / target_fps > original_t[t_idx]:
            t_idx += 1
            if t_idx >= L:
                break
        target_frames.append(frames[min(t_idx, L - 1)])

    imageio.mimwrite(
        output_path, target_frames, "FFMPEG",
        fps=target_fps, codec="libx264", quality=9, pixelformat="yuv420p",
    )


# ---------------------------------------------------------------------------
# Core synthesis endpoint
# ---------------------------------------------------------------------------
@app.post("/api/v1/synthesize")
def synthesize(
    video: UploadFile = File(..., description="Template video file (mp4/avi/mov)"),
    audio: UploadFile = File(..., description="Driving audio file (wav/mp3/m4a)"),
    bbox_shift: int = Form(0, description="Bounding box vertical shift (px)"),
    extra_margin: int = Form(10, ge=0, le=40, description="Extra margin for jaw region"),
    parsing_mode: str = Form("jaw", description="Face parsing mode: jaw or raw"),
    left_cheek_width: int = Form(90, ge=20, le=160, description="Left cheek mask width"),
    right_cheek_width: int = Form(90, ge=20, le=160, description="Right cheek mask width"),
    batch_size: int = Form(8, ge=1, le=32, description="Inference batch size"),
):
    """
    Synthesize a talking-head video from a template video and driving audio.
    """
    if not models.is_ready:
        raise HTTPException(status_code=503, detail="Models are still loading. Please retry later.")

    # Validate file extensions
    video_ext = os.path.splitext(video.filename or "")[1].lower()
    audio_ext = os.path.splitext(audio.filename or "")[1].lower()
    if video_ext not in ALLOWED_VIDEO_EXTS:
        raise HTTPException(400, f"Unsupported video format '{video_ext}'. Allowed: {ALLOWED_VIDEO_EXTS}")
    if audio_ext not in ALLOWED_AUDIO_EXTS:
        raise HTTPException(400, f"Unsupported audio format '{audio_ext}'. Allowed: {ALLOWED_AUDIO_EXTS}")
    if parsing_mode not in ("jaw", "raw"):
        raise HTTPException(400, "parsing_mode must be 'jaw' or 'raw'")

    # Create isolated working directory
    work_dir = tempfile.mkdtemp(prefix="musetalk_", dir=os.path.join(os.getcwd(), "temp"))

    try:
        with _inference_lock:
            return _run_inference(
                work_dir=work_dir,
                video_upload=video,
                audio_upload=audio,
                video_ext=video_ext,
                audio_ext=audio_ext,
                bbox_shift=bbox_shift,
                extra_margin=extra_margin,
                parsing_mode=parsing_mode,
                left_cheek_width=left_cheek_width,
                right_cheek_width=right_cheek_width,
                batch_size=batch_size,
            )
    except HTTPException:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise
    except Exception as e:
        logger.exception("Synthesis failed")
        shutil.rmtree(work_dir, ignore_errors=True)
        raise HTTPException(500, f"Synthesis failed: {str(e)}")


@torch.no_grad()
def _run_inference(
    *,
    work_dir: str,
    video_upload: UploadFile,
    audio_upload: UploadFile,
    video_ext: str,
    audio_ext: str,
    bbox_shift: int,
    extra_margin: int,
    parsing_mode: str,
    left_cheek_width: int,
    right_cheek_width: int,
    batch_size: int,
) -> FileResponse:
    device = models.device
    vae = models.vae
    unet = models.unet
    pe = models.pe
    timesteps = models.timesteps
    audio_processor = models.audio_processor
    whisper = models.whisper
    weight_dtype = models.weight_dtype

    # ---- 1. Save uploaded files ----
    raw_video_path = os.path.join(work_dir, f"input_video{video_ext}")
    raw_audio_path = os.path.join(work_dir, f"input_audio{audio_ext}")
    _save_upload(video_upload, raw_video_path)
    _save_upload(audio_upload, raw_audio_path)

    # ---- 2. Convert audio to WAV 16kHz PCM ----
    wav_path = os.path.join(work_dir, "input_audio_16k.wav")
    ensure_wav(raw_audio_path, target_path=wav_path)

    # ---- 3. Normalize video to 25fps ----
    norm_video_path = os.path.join(work_dir, "input_video_25fps.mp4")
    _normalize_video_fps(raw_video_path, norm_video_path, target_fps=25)

    # ---- 4. Extract frames ----
    frames_dir = os.path.join(work_dir, "frames")
    os.makedirs(frames_dir)
    reader = imageio.get_reader(norm_video_path)
    for i, im in enumerate(reader):
        imageio.imwrite(os.path.join(frames_dir, f"{i:08d}.png"), im)
    reader.close()
    fps = get_video_fps(norm_video_path)
    input_img_list = sorted(glob.glob(os.path.join(frames_dir, "*.[jpJP][pnPN]*[gG]")))

    if not input_img_list:
        raise HTTPException(422, "Failed to extract frames from video")

    # ---- 5. Extract audio features ----
    whisper_input_features, librosa_length = audio_processor.get_audio_feature(wav_path)
    whisper_chunks = audio_processor.get_whisper_chunk(
        whisper_input_features,
        device,
        weight_dtype,
        whisper,
        librosa_length,
        fps=fps,
        audio_padding_length_left=2,
        audio_padding_length_right=2,
    )

    # ---- 6. Face landmarks + bounding boxes ----
    logger.info("Extracting face landmarks...")
    coord_list, frame_list = get_landmark_and_bbox(input_img_list, bbox_shift)

    if all(c == coord_placeholder for c in coord_list):
        raise HTTPException(422, "No face detected in the video. Try adjusting bbox_shift.")

    # ---- 7. VAE encode face crops ----
    input_latent_list = []
    for bbox, frame in zip(coord_list, frame_list):
        if bbox == coord_placeholder:
            continue
        x1, y1, x2, y2 = bbox
        y2 = y2 + extra_margin
        y2 = min(y2, frame.shape[0])
        crop_frame = frame[y1:y2, x1:x2]
        crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        latents = vae.get_latents_for_unet(crop_frame)
        input_latent_list.append(latents)

    # ---- 8. Cycle lists for smooth looping ----
    frame_list_cycle = frame_list + frame_list[::-1]
    coord_list_cycle = coord_list + coord_list[::-1]
    input_latent_list_cycle = input_latent_list + input_latent_list[::-1]

    # ---- 9. Batch inference ----
    logger.info("Running inference (%d audio chunks, batch_size=%d)...", len(whisper_chunks), batch_size)
    video_num = len(whisper_chunks)
    gen = datagen(
        whisper_chunks=whisper_chunks,
        vae_encode_latents=input_latent_list_cycle,
        batch_size=batch_size,
        delay_frame=0,
        device=device,
    )

    res_frame_list = []
    for whisper_batch, latent_batch in tqdm(gen, total=int(np.ceil(float(video_num) / batch_size))):
        audio_feature_batch = pe(whisper_batch)
        latent_batch = latent_batch.to(dtype=weight_dtype)
        pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
        recon = vae.decode_latents(pred_latents)
        for res_frame in recon:
            res_frame_list.append(res_frame)

    # ---- 10. Face blending / compositing ----
    logger.info("Blending %d frames...", len(res_frame_list))
    fp = FaceParsing(left_cheek_width=left_cheek_width, right_cheek_width=right_cheek_width)
    result_frames_dir = os.path.join(work_dir, "result_frames")
    os.makedirs(result_frames_dir)

    for i, res_frame in enumerate(res_frame_list):
        bbox = coord_list_cycle[i % len(coord_list_cycle)]
        ori_frame = copy.deepcopy(frame_list_cycle[i % len(frame_list_cycle)])
        x1, y1, x2, y2 = bbox
        y2 = y2 + extra_margin
        y2 = min(y2, ori_frame.shape[0])
        try:
            res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
        except Exception:
            continue
        combine_frame = get_image(ori_frame, res_frame, [x1, y1, x2, y2], mode=parsing_mode, fp=fp)
        cv2.imwrite(os.path.join(result_frames_dir, f"{i:08d}.png"), combine_frame)

    # ---- 11. Assemble frames into video ----
    valid_pattern = re.compile(r"\d{8}\.png")
    files = sorted(
        [f for f in os.listdir(result_frames_dir) if valid_pattern.match(f)],
        key=lambda x: int(x.split(".")[0]),
    )
    images = [imageio.imread(os.path.join(result_frames_dir, f)) for f in files]

    if not images:
        raise HTTPException(500, "No result frames generated")

    temp_video_path = os.path.join(work_dir, "temp_no_audio.mp4")
    imageio.mimwrite(temp_video_path, images, "FFMPEG", fps=25, codec="libx264", pixelformat="yuv420p")

    # ---- 12. Mux audio with ffmpeg ----
    output_path = os.path.join(work_dir, "output.mp4")
    cmd = [
        "ffmpeg", "-y", "-v", "warning",
        "-i", wav_path,
        "-i", temp_video_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("ffmpeg mux failed: %s", result.stderr)
        raise HTTPException(500, f"ffmpeg audio mux failed: {result.stderr}")

    logger.info("Synthesis complete: %s", output_path)

    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=f"synthesized_{uuid.uuid4().hex[:8]}.mp4",
        background=BackgroundTask(shutil.rmtree, work_dir),
    )
