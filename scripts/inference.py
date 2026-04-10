import argparse
import copy
import glob
import os
import pickle
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import WhisperModel

from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.blending import get_image
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.preprocessing import coord_placeholder, get_landmark_and_bbox, read_imgs
from musetalk.utils.utils import datagen, get_file_type, get_video_fps, load_all_model


class MuseTalkSynthesizer:
    """Class-based MuseTalk interface: audio path + template path -> synthesized video."""

    def __init__(
        self,
        *,
        ffmpeg_path: str = "./ffmpeg-4.4-amd64-static/",
        gpu_id: int = 0,
        vae_type: str = "sd-vae",
        unet_config: str = "./models/musetalk/config.json",
        unet_model_path: str = "./models/musetalkV15/unet.pth",
        whisper_dir: str = "./models/whisper",
        use_float16: bool = False,
        version: str = "v15",
        left_cheek_width: int = 90,
        right_cheek_width: int = 90,
    ):
        if version not in {"v1", "v15"}:
            raise ValueError("version must be 'v1' or 'v15'")

        self.version = version
        self._configure_ffmpeg(ffmpeg_path)

        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        self.vae, self.unet, self.pe = load_all_model(
            unet_model_path=unet_model_path,
            vae_type=vae_type,
            unet_config=unet_config,
            device=self.device,
        )
        self.timesteps = torch.tensor([0], device=self.device)

        if use_float16:
            self.pe = self.pe.half()
            self.vae.vae = self.vae.vae.half()
            self.unet.model = self.unet.model.half()

        self.pe = self.pe.to(self.device)
        self.vae.vae = self.vae.vae.to(self.device)
        self.unet.model = self.unet.model.to(self.device)

        self.audio_processor = AudioProcessor(feature_extractor_path=whisper_dir)
        self.weight_dtype = self.unet.model.dtype
        self.whisper = WhisperModel.from_pretrained(whisper_dir)
        self.whisper = self.whisper.to(device=self.device, dtype=self.weight_dtype).eval()
        self.whisper.requires_grad_(False)

        if version == "v15":
            self.fp = FaceParsing(
                left_cheek_width=left_cheek_width,
                right_cheek_width=right_cheek_width,
            )
        else:
            self.fp = FaceParsing()

    @staticmethod
    def _ffmpeg_available() -> bool:
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
            return True
        except Exception:
            return False

    def _configure_ffmpeg(self, ffmpeg_path: str) -> None:
        if self._ffmpeg_available():
            return
        path_separator = ";" if os.name == "nt" else ":"
        os.environ["PATH"] = f"{ffmpeg_path}{path_separator}{os.environ.get('PATH', '')}"
        if not self._ffmpeg_available():
            raise RuntimeError("Unable to find ffmpeg. Please install ffmpeg or set --ffmpeg_path.")

    @staticmethod
    def _collect_images_from_directory(image_dir: str):
        image_list = glob.glob(os.path.join(image_dir, "*.[jpJP][pnPN]*[gG]"))

        def _sort_key(path: str):
            stem = os.path.splitext(os.path.basename(path))[0]
            return (0, int(stem)) if stem.isdigit() else (1, stem)

        return sorted(image_list, key=_sort_key)

    @torch.no_grad()
    def synthesize(
        self,
        *,
        audio_path: str,
        video_path: str,
        output_path: str,
        result_dir: str = "./results",
        bbox_shift: int = 0,
        extra_margin: int = 10,
        fps: int = 25,
        batch_size: int = 8,
        audio_padding_length_left: int = 2,
        audio_padding_length_right: int = 2,
        use_saved_coord: bool = False,
        saved_coord: bool = False,
        parsing_mode: str = "jaw",
    ) -> str:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Template video/image not found: {video_path}")

        output_path = str(Path(output_path))
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        input_basename = os.path.basename(video_path).split(".")[0]
        audio_basename = os.path.basename(audio_path).split(".")[0]
        output_basename = f"{input_basename}_{audio_basename}"

        work_dir = os.path.join(result_dir, self.version, f"tmp_{output_basename}")
        input_frames_dir = os.path.join(work_dir, "input_frames")
        result_frames_dir = os.path.join(work_dir, "result_frames")
        crop_coord_save_path = os.path.join(result_dir, f"{input_basename}.pkl")

        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)
        os.makedirs(input_frames_dir, exist_ok=True)
        os.makedirs(result_frames_dir, exist_ok=True)

        try:
            file_type = get_file_type(video_path)
            if file_type == "video":
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-v",
                        "fatal",
                        "-i",
                        video_path,
                        os.path.join(input_frames_dir, "%08d.png"),
                    ],
                    check=True,
                )
                input_img_list = self._collect_images_from_directory(input_frames_dir)
                fps = get_video_fps(video_path)
            elif file_type == "image":
                input_img_list = [video_path]
            elif os.path.isdir(video_path):
                input_img_list = self._collect_images_from_directory(video_path)
            else:
                raise ValueError(f"{video_path} should be a video file, image file or image directory")

            if not input_img_list:
                raise RuntimeError("No input images were prepared from template video")

            whisper_input_features, librosa_length = self.audio_processor.get_audio_feature(audio_path)
            whisper_chunks = self.audio_processor.get_whisper_chunk(
                whisper_input_features,
                self.device,
                self.weight_dtype,
                self.whisper,
                librosa_length,
                fps=fps,
                audio_padding_length_left=audio_padding_length_left,
                audio_padding_length_right=audio_padding_length_right,
            )

            if os.path.exists(crop_coord_save_path) and use_saved_coord:
                with open(crop_coord_save_path, "rb") as f:
                    coord_list = pickle.load(f)
                frame_list = read_imgs(input_img_list)
            else:
                coord_list, frame_list = get_landmark_and_bbox(input_img_list, bbox_shift)
                if saved_coord:
                    with open(crop_coord_save_path, "wb") as f:
                        pickle.dump(coord_list, f)

            input_latent_list = []
            for bbox, frame in zip(coord_list, frame_list):
                if bbox == coord_placeholder:
                    continue
                x1, y1, x2, y2 = bbox
                if self.version == "v15":
                    y2 = min(y2 + extra_margin, frame.shape[0])
                crop_frame = frame[y1:y2, x1:x2]
                crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
                latents = self.vae.get_latents_for_unet(crop_frame)
                input_latent_list.append(latents)

            if not input_latent_list:
                raise RuntimeError("No valid face regions were detected")

            frame_list_cycle = frame_list + frame_list[::-1]
            coord_list_cycle = coord_list + coord_list[::-1]
            input_latent_list_cycle = input_latent_list + input_latent_list[::-1]

            gen = datagen(
                whisper_chunks=whisper_chunks,
                vae_encode_latents=input_latent_list_cycle,
                batch_size=batch_size,
                delay_frame=0,
                device=self.device,
            )

            res_frame_list = []
            total = int(np.ceil(float(len(whisper_chunks)) / batch_size))
            for whisper_batch, latent_batch in tqdm(gen, total=total):
                audio_feature_batch = self.pe(whisper_batch)
                latent_batch = latent_batch.to(dtype=self.weight_dtype)
                pred_latents = self.unet.model(
                    latent_batch,
                    self.timesteps,
                    encoder_hidden_states=audio_feature_batch,
                ).sample
                recon = self.vae.decode_latents(pred_latents)
                res_frame_list.extend(recon)

            for i, res_frame in enumerate(tqdm(res_frame_list)):
                bbox = coord_list_cycle[i % len(coord_list_cycle)]
                ori_frame = copy.deepcopy(frame_list_cycle[i % len(frame_list_cycle)])
                x1, y1, x2, y2 = bbox
                if self.version == "v15":
                    y2 = min(y2 + extra_margin, ori_frame.shape[0])
                try:
                    res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
                except Exception:
                    continue

                if self.version == "v15":
                    combine_frame = get_image(ori_frame, res_frame, [x1, y1, x2, y2], mode=parsing_mode, fp=self.fp)
                else:
                    combine_frame = get_image(ori_frame, res_frame, [x1, y1, x2, y2], fp=self.fp)
                cv2.imwrite(os.path.join(result_frames_dir, f"{i:08d}.png"), combine_frame)

            temp_vid_path = os.path.join(work_dir, "temp_video.mp4")
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-v",
                    "warning",
                    "-r",
                    str(fps),
                    "-f",
                    "image2",
                    "-i",
                    os.path.join(result_frames_dir, "%08d.png"),
                    "-vcodec",
                    "libx264",
                    "-vf",
                    "format=yuv420p",
                    "-crf",
                    "18",
                    temp_vid_path,
                ],
                check=True,
            )

            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-v",
                    "warning",
                    "-i",
                    audio_path,
                    "-i",
                    temp_vid_path,
                    "-shortest",
                    output_path,
                ],
                check=True,
            )

            return output_path
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    def close(self):
        del self.vae
        del self.unet
        del self.pe
        del self.whisper
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main(args):
    inference_config = OmegaConf.load(args.inference_config)
    print("Loaded inference config:", inference_config)

    synthesizer = MuseTalkSynthesizer(
        ffmpeg_path=args.ffmpeg_path,
        gpu_id=args.gpu_id,
        vae_type=args.vae_type,
        unet_config=args.unet_config,
        unet_model_path=args.unet_model_path,
        whisper_dir=args.whisper_dir,
        use_float16=args.use_float16,
        version=args.version,
        left_cheek_width=args.left_cheek_width,
        right_cheek_width=args.right_cheek_width,
    )

    try:
        for task_id in inference_config:
            try:
                video_path = inference_config[task_id]["video_path"]
                audio_path = inference_config[task_id]["audio_path"]
                task_output_name = inference_config[task_id].get("result_name", args.output_vid_name)

                if args.version == "v15":
                    bbox_shift = 0
                else:
                    bbox_shift = inference_config[task_id].get("bbox_shift", args.bbox_shift)

                input_basename = os.path.basename(video_path).split(".")[0]
                audio_basename = os.path.basename(audio_path).split(".")[0]
                output_basename = f"{input_basename}_{audio_basename}.mp4"

                output_dir = os.path.join(args.result_dir, args.version)
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, task_output_name or output_basename)

                result_path = synthesizer.synthesize(
                    audio_path=audio_path,
                    video_path=video_path,
                    output_path=output_path,
                    result_dir=args.result_dir,
                    bbox_shift=bbox_shift,
                    extra_margin=args.extra_margin,
                    fps=args.fps,
                    batch_size=args.batch_size,
                    audio_padding_length_left=args.audio_padding_length_left,
                    audio_padding_length_right=args.audio_padding_length_right,
                    use_saved_coord=args.use_saved_coord,
                    saved_coord=args.saved_coord,
                    parsing_mode=args.parsing_mode,
                )
                print(f"Results saved to {result_path}")
            except Exception as exc:
                print(f"Error occurred during processing task {task_id}: {exc}")
    finally:
        synthesizer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ffmpeg_path", type=str, default="./ffmpeg-4.4-amd64-static/", help="Path to ffmpeg executable")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--vae_type", type=str, default="sd-vae", help="Type of VAE model")
    parser.add_argument("--unet_config", type=str, default="./models/musetalk/config.json", help="Path to UNet configuration file")
    parser.add_argument("--unet_model_path", type=str, default="./models/musetalkV15/unet.pth", help="Path to UNet model weights")
    parser.add_argument("--whisper_dir", type=str, default="./models/whisper", help="Directory containing Whisper model")
    parser.add_argument("--inference_config", type=str, default="configs/inference/test_img.yaml", help="Path to inference configuration file")
    parser.add_argument("--bbox_shift", type=int, default=0, help="Bounding box shift value")
    parser.add_argument("--result_dir", default="./results", help="Directory for output results")
    parser.add_argument("--extra_margin", type=int, default=10, help="Extra margin for face cropping")
    parser.add_argument("--fps", type=int, default=25, help="Video frames per second")
    parser.add_argument("--audio_padding_length_left", type=int, default=2, help="Left padding length for audio")
    parser.add_argument("--audio_padding_length_right", type=int, default=2, help="Right padding length for audio")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--output_vid_name", type=str, default=None, help="Name of output video file")
    parser.add_argument("--use_saved_coord", action="store_true", help="Use saved coordinates to save time")
    parser.add_argument("--saved_coord", action="store_true", help="Save coordinates for future use")
    parser.add_argument("--use_float16", action="store_true", help="Use float16 for faster inference")
    parser.add_argument("--parsing_mode", default="jaw", help="Face blending parsing mode")
    parser.add_argument("--left_cheek_width", type=int, default=90, help="Width of left cheek region")
    parser.add_argument("--right_cheek_width", type=int, default=90, help="Width of right cheek region")
    parser.add_argument("--version", type=str, default="v15", choices=["v1", "v15"], help="Model version to use")
    main(parser.parse_args())
