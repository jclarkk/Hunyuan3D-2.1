import logging
import os
from contextlib import suppress
from typing import Iterable, List, Optional, Sequence

import torch
from PIL import Image
from diffusers import DiffusionPipeline


LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    LOGGER.addHandler(_handler)
LOGGER.setLevel(logging.INFO)


class QwenEditQuantPipelineWrapper:
    """
    Wrapper around the quantized Qwen image-edit pipeline with LoRA support.
    Generates canonical six-view coverage (front, right, back, left, top, bottom)
    and returns PIL images compatible with downstream multiview processing.
    """

    def __init__(self, pipeline: DiffusionPipeline, device: str, config) -> None:
        self.pipeline = pipeline
        self.device = device
        self.config = config

        self.primary_view_count = int(getattr(config, "qwen_edit_primary_view_count", 6))
        self.reference_size = getattr(config, "qwen_edit_reference_size", None)
        self.prompt_template = getattr(config, "qwen_edit_prompt_template", "{}")
        self.camera_prompts = getattr(config, "qwen_edit_camera_prompts", {})
        self.negative_prompt = getattr(config, "qwen_edit_negative_prompt", "")
        self.guidance_scale = float(getattr(config, "qwen_edit_guidance_scale", 4.5))
        self.num_inference_steps = int(getattr(config, "qwen_edit_num_inference_steps", 30))
        self.control_scale = float(getattr(config, "qwen_edit_control_scale", 1.0))

    def to(self, device: str) -> "QwenEditQuantPipelineWrapper":
        if hasattr(self.pipeline, "to"):
            self.pipeline.to(device=device)
        self.device = device
        return self

    @classmethod
    def from_config(cls, config, device: str = "cuda") -> "QwenEditQuantPipelineWrapper":
        base_model = getattr(config, "qwen_edit_base_model", None)
        if not base_model:
            raise ValueError(
                "qwen-edit-quant selected but `config.qwen_edit_base_model` is not set. "
                "Please provide the base Qwen image-edit model identifier or path."
            )

        pipeline_kwargs = getattr(config, "qwen_edit_pipeline_kwargs", {}) or {}
        fuse_lora = getattr(config, "qwen_edit_fuse_lora", True)

        dtype = getattr(config, "qwen_edit_dtype", None)
        if dtype is not None and not isinstance(dtype, torch.dtype):
            dtype = getattr(torch, dtype, None)
        if dtype is not None:
            pipeline_kwargs["torch_dtype"] = dtype

        pipeline_kwargs["local_files_only"] = getattr(config, "local_files_only", False)
        custom_pipeline = getattr(config, "qwen_edit_custom_pipeline", None)
        pipeline_kwargs = QwenEditQuantPipelineWrapper._sanitize_kwargs(pipeline_kwargs, custom_pipeline=custom_pipeline)

        LOGGER.info("Loading Qwen edit pipeline from %s", base_model)
        pipeline = DiffusionPipeline.from_pretrained(base_model, **pipeline_kwargs)
        pipeline.set_progress_bar_config(disable=True)

        if hasattr(pipeline, "to"):
            pipeline.to(device)

        cls._apply_loras(
            pipeline,
            getattr(config, "qwen_edit_lora_paths", None),
            fuse=fuse_lora,
        )
        if fuse_lora:
            LOGGER.info("Fused Qwen LoRA weights into base pipeline")
        else:
            LOGGER.info("Loaded Qwen LoRA weights without fusion")

        with suppress(AttributeError):
            pipeline.enable_attention_slicing()

        return cls(pipeline, device, config)

    @staticmethod
    def _sanitize_kwargs(original_kwargs, custom_pipeline=None):
        sanitized = dict(original_kwargs)
        if custom_pipeline in (None, "", False):
            sanitized.pop("custom_pipeline", None)
        else:
            if isinstance(custom_pipeline, str) and (custom_pipeline.endswith(".py") or "/" in custom_pipeline):
                sanitized["custom_pipeline"] = custom_pipeline
            else:
                sanitized.pop("custom_pipeline", None)

        return sanitized

    @staticmethod
    def _apply_loras(pipeline: DiffusionPipeline, lora_paths: Optional[Iterable[str]], fuse: bool = False) -> None:
        if not lora_paths:
            return

        adapters: List[str] = []
        for raw_path in lora_paths:
            if not raw_path:
                continue
            path = os.path.expanduser(raw_path)
            if os.path.isdir(path):
                weight_name = QwenEditQuantPipelineWrapper._first_lora_file(path)
                load_path = path
            else:
                load_path, weight_name = os.path.split(path)
                load_path = load_path or "."

            adapter_name = QwenEditQuantPipelineWrapper._adapter_name(weight_name or path)

            if weight_name and not os.path.exists(os.path.join(load_path, weight_name)):
                LOGGER.warning("LoRA weight %s not found in %s", weight_name, load_path)
                continue
            if weight_name is None and not os.path.exists(path):
                LOGGER.warning("LoRA path %s is missing", path)
                continue

            try:
                if os.path.isdir(path) and weight_name is None:
                    pipeline.load_lora_weights(path, adapter_name=adapter_name)
                else:
                    pipeline.load_lora_weights(load_path, weight_name=weight_name, adapter_name=adapter_name)
                adapters.append(adapter_name)
                LOGGER.info("Loaded Qwen LoRA: %s", adapter_name)
            except Exception as exc:  # pylint: disable=broad-except
                LOGGER.warning("Failed to load LoRA %s: %s", path, exc)

        if not adapters:
            return

        with suppress(AttributeError):
            pipeline.set_adapters(adapters, adapter_weights=[1.0] * len(adapters))
        if fuse:
            with suppress(Exception):
                pipeline.fuse_lora()

    @staticmethod
    def _first_lora_file(directory: str) -> Optional[str]:
        for filename in os.listdir(directory):
            if filename.endswith((".safetensors", ".bin", ".pt")):
                return filename
        return None

    @staticmethod
    def _adapter_name(identifier: str) -> str:
        base = os.path.basename(identifier)
        name, _ = os.path.splitext(base)
        return name.replace(".", "_")

    @torch.no_grad()
    def __call__(
        self,
        reference_images: Sequence[Image.Image],
        base_prompt: Optional[str],
        camera_elevations: Sequence[float],
        camera_azimuths: Sequence[float],
        num_views: int,
        seed: int = 42,
        negative_prompt: Optional[str] = None,
        control_images: Optional[Sequence[Image.Image]] = None,
    ) -> List[Image.Image]:
        if not reference_images:
            raise ValueError("QwenEditQuantPipelineWrapper requires at least one reference image.")

        negative = negative_prompt or self.negative_prompt

        view_count = min(num_views, self.primary_view_count, len(camera_azimuths), len(camera_elevations), len(reference_images))
        outputs: List[Image.Image] = []

        LOGGER.info("Starting Qwen stylisation for %d view(s)", view_count)
        for view_idx in range(view_count):
            reference_batch = reference_images[view_idx]
            if not isinstance(reference_batch, (list, tuple)):
                reference_batch = [reference_batch]

            processed_batch: List[Image.Image] = []
            for idx, img in enumerate(reference_batch):
                if idx == 0:
                    processed_batch.append(self._prepare_reference(img))
                else:
                    processed_batch.append(self._prepare_aux_image(img))

            reference = processed_batch[0]

            LOGGER.info(
                "Qwen stylising view %d/%d (azimuth=%.2f°, elevation=%.2f°)",
                view_idx + 1,
                view_count,
                camera_azimuths[view_idx],
                camera_elevations[view_idx],
            )
            prompt_text = self._compose_prompt(
                base_prompt,
                camera_azimuths[view_idx],
                camera_elevations[view_idx],
            )
            if len(processed_batch) > 1:
                prompt_text = f"{prompt_text} 第二张图给出了位置边界，请严格贴合。"

            generator = None
            if seed != -1:
                generator = torch.Generator(device=self.device).manual_seed(seed + view_idx)

            result = self.pipeline(
                prompt=prompt_text,
                image=processed_batch,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                negative_prompt=negative,
                generator=generator,
            )

            images = getattr(result, "images", result)
            outputs.append(images[0])
            LOGGER.info("Completed Qwen view %d/%d", view_idx + 1, view_count)

        return outputs

    def _prepare_reference(self, image: Image.Image) -> Image.Image:
        if not isinstance(image, Image.Image):
            raise TypeError(f"Expected PIL.Image, received {type(image).__name__}")

        reference = image.convert("RGB")
        if self.reference_size:
            reference = reference.resize(
                (self.reference_size, self.reference_size),
                resample=Image.LANCZOS,
            )
        return reference

    def _compose_prompt(self, base_prompt: Optional[str], azimuth: float, elevation: float) -> str:
        key = self._view_key(azimuth, elevation)
        instruction = self.camera_prompts.get(key)
        if instruction is None:
            instruction = (
                f"Move the camera so that the subject is viewed at azimuth {azimuth:.1f}° "
                f"and elevation {elevation:.1f}°."
            )

        template = self.prompt_template or "{}"
        if "{}" in template:
            composed = template.format(instruction)
        else:
            composed = f"{template} {instruction}"

        if base_prompt:
            return f"{base_prompt.rstrip('.')}. {composed}"
        return composed

    @staticmethod
    def _view_key(azimuth: float, elevation: float) -> str:
        if elevation >= 45:
            return "top"
        if elevation <= -45:
            return "bottom"

        azimuth_norm = azimuth % 360
        if azimuth_norm < 45 or azimuth_norm >= 315:
            return "front"
        if azimuth_norm < 135:
            return "right"
        if azimuth_norm < 225:
            return "back"
        return "left"

    def _prepare_aux_image(self, image: Image.Image) -> Image.Image:
        if not isinstance(image, Image.Image):
            raise TypeError(f"Expected PIL.Image, received {type(image).__name__}")
        aux = image.convert("RGB")
        if self.reference_size:
            aux = aux.resize((self.reference_size, self.reference_size), Image.NEAREST)
        return aux

