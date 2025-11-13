# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import gc
import os
import logging
import torch
import copy
import time
import trimesh
import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Union
from DifferentiableRenderer.MeshRender import MeshRender
from hy3dpaint.mvadapter.pipeline import MVAdapterPipelineWrapper
from hy3dpaint.mvadapter.pipelines.pipeline_mvadapter_i2mv_sdxl import MVAdapterI2MVSDXLPipeline
from hy3dpaint.mvadapter.pipelines.pipeline_mvadapter_t2mv_sdxl import MVAdapterT2MVSDXLPipeline
from hy3dpaint.qwenedit.pipeline import QwenEditQuantPipelineWrapper
from utils.multiview_utils import multiviewDiffusionNet, MetalRoughnessOnlyNet
from utils.pipeline_utils import ViewProcessor
import warnings

warnings.filterwarnings("ignore")
from diffusers.utils import logging as diffusers_logging

diffusers_logging.set_verbosity(50)

LOGGER = logging.getLogger(__name__)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"


def aggressive_memory_cleanup():
    """Aggressively clean up GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    # Force PyTorch to release cached memory
    if hasattr(torch.cuda, 'memory._dump_snapshot'):
        torch.cuda.memory._dump_snapshot()


def move_model_to_cpu(model):
    """Move model to CPU and clean up GPU memory"""
    model.to('cpu')
    aggressive_memory_cleanup()


def delete_model_and_cleanup(model_dict, key):
    """Delete model and clean up all references"""
    if key in model_dict:
        model = model_dict[key]
        # Move to CPU first to ensure GPU memory is freed
        try:
            model.to('cpu')
        except:
            pass
        # Delete the model
        del model_dict[key]
        del model
        aggressive_memory_cleanup()


def _resolve_torch_dtype(dtype: Optional[Union[str, torch.dtype]]) -> Optional[torch.dtype]:
    if dtype is None:
        return None
    if isinstance(dtype, torch.dtype):
        return dtype

    dtype_str = str(dtype).lower()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping.get(dtype_str, None)


class Hunyuan3DPaintConfig:
    def __init__(
        self,
        hypaint_resolution: int = 1024,
        local_files_only: bool = False,
        optimization_level: str = "balanced",
        multiview_pretrained_path: str = "tencent/Hunyuan3D-2.1",
        qwen_edit_base_model: Optional[str] = None,
        qwen_edit_lora_paths: Optional[List[str]] = None,
        qwen_edit_negative_prompt: Optional[str] = None,
        qwen_edit_guidance_scale: float = 4.5,
        qwen_edit_strength: float = 0.6,
        qwen_edit_num_inference_steps: int = 30,
        qwen_edit_custom_pipeline: Optional[str] = None,
        qwen_edit_dtype: Optional[Union[str, torch.dtype]] = torch.bfloat16,
        qwen_edit_fuse_lora: bool = True,
        qwen_edit_use_control: bool = True,
        qwen_edit_control_scale: float = 1.0,
    ) -> None:
        self.device = "cuda"
        self.local_files_only = local_files_only

        self.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
        self.custom_pipeline = "hunyuanpaintpbr"
        self.multiview_pretrained_path = multiview_pretrained_path
        self.dino_ckpt_path = "facebook/dinov2-giant"

        self.raster_mode = "cr"
        self.bake_mode = "back_sample"
        self.render_size = 1024 * 2
        self.texture_size = 1024 * 4
        self.hypaint_resolution = hypaint_resolution
        self.bake_exp = 4
        self.merge_method = "fast"

        # view selection
        self.candidate_camera_azims = [0, 90, 180, 270, 0, 180]
        self.candidate_camera_elevs = [0, 0, 0, 0, 90, -90]
        self.candidate_view_weights = [1, 0.1, 0.5, 0.1, 0.05, 0.05]

        for azim in range(0, 360, 30):
            self.candidate_camera_azims.append(azim)
            self.candidate_camera_elevs.append(20)
            self.candidate_view_weights.append(0.01)

            self.candidate_camera_azims.append(azim)
            self.candidate_camera_elevs.append(-20)
            self.candidate_view_weights.append(0.01)

        self.optimization_level = optimization_level
        self.continuous_inference = True

        # Quantized Qwen edit defaults
        self.qwen_edit_base_model = qwen_edit_base_model
        self.qwen_edit_lora_paths = qwen_edit_lora_paths or [
            os.path.join("weights", "loras", "Qwen-Edit-2509-Multiple-angles.safetensors"),
            os.path.join("weights", "loras", "Qwen-Image-Lightning.safetensors"),
        ]
        self.qwen_edit_negative_prompt = qwen_edit_negative_prompt or (
            "watermark, ugly, deformed, noisy, blurry, low contrast, baked lighting, ambient occlusion, shadow artifacts"
        )
        self.qwen_edit_guidance_scale = qwen_edit_guidance_scale
        self.qwen_edit_strength = qwen_edit_strength
        self.qwen_edit_num_inference_steps = qwen_edit_num_inference_steps
        self.qwen_edit_primary_view_count = 6
        self.qwen_edit_reference_size = 1024
        self.qwen_edit_prompt_template = "高质量3D参考图，{}"
        self.qwen_edit_custom_pipeline = qwen_edit_custom_pipeline
        self.qwen_edit_dtype = _resolve_torch_dtype(qwen_edit_dtype)
        self.qwen_edit_fuse_lora = qwen_edit_fuse_lora
        self.qwen_edit_use_control = qwen_edit_use_control
        self.qwen_edit_control_scale = qwen_edit_control_scale
        self.qwen_edit_camera_prompts: Dict[str, str] = {
            "front": "保持镜头正面对准主体。",
            "right": "将镜头向右旋转90度。",
            "back": "将镜头向后旋转180度观看主体背面。",
            "left": "将镜头向左旋转90度。",
            "top": "将镜头转为俯视视角。",
            "bottom": "将镜头转为仰视视角。",
        }
        self.qwen_edit_pipeline_kwargs: Dict[str, object] = {
            "trust_remote_code": True,
            "use_safetensors": True,
            "low_cpu_mem_usage": True,
        }
        if self.qwen_edit_custom_pipeline:
            self.qwen_edit_pipeline_kwargs.setdefault("custom_pipeline", self.qwen_edit_custom_pipeline)
        if self.qwen_edit_dtype is not None:
            self.qwen_edit_pipeline_kwargs["torch_dtype"] = self.qwen_edit_dtype


class Hunyuan3DPaintPipeline:

    def __init__(self, config=None) -> None:
        self.config = config if config is not None else Hunyuan3DPaintConfig()
        self.models = {}
        self.stats_logs = {}
        self.render = MeshRender(
            default_resolution=self.config.render_size,
            texture_size=self.config.texture_size,
            bake_mode=self.config.bake_mode,
            raster_mode=self.config.raster_mode,
        )
        self.view_processor = ViewProcessor(self.config, self.render)
        self.load_models()

    def _ensure_mvadapter_fallback(self):
        model = self.models.get("mv_adapter_fallback")
        if model is None:
            print("Loading MVAdapter fallback pipeline...")
            model = MVAdapterPipelineWrapper.from_pretrained(
                device=self.config.device,
                local_files_only=self.config.local_files_only,
                model_cls=MVAdapterI2MVSDXLPipeline,
            )
            self.models["mv_adapter_fallback"] = model
        else:
            model.to(self.config.device)
        return model

    def _generate_additional_views_with_mvadapter(
        self,
        mesh,
        image_prompt,
        normal_maps,
        position_maps,
        camera_elevations,
        camera_azimuths,
        seed,
    ):
        if not normal_maps:
            return []

        mv_adapter = self._ensure_mvadapter_fallback()
        reference_image = image_prompt
        if isinstance(image_prompt, list):
            reference_image = image_prompt[0] if image_prompt else None

        view_count = len(camera_elevations)
        result = mv_adapter(
            mesh,
            image_prompt=reference_image,
            normal_maps=normal_maps,
            position_maps=position_maps,
            camera_elevation_deg=camera_elevations,
            camera_azimuth_deg=camera_azimuths,
            num_views=view_count,
            seed=seed,
            height=self.config.hypaint_resolution,
            width=self.config.hypaint_resolution,
            use_mesh_renderer=False,
        )

        albedo = (result or {}).get("albedo", [])[:view_count]

        if self.config.continuous_inference:
            move_model_to_cpu(self.models["mv_adapter_fallback"])
        else:
            delete_model_and_cleanup(self.models, "mv_adapter_fallback")

        return albedo

    def _run_pbr_generation(self, multiviews, normal_maps, position_maps):
        print("Preparing for PBR generation...")

        albedo_views_cpu = [img.copy() for img in multiviews["albedo"]]

        if self.config.continuous_inference and "multiview_model" in self.models:
            print("Moving multiview model to CPU...")
            move_model_to_cpu(self.models["multiview_model"])
        elif "multiview_model" in self.models:
            print("Deleting multiview model...")
            delete_model_and_cleanup(self.models, "multiview_model")

        del multiviews

        aggressive_memory_cleanup()
        time.sleep(1)

        print(f"GPU memory before PBR: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB allocated")

        t2 = time.time()
        print("Loading PBR model...")
        self.models["pbr_model"] = MetalRoughnessOnlyNet(self.config)

        mr_views = self._execute_pbr_batches(albedo_views_cpu, normal_maps, position_maps)

        for i in range(len(mr_views["mr"])):
            mr_views["mr"][i] = mr_views["mr"][i].resize(
                (self.config.hypaint_resolution, self.config.hypaint_resolution)
            )

        result = {
            "albedo": albedo_views_cpu,
            "mr": mr_views["mr"],
        }

        delete_model_and_cleanup(self.models, "pbr_model")

        if self.config.continuous_inference and "multiview_model" in self.models:
            print("Restoring multiview model to GPU...")
            self.models["multiview_model"].to(self.config.device)

        t3 = time.time()
        print(f"PBR generation took {t3 - t2:.2f} seconds")

        return result

    def _execute_pbr_batches(self, albedo_views_cpu, normal_maps, position_maps):
        view_count = len(albedo_views_cpu)

        if (
            view_count > 6
            and getattr(self.config, "optimization_level", "balanced") == "aggressive"
        ):
            mr_views_list = []
            for i in range(0, view_count, 6):
                batch_end = min(i + 6, view_count)
                batch_albedo = albedo_views_cpu[i:batch_end]
                batch_normal = normal_maps[i:batch_end]
                batch_position = position_maps[i:batch_end]

                if len(batch_albedo) < 6:
                    pad_count = 6 - len(batch_albedo)
                    batch_albedo += [batch_albedo[-1]] * pad_count
                    batch_normal += [batch_normal[-1]] * pad_count
                    batch_position += [batch_position[-1]] * pad_count

                with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                    batch_mr = self.models["pbr_model"](
                        batch_albedo[0],
                        batch_normal + batch_position,
                        prompt="material roughness and metallic map",
                        custom_view_size=512,
                        resize_input=True,
                    )

                mr_views_list.extend(batch_mr["mr"][: batch_end - i])
                aggressive_memory_cleanup()

            return {"mr": mr_views_list[:view_count]}

        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
            mr_views = self.models["pbr_model"](
                albedo_views_cpu[0],
                normal_maps + position_maps,
                prompt="material roughness and metallic map",
                custom_view_size=512,
                resize_input=True,
            )

        return mr_views


    def load_models(self):
        torch.cuda.empty_cache()
        if self.config.multiview_pretrained_path == "tencent/Hunyuan3D-2.1":
            print("Loading Hunyuan3D-2.1 Multiview Diffusion Model...")
            self.models["multiview_model"] = multiviewDiffusionNet(self.config)
        elif self.config.multiview_pretrained_path == "mv-adapter":
            self.models["multiview_model"] = MVAdapterPipelineWrapper.from_pretrained(device=self.config.device,
                                                                                      local_files_only=self.config.local_files_only,
                                                                                      model_cls=MVAdapterI2MVSDXLPipeline)
        elif self.config.multiview_pretrained_path == "mv-adapter-t2mv":
            self.models["multiview_model"] = MVAdapterPipelineWrapper.from_pretrained(device=self.config.device,
                                                                                      local_files_only=self.config.local_files_only,
                                                                                      model_cls=MVAdapterT2MVSDXLPipeline)
        elif self.config.multiview_pretrained_path == "qwen-edit-quant":
            self.models["multiview_model"] = QwenEditQuantPipelineWrapper.from_config(
                self.config, device=self.config.device
            )
            self.models["mv_adapter_fallback"] = None

        print("Models Loaded.")

    @torch.no_grad()
    def __call__(self, mesh_path=None, image_path=None, prompt=None, use_remesh=False, upscale_model='NMKD', pbr=True,
                 texture_size=4096, seed=42, unwrap_method='xatlas', num_views=6):
        self.config.texture_size = texture_size
        self.render.set_default_texture_resolution(texture_size)

        """Generate texture for 3D mesh using multiview diffusion"""
        image_prompt = None
        if image_path is not None:
            # Ensure image_prompt is a PIL image if it's not a list.
            if isinstance(image_path, str):
                image_prompt = [Image.open(image_path)]
            elif isinstance(image_path, Image.Image):
                image_prompt = [image_path]
            elif isinstance(image_path, List):
                image_prompt = []
                for img in image_path:
                    if isinstance(img, str):
                        img = Image.open(img)
                    elif not isinstance(img, Image.Image):
                        raise ValueError("image_path must be a string or a PIL.Image object")
                    image_prompt.append(img)

        # Load mesh
        if isinstance(mesh_path, str):
            mesh = trimesh.load(mesh_path, force='mesh')
        elif isinstance(mesh_path, trimesh.Trimesh):
            mesh = mesh_path
        else:
            raise ValueError("mesh_path must be a string or a trimesh.Trimesh object")

        print('Wrapping UV...')
        t0 = time.time()
        if unwrap_method == 'open3d':
            from utils.uvwrap_utils import open3d_mesh_uv_wrap
            mesh = open3d_mesh_uv_wrap(mesh, resolution=texture_size)
        elif unwrap_method == 'bpy':
            from utils.uvwrap_utils import bpy_unwrap_mesh
            mesh = bpy_unwrap_mesh(mesh)
        elif unwrap_method == 'xatlas':
            from utils.uvwrap_utils import mesh_uv_wrap
            mesh = mesh_uv_wrap(mesh, resolution=texture_size)
        elif unwrap_method == 'sf':
            from utils.uvwrap_utils import sf_mesh_uv_wrap
            mesh = sf_mesh_uv_wrap(mesh)
        else:
            raise ValueError(f"Invalid unwrap method {unwrap_method}")
        t1 = time.time()
        print(f"UV wrapping took {t1 - t0:.2f} seconds")

        self.render.load_mesh(mesh=mesh)

        ########### View Selection #########
        selected_camera_elevs, selected_camera_azims, selected_view_weights = self.view_processor.bake_view_selection(
            self.config.candidate_camera_elevs,
            self.config.candidate_camera_azims,
            self.config.candidate_view_weights,
            num_views,
        )

        normal_maps = self.view_processor.render_normal_multiview(
            selected_camera_elevs, selected_camera_azims, use_abs_coor=True
        )
        position_maps = self.view_processor.render_position_multiview(selected_camera_elevs, selected_camera_azims)

        t0 = time.time()
        if self.config.multiview_pretrained_path == "tencent/Hunyuan3D-2.1":
            if image_prompt is None:
                raise ValueError("Image prompt is required for Hunyuan3D-2.1 model.")

            ##########  Style  ###########
            image_caption = "high quality"
            image_style = []
            for image in image_prompt:
                image = image.resize((512, 512))
                if image.mode == "RGBA":
                    white_bg = Image.new("RGB", image.size, (255, 255, 255))
                    white_bg.paste(image, mask=image.getchannel("A"))
                    image = white_bg
                image_style.append(image)
            image_style = [image.convert("RGB") for image in image_style]

            ###########  Multiview  ##########
            multiviews = self.models["multiview_model"](
                image_style,
                normal_maps + position_maps,
                prompt=image_caption,
                custom_view_size=self.config.hypaint_resolution,
                resize_input=True,
            )

        elif self.config.multiview_pretrained_path == "qwen-edit-quant":
            if not image_prompt:
                raise ValueError("Image prompt is required for qwen-edit-quant pipeline.")

            qwen_wrapper = self.models["multiview_model"]
            LOGGER.info("Moving Qwen pipeline to GPU for stylisation")
            qwen_wrapper = qwen_wrapper.to(self.config.device)
            self.models["multiview_model"] = qwen_wrapper
            control_inputs = position_maps if self.config.qwen_edit_use_control else None
            primary_views = qwen_wrapper(
                image_prompt,
                prompt,
                selected_camera_elevs,
                selected_camera_azims,
                num_views=num_views,
                seed=seed,
                negative_prompt=self.config.qwen_edit_negative_prompt,
                control_images=control_inputs,
            )
            LOGGER.info("Moving Qwen pipeline back to CPU")
            qwen_wrapper = qwen_wrapper.to("cpu")
            self.models["multiview_model"] = qwen_wrapper
            aggressive_memory_cleanup()

            if len(primary_views) < num_views:
                remaining = num_views - len(primary_views)
                LOGGER.info("Generating %d additional fallback view(s) via MVAdapter", remaining)
                extra_views = self._generate_additional_views_with_mvadapter(
                    mesh,
                    image_prompt,
                    normal_maps[len(primary_views):],
                    position_maps[len(primary_views):],
                    selected_camera_elevs[len(primary_views):],
                    selected_camera_azims[len(primary_views):],
                    seed,
                )
                primary_views.extend(extra_views)
                while len(primary_views) < num_views and primary_views:
                    primary_views.append(primary_views[-1])

            multiviews = {"albedo": primary_views[:num_views]}

            t1 = time.time()
            print(f"Qwen multiview generation took {t1 - t0:.2f} seconds")

            if pbr:
                multiviews = self._run_pbr_generation(multiviews, normal_maps, position_maps)

        elif self.config.multiview_pretrained_path in ["mv-adapter", "mv-adapter-t2mv"]:
            if self.config.multiview_pretrained_path == "mv-adapter":
                multiviews = self.models["multiview_model"](
                    mesh,
                    image_prompt[0],
                    normal_maps=normal_maps,
                    position_maps=position_maps,
                    camera_elevation_deg=selected_camera_elevs,
                    camera_azimuth_deg=selected_camera_azims,
                    num_views=num_views,
                    seed=seed,
                    use_mesh_renderer=False,
                )
            else:  # mv-adapter-t2mv
                multiviews = self.models["multiview_model"](
                    mesh,
                    normal_maps=normal_maps,
                    position_maps=position_maps,
                    prompt=prompt,
                    camera_elevation_deg=selected_camera_elevs,
                    camera_azimuth_deg=selected_camera_azims,
                    num_views=num_views,
                    seed=seed,
                    use_mesh_renderer=False,
                )

            t1 = time.time()
            print(f"Multiview generation took {t1 - t0:.2f} seconds")

            if pbr:
                multiviews = self._run_pbr_generation(multiviews, normal_maps, position_maps)
        else:
            raise ValueError("Unsupported multiview model path: {}".format(self.config.multiview_pretrained_path))

        t0 = time.time()
        ###########  Enhance  ##########
        enhance_images = {}
        enhance_images["albedo"] = copy.deepcopy(multiviews["albedo"])
        if pbr:
            enhance_images["mr"] = copy.deepcopy(multiviews["mr"])

        if upscale_model == 'Aura':
            from hy3dpaint.upscalers.pipelines import AuraSRUpscalerPipeline
            upscaler = AuraSRUpscalerPipeline.from_pretrained()
        elif upscale_model == 'NMKD':
            from hy3dpaint.upscalers.pipelines import NMKDSiaxUpscalerPipeline
            upscaler = NMKDSiaxUpscalerPipeline.from_pretrained(self.config.device)
        elif upscale_model == 'Flux':
            from hy3dpaint.upscalers.pipelines import FluxUpscalerPipeline
            upscaler = FluxUpscalerPipeline.from_pretrained(self.config.device)
        elif upscale_model == 'Topaz':
            from hy3dpaint.upscalers.pipelines import TopazAPIUpscalerPipeline
            upscaler = TopazAPIUpscalerPipeline()
        else:
            upscaler = None

        if upscaler is not None:
            resized_images = []
            for i, img in enumerate(enhance_images["albedo"]):
                if i < 6:
                    resized_images.append(upscaler(img))
                else:
                    multiplier = int(texture_size // 1024)
                    new_size = (img.width * multiplier, img.height * multiplier)
                    resized_images.append(img.resize(new_size, resample=Image.LANCZOS))
            enhance_images["albedo"] = resized_images

            # Process mr (if available)
            if pbr and "mr" in enhance_images:
                resized_mr = []
                for i, img in enumerate(enhance_images["mr"]):
                    multiplier = int(texture_size // 1024)
                    new_size = (img.width * multiplier, img.height * multiplier)
                    resized_mr.append(img.resize(new_size, resample=Image.LANCZOS))
                enhance_images["mr"] = resized_mr

        t1 = time.time()
        print(f"Upscaling took {t1 - t0:.2f} seconds")

        t0 = time.time()
        ###########  Bake  ##########
        texture_mr, mask_mr_np = None, None
        for i in range(len(enhance_images)):
            enhance_images["albedo"][i] = enhance_images["albedo"][i].resize(
                (self.config.render_size, self.config.render_size)
            )
            if pbr:
                enhance_images["mr"][i] = enhance_images["mr"][i].resize(
                    (self.config.render_size, self.config.render_size))
        texture, mask = self.view_processor.bake_from_multiview(
            enhance_images["albedo"], selected_camera_elevs, selected_camera_azims, selected_view_weights
        )
        mask_np = (mask.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)
        if pbr:
            texture_mr, mask_mr = self.view_processor.bake_from_multiview(
                enhance_images["mr"], selected_camera_elevs, selected_camera_azims, selected_view_weights
            )
            mask_mr_np = (mask_mr.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)
        t1 = time.time()
        print(f"Baking textures took {t1 - t0:.2f} seconds")

        t0 = time.time()
        ##########  inpaint  ###########
        texture = self.view_processor.texture_inpaint(texture, mask_np)
        self.render.set_texture(texture, force_set=True)
        if "mr" in enhance_images:
            texture_mr = self.view_processor.texture_inpaint(texture_mr, mask_mr_np)
            self.render.set_texture_mr(texture_mr)

        mesh = self.render.get_trimesh()
        t1 = time.time()
        print('Inpainting and saving mesh took {:.2f} seconds'.format(t1 - t0))

        return mesh
