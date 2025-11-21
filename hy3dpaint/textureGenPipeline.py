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
import torch
import copy
import time
import trimesh
import numpy as np
from PIL import Image
from typing import List
from DifferentiableRenderer.MeshRender import MeshRender
from hy3dpaint.mvadapter.pipeline import MVAdapterPipelineWrapper
from hy3dpaint.mvadapter.pipelines.pipeline_mvadapter_i2mv_sdxl import MVAdapterI2MVSDXLPipeline
from hy3dpaint.mvadapter.pipelines.pipeline_mvadapter_t2mv_sdxl import MVAdapterT2MVSDXLPipeline
from utils.multiview_utils import multiviewDiffusionNet, MetalRoughnessOnlyNet
from utils.pipeline_utils import ViewProcessor
import warnings

warnings.filterwarnings("ignore")
from diffusers.utils import logging as diffusers_logging

diffusers_logging.set_verbosity(50)

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


class Hunyuan3DPaintConfig:
    def __init__(self, hypaint_resolution, local_files_only=False, optimization_level="balanced") -> None:
        self.device = "cuda"
        self.local_files_only = local_files_only

        self.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
        self.custom_pipeline = "hunyuanpaintpbr"
        self.multiview_pretrained_path = "tencent/Hunyuan3D-2.1"
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

        elif self.config.multiview_pretrained_path in ["mv-adapter", "mv-adapter-t2mv"]:
            ############  Multiview  ##########
            if self.config.multiview_pretrained_path == "mv-adapter":
                multiviews = self.models['multiview_model'](
                    mesh,
                    image_prompt[0],
                    normal_maps=normal_maps,
                    position_maps=position_maps,
                    camera_elevation_deg=selected_camera_elevs,
                    camera_azimuth_deg=selected_camera_azims,
                    num_views=num_views,
                    seed=seed,
                    use_mesh_renderer=False
                )
            else:  # mv-adapter-t2mv
                multiviews = self.models['multiview_model'](
                    mesh,
                    normal_maps=normal_maps,
                    position_maps=position_maps,
                    prompt=prompt,
                    camera_elevation_deg=selected_camera_elevs,
                    camera_azimuth_deg=selected_camera_azims,
                    num_views=num_views,
                    seed=seed,
                    use_mesh_renderer=False
                )

            t1 = time.time()
            print(f"Multiview generation took {t1 - t0:.2f} seconds")

            if pbr:
                print("Preparing for PBR generation...")

                # Store albedo views in CPU memory to free GPU
                albedo_views_cpu = [img.copy() for img in multiviews['albedo']]

                # Aggressive cleanup before loading PBR model
                if self.config.continuous_inference:
                    print("Moving multiview model to CPU...")
                    move_model_to_cpu(self.models['multiview_model'])
                else:
                    print("Deleting multiview model...")
                    delete_model_and_cleanup(self.models, 'multiview_model')

                # Additional cleanup
                del multiviews
                aggressive_memory_cleanup()

                # Wait a moment for memory to be fully released
                time.sleep(1)

                # Print memory status
                print(f"GPU memory before PBR: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB allocated")

                t2 = time.time()

                # Load PBR model with optimizations
                print("Loading PBR model...")
                self.models['pbr_model'] = MetalRoughnessOnlyNet(self.config)

                mr_views = None
                # Process in smaller batches if needed
                if len(albedo_views_cpu) > 6 and self.config.optimization_level == "aggressive":
                    # Process PBR in batches to save memory
                    mr_views_list = []
                    for i in range(0, len(albedo_views_cpu), 6):
                        batch_end = min(i + 6, len(albedo_views_cpu))
                        batch_albedo = albedo_views_cpu[i:batch_end]
                        batch_normal = normal_maps[i:batch_end]
                        batch_position = position_maps[i:batch_end]

                        if len(batch_albedo) < 6:
                            # Pad to 6
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

                        mr_views_list.extend(batch_mr["mr"][:batch_end - i])
                        aggressive_memory_cleanup()

                    mr_views = {"mr": mr_views_list[:len(albedo_views_cpu)]}
                else:
                    with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                        # Process all at once
                        mr_views = self.models["pbr_model"](
                            albedo_views_cpu[0],
                            normal_maps + position_maps,
                            prompt="material roughness and metallic map",
                            custom_view_size=512,
                            resize_input=True,
                        )

                # Resize the MR views to match the albedo views resolution
                for i in range(len(mr_views["mr"])):
                    mr_views["mr"][i] = mr_views["mr"][i].resize(
                        (self.config.hypaint_resolution, self.config.hypaint_resolution))

                # Recreate multiviews dict with CPU albedo and new MR
                multiviews = {
                    "albedo": albedo_views_cpu,
                    "mr": mr_views["mr"]
                }

                # Clean up PBR model
                delete_model_and_cleanup(self.models, 'pbr_model')

                # Restore multiview model if continuous inference
                if self.config.continuous_inference:
                    print("Restoring multiview model to GPU...")
                    self.models['multiview_model'].to(self.config.device)

                t3 = time.time()
                print(f"PBR generation took {t3 - t2:.2f} seconds")

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
            upscaler = NMKDSiaxUpscalerPipeline.from_pretrained(self.config.device, texture_size=texture_size)
        elif upscale_model == 'Flux':
            from hy3dpaint.upscalers.pipelines import FluxUpscalerPipeline
            upscaler = FluxUpscalerPipeline.from_pretrained(self.config.device)
        elif upscale_model == 'Topaz':
            from hy3dpaint.upscalers.pipelines import TopazAPIUpscalerPipeline
            upscaler = TopazAPIUpscalerPipeline(texture_size=texture_size)
        elif upscale_model == 'Gemini':
            from hy3dpaint.upscalers.pipelines import GeminiAPIPipeline
            upscaler = GeminiAPIPipeline()
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
