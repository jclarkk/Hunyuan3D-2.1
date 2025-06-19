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

import os
import torch
import copy
import trimesh
import numpy as np
from PIL import Image
from typing import List
from DifferentiableRenderer.MeshRender import MeshRender
from hy3dpaint.mvadapter.pipeline import MVAdapterPipelineWrapper
from hy3dpaint.mvadapter.pipelines.pipeline_mvadapter_i2mv_sdxl import MVAdapterI2MVSDXLPipeline
from hy3dpaint.mvadapter.pipelines.pipeline_mvadapter_t2mv_sdxl import MVAdapterT2MVSDXLPipeline
from utils.simplify_mesh_utils import remesh_mesh
from utils.multiview_utils import multiviewDiffusionNet
from utils.pipeline_utils import ViewProcessor
from utils.uvwrap_utils import mesh_uv_wrap
import warnings

warnings.filterwarnings("ignore")
from diffusers.utils import logging as diffusers_logging

diffusers_logging.set_verbosity(50)


class Hunyuan3DPaintConfig:
    def __init__(self, max_num_view, resolution, local_files_only=False) -> None:
        self.device = "cuda"
        self.local_files_only = local_files_only

        self.multiview_cfg_path = "cfgs/hunyuan-paint-pbr.yaml"
        self.custom_pipeline = "hunyuanpaintpbr"
        self.multiview_pretrained_path = "tencent/Hunyuan3D-2.1"
        self.dino_ckpt_path = "facebook/dinov2-giant"

        self.raster_mode = "cr"
        self.bake_mode = "back_sample"
        self.render_size = 1024 * 2
        self.texture_size = 1024 * 4
        self.max_selected_view_num = max_num_view
        self.resolution = resolution
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
    def __call__(self, mesh_path=None, image_path=None, prompt=None, output_mesh_path=None, use_remesh=True, save_glb=True,
                 upscale_model='NMKD', pbr=True, texture_size=4096, seed=42):
        self.config.texture_size = texture_size
        self.render.set_default_texture_resolution(texture_size)

        """Generate texture for 3D mesh using multiview diffusion"""
        image_prompt = None
        if image_path is not None:
            # Ensure image_prompt is a list
            if isinstance(image_path, str):
                image_prompt = Image.open(image_path)
            elif isinstance(image_path, Image.Image):
                image_prompt = image_path
            if not isinstance(image_prompt, List):
                image_prompt = [image_prompt]
            else:
                image_prompt = image_path

        # Process mesh
        path = os.path.dirname(mesh_path)
        if use_remesh:
            processed_mesh_path = os.path.join(path, "white_mesh_remesh.obj")
            remesh_mesh(mesh_path, processed_mesh_path)
        else:
            processed_mesh_path = mesh_path

        # Output path
        if output_mesh_path is None:
            output_mesh_path = os.path.join(path, f"textured_mesh.obj")

        # Load mesh
        mesh = trimesh.load(processed_mesh_path)
        mesh = mesh_uv_wrap(mesh)
        self.render.load_mesh(mesh=mesh)

        ########### View Selection #########
        selected_camera_elevs, selected_camera_azims, selected_view_weights = self.view_processor.bake_view_selection(
            self.config.candidate_camera_elevs,
            self.config.candidate_camera_azims,
            self.config.candidate_view_weights,
            self.config.max_selected_view_num,
        )

        normal_maps = self.view_processor.render_normal_multiview(
            selected_camera_elevs, selected_camera_azims, use_abs_coor=True
        )
        position_maps = self.view_processor.render_position_multiview(selected_camera_elevs, selected_camera_azims)

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
                custom_view_size=self.config.resolution,
                resize_input=True,
            )
        elif self.config.multiview_pretrained_path == "mv-adapter":
            ############  Multiview  ##########
            multiviews = self.models['multiview_model'](mesh,
                                                        image_prompt[0],
                                                        normal_maps=normal_maps,
                                                        position_maps=position_maps,
                                                        camera_elevation_deg=selected_camera_elevs,
                                                        camera_azimuth_deg=selected_camera_azims,
                                                        num_views=len(selected_camera_azims),
                                                        seed=seed,
                                                        use_mesh_renderer=False)
        elif self.config.multiview_pretrained_path == "mv-adapter-t2mv":
            ############  Multiview  ##########
            multiviews = self.models['multiview_model'](mesh,
                                                        normal_maps=normal_maps,
                                                        position_maps=position_maps,
                                                        prompt=prompt,
                                                        camera_elevation_deg=selected_camera_elevs,
                                                        camera_azimuth_deg=selected_camera_azims,
                                                        num_views=len(selected_camera_azims),
                                                        seed=seed,
                                                        use_mesh_renderer=False)
        else:
            raise ValueError("Unsupported multiview model path: {}".format(self.config.multiview_pretrained_path))

        ###########  Enhance  ##########
        enhance_images = {}
        enhance_images["albedo"] = copy.deepcopy(multiviews["albedo"])
        if pbr:
            enhance_images["mr"] = copy.deepcopy(multiviews["mr"])

        if upscale_model == 'Aura':
            from .upscalers.pipelines import AuraSRUpscalerPipeline
            upscaler = AuraSRUpscalerPipeline.from_pretrained()
        elif upscale_model == 'NMKD':
            from .upscalers.pipelines import NMKDSiaxUpscalerPipeline
            upscaler = NMKDSiaxUpscalerPipeline.from_pretrained(self.config.device)
        elif upscale_model == 'Flux':
            from .upscalers.pipelines import FluxUpscalerPipeline
            upscaler = FluxUpscalerPipeline.from_pretrained(self.config.device)
        elif upscale_model == 'Topaz':
            from .upscalers.pipelines import TopazAPIUpscalerPipeline
            upscaler = TopazAPIUpscalerPipeline()
        else:
            upscaler = None

        if upscaler is not None:
            for i in range(len(enhance_images["albedo"])):
                enhance_images["albedo"][i] = upscaler(enhance_images["albedo"][i])
                if pbr:
                    enhance_images["mr"][i] = upscaler(enhance_images["mr"][i])

            del upscaler

        ###########  Bake  ##########
        texture_mr, mask_mr_np = None, None
        for i in range(len(enhance_images)):
            enhance_images["albedo"][i] = enhance_images["albedo"][i].resize(
                (self.config.render_size, self.config.render_size)
            )
            if pbr:
                enhance_images["mr"][i] = enhance_images["mr"][i].resize((self.config.render_size, self.config.render_size))
        texture, mask = self.view_processor.bake_from_multiview(
            enhance_images["albedo"], selected_camera_elevs, selected_camera_azims, selected_view_weights
        )
        mask_np = (mask.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)
        if pbr:
            texture_mr, mask_mr = self.view_processor.bake_from_multiview(
                enhance_images["mr"], selected_camera_elevs, selected_camera_azims, selected_view_weights
            )
            mask_mr_np = (mask_mr.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)

        ##########  inpaint  ###########
        texture = self.view_processor.texture_inpaint(texture, mask_np)
        self.render.set_texture(texture, force_set=True)
        if "mr" in enhance_images:
            texture_mr = self.view_processor.texture_inpaint(texture_mr, mask_mr_np)
            self.render.set_texture_mr(texture_mr)

        self.render.save_glb_mesh(output_mesh_path, downsample=True)

        return output_mesh_path
