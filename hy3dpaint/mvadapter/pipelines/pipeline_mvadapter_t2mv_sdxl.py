# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.models import AutoencoderKL, T2IAdapter, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import (
    StableDiffusionXLPipelineOutput,
)
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    StableDiffusionXLPipeline,
    rescale_noise_cfg,
    retrieve_timesteps,
)
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import deprecate, logging
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

from ..loaders import CustomAdapterMixin
from ..models.attention_processor import (
    DecoupledMVRowSelfAttnProcessor2_0,
    set_unet_2d_condition_attn_processor,
)

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class MVAdapterT2MVSDXLPipeline(StableDiffusionXLPipeline, CustomAdapterMixin):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        image_encoder: CLIPVisionModelWithProjection = None,
        feature_extractor: CLIPImageProcessor = None,
        force_zeros_for_empty_prompt: bool = True,
        add_watermarker: Optional[bool] = None,
    ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=unet,
            scheduler=scheduler,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
            force_zeros_for_empty_prompt=force_zeros_for_empty_prompt,
            add_watermarker=add_watermarker,
        )

        self.control_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor,
            do_convert_rgb=True,
            do_normalize=False,
        )

    def prepare_control_image(
        self,
        image,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance=False,
    ):
        assert hasattr(
            self, "control_image_processor"
        ), "control_image_processor is not initialized"

        image = self.control_image_processor.preprocess(
            image, height=height, width=width
        ).to(dtype=torch.float32)
        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt  # always 1 for control image

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        if do_classifier_free_guidance:
            image = torch.cat([image] * 2)

        return image

    def combine_metallic_roughness(self, metallic_maps, roughness_maps, output_type="pil"):
        """Combine metallic and roughness into a single MR map.
        G channel = roughness, B channel = metallic, R channel = 0
        """
        if output_type == "pil":
            # Convert PIL images to tensors for processing
            metallic_tensors = []
            roughness_tensors = []

            for m, r in zip(metallic_maps, roughness_maps):
                # Convert PIL to numpy then to tensor
                m_array = np.array(m).astype(np.float32) / 255.0
                r_array = np.array(r).astype(np.float32) / 255.0

                metallic_tensors.append(torch.from_numpy(m_array).permute(2, 0, 1))
                roughness_tensors.append(torch.from_numpy(r_array).permute(2, 0, 1))

            metallic_tensor = torch.stack(metallic_tensors)
            roughness_tensor = torch.stack(roughness_tensors)
        else:
            metallic_tensor = metallic_maps
            roughness_tensor = roughness_maps

        # Create MR map: R=0, G=roughness, B=metallic
        batch_size = metallic_tensor.shape[0]
        height = metallic_tensor.shape[2]
        width = metallic_tensor.shape[3]

        mr_tensor = torch.zeros(batch_size, 3, height, width,
                                dtype=metallic_tensor.dtype,
                                device=metallic_tensor.device)

        # G channel = roughness (take first channel from roughness maps)
        mr_tensor[:, 1, :, :] = roughness_tensor[:, 0, :, :]

        # B channel = metallic (take first channel from metallic maps)
        mr_tensor[:, 2, :, :] = metallic_tensor[:, 0, :, :]

        # Convert back to PIL if needed
        if output_type == "pil":
            mr_images = []
            for i in range(batch_size):
                # Convert to numpy and then PIL
                mr_array = mr_tensor[i].permute(1, 2, 0).cpu().numpy()
                mr_array = (mr_array * 255).astype(np.uint8)
                mr_images.append(Image.fromarray(mr_array))
            return mr_images
        else:
            return mr_tensor

    @torch.no_grad()
    def __call__(
            self,
            prompt: Union[str, List[str]] = None,
            prompt_2: Optional[Union[str, List[str]]] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            timesteps: List[int] = None,
            denoising_end: Optional[float] = None,
            guidance_scale: float = 5.0,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            negative_prompt_2: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            ip_adapter_image: Optional[PipelineImageInput] = None,
            ip_adapter_image_embeds: Optional[List[torch.FloatTensor]] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            guidance_rescale: float = 0.0,
            original_size: Optional[Tuple[int, int]] = None,
            crops_coords_top_left: Tuple[int, int] = (0, 0),
            target_size: Optional[Tuple[int, int]] = None,
            negative_original_size: Optional[Tuple[int, int]] = None,
            negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
            negative_target_size: Optional[Tuple[int, int]] = None,
            clip_skip: Optional[int] = None,
            callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            mv_scale: float = 1.0,
            control_image: Optional[PipelineImageInput] = None,
            control_conditioning_scale: Optional[float] = 1.0,
            control_conditioning_factor: float = 1.0,
            controlnet_image: Optional[PipelineImageInput] = None,
            controlnet_conditioning_scale: Optional[float] = 1.0,
            # PBR generation parameters
            output_pbr_maps: bool = True,
            pbr_prompts: Optional[Dict[str, str]] = None,
            pbr_strength: float = 0.8,  # Control how much PBR affects the latents
            **kwargs,
    ):
        """
        Returns a dictionary with 'albedo' and 'mr' keys when output_pbr_maps=True
        """

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )

        # 0. Default height and width to unet
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._denoising_end = denoising_end
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None)
            if self.cross_attention_kwargs is not None
            else None
        )

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps
        )

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
        else:
            negative_add_time_ids = add_time_ids

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat(
                [negative_pooled_prompt_embeds, add_text_embeds], dim=0
            )
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(
            batch_size * num_images_per_prompt, 1
        )

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

        # Preprocess control image
        control_image_feature = self.prepare_control_image(
            image=control_image,
            width=width,
            height=height,
            batch_size=batch_size * num_images_per_prompt,
            num_images_per_prompt=1,  # NOTE: always 1 for control images
            device=device,
            dtype=latents.dtype,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
        )
        control_image_feature = control_image_feature.to(
            device=device, dtype=latents.dtype
        )

        adapter_state = self.cond_encoder(control_image_feature)
        for i, state in enumerate(adapter_state):
            adapter_state[i] = state * control_conditioning_scale

        # Preprocess controlnet image if provided
        do_controlnet = controlnet_image is not None and hasattr(self, "controlnet")
        if do_controlnet:
            controlnet_image = self.prepare_control_image(
                image=controlnet_image,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=1,  # NOTE: always 1 for control images
                device=device,
                dtype=latents.dtype,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
            )
            controlnet_image = controlnet_image.to(device=device, dtype=latents.dtype)

        # 8. Denoising loop
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )

        # 8.1 Apply denoising_end
        if (
            self.denoising_end is not None
            and isinstance(self.denoising_end, float)
            and self.denoising_end > 0
            and self.denoising_end < 1
        ):
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (self.denoising_end * self.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(
                list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps))
            )
            timesteps = timesteps[:num_inference_steps]

        # 9. Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(
                batch_size * num_images_per_prompt
            )
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2)
                    if self.do_classifier_free_guidance
                    else latents
                )

                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                added_cond_kwargs = {
                    "text_embeds": add_text_embeds,
                    "time_ids": add_time_ids,
                }
                if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
                    added_cond_kwargs["image_embeds"] = image_embeds

                if i < int(num_inference_steps * control_conditioning_factor):
                    down_intrablock_additional_residuals = [
                        state.clone() for state in adapter_state
                    ]
                else:
                    down_intrablock_additional_residuals = None

                unet_add_kwargs = {}

                # Do controlnet if provided
                if do_controlnet:
                    down_block_res_samples, mid_block_res_sample = self.controlnet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        controlnet_cond=controlnet_image,
                        conditioning_scale=controlnet_conditioning_scale,
                        guess_mode=False,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )
                    unet_add_kwargs.update(
                        {
                            "down_block_additional_residuals": down_block_res_samples,
                            "mid_block_additional_residual": mid_block_res_sample,
                        }
                    )

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs={
                        "mv_scale": mv_scale,
                        **(self.cross_attention_kwargs or {}),
                    },
                    down_intrablock_additional_residuals=down_intrablock_additional_residuals,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                    **unet_add_kwargs,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(
                        noise_pred,
                        noise_pred_text,
                        guidance_rescale=self.guidance_rescale,
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]
                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop(
                        "negative_prompt_embeds", negative_prompt_embeds
                    )
                    add_text_embeds = callback_outputs.pop(
                        "add_text_embeds", add_text_embeds
                    )
                    negative_pooled_prompt_embeds = callback_outputs.pop(
                        "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                    )
                    add_time_ids = callback_outputs.pop("add_time_ids", add_time_ids)
                    negative_add_time_ids = callback_outputs.pop(
                        "negative_add_time_ids", negative_add_time_ids
                    )

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        # Store final latents for PBR generation
        final_latents = latents.clone()

        # Decode main images
        if not output_type == "latent":
            # make sure the VAE is in float32 mode, as it overflows in float16
            needs_upcasting = (
                self.vae.dtype == torch.float16 and self.vae.config.force_upcast
            )

            if needs_upcasting:
                self.upcast_vae()
                latents = latents.to(
                    next(iter(self.vae.post_quant_conv.parameters())).dtype
                )
            elif latents.dtype != self.vae.dtype:
                if torch.backends.mps.is_available():
                    # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                    self.vae = self.vae.to(latents.dtype)

            # unscale/denormalize the latents
            # denormalize with the mean and std if available and not None
            has_latents_mean = (
                hasattr(self.vae.config, "latents_mean")
                and self.vae.config.latents_mean is not None
            )
            has_latents_std = (
                hasattr(self.vae.config, "latents_std")
                and self.vae.config.latents_std is not None
            )
            if has_latents_mean and has_latents_std:
                latents_mean = (
                    torch.tensor(self.vae.config.latents_mean)
                    .view(1, 4, 1, 1)
                    .to(latents.device, latents.dtype)
                )
                latents_std = (
                    torch.tensor(self.vae.config.latents_std)
                    .view(1, 4, 1, 1)
                    .to(latents.device, latents.dtype)
                )
                latents = (
                    latents * latents_std / self.vae.config.scaling_factor
                    + latents_mean
                )
            else:
                latents = latents / self.vae.config.scaling_factor

            image = self.vae.decode(latents, return_dict=False)[0]

            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
        else:
            image = latents

        # Post-process main images
        image = self.image_processor.postprocess(image, output_type=output_type)

        # Generate PBR maps if requested
        if output_pbr_maps:
            # Default PBR prompts if not provided
            if pbr_prompts is None:
                pbr_prompts = {
                    'metallic': 'metallic surface, chrome material, steel, reflective metal, shiny, mirror finish',
                    'roughness': 'rough texture, matte surface, bumpy, coarse material, grainy, textured'
                }

            # Generate metallic map using modified prompts
            metallic_prompt = f"{prompt}, {pbr_prompts.get('metallic', 'metallic chrome steel reflective shiny metal')}"
            metallic_prompt_embeds, _, metallic_pooled_embeds, _ = self.encode_prompt(
                prompt=metallic_prompt,
                prompt_2=metallic_prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=False,
            )

            # Use the final latents as starting point
            metallic_latents = final_latents.clone()

            # Add noise and denoise with metallic-specific prompts
            noise = torch.randn_like(metallic_latents)
            metallic_timesteps = timesteps[-8:-3] if len(timesteps) > 8 else timesteps[-5:]

            # Mix original latents with noise
            init_timestep = metallic_timesteps[0]
            metallic_latents = self.scheduler.add_noise(
                metallic_latents,
                noise * pbr_strength,
                init_timestep
            )

            for t in metallic_timesteps:
                metallic_noise_pred = self.unet(
                    metallic_latents,
                    t,
                    encoder_hidden_states=metallic_prompt_embeds,
                    added_cond_kwargs={
                        "text_embeds": metallic_pooled_embeds,
                        "time_ids": add_time_ids[:metallic_latents.shape[0]],
                    },
                    cross_attention_kwargs={
                        "mv_scale": mv_scale,
                        **(self.cross_attention_kwargs or {}),
                    },
                    return_dict=False,
                )[0]

                metallic_latents = self.scheduler.step(
                    metallic_noise_pred, t, metallic_latents, **extra_step_kwargs, return_dict=False
                )[0]

            # Decode metallic
            if has_latents_mean and has_latents_std:
                metallic_latents = (
                        metallic_latents * latents_std / self.vae.config.scaling_factor
                        + latents_mean
                )
            else:
                metallic_latents = metallic_latents / self.vae.config.scaling_factor

            metallic_maps = self.vae.decode(metallic_latents, return_dict=False)[0]

            # Convert to grayscale based on perceived metallic properties
            # Emphasize blue and green channels (often metallic in diffusion models)
            metallic_maps = 0.1 * metallic_maps[:, 0:1] + \
                            0.3 * metallic_maps[:, 1:2] + \
                            0.6 * metallic_maps[:, 2:3]
            metallic_maps = torch.sigmoid((metallic_maps - 0.5) * 3)

            # Generate roughness map
            roughness_prompt = f"{prompt}, {pbr_prompts.get('roughness', 'rough matte texture bumpy coarse surface')}"
            roughness_prompt_embeds, _, roughness_pooled_embeds, _ = self.encode_prompt(
                prompt=roughness_prompt,
                prompt_2=roughness_prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=False,
            )

            roughness_latents = final_latents.clone()
            roughness_latents = self.scheduler.add_noise(
                roughness_latents,
                noise * pbr_strength,
                init_timestep
            )

            for t in metallic_timesteps:
                roughness_noise_pred = self.unet(
                    roughness_latents,
                    t,
                    encoder_hidden_states=roughness_prompt_embeds,
                    added_cond_kwargs={
                        "text_embeds": roughness_pooled_embeds,
                        "time_ids": add_time_ids[:roughness_latents.shape[0]],
                    },
                    cross_attention_kwargs={
                        "mv_scale": mv_scale,
                        **(self.cross_attention_kwargs or {}),
                    },
                    return_dict=False,
                )[0]

                roughness_latents = self.scheduler.step(
                    roughness_noise_pred, t, roughness_latents, **extra_step_kwargs, return_dict=False
                )[0]

            # Decode roughness
            if has_latents_mean and has_latents_std:
                roughness_latents = (
                        roughness_latents * latents_std / self.vae.config.scaling_factor
                        + latents_mean
                )
            else:
                roughness_latents = roughness_latents / self.vae.config.scaling_factor

            roughness_maps = self.vae.decode(roughness_latents, return_dict=False)[0]

            # Convert to grayscale based on texture variance
            # Use standard deviation across channels as roughness indicator
            roughness_std = torch.std(roughness_maps, dim=1, keepdim=True)
            roughness_maps = 0.299 * roughness_maps[:, 0:1] + \
                             0.587 * roughness_maps[:, 1:2] + \
                             0.114 * roughness_maps[:, 2:3]
            # Combine luminance with variance for better roughness estimation
            roughness_maps = 0.7 * roughness_maps + 0.3 * roughness_std
            roughness_maps = torch.sigmoid((roughness_maps - 0.5) * 3)

            # Convert to RGB format for compatibility
            metallic_maps = metallic_maps.repeat(1, 3, 1, 1)
            roughness_maps = roughness_maps.repeat(1, 3, 1, 1)

            # Post-process individual maps
            metallic_maps = self.image_processor.postprocess(metallic_maps, output_type=output_type)
            roughness_maps = self.image_processor.postprocess(roughness_maps, output_type=output_type)

            # Combine into MR maps
            mr_maps = self.combine_metallic_roughness(metallic_maps, roughness_maps, output_type)

            # Offload all models
            self.maybe_free_model_hooks()

            # Return dictionary format
            return {
                'albedo': image,  # List of 6 images
                'mr': mr_maps  # List of 6 MR images (G=roughness, B=metallic)
            }

        # Offload all models
        self.maybe_free_model_hooks()

        # Original return format when not generating PBR
        if not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image)

    ### NEW: adapters ###
    def _init_custom_adapter(
        self,
        # Multi-view adapter
        num_views: int = 1,
        self_attn_processor: Any = DecoupledMVRowSelfAttnProcessor2_0,
        # Condition encoder
        cond_in_channels: int = 6,
        # For training
        copy_attn_weights: bool = True,
        zero_init_module_keys: List[str] = [],
    ):
        # Condition encoder
        self.cond_encoder = T2IAdapter(
            in_channels=cond_in_channels,
            channels=(320, 640, 1280, 1280),
            num_res_blocks=2,
            downscale_factor=16,
            adapter_type="full_adapter_xl",
        )

        # set custom attn processor for multi-view attention
        self.unet: UNet2DConditionModel
        set_unet_2d_condition_attn_processor(
            self.unet,
            set_self_attn_proc_func=lambda name, hs, cad, ap: self_attn_processor(
                query_dim=hs,
                inner_dim=hs,
                num_views=num_views,
                name=name,
                use_mv=True,
                use_ref=False,
            ),
            set_cross_attn_proc_func=lambda name, hs, cad, ap: self_attn_processor(
                query_dim=hs,
                inner_dim=hs,
                num_views=num_views,
                name=name,
                use_mv=False,
                use_ref=False,
            ),
        )

        # copy decoupled attention weights from original unet
        if copy_attn_weights:
            state_dict = self.unet.state_dict()
            for key in state_dict.keys():
                if "_mv" in key:
                    compatible_key = key.replace("_mv", "").replace("processor.", "")
                else:
                    compatible_key = key

                is_zero_init_key = any([k in key for k in zero_init_module_keys])
                if is_zero_init_key:
                    state_dict[key] = torch.zeros_like(state_dict[compatible_key])
                else:
                    state_dict[key] = state_dict[compatible_key].clone()
            self.unet.load_state_dict(state_dict)

    def _load_custom_adapter(self, state_dict):
        self.unet.load_state_dict(state_dict, strict=False)
        self.cond_encoder.load_state_dict(state_dict, strict=False)

    def _save_custom_adapter(
        self,
        include_keys: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
    ):
        def include_fn(k):
            is_included = False

            if include_keys is not None:
                is_included = is_included or any([key in k for key in include_keys])
            if exclude_keys is not None:
                is_included = is_included and not any(
                    [key in k for key in exclude_keys]
                )

            return is_included

        state_dict = {k: v for k, v in self.unet.state_dict().items() if include_fn(k)}
        state_dict.update(self.cond_encoder.state_dict())

        return state_dict
