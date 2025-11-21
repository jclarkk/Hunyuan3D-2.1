import shutil
import time

import os
import requests
import torch
from PIL import Image
from io import BytesIO
from typing import Union


class FluxUpscalerPipeline:
    """
        Highest quality but slow
    """

    @classmethod
    def from_pretrained(cls, device):
        from diffusers import FluxControlNetModel
        from diffusers.pipelines import FluxControlNetPipeline
        # Load pipeline
        controlnet = FluxControlNetModel.from_pretrained(
            "jasperai/Flux.1-dev-Controlnet-Upscaler",
            torch_dtype=torch.bfloat16
        )
        pipe = FluxControlNetPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            controlnet=controlnet,
            torch_dtype=torch.bfloat16
        )
        pipe.enable_model_cpu_offload()

        return cls(pipe, device)

    def __init__(self, pipe, device):
        self.pipe = pipe
        self.device = device

    def __call__(self, input_image: Image.Image) -> Image.Image:
        w, h = input_image.size
        input_image = input_image.resize((w * 4, h * 4))

        return self.pipe(
            prompt="",
            control_image=input_image,
            controlnet_conditioning_scale=0.8,
            num_inference_steps=28,
            guidance_scale=3.5,
            height=input_image.size[1],
            width=input_image.size[0]
        ).images[0]


class NMKDSiaxUpscalerPipeline:
    """
        High quality upscaling with good performance using NMKD
    """

    @classmethod
    def from_pretrained(cls, device, texture_size: int = 4096):
        from spandrel import ModelLoader
        # Initialize Real-ESRGAN with 4x upscaling model
        model = ModelLoader().load_from_file('./weights/4x_NMKD-Siax_200k.pth')
        model.to(device).eval().half()
        return cls(model, device, texture_size)

    def __init__(self, model, device, texture_size):
        self.model = model
        self.device = device
        self.texture_size = texture_size

    def upscale_once(self, image: Image.Image) -> Image.Image:
        from torchvision import transforms
        to_pil = transforms.ToPILImage()
        to_tensor = transforms.ToTensor()

        input_tensor = to_tensor(image).unsqueeze(0).to(self.device).half()
        with torch.no_grad():
            output = self.model(input_tensor).float().clamp(0, 1).squeeze(0).cpu()
        return to_pil(output)

    def __call__(self, input_image: Image.Image) -> Image.Image:
        # --- Single or double pass upscaling ---
        if self.texture_size in (6144, 8192):
            # 1st upscale → 4096
            img = self.upscale_once(input_image)
            # 2nd upscale → ~16384, then resize down
            img = self.upscale_once(img)
        else:
            img = self.upscale_once(input_image)

        # --- Final resize to exact target size ---
        if img.size[0] != self.texture_size:
            img = img.resize((self.texture_size, self.texture_size), Image.LANCZOS)

        return img


class AuraSRUpscalerPipeline:
    """
        Medium quality but fast
    """

    @classmethod
    def from_pretrained(cls):
        from aura_sr import AuraSR
        return cls(AuraSR.from_pretrained("fal/AuraSR-v2"))

    def __init__(self, pipe):
        self.pipe = pipe

    def __call__(self, input_image: Image.Image) -> Image.Image:
        return self.pipe.upscale_4x_overlapped(input_image, max_batch_size=16)


class TopazAPIUpscalerPipeline:
    """
    High quality upscaling using Topaz synchronous API.
    """

    def __init__(self, texture_size: int = 4096):
        self.topaz_api_key = os.getenv('TOPAZ_API_KEY')
        self.topaz_url = 'https://api.topazlabs.com/image/v1/enhance'
        self.output_height = texture_size
        self.output_width = texture_size
        self.model = 'Standard V2'
        self.output_format = 'png'
        self.max_retries = 5
        self.backoff_base = 2

    def __call__(self, input_image: Image.Image) -> Image.Image:
        image_bytes = BytesIO()
        input_image.save(image_bytes, format='PNG')
        image_bytes.seek(0)

        headers = {
            'X-API-Key': self.topaz_api_key,
            'accept': f'image/{self.output_format}',
        }

        files = {
            'image': ('input.png', image_bytes, 'image/png')
        }

        data = {
            'model': self.model,
            'output_height': self.output_height,
            'output_width': self.output_width,
            'output_format': self.output_format
        }

        for attempt in range(self.max_retries):
            response = requests.post(self.topaz_url, headers=headers, files=files, data=data)

            if response.status_code == 200:
                return Image.open(BytesIO(response.content))

            elif response.status_code in [429, 500, 502, 503, 504]:
                sleep_time = self.backoff_base ** attempt
                time.sleep(sleep_time)
                continue

            else:
                response.raise_for_status()

        raise Exception("Topaz sync upscaling failed after retries.")


class GeminiAPIPipeline:
    def __init__(self, original_input_image: Union[str, Image.Image]):
        self.original_input_image = original_input_image
        if type(self.original_input_image) is str:
            self.original_input_image = Image.open(self.original_input_image)

        genai_key = os.getenv('GOOGLE_GENAI_KEY')
        if not genai_key:
            raise ValueError("GOOGLE_GENAI_KEY environment variable not set")

        from google import genai
        self.client = genai.Client(api_key=genai_key)

    def __call__(self, input_image: Image.Image, resolution=1024) -> Image.Image:

        google_resolution = "1K"
        if resolution == 2048:
            google_resolution = "2K"
        elif resolution in [4096, 6144, 8192]:
            google_resolution = "4K"

        contents = [
            f"Upscale the image quality while maintaining pixel perfect object position from the first image. "
            f"The aspect ratio must remain the same. The object in the first image must not be distorted in any way. ",
            f"The object position, translation, zoom and shape in the first image must be preserved exactly as it is. ",
            f"The second provided image is the ground truth reference for the object details. "
            f"You must preserve the original colors and details of the second image exactly as they are. "
            f"If the second image has text then please use it since the first image may have messed it up.",
            input_image,
            self.original_input_image]

        from google.genai import types

        response = self.client.models.generate_content(
            model="gemini-3-pro-image-preview",
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=['IMAGE'],
                image_config=types.ImageConfig(
                    image_size=google_resolution
                ),
            )
        )

        output_image = None
        for part in response.parts:
            if image := part.as_image():
                os.makedirs("tmp/", exist_ok=True)
                image.save("tmp/gemini_upscaled.png")
                output_image = Image.open("tmp/gemini_upscaled.png")
                shutil.rmtree("tmp/", ignore_errors=True)

        if output_image is None:
            raise RuntimeError("Failed to upscale image; no image returned from Gemini API")

        if resolution in [6144, 8192]:
            output_image = output_image.resize((resolution, resolution), Image.LANCZOS)

        return output_image
