import atexit
import time
from concurrent.futures import ThreadPoolExecutor

import os
import requests
import shutil
import threading
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


import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from spandrel import ModelLoader


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
        self.to_tensor = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()

    def upscale_tensor(self, tensor: torch.Tensor, use_tiling: bool = False) -> torch.Tensor:
        """
        Runs inference on a GPU tensor.
        Returns a GPU tensor (no CPU transfer).
        """
        if use_tiling:
            return self.tiled_inference(tensor)

        with torch.no_grad():
            return self.model(tensor)

    def tiled_inference(self, tensor: torch.Tensor, tile_size=1024, overlap=32) -> torch.Tensor:
        """
        Splits large tensor into tiles, upscales them, and stitches them back.
        Crucial for 2nd pass (2048 -> 8192) to keep GPU compute efficient.
        """
        b, c, h, w = tensor.shape
        # Calculate target size (model is 4x)
        scale = 4
        out_h, out_w = h * scale, w * scale
        output = torch.zeros((b, c, out_h, out_w), device=self.device, dtype=tensor.dtype)

        # Simple sliding window tiling
        for i in range(0, h, tile_size - overlap):
            for j in range(0, w, tile_size - overlap):
                # Crop input
                h_end = min(i + tile_size, h)
                w_end = min(j + tile_size, w)
                h_start = max(0, h_end - tile_size)
                w_start = max(0, w_end - tile_size)

                tile = tensor[:, :, h_start:h_end, w_start:w_end]

                # Inference
                with torch.no_grad():
                    out_tile = self.model(tile)

                out_y_start = h_start * scale
                out_x_start = w_start * scale
                out_y_end = h_end * scale
                out_x_end = w_end * scale

                output[:, :, out_y_start:out_y_end, out_x_start:out_x_end] = out_tile

        return output

    def __call__(self, input_image: Image.Image) -> Image.Image:
        # Move to GPU immediately
        img_tensor = self.to_tensor(input_image).unsqueeze(0).to(self.device).half()

        # --- Pipeline Logic ---
        if self.texture_size in (6144, 8192):
            # Pass 1: 1024 -> 4096 (Fast, no tiling needed usually)
            img_tensor = self.upscale_tensor(img_tensor, use_tiling=False)
            # Downscale 4096 -> 2048
            img_tensor = F.interpolate(img_tensor, size=(2048, 2048), mode='bicubic', antialias=True)
            # Pass 2: 2048 -> 8192
            img_tensor = self.upscale_tensor(img_tensor, use_tiling=True)
        else:
            img_tensor = self.upscale_tensor(img_tensor)

        if img_tensor.shape[-1] != self.texture_size:
            img_tensor = F.interpolate(
                img_tensor,
                size=(self.texture_size, self.texture_size),
                mode='bicubic',
                antialias=True
            )

        output = img_tensor.float().clamp(0, 1).squeeze(0).cpu()
        return self.to_pil(output)


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

    def __init__(self, texture_size: int = 4096, concurrency: int = 3):
        self.topaz_api_key = os.getenv("TOPAZ_API_KEY")
        self.topaz_url = "https://api.topazlabs.com/image/v1/enhance"
        self.output_height = texture_size
        self.output_width = texture_size
        self.model = "Standard V2"
        self.output_format = "png"
        self.max_retries = 5
        self.backoff_base = 2

        # Internal concurrency control (max 3 in-flight requests)
        self._executor = ThreadPoolExecutor(max_workers=concurrency)
        self._sem = threading.Semaphore(concurrency)

        # Optional: reuse connections
        self._session = requests.Session()

        atexit.register(self._shutdown)

    def _shutdown(self):
        # Safe shutdown at process exit
        try:
            self._executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            # cancel_futures not available on older python versions
            self._executor.shutdown(wait=False)

    def __call__(self, input_image: Image.Image) -> Image.Image:
        # Submit work to internal pool and block for result
        fut = self._executor.submit(self._process_one, input_image)
        return fut.result()

    def _process_one(self, input_image: Image.Image) -> Image.Image:
        if not self.topaz_api_key:
            raise RuntimeError("TOPAZ_API_KEY is not set")

        # Limit number of concurrent HTTP requests
        with self._sem:
            raw = BytesIO()
            input_image.save(raw, format="PNG")
            payload = raw.getvalue()

            headers = {
                "X-API-Key": self.topaz_api_key,
                "accept": f"image/{self.output_format}",
            }

            data = {
                "model": self.model,
                "output_height": self.output_height,
                "output_width": self.output_width,
                "output_format": self.output_format,
            }

            last_exc = None
            for attempt in range(self.max_retries):
                try:
                    files = {"image": ("input.png", BytesIO(payload), "image/png")}
                    response = self._session.post(
                        self.topaz_url,
                        headers=headers,
                        files=files,
                        data=data,
                        timeout=(10, 300),
                    )

                    if response.status_code == 200:
                        img = Image.open(BytesIO(response.content))
                        img.load()
                        return img

                    if response.status_code in (429, 500, 502, 503, 504):
                        time.sleep(self.backoff_base ** attempt)
                        continue

                    response.raise_for_status()

                except Exception as e:
                    last_exc = e
                    time.sleep(self.backoff_base ** attempt)

            raise RuntimeError("Topaz sync upscaling failed after retries.") from last_exc


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
            f"Upscale the image quality while maintaining pixel perfect object position from image 1. "
            f"The aspect ratio must remain the same. The object in image 1 must not be distorted in any way. ",
            f"The object position, translation, zoom and shape in image 1 must be preserved exactly as it is. ",
            f"The second provided image is the ground truth reference for the object details. "
            f"You must preserve the original colors and details of image 2 exactly as they are. "
            f"If image 2 has text then please use it since the first image may have messed it up.",
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
