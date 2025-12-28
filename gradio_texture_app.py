import gradio as gr
import trimesh
import os
import uuid
import argparse
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn

import sys

sys.path.insert(0, './hy3dpaint')

from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

parser = argparse.ArgumentParser()
parser.add_argument('--host', type=str, default='0.0.0.0')
parser.add_argument('--port', type=int, default=8080)
parser.add_argument('--texgen_model_path', type=str, default='tencent/Hunyuan3D-2.1')
args = parser.parse_args()

# Init texture generation pipeline
conf = Hunyuan3DPaintConfig(
    hypaint_resolution=1024
)
conf.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
conf.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"
conf.multiview_pretrained_path = args.texgen_model_path
tex_pipeline = Hunyuan3DPaintPipeline(conf)

SAVE_DIR = "./save_dir"
os.makedirs(SAVE_DIR, exist_ok=True)


def run_texturing(glb_file, reference_image, uv_unwrap_method, texture_size, pbr, super_resolution, num_views, seed):
    save_folder = os.path.join(SAVE_DIR, str(uuid.uuid4()))
    os.makedirs(save_folder, exist_ok=True)

    # Load GLB mesh
    mesh = trimesh.load(glb_file, force='mesh')

    # Run texturing
    textured_mesh = tex_pipeline(
        mesh_path=mesh,
        image_path=reference_image,
        texture_size=texture_size,
        pbr=pbr,
        upscale_model=super_resolution,
        num_views=num_views,
        unwrap_method=uv_unwrap_method,
        seed=seed
    )

    glb_out_path = os.path.join(save_folder, "textured_mesh.glb")
    textured_mesh.export(glb_out_path)

    return glb_out_path


# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 🎨 Hunyuan 3D Texturing - GLB + Reference Image")
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Column():
                input_preview = gr.Model3D(label="Input Mesh (.glb, .obj)")
                reference_image = gr.Image(type="filepath", label="Reference Image")

            with gr.Column():
                seed = gr.Number(value=-1, label="Seed (use -1 for random)", precision=0)
                uv_unwrap_method = gr.Radio(['xatlas', 'open3d', 'bpy', 'sf', 'cuda_xatlas'], label='UV Unwrap Method',
                                            value='xatlas')
                texture_size = gr.Slider(1024, 8192, step=1024, value=4096, label="Texture Size")
                pbr = gr.Checkbox(value=True, label="Enable PBR Texturing")
                super_resolution = gr.Radio(["None", "NMKD", "Aura", "Flux", "Topaz", "Gemini"],
                                            value="NMKD", label="Super-Resolution")
                num_views = gr.Slider(6, 24, step=2, value=6, label="Number of Views")
                submit = gr.Button("Generate Texture")

        with gr.Column(scale=1):
            output_preview = gr.Model3D(label="Refined Output Mesh")

    submit.click(run_texturing,
                 inputs=[input_preview, reference_image, uv_unwrap_method, texture_size, pbr, super_resolution,
                         num_views,
                         seed],
                 outputs=[output_preview])

if os.getenv("USE_GRADIO_SHARE", "0") == "1":
    demo.launch(server_name="0.0.0.0", server_port=8080, share=True)
else:
    app = FastAPI()
    app.mount("/static", StaticFiles(directory=SAVE_DIR, html=True), name="static")
    app = gr.mount_gradio_app(app, demo, path="/")
    uvicorn.run(app, host="0.0.0.0", port=8080, proxy_headers=True, forwarded_allow_ips="*")
