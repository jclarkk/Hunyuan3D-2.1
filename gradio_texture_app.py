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


def build_model_viewer_html(glb_path: str, height: int = 480, textured=False) -> str:
    rel_path = os.path.relpath(glb_path, SAVE_DIR)
    html_file = os.path.splitext(glb_path)[0] + (".textured.html" if textured else ".preview.html")

    with open(html_file, "w") as f:
        f.write(f"""
        <html>
        <head>
            <script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>
        </head>
        <body style="margin:0;">
            <model-viewer 
                src="/static/{rel_path}" 
                camera-controls 
                auto-rotate 
                background-color="#ffffff"
                style="height:{height}px; width:100%;">
            </model-viewer>
        </body>
        </html>
        """)

    rel_html = os.path.relpath(html_file, SAVE_DIR)
    return f'<iframe src="/static/{rel_html}" height="{height}" width="100%" frameborder="0"></iframe>'


def preview_uploaded_glb(glb_file):
    if glb_file is None:
        # File cleared
        return gr.update(visible=False, value="")

    save_folder = os.path.join(SAVE_DIR, str(uuid.uuid4()))
    os.makedirs(save_folder, exist_ok=True)

    glb_preview_path = os.path.join(save_folder, "input_preview.glb")
    with open(glb_file.name, "rb") as src, open(glb_preview_path, "wb") as dst:
        dst.write(src.read())

    html = build_model_viewer_html(glb_preview_path)
    return gr.update(visible=True, value=html)


parser = argparse.ArgumentParser()
parser.add_argument('--host', type=str, default='0.0.0.0')
parser.add_argument('--port', type=int, default=8080)
parser.add_argument('--texgen_model_path', type=str, default='tencent/Hunyuan3D-2.1')
args = parser.parse_args()

# Init texture generation pipeline
conf = Hunyuan3DPaintConfig(hypaint_resolution=1024)
conf.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
conf.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"
conf.multiview_pretrained_path = args.texgen_model_path
tex_pipeline = Hunyuan3DPaintPipeline(conf)

SAVE_DIR = "./save_dir"
os.makedirs(SAVE_DIR, exist_ok=True)


def run_texturing(glb_file, reference_image, uv_unwrap_method, texture_size, pbr, super_resolution, num_views):
    save_folder = os.path.join(SAVE_DIR, str(uuid.uuid4()))
    os.makedirs(save_folder, exist_ok=True)

    # Load GLB mesh
    mesh = trimesh.load(glb_file.name, force='mesh')

    # Run texturing
    textured_mesh = tex_pipeline(
        mesh_path=mesh,
        image_path=reference_image,
        texture_size=texture_size,
        pbr=pbr,
        upscale_model=super_resolution,
        num_views=num_views,
        unwrap_method=uv_unwrap_method
    )

    glb_out_path = os.path.join(save_folder, "textured_mesh.glb")
    textured_mesh.export(glb_out_path)

    return (
        glb_out_path,
        build_model_viewer_html(glb_out_path)
    )


# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 🎨 Hunyuan 3D Texturing - GLB + Reference Image")
    with gr.Row():
        with gr.Column():
            glb_file = gr.File(label="Upload GLB", file_types=[".glb"], interactive=True, visible=True)
            input_preview = gr.HTML(label="Preview", visible=False)
        reference_image = gr.Image(type="filepath", label="Reference Image")

    with gr.Row():
        uv_unwrap_method = gr.Radio(['xatlas', 'open3d', 'bpy'], label='UV Unwrap Method', value='xatlas')
        texture_size = gr.Slider(1024, 8192, step=1024, value=4096, label="Texture Size")
        pbr = gr.Checkbox(value=True, label="Enable PBR Texturing")
        super_resolution = gr.Radio(["None", "NMKD", "Aura", "Flux", "Topaz"], value="NMKD", label="Super-Resolution")
        num_views = gr.Slider(6, 24, step=2, value=6, label="Number of Views")

    submit = gr.Button("Generate Texture")
    output_file = gr.File(label="Download Textured GLB")

    glb_file.change(
        fn=preview_uploaded_glb,
        inputs=[glb_file],
        outputs=[input_preview]
    )

    output_preview = gr.HTML(label="Output Preview")

    submit.click(run_texturing,
                 inputs=[glb_file, reference_image, uv_unwrap_method, texture_size, pbr, super_resolution, num_views],
                 outputs=[output_file, output_preview])

app = FastAPI()
app.mount("/static", StaticFiles(directory=SAVE_DIR, html=True), name="static")
app = gr.mount_gradio_app(app, demo, path="/")

uvicorn.run(app, host=args.host, port=args.port)
