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
    template_name = './assets/modelviewer-template.html'
    out_html = os.path.join('current_mesh.html')

    # read template
    with open(os.path.join(template_name), 'r', encoding='utf-8') as f:
        html = f.read()

    # IMPORTANT: use file= paths (works locally, with Runpod proxy, and with share=True)
    # your template should reference #src# where the model URL goes
    html = html.replace('#src#', f'file={glb_path}')

    # if your template references env maps or other assets, also replace those to file=ABS_PATH
    # e.g.: html = html.replace('#env_map#', f'file={os.path.abspath("assets/env_maps/studio.hdr")}')

    with open(out_html, 'w', encoding='utf-8') as f:
        f.write(html)

    # Point iframe to the HTML file via Gradio's file server
    return f"""
      <div style='height:{height}px;width:100%;'>
        <iframe src="file={out_html}" height="{height}" width="100%" frameborder="0"></iframe>
      </div>
    """


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
parser.add_argument("--qwen_edit_base_model", type=str, default=None, help='Qwen edit base model')
parser.add_argument("--qwen_edit_disable_control", action="store_true",
                    help="Disable passing position ControlNet maps to Qwen.")
parser.add_argument("--qwen_edit_dtype", type=str, default=None,
                    help="Optional dtype override for the Qwen pipeline (e.g. 'float16').")
parser.add_argument("--qwen_edit_disable_fuse_lora", action="store_true",
                    help="Disable LoRA fusion inside the Qwen pipeline.")
args = parser.parse_args()

# Init texture generation pipeline
conf = Hunyuan3DPaintConfig(
    hypaint_resolution=1024,
    qwen_edit_base_model=args.qwen_edit_base_model,
    qwen_edit_use_control=not args.qwen_edit_disable_control,
)
if args.qwen_edit_dtype:
    conf.qwen_edit_dtype = args.qwen_edit_dtype
if args.qwen_edit_disable_fuse_lora:
    conf.qwen_edit_fuse_lora = False
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
    mesh = trimesh.load(glb_file.name, force='mesh')

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

    return (
        glb_out_path,
        build_model_viewer_html(glb_out_path)
    )


# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 🎨 Hunyuan 3D Texturing - GLB + Reference Image")
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Column():
                glb_file = gr.File(label="Upload GLB", file_types=[".glb"], interactive=True)
                input_preview = gr.HTML(label="Input Preview", visible=False)
                reference_image = gr.Image(type="filepath", label="Reference Image")

            with gr.Column():
                seed = gr.Number(value=-1, label="Seed (use -1 for random)", precision=0)
                uv_unwrap_method = gr.Radio(['xatlas', 'open3d', 'bpy', 'sf'], label='UV Unwrap Method', value='xatlas')
                texture_size = gr.Slider(1024, 8192, step=1024, value=4096, label="Texture Size")
                pbr = gr.Checkbox(value=True, label="Enable PBR Texturing")
                super_resolution = gr.Radio(["None", "NMKD", "Aura", "Flux", "Topaz"],
                                            value="NMKD", label="Super-Resolution")
                num_views = gr.Slider(6, 24, step=2, value=6, label="Number of Views")
                submit = gr.Button("Generate Texture")

        with gr.Column(scale=1):  # RIGHT COLUMN: Output
            output_file = gr.File(label="Download Textured GLB")
            output_preview = gr.HTML(label="Output Preview")

    glb_file.change(
        fn=preview_uploaded_glb,
        inputs=[glb_file],
        outputs=[input_preview]
    )

    submit.click(run_texturing,
                 inputs=[glb_file, reference_image, uv_unwrap_method, texture_size, pbr, super_resolution, num_views,
                         seed],
                 outputs=[output_file, output_preview])

if os.getenv("USE_GRADIO_SHARE", "0") == "1":
    demo.launch(server_name="0.0.0.0", server_port=8080, share=True)
else:
    app = FastAPI()
    app.mount("/static", StaticFiles(directory=SAVE_DIR, html=True), name="static")
    app = gr.mount_gradio_app(app, demo, path="/")
    uvicorn.run(app, host="0.0.0.0", port=8080, proxy_headers=True, forwarded_allow_ips="*")
