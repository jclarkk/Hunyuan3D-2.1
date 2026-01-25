import argparse
import os
import sys
import time

import trimesh
from PIL import Image

sys.path.insert(0, './hy3dshape')
sys.path.insert(0, './hy3dpaint')

from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
from hy3dshape.rmbg import RMBGRemover
from hy3dshape.utils.utils import normalize_mesh


def run(args):
    if args.prompt is None and args.image_paths is None:
        raise ValueError("Please provide either a prompt or an image")

    if args.prompt is not None and args.image_paths is not None:
        raise ValueError("Please provide either a prompt or an image, not both")

    if args.remesh_method not in [None, 'im', 'bpt', 'None']:
        raise ValueError("Re-mesh type must be either 'im' or 'bpt'")

    if args.texture_size not in [1024, 2048, 3072, 4096, 6144, 8192]:
        raise ValueError("Texture size must be one of 1024, 2048, 3072, 4096, 6144, 8192")

    if args.unwrap_method not in ['xatlas', 'open3d', 'bpy', 'sf', 'cuda_xatlas']:
        raise ValueError("Unwrap method must be either 'xatlas', 'open3d' or 'bpy' or 'sf' or 'cuda_xatlas'")

    t0 = time.time()
    # Load mesh
    mesh = trimesh.load(args.mesh_path, force='mesh')

    # Reduce face count
    face_limit = 550000
    if len(mesh.faces) > face_limit:
        # Try reducing first
        try:
            from hy3dshape.hy3dshape.postprocessors import reduce_face_with_meshlib
            mesh = reduce_face_with_meshlib(mesh, 500000)
            if len(mesh.faces) > face_limit:
                raise ValueError(f"Reduction failed - Face count must be less than or equal to {face_limit}")
        except Exception as e:
            print(f"Mesh reduction with meshlib failed: {e}.")
        raise ValueError(f"Face count must be less than or equal to {face_limit}")

    t1 = time.time()
    print(f"Mesh pre-processing took {t1 - t0:.2f} seconds")

    t2 = time.time()
    # Load models
    conf = Hunyuan3DPaintConfig(
        hypaint_resolution=1024
    )
    conf.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
    conf.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"
    conf.multiview_pretrained_path = args.texgen_model_path
    conf.continuous_inference = False
    texture_pipeline = Hunyuan3DPaintPipeline(conf)
    print('3D Paint pipeline loaded')

    t3 = time.time()
    print(f"Model loading took {t3 - t2:.2f} seconds")

    images = None
    if args.prompt is None:
        # Only one image supported right now
        images = [Image.open(image_path) for image_path in args.image_paths]

    t4 = time.time()
    # Preprocess the image
    if not args.texgen_model_path.startswith('mv-adapter') and images is not None:
        processed_images = []
        for image in images:
            rmbg_remover = RMBGRemover(local_files_only=args.local_files_only)
            image = rmbg_remover(image)
            processed_images.append(image)
    else:
        processed_images = images

    t5 = time.time()
    print(f"Image processing took {t5 - t4:.2f} seconds")

    # Use mesh file name as output name
    output_name = os.path.splitext(os.path.basename(args.mesh_path))[0] + '_textured'
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate texture
    t6 = time.time()
    mesh = texture_pipeline(
        mesh_path=mesh,
        image_path=processed_images,
        prompt=args.prompt,
        unwrap_method=args.unwrap_method,
        upscale_model=args.upscale_model,
        pbr=args.pbr,
        texture_size=args.texture_size,
        num_views=args.num_views,
        seed=args.seed
    )
    t7 = time.time()

    print(f"Texture generation took {t7 - t6:.2f} seconds")

    mesh = normalize_mesh(mesh)

    mesh.export(os.path.join(args.output_dir, '{}.glb'.format(output_name)))

    t8 = time.time()

    print(f"Mesh export took {t8 - t7:.2f} seconds")

    print(f"Output saved to {args.output_dir}/{output_name}.glb")

    print(f"Total time taken: {t8 - t0:.2f} seconds")


if __name__ == "__main__":
    # Parse arguments and then call run
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_files_only', action='store_true', help='Use local models only')
    parser.add_argument('--image_paths', type=str, nargs='+', default=None,
                        help='Path to input images. Can specify multiple paths separated by spaces')
    parser.add_argument('--prompt', type=str, default=None, help='Prompt for the image')
    parser.add_argument('--mesh_path', type=str, help='Path to input mesh', required=True)
    parser.add_argument('--output_dir', type=str, default='./output', help='Path to output directory')
    parser.add_argument('--seed', type=int, default=0, help='Seed for the random number generator')
    parser.add_argument('--texture_size', type=int, default=1024, help='Texture size')
    parser.add_argument('--remesh_method', type=str, help='Re-mesh method. Must be either "im" or "bpt" if used.',
                        default=None)
    parser.add_argument('--unwrap_method', type=str,
                        help='UV unwrap method. Must be either "xatlas", "open3d" or "bpy" or "sf" or "cuda_xatlas"', default='xatlas')
    parser.add_argument("--texgen_model_path", type=str, default='tencent/Hunyuan3D-2.1')
    parser.add_argument('--upscale_model', type=str, default=None, help='Upscale model to use')
    parser.add_argument('--num_views', type=int, help='Number of texture projection views', default=8)
    parser.add_argument('--pbr', action='store_true', help='Generate PBR textures', default=False)
    parser.add_argument('--debug', action='store_true', help='Debug mode', default=False)

    args = parser.parse_args()

    run(args)
