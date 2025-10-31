import numpy as np
import torch
import trimesh

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from accelerate.utils import set_seed
from fastmesh.models import MODELS


class FastMeshPipeline:

    def __init__(self, seed=42):
        accelerator = Accelerator(
            mixed_precision="fp16",
            kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)]
        )

        set_seed(seed, device_specific=True)

        self.model = MODELS["MeshGen"].from_pretrained("WopperSet/FastMesh-V4K")
        self.model = accelerator.prepare(self.model)
        self.model.eval()

    def __call__(self, mesh: trimesh.Trimesh, input_pc_num: int = 16384) -> trimesh.Trimesh:
        accelerator = Accelerator(
            mixed_precision="fp16",
            kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)]
        )

        input_shape = torch.empty((0, input_pc_num, 6)).cuda()

        gt_mesh = self.apply_normalize(mesh)
        pc_normal = self.sample_pc(gt_mesh, input_pc_num, with_normal=True)
        pc_normal = torch.from_numpy(pc_normal).unsqueeze(0).cuda()
        input_shape = torch.cat([input_shape, pc_normal])

        input_dict = {
            "pc_normal": input_shape
        }

        with accelerator.autocast():
            recon_meshes = self.model(input_dict, is_eval=True)

        return recon_meshes[0]

    def sample_pc(self, mesh, pc_num, with_normal=False):
        if not with_normal:
            points, _ = mesh.sample(pc_num, return_index=True)
            return points

        points, face_idx = mesh.sample(200000, return_index=True)
        normals = mesh.face_normals[face_idx]
        pc_normal = np.concatenate([points, normals], axis=-1, dtype=np.float16)

        # random sample point cloud
        ind = np.random.choice(pc_normal.shape[0], pc_num, replace=False)
        pc_normal = pc_normal[ind]

        return pc_normal

    @staticmethod
    def apply_normalize(mesh):
        '''
        normalize mesh to [-1, 1]
        '''
        bbox = mesh.bounds
        center = (bbox[1] + bbox[0]) / 2
        scale = (bbox[1] - bbox[0]).max()

        mesh.apply_translation(-center)
        mesh.apply_scale(1 / scale * 2 * 0.95)

        return mesh
