import traceback

import torch
import trimesh
from safetensors.torch import load_file

from silksong.model.data_provider_infer import InferDataset, joint_filter, max_filter, collate_fn_infer
from silksong.model.model import SSMeshTransformer
from silksong.silkutils.silksong_tokenization import get_tokenizer_silksong, detokenize_mesh_ss


class MeshSilkSongPipeline:

    def __init__(self, device='cuda'):
        # tokenizer
        self.tokenizer, _ = get_tokenizer_silksong()

        # model
        pipeline = SSMeshTransformer(
            dim=1024,
            attn_depth=24,
            attn_dim_head=64,
            attn_heads=16,
            max_seq_len=10240,
            dropout=0.0,
            mode="vertices",
            num_discrete_coors=128,
            block_size=8,
            offset_size=16,
            conditioned_on_pc=1,
            encoder_name="miche-256-feature",
            encoder_freeze=0,
        )

        ckpt = load_file("./weights/silksong.safetensors", device='cpu')
        pipeline.load_state_dict(ckpt, strict=False)

        self.pipeline = pipeline.half().eval().to(device)

        num_params = sum([param.nelement() for param in self.pipeline.decoder.parameters()])
        print('Silksong Number of parameters: %.2f M' % (num_params / 1e6))

    def __call__(self, mesh: trimesh.Trimesh, batch_size=1):

        infer_dataset = InferDataset(mesh=mesh)

        infer_dataloader = torch.utils.data.DataLoader(
            infer_dataset,
            batch_size=batch_size,
            drop_last=False,
            shuffle=False,
            collate_fn=collate_fn_infer,
        )

        for it, data in enumerate(infer_dataloader):
            codes = self.pipeline.generate(
                batch_size=1,
                temperature=0.5,
                pc=data['pc_normal'].cuda().half(),
                filter_logits_fn=joint_filter,
                filter_kwargs=dict(k=50, p=0.95),
                return_codes=True,
            )

            coords = []

            # decoding codes to coordinates
            for i in range(len(codes)):
                code = codes[i]
                full_path = data['full_path'][i]
                code = code[code != self.pipeline.pad_id].cpu().numpy()
                try:
                    verts, faces = detokenize_mesh_ss(self.tokenizer, code, colorful=True, mani_fix=True)
                    coords.append({'v': verts, 'f': faces, 'tokens': code})
                except Exception as e:
                    print(f'path generation failed: {full_path}, {str(e)}')
                    traceback.print_exc()
                    coords.append({'tokens': code})

            # convert coordinates to mesh
            for i in range(batch_size):
                gt_v = data['gt_mesh'][i]['v']
                gt_f = data['gt_mesh'][i]['f']

                gt_mesh = trimesh.Trimesh(vertices=gt_v, faces=gt_f)

                return gt_mesh

        return None # If no mesh is generated, return None
