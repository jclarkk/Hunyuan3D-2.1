# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.
import numpy as np
import os
import torch
import torch.nn.functional as F
import trimesh


# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.


def cuda_xatlas_unwrap(mesh, padding=2, resolution=1024, max_iterations=4):
    """
    UV unwrap using the CUDA-backed xatlas binding (cumesh.xatlas).

    Args:
        mesh (trimesh.Trimesh | trimesh.Scene): Input mesh.
        padding (int): Pixel padding between islands (chart separation).
        resolution (int): Target square atlas resolution (used for padding calc).
        max_iterations (int): Chart growing/seeding iterations (higher = better, slower).

    Returns:
        trimesh.Trimesh: New mesh with reindexed vertices/faces and mesh.visual.uv set.
    """
    try:
        # We import CuMesh here as per the reference logic provided
        import cumesh
        import torch
    except ImportError as e:
        raise ImportError(
            "cumesh not found. Install CuMesh (e.g., `pip install CuMesh/ --no-build-isolation`)."
        ) from e

    # --- 1. Data Preparation ---
    # Handle Scene vs Trimesh
    if isinstance(mesh, trimesh.Scene):
        # Concatenate scene into a single mesh for unwrapping
        mesh = trimesh.util.concatenate(
            tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                  for g in mesh.geometry.values())
        )

    # Ensure mesh is a Trimesh object
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Input must be trimesh.Trimesh or trimesh.Scene, got {type(mesh)}")

    vertices = torch.from_numpy(mesh.vertices).float().cuda()
    faces = torch.from_numpy(mesh.faces).int().cuda()

    # --- 2. CuMesh Initialization ---
    # The reference logic initializes CuMesh with the vertices/faces
    cm = cumesh.CuMesh()
    cm.init(vertices, faces)

    # --- 3. UV Unwrapping ---
    # We map the function arguments to the kwargs expected by cumesh/xatlas.
    # Note: 'padding' and 'resolution' usually go into pack_charts,
    # while 'max_iterations' affects the chart generation.

    # Based on to_glb logic, we call uv_unwrap.
    # We assume uv_unwrap accepts pack_charts_kwargs (standard xatlas pattern).
    out_vertices, out_faces, out_uvs, _ = cm.uv_unwrap(
        compute_charts_kwargs={
            "global_iterations": max_iterations,
            # Defaults from to_glb or reasonable xatlas defaults:
            "threshold_cone_half_angle_rad": np.radians(90.0),
            "refine_iterations": 0,
            "smooth_strength": 1,
        },
        pack_charts_kwargs={
            "padding": padding,
            "resolution": resolution,
            "texels_per_unit": 0.0,  # 0 = use resolution/padding to pack
            "brute_force": False,
        },
        return_vmaps=True,  # Required to unpack the tuple correctly even if unused
        verbose=False,
    )

    # --- 4. Result Reconstruction ---
    # Move results back to CPU
    new_vertices = out_vertices.cpu().numpy()
    new_faces = out_faces.cpu().numpy()
    new_uvs = out_uvs.cpu().numpy()

    # Apply the UV V-flip as seen in the to_glb reference logic
    # (GLTF/OpenGL standard often requires V to be inverted relative to xatlas default)
    if new_uvs.shape[0] > 0:
        new_uvs[:, 1] = 1 - new_uvs[:, 1]

    # Construct the final Trimesh
    # Note: xatlas splits vertices at UV seams, so we must return the NEW topology.
    unwrapped_mesh = trimesh.Trimesh(
        vertices=new_vertices,
        faces=new_faces,
        visual=trimesh.visual.TextureVisuals(uv=new_uvs),
        process=False  # Don't re-merge vertices or we lose the UV seams
    )

    return unwrapped_mesh


def sf_mesh_uv_wrap(mesh, island_padding=0.05, device='cuda', y_flip=False):
    try:
        from uv_unwrapper import Unwrapper
    except ImportError:
        import logging
        logging.warning(
            "Could not import uv_unwrapper. Please install it via `pip install hy3dpaint/utils/uv_unwrapper/`"
        )
        raise ImportError("uv_unwrapper not found")

    if not mesh.is_watertight or not mesh.faces.shape[1] == 3:
        mesh = ensure_triangles(mesh)

    v_pos = torch.from_numpy(np.asarray(mesh.vertices, dtype=np.float32)).to(device)
    t_pos_idx = torch.from_numpy(np.asarray(mesh.faces, dtype=np.int64)).to(device)
    v_nrm = _torch_vertex_normals(v_pos, t_pos_idx)

    unwrapper = Unwrapper()

    with torch.no_grad():
        uv, indices = unwrapper(v_pos, v_nrm, t_pos_idx, float(island_padding))
        # uv: [Nuv, 2] unique UVs
        # indices: [Nf, 3] per-corner indices into uv
        t_pos_idx_flat = t_pos_idx.reshape(-1)
        indices_flat = indices.reshape(-1)
        combined = torch.stack([t_pos_idx_flat, indices_flat], dim=1)  # [Nf*3, 2]
        unique_combined, new_indices_flat = torch.unique(combined, return_inverse=True, dim=0)
        new_vertices = v_pos[unique_combined[:, 0]]
        new_uvs = uv[unique_combined[:, 1]]
        if y_flip:
            new_uvs[:, 1] = 1.0 - new_uvs[:, 1]
        new_faces = new_indices_flat.reshape(-1, 3)
        v_nrm_dup = _torch_vertex_normals(new_vertices, new_faces)

    # To numpy
    new_vertices_np = new_vertices.detach().cpu().numpy().astype(np.float32)
    new_faces_np = new_faces.detach().cpu().numpy().astype(np.int64)
    new_uvs_np = new_uvs.detach().cpu().numpy().astype(np.float32)
    new_vn_np = v_nrm_dup.detach().cpu().numpy().astype(np.float32)

    # Build new trimesh
    new_mesh = trimesh.Trimesh(vertices=new_vertices_np, faces=new_faces_np, process=False)
    new_mesh.visual = trimesh.visual.TextureVisuals(uv=new_uvs_np)
    new_mesh._vertex_normals = new_vn_np

    return new_mesh


def mesh_uv_wrap(mesh, padding=2, resolution=1024, max_iterations=4):
    import xatlas

    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    large_mesh_mode = False
    if len(mesh.faces) > 100000 and len(mesh.faces) < 200000:
        large_mesh_mode = True
        print("Warning: The mesh has more than 100,000 faces, which may cause slowdowns.")
    if len(mesh.faces) > 200000:
        # Try with open3d if the mesh is too large
        return open3d_mesh_uv_wrap(mesh, resolution=resolution, use_fallback=False)

    print('Using xatlas for UV unwrapping')

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.uint32)

    atlas = xatlas.Atlas()

    # Add the mesh to the atlas
    atlas.add_mesh(vertices, faces)

    chart_options = xatlas.ChartOptions()
    chart_options.max_iterations = 1
    chart_options.max_cost = 3.0
    chart_options.normal_seam_weight = 0.5
    chart_options.texture_seam_weight = 1.0

    pack_options = xatlas.PackOptions()
    pack_options.padding = padding
    pack_options.resolution = resolution
    pack_options.bilinear = True
    if large_mesh_mode:
        pack_options.rotate_charts = False

    atlas.generate(chart_options=chart_options, pack_options=pack_options)

    vmapping, indices, uvs = atlas[0]

    # Update the mesh
    mesh.vertices = mesh.vertices[vmapping]
    mesh.faces = indices
    mesh.visual.uv = uvs

    return mesh


def open3d_mesh_uv_wrap(mesh, gutter_size=6, max_stretch=0.5, resolution=1024, use_fallback=True):
    try:
        import open3d as o3d
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)

        o3d_mesh = o3d.t.geometry.TriangleMesh()
        o3d_mesh.vertex.positions = o3d.core.Tensor(mesh.vertices)
        o3d_mesh.triangle.indices = o3d.core.Tensor(mesh.faces)

        core_count = max(os.cpu_count(), 16)
        print('Using Open3D for UV unwrapping with {} threads'.format(core_count))

        o3d_mesh.compute_uvatlas(
            size=resolution,
            parallel_partitions=4,
            gutter=gutter_size,
            max_stretch=max_stretch,
            nthreads=core_count
        )

        new_v = mesh.vertices[mesh.faces.reshape(-1)]
        new_f = np.arange(len(new_v)).reshape(-1, 3)
        new_uv = o3d_mesh.triangle.texture_uvs.numpy().reshape(-1, 2)

        mesh = trimesh.Trimesh(
            vertices=new_v,
            faces=new_f,
            process=False
        )
        mesh.visual = trimesh.visual.TextureVisuals(
            uv=new_uv.astype(np.float32),
        )
    except Exception as e:
        if use_fallback:
            # Open3D might fail on mesh conditions so we will fallback to xatlas
            print('Open3D failed to unwrap mesh, falling back to xatlas. Error: ', e)
            return mesh_uv_wrap(mesh)
        else:
            raise e

    return mesh


def bpy_unwrap_mesh(mesh):
    import bpy
    import bmesh

    # Store original vertices and faces
    vertices = mesh.vertices
    faces = mesh.faces

    # Clear any existing mesh with the same name
    if "TempMesh" in bpy.data.meshes:
        bpy.data.meshes.remove(bpy.data.meshes["TempMesh"])
    if "TempObject" in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects["TempObject"])

    # Create new mesh and object
    bpy_mesh = bpy.data.meshes.new(name="TempMesh")
    obj = bpy.data.objects.new(name="TempObject", object_data=bpy_mesh)

    # Link to scene
    bpy.context.collection.objects.link(obj)

    # Set as active object with proper context
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # Create BMesh and populate
    bm = bmesh.new()
    vert_list = [bm.verts.new(tuple(v)) for v in vertices]
    bm.verts.ensure_lookup_table()

    # Add faces with error checking
    for f in faces:
        try:
            face_verts = [vert_list[i] for i in f]
            bm.faces.new(face_verts)
        except ValueError as e:
            print(f"Skipping invalid face: {e}")
            continue

    # Update mesh
    bm.to_mesh(bpy_mesh)
    bm.free()

    # Ensure UV layer exists
    if not bpy_mesh.uv_layers:
        bpy_mesh.uv_layers.new(name="UVMap")

    # Switch to edit mode and unwrap with proper context
    override = bpy.context.copy()
    override['active_object'] = obj
    override['object'] = obj
    override['edit_object'] = obj
    override['scene'] = bpy.context.scene

    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(
        override,
        angle_limit=66.0,
        island_margin=0.03
    )
    bpy.ops.object.mode_set(mode='OBJECT')

    # Get UV data
    uv_layer = bpy_mesh.uv_layers.active
    if not uv_layer:
        print("Failed to create UV layer")
        return mesh

    # Extract UV coordinates
    uv_data = uv_layer.data
    uvs = np.zeros((len(faces), 3, 2), dtype=np.float32)

    for poly in bpy_mesh.polygons:
        for loop_idx, loop in enumerate(poly.loop_indices):
            uv = uv_data[loop].uv
            uvs[poly.index, loop_idx] = [uv.x, uv.y]

    # Create averaged UVs per vertex
    vertex_uvs = np.zeros((len(vertices), 2), dtype=np.float32)
    counts = np.zeros(len(vertices), dtype=np.int32)

    for face_idx, face in enumerate(faces):
        for vert_idx, uv in zip(face, uvs[face_idx]):
            vertex_uvs[vert_idx] += uv
            counts[vert_idx] += 1

    # Avoid division by zero and compute average
    mask = counts > 0
    vertex_uvs[mask] /= counts[mask, None]

    # Create new TextureVisuals object
    mesh.visual = trimesh.visual.TextureVisuals(uv=vertex_uvs)

    # Clean up
    bpy.data.objects.remove(obj)
    bpy.data.meshes.remove(bpy_mesh)

    return mesh


def ensure_triangles(tm: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Return a triangle-only Trimesh.
    - If already triangles: returns a copy with process=False (no topology edits).
    - If faces are quads/ngons: fan-triangulates each polygon [v0, v1, v2, ...] into
      triangles (v0, v1, v2), (v0, v2, v3), ...
    Preserves visual.uv and material when they are per-vertex.
    """
    V = np.asarray(tm.vertices, dtype=np.float64)
    F = np.asarray(tm.faces)

    # If faces are already triangles, just copy (and keep uv/material)
    if F.ndim == 2 and F.shape[1] == 3:
        new_tm = trimesh.Trimesh(vertices=V.copy(), faces=F.copy(), process=False)
    else:
        # Some formats pad with -1: strip them
        tris = []
        for face in F:
            idx = [int(i) for i in face if i != -1]  # drop padding
            if len(idx) < 3:
                continue
            # fan triangulation: (v0, v1, v2), (v0, v2, v3), ...
            for i in range(1, len(idx) - 1):
                tris.append([idx[0], idx[i], idx[i + 1]])
        if not tris:
            raise ValueError("Mesh has no valid faces to triangulate.")
        tris = np.asarray(tris, dtype=np.int64)
        new_tm = trimesh.Trimesh(vertices=V.copy(), faces=tris, process=False)

    # Preserve per-vertex UVs/material if present
    uv = getattr(getattr(tm, "visual", None), "uv", None)
    if uv is not None and len(uv) == len(V):
        new_tm.visual = trimesh.visual.TextureVisuals(uv=np.asarray(uv))
        mat = getattr(tm.visual, "material", None)
        if mat is not None:
            new_tm.visual.material = mat

    return new_tm


def _torch_vertex_normals(v_pos: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """
    Compute per-vertex normals in torch (same math as in your Mesh class).
    v_pos:  [Nv, 3]
    faces:  [Nf, 3]
    """
    i0, i1, i2 = faces[:, 0], faces[:, 1], faces[:, 2]
    v0, v1, v2 = v_pos[i0], v_pos[i1], v_pos[i2]

    face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
    v_nrm = torch.zeros_like(v_pos)
    v_nrm.scatter_add_(0, i0[:, None].expand(-1, 3), face_normals)
    v_nrm.scatter_add_(0, i1[:, None].expand(-1, 3), face_normals)
    v_nrm.scatter_add_(0, i2[:, None].expand(-1, 3), face_normals)

    mask = (v_nrm * v_nrm).sum(dim=1) <= 1e-20
    if mask.any():
        v_nrm[mask] = v_nrm.new_tensor([0.0, 0.0, 1.0])
    v_nrm = F.normalize(v_nrm, dim=1)
    return v_nrm
