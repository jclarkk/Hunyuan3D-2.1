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
        mesh (trimesh.Trimesh | trimesh.Scene): Input mesh (tri or not).
        padding (int): Pixel padding between islands.
        resolution (int): Target square atlas resolution.
        max_iterations (int): Chart growing/seeding iterations (higher = better, slower).

    Returns:
        trimesh.Trimesh: New mesh with reindexed vertices/faces and mesh.visual.uv set.
    """
    try:
        from cumesh.xatlas import Atlas  # your provided wrapper
        import torch
        import numpy as np
        import trimesh
    except ImportError as e:
        raise ImportError(
            "cumesh.xatlas not found. Install CuMesh (e.g., `pip install CuMesh/ --no-build-isolation`)."
        ) from e

    # Flatten scenes and ensure triangles (preserve any per-vertex UV/material if present)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("cuda_xatlas_unwrap expects a trimesh.Trimesh or trimesh.Scene")

    tm = mesh
    if not (hasattr(tm, "faces") and tm.faces is not None and tm.faces.ndim == 2 and tm.faces.shape[1] == 3):
        tm = ensure_triangles(tm)

    # Heuristics for big meshes (helps speed/robustness)
    face_count = int(len(tm.faces))
    rotate_charts = True
    if face_count > 100_000:
        rotate_charts = False  # avoid extra work on very large meshes

    # Prepare CPU tensors in the dtypes/layout expected by the wrapper
    V_np = np.asarray(tm.vertices, dtype=np.float32, order="C")
    F_np = np.asarray(tm.faces, dtype=np.int32, order="C")

    V = torch.from_numpy(V_np)  # [V,3], float32, CPU, contiguous
    F = torch.from_numpy(F_np)  # [F,3], int32,  CPU, contiguous

    # Optionally pass normals as hints—compute if not present
    normals_np = None
    try:
        if getattr(tm, "_vertex_normals", None) is not None and len(tm._vertex_normals) == len(V_np):
            normals_np = np.asarray(tm._vertex_normals, dtype=np.float32, order="C")
        elif getattr(getattr(tm, "vertex_normals", None), "__array__", None) is not None:
            n_tmp = np.asarray(tm.vertex_normals, dtype=np.float32)
            if len(n_tmp) == len(V_np):
                normals_np = np.ascontiguousarray(n_tmp, dtype=np.float32)
    except Exception:
        normals_np = None

    if normals_np is None:
        # CPU torch normals (wrapper requires CPU anyway)
        v_pos = torch.from_numpy(V_np.copy())
        t_pos_idx = torch.from_numpy(F_np.copy().astype(np.int64))
        nrm = _torch_vertex_normals(v_pos, t_pos_idx).contiguous().to(torch.float32)
        normals_np = nrm.numpy()

    N = torch.from_numpy(normals_np)  # [V,3], float32, CPU

    # If the source had UVs, we can pass them as hints (optional)
    UV_hint = None
    try:
        if getattr(getattr(tm, "visual", None), "uv", None) is not None:
            uv_arr = np.asarray(tm.visual.uv, dtype=np.float32)
            if uv_arr.shape[0] == V_np.shape[0] and uv_arr.shape[1] == 2:
                UV_hint = torch.from_numpy(np.ascontiguousarray(uv_arr, dtype=np.float32))
    except Exception:
        UV_hint = None

    # Build atlas and add the mesh
    atlas = Atlas()
    atlas.add_mesh(V, F, N, UV_hint)

    # Chart (parameterization) options — tuned for quality/speed balance
    # You can surface these as function args if you want finer control.
    atlas.compute_charts(
        max_chart_area=0.0,
        max_boundary_length=0.0,
        normal_deviation_weight=2.0,
        roundness_weight=0.01,
        straightness_weight=6.0,
        normal_seam_weight=4.0,
        texture_seam_weight=0.5,
        max_cost=2.0,
        max_iterations=int(max_iterations),
        use_input_mesh_uvs=(UV_hint is not None),
        fix_winding=False,
        verbose=False,
    )

    # Pack charts into a square texture
    atlas.pack_charts(
        max_chart_size=0,  # no per-chart limit
        padding=int(padding),
        texels_per_unit=0.0,  # auto-estimate to match resolution
        resolution=int(resolution),
        bilinear=True,
        block_align=False,
        brute_force=False,
        rotate_charts=rotate_charts,
        rotate_charts_to_axis=True,
        verbose=False,
    )

    # Retrieve results for mesh 0
    xrefs, faces_out, uvs_out = atlas.get_mesh(0)  # xrefs:[NewV], faces:[NewF,3], uvs:[NewV,2]
    # to numpy
    xrefs = xrefs.cpu().numpy().astype(np.int64, copy=False)
    faces_out = faces_out.cpu().numpy().astype(np.int64, copy=False)
    uvs_out = uvs_out.cpu().numpy().astype(np.float32, copy=False)

    # Map original vertices -> new vertex order
    new_vertices = V_np[xrefs]  # [NewV,3]
    new_faces = faces_out  # [NewF,3]
    new_uvs = uvs_out  # [NewV,2]

    # Safety: clamp UVs to [0,1] (xatlas should already pack properly)
    np.clip(new_uvs, 0.0, 1.0, out=new_uvs)

    # Build a fresh trimesh
    out = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)
    out.visual = trimesh.visual.TextureVisuals(uv=new_uvs)

    # Recompute/transfer vertex normals for the new indexing
    try:
        # Duplicate normals to new indexing using xrefs
        if normals_np is not None and len(normals_np) == len(V_np):
            out._vertex_normals = normals_np[xrefs].astype(np.float32, copy=False)
        else:
            # fallback recompute
            _ = out.vertex_normals  # triggers trimesh compute if needed
    except Exception:
        pass

    return out


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
