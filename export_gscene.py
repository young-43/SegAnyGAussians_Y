"""
Stage 3 - Step 3: Export .gscene format.

Exports:
  1. Standard 3DGS model PLY file (full scene, compatible with existing tools).
  2. Per-object PLY files (each object in local coordinates).
  3. Structured metadata JSON (.gscene) with:
     - Per-Gaussian: global index, semanticID, objectID
     - Per-object: ID, name, localToWorld transform, gaussian indices, AABB bounds.
"""

import os
import json
import numpy as np
from plyfile import PlyData, PlyElement
from utils.system_utils import mkdir_p


def _numpy_to_list(obj):
    """Recursively convert numpy arrays and types to JSON-serializable lists."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: _numpy_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_numpy_to_list(i) for i in obj]
    return obj


def export_scene_ply(gaussian_model, output_path):
    """
    Export the full scene Gaussian model as a standard PLY file.
    Delegates to the model's own save_ply method.

    Args:
        gaussian_model: GaussianModel instance.
        output_path: Path to write the PLY file.
    """
    mkdir_p(os.path.dirname(output_path))
    gaussian_model.save_ply(output_path)
    print(f"Exported full scene PLY: {output_path}")


def export_object_ply(gaussian_model, object_ids, objects_info, output_dir):
    """
    Export individual PLY files for each object, in local coordinates.

    Args:
        gaussian_model: GaussianModel instance.
        object_ids: (N,) numpy array of object IDs.
        objects_info: Dict from compute_object_info.
        output_dir: Directory to write per-object PLY files.
    """
    mkdir_p(output_dir)

    xyz = gaussian_model._xyz.detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    f_dc = gaussian_model._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    f_rest = gaussian_model._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = gaussian_model._opacity.detach().cpu().numpy()
    scale = gaussian_model._scaling.detach().cpu().numpy()
    rotation = gaussian_model._rotation.detach().cpu().numpy()

    attr_names = gaussian_model.construct_list_of_attributes()
    dtype_full = [(attr, 'f4') for attr in attr_names]

    for obj_id, info in objects_info.items():
        indices = np.array(info["gaussian_indices"])
        if len(indices) == 0:
            continue

        centroid = info["centroid"]

        # Extract subset and shift to local coordinates
        obj_xyz = xyz[indices] - centroid
        obj_normals = normals[indices]
        obj_f_dc = f_dc[indices]
        obj_f_rest = f_rest[indices]
        obj_opacities = opacities[indices]
        obj_scale = scale[indices]
        obj_rotation = rotation[indices]

        attributes = np.concatenate(
            (obj_xyz, obj_normals, obj_f_dc, obj_f_rest, obj_opacities, obj_scale, obj_rotation),
            axis=1
        )

        elements = np.empty(len(indices), dtype=dtype_full)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')

        ply_path = os.path.join(output_dir, f"{info['name']}.ply")
        PlyData([el]).write(ply_path)
        print(f"  Exported object PLY: {ply_path} ({len(indices)} gaussians)")


def export_gscene_metadata(object_ids, objects_info, output_path,
                           semantic_ids=None):
    """
    Export structured .gscene metadata JSON.

    Schema:
    {
      "version": "1.0",
      "total_gaussians": int,
      "total_objects": int,
      "gaussians": [
        { "index": int, "semanticID": int, "objectID": int }, ...
      ],
      "objects": [
        {
          "id": int,
          "name": str,
          "gaussian_count": int,
          "gaussian_indices": [int, ...],
          "centroid": [x, y, z],
          "aabb_min": [x, y, z],
          "aabb_max": [x, y, z],
          "local_to_world": [[4x4 matrix]]
        }, ...
      ]
    }

    Args:
        object_ids: (N,) numpy array of object IDs.
        objects_info: Dict from compute_object_info.
        output_path: Path to write JSON file.
        semantic_ids: Optional (N,) array of semantic class IDs.
                      If None, uses object_ids as semantic IDs.
    """
    mkdir_p(os.path.dirname(output_path))

    if semantic_ids is None:
        semantic_ids = object_ids.copy()
        # Map: 0 -> 0 (background), nonzero -> 1 (object)
        semantic_ids[semantic_ids > 0] = 1

    N = len(object_ids)

    # Build per-gaussian records
    gaussians_list = []
    for i in range(N):
        gaussians_list.append({
            "index": int(i),
            "semanticID": int(semantic_ids[i]),
            "objectID": int(object_ids[i]),
        })

    # Build per-object records
    objects_list = []
    for obj_id in sorted(objects_info.keys()):
        info = objects_info[obj_id]
        objects_list.append({
            "id": info["id"],
            "name": info["name"],
            "gaussian_count": info["gaussian_count"],
            "gaussian_indices": info["gaussian_indices"],
            "centroid": _numpy_to_list(info["centroid"]),
            "aabb_min": _numpy_to_list(info["aabb_min"]),
            "aabb_max": _numpy_to_list(info["aabb_max"]),
            "local_to_world": _numpy_to_list(info["local_to_world"]),
        })

    gscene = {
        "version": "1.0",
        "total_gaussians": N,
        "total_objects": len(objects_info),
        "gaussians": gaussians_list,
        "objects": objects_list,
    }

    with open(output_path, 'w') as f:
        json.dump(gscene, f, indent=2)

    print(f"Exported .gscene metadata: {output_path}")
    print(f"  Total gaussians: {N}, Total objects: {len(objects_info)}")


def export_gscene(gaussian_model, object_ids, objects_info, output_dir,
                  scene_name="scene", semantic_ids=None):
    """
    Full .gscene export pipeline.

    Creates:
      output_dir/
        {scene_name}.ply              - Full scene PLY
        objects/
          {object_name}.ply           - Per-object PLYs in local coordinates
        {scene_name}.gscene           - Metadata JSON

    Args:
        gaussian_model: GaussianModel instance.
        object_ids: (N,) numpy array of object IDs.
        objects_info: Dict from compute_object_info.
        output_dir: Output directory path.
        scene_name: Name prefix for output files.
        semantic_ids: Optional (N,) array of semantic class IDs.
    """
    mkdir_p(output_dir)

    # 1. Export full scene PLY
    scene_ply_path = os.path.join(output_dir, f"{scene_name}.ply")
    export_scene_ply(gaussian_model, scene_ply_path)

    # 2. Export per-object PLYs
    objects_dir = os.path.join(output_dir, "objects")
    export_object_ply(gaussian_model, object_ids, objects_info, objects_dir)

    # 3. Export metadata JSON
    gscene_path = os.path.join(output_dir, f"{scene_name}.gscene")
    export_gscene_metadata(object_ids, objects_info, gscene_path, semantic_ids)

    print(f"\n=== .gscene export complete: {output_dir} ===")
