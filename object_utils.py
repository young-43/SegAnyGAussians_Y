"""
Stage 3 - Step 2: Object encapsulation utilities.

For each segmented object:
  - Compute geometric centroid
  - Build local coordinate system (translate to local origin)
  - Compute AABB bounding box
  - Build localToWorld 4x4 transform matrix
"""

import numpy as np


def compute_object_info(gaussians_xyz_np, object_ids, id_to_name=None):
    """
    Compute per-object metadata: centroid, local coordinates, AABB, and transform.

    Args:
        gaussians_xyz_np: (N, 3) numpy array of Gaussian positions (world space).
        object_ids: (N,) numpy array of integer object IDs.
        id_to_name: Optional dict mapping object_id -> human-readable name.
                    If None, names are auto-generated as "object_{id}".

    Returns:
        objects_info: dict mapping object_id -> {
            "id": int,
            "name": str,
            "centroid": (3,) array,
            "aabb_min": (3,) array,
            "aabb_max": (3,) array,
            "local_to_world": (4, 4) array,
            "gaussian_indices": list of int,
            "gaussian_count": int,
        }
    """
    unique_ids = np.unique(object_ids)
    objects_info = {}

    for obj_id in unique_ids:
        obj_id = int(obj_id)
        mask = object_ids == obj_id
        indices = np.where(mask)[0]
        points = gaussians_xyz_np[mask]  # (M, 3)

        # Compute centroid
        centroid = points.mean(axis=0)  # (3,)

        # Compute AABB
        aabb_min = points.min(axis=0)
        aabb_max = points.max(axis=0)

        # Build localToWorld transform (translation only, identity rotation)
        local_to_world = np.eye(4, dtype=np.float64)
        local_to_world[:3, 3] = centroid

        # Generate name
        if id_to_name and obj_id in id_to_name:
            name = id_to_name[obj_id]
        else:
            if obj_id == 0:
                name = "background"
            else:
                name = f"object_{obj_id}"

        objects_info[obj_id] = {
            "id": obj_id,
            "name": name,
            "centroid": centroid,
            "aabb_min": aabb_min,
            "aabb_max": aabb_max,
            "local_to_world": local_to_world,
            "gaussian_indices": indices.tolist(),
            "gaussian_count": len(indices),
        }

    return objects_info


def compute_local_coordinates(gaussians_xyz_np, object_ids, objects_info):
    """
    Compute local coordinates for each Gaussian by subtracting its object's centroid.

    Args:
        gaussians_xyz_np: (N, 3) numpy array of world positions.
        object_ids: (N,) numpy array of object IDs.
        objects_info: Dict from compute_object_info.

    Returns:
        local_xyz: (N, 3) numpy array of local coordinates.
    """
    local_xyz = np.copy(gaussians_xyz_np)
    for obj_id, info in objects_info.items():
        mask = object_ids == obj_id
        local_xyz[mask] -= info["centroid"]
    return local_xyz
