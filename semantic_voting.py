"""
Stage 3 - Step 1: Multi-view voting for 3D Gaussian semantic/object ID assignment.

Projects 3D Gaussian centers onto 2D SAM masks across all training views,
accumulates votes, and assigns final object IDs via argmax.
Includes KNN-based ID smoothing for boundary noise reduction.
"""

import os
import torch
import numpy as np
from tqdm import tqdm
from utils.graphics_utils import getWorld2View2, fov2focal


def project_gaussians_to_image(xyz, camera):
    """
    Project 3D Gaussian centers onto a 2D image plane using camera parameters.

    Args:
        xyz: (N, 3) tensor of 3D Gaussian centers (world coordinates).
        camera: Camera object with R, T, FoVx, FoVy, image_width, image_height.

    Returns:
        uv: (N, 2) tensor of pixel coordinates (u, v).
        valid_mask: (N,) boolean tensor indicating points within image bounds and in front of camera.
    """
    N = xyz.shape[0]
    device = xyz.device

    # Build world-to-camera transform
    W2C = torch.tensor(
        getWorld2View2(camera.R, camera.T), dtype=torch.float32, device=device
    )  # (4, 4)

    # Transform to camera space
    ones = torch.ones(N, 1, dtype=torch.float32, device=device)
    xyz_hom = torch.cat([xyz, ones], dim=1)  # (N, 4)
    xyz_cam = (W2C[:3, :3] @ xyz.T + W2C[:3, 3:4]).T  # (N, 3)

    # Depth check (must be in front of camera)
    depth = xyz_cam[:, 2]
    in_front = depth > 0.01

    # Compute focal lengths from FoV
    fx = fov2focal(camera.FoVx, camera.image_width)
    fy = fov2focal(camera.FoVy, camera.image_height)
    cx = camera.image_width / 2.0
    cy = camera.image_height / 2.0

    # Project to pixel coordinates
    u = (fx * xyz_cam[:, 0] / depth + cx)
    v = (fy * xyz_cam[:, 1] / depth + cy)

    # Bounds check
    in_bounds = (
        (u >= 0) & (u < camera.image_width) &
        (v >= 0) & (v < camera.image_height)
    )

    valid_mask = in_front & in_bounds
    uv = torch.stack([u, v], dim=1)

    return uv, valid_mask


def multi_view_voting(gaussians_xyz, cameras, sam_masks_dir, image_dir,
                      device='cuda', batch_size=100000):
    """
    Assign object IDs to 3D Gaussians via multi-view voting on SAM masks.

    For each training view:
      1. Project Gaussian centers to image plane.
      2. Look up SAM mask ID at projected pixel.
      3. Accumulate per-class vote counts.

    Args:
        gaussians_xyz: (N, 3) tensor of 3D Gaussian centers.
        cameras: List of Camera objects (training views).
        sam_masks_dir: Path to directory containing SAM masks (.pt files).
        image_dir: Path to directory containing source images (for name mapping).
        device: Torch device.
        batch_size: Batch size for projection (memory management).

    Returns:
        object_ids: (N,) numpy array of assigned object IDs per Gaussian.
        vote_counts: (N, max_id+1) numpy array of vote counts.
    """
    N = gaussians_xyz.shape[0]
    xyz = gaussians_xyz.detach().clone().to(device)

    # First pass: determine the maximum number of mask IDs across all views
    print("Scanning SAM masks to determine ID space...")
    max_mask_count = 0
    mask_cache = {}
    for cam in tqdm(cameras, desc="Loading masks"):
        mask_path = os.path.join(
            sam_masks_dir,
            cam.image_name + '.pt'
        )
        if not os.path.exists(mask_path):
            # Try common extensions
            for ext in ['.pt']:
                alt = os.path.join(sam_masks_dir, cam.image_name + ext)
                if os.path.exists(alt):
                    mask_path = alt
                    break

        if os.path.exists(mask_path):
            masks = torch.load(mask_path, map_location='cpu')  # (num_masks, H, W) bool
            mask_cache[cam.image_name] = masks
            max_mask_count = max(max_mask_count, masks.shape[0])
        else:
            print(f"Warning: No mask found for {cam.image_name}")

    # ID 0 = background (no mask matched), IDs 1..max_mask_count = SAM mask IDs
    num_classes = max_mask_count + 1
    print(f"Total mask classes: {num_classes} (0=background, 1-{max_mask_count}=objects)")

    # Voting accumulation
    # Use float32 for accumulation to avoid overflow issues
    vote_counts = torch.zeros(N, num_classes, dtype=torch.float32, device='cpu')

    print("Running multi-view voting...")
    for cam in tqdm(cameras, desc="Voting"):
        if cam.image_name not in mask_cache:
            continue

        masks = mask_cache[cam.image_name]  # (num_masks, H, W)
        num_masks = masks.shape[0]
        H, W = masks.shape[1], masks.shape[2]

        # Build a label map: pixel -> mask_id (1-indexed), 0 for background
        # Later masks overwrite earlier ones (SAM returns sorted by area desc)
        label_map = torch.zeros(H, W, dtype=torch.long)
        for mid in range(num_masks):
            label_map[masks[mid]] = mid + 1  # 1-indexed

        label_map = label_map.to(device)

        # Project gaussians in batches
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            xyz_batch = xyz[start:end]

            uv, valid = project_gaussians_to_image(xyz_batch, cam)

            if valid.sum() == 0:
                continue

            # Get pixel coordinates for valid points
            u_valid = uv[valid, 0].long().clamp(0, W - 1)
            v_valid = uv[valid, 1].long().clamp(0, H - 1)

            # Look up label
            labels = label_map[v_valid, u_valid]  # (num_valid,)

            # Scatter-add votes
            valid_indices = torch.arange(start, end, device=device)[valid]
            for i in range(valid_indices.shape[0]):
                vote_counts[valid_indices[i].cpu(), labels[i].cpu()] += 1.0

    # Assign IDs via argmax
    object_ids = vote_counts.argmax(dim=1).numpy()

    return object_ids, vote_counts.numpy()


def knn_smooth_ids(gaussians_xyz, object_ids, k=16, device='cuda'):
    """
    KNN-based ID smoothing: replace isolated point IDs with the dominant
    ID among their K nearest neighbors.

    Args:
        gaussians_xyz: (N, 3) tensor of Gaussian positions.
        object_ids: (N,) numpy array of object IDs.
        k: Number of nearest neighbors to consider.
        device: Torch device.

    Returns:
        smoothed_ids: (N,) numpy array of smoothed object IDs.
    """
    from sklearn.neighbors import NearestNeighbors
    import numpy as np

    print(f"KNN smoothing with k={k}...")
    xyz_np = gaussians_xyz.detach().cpu().numpy()

    nn = NearestNeighbors(n_neighbors=k, algorithm='auto')
    nn.fit(xyz_np)
    _, indices = nn.kneighbors(xyz_np)  # (N, k)

    # For each point, find the dominant ID among neighbors
    neighbor_ids = object_ids[indices]  # (N, k)
    smoothed_ids = np.zeros(len(object_ids), dtype=object_ids.dtype)

    for i in range(len(object_ids)):
        unique, counts = np.unique(neighbor_ids[i], return_counts=True)
        smoothed_ids[i] = unique[np.argmax(counts)]

    changed = np.sum(smoothed_ids != object_ids)
    print(f"KNN smoothing changed {changed}/{len(object_ids)} point IDs")

    return smoothed_ids
