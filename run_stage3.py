"""
Stage 3: End-to-end pipeline for semantic objectification and .gscene export.

Usage:
    python run_stage3.py \
        --model_path ./output/my_scene/ \
        --source_path ./data/my_scene/ \
        --output_dir ./output/my_scene/gscene_export/ \
        --iteration -1 \
        --knn_smooth_k 16 \
        --scene_name my_scene

This script:
  1. Loads a pre-trained 3DGS scene and SAM masks.
  2. Projects 3D Gaussian centers onto 2D SAM masks across all training views (multi-view voting).
  3. Assigns object IDs to each Gaussian via majority vote.
  4. Applies KNN smoothing to reduce boundary noise.
  5. Computes per-object metadata (centroid, AABB, local-to-world transform).
  6. Exports the full .gscene package:
       - Full scene PLY
       - Per-object PLY files (local coordinates)
       - Structured metadata JSON (.gscene)
"""

import os
import sys
import json
import torch
import numpy as np
from argparse import ArgumentParser, Namespace

from arguments import ModelParams, PipelineParams
from scene import Scene, GaussianModel
from semantic_voting import multi_view_voting, knn_smooth_ids
from object_utils import compute_object_info, compute_local_coordinates
from export_gscene import export_gscene


def get_combined_args(parser: ArgumentParser):
    """Load and merge config file args with command-line args."""
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args()

    target_cfg_file = "cfg_args"

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, target_cfg_file)
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except (TypeError, FileNotFoundError):
        print("Config file not found, using command-line args only.")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k, v in vars(args_cmdline).items():
        if v is not None:
            merged_dict[k] = v

    return Namespace(**merged_dict)


def main():
    parser = ArgumentParser(description="Stage 3: Semantic objectification & .gscene export")

    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--iteration", default=-1, type=int,
                        help="Iteration to load (-1 for latest)")
    parser.add_argument("--output_dir", default=None, type=str,
                        help="Output directory for .gscene export (default: model_path/gscene_export)")
    parser.add_argument("--scene_name", default="scene", type=str,
                        help="Name prefix for output files")
    parser.add_argument("--knn_smooth_k", default=16, type=int,
                        help="KNN neighbor count for ID smoothing (0 to disable)")
    parser.add_argument("--id_to_name_json", default=None, type=str,
                        help="Optional JSON file mapping object_id -> name")
    parser.add_argument("--batch_size", default=100000, type=int,
                        help="Batch size for Gaussian projection")

    args = get_combined_args(parser)

    # Setup dataset params
    dataset = model.extract(args)
    dataset.need_features = False
    dataset.need_masks = False
    dataset.allow_principle_point_shift = getattr(args, 'allow_principle_point_shift', False)

    # Load scene
    print("=" * 60)
    print("Stage 3: Semantic Objectification & .gscene Export")
    print("=" * 60)

    scene_gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(
        dataset, scene_gaussians, None,
        load_iteration=args.iteration,
        feature_load_iteration=-1,
        shuffle=False,
        mode='eval',
        target='scene'
    )

    cameras = scene.getTrainCameras()
    print(f"Loaded {len(cameras)} training cameras")

    xyz = scene_gaussians.get_xyz  # (N, 3)
    N = xyz.shape[0]
    print(f"Total Gaussians: {N}")

    # SAM masks directory
    sam_masks_dir = os.path.join(dataset.source_path, 'sam_masks')
    image_dir = os.path.join(dataset.source_path, 'images')

    if not os.path.exists(sam_masks_dir):
        print(f"ERROR: SAM masks not found at {sam_masks_dir}")
        print("Please run extract_segment_everything_masks.py first.")
        sys.exit(1)

    # Step 1: Multi-view voting
    print("\n--- Step 1: Multi-view Voting ---")
    object_ids, vote_counts = multi_view_voting(
        xyz, cameras, sam_masks_dir, image_dir,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        batch_size=args.batch_size
    )
    print(f"Unique object IDs assigned: {len(np.unique(object_ids))}")

    # Step 2: KNN smoothing
    if args.knn_smooth_k > 0:
        print(f"\n--- Step 2: KNN Smoothing (k={args.knn_smooth_k}) ---")
        object_ids = knn_smooth_ids(
            xyz, object_ids, k=args.knn_smooth_k,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        print(f"Unique object IDs after smoothing: {len(np.unique(object_ids))}")
    else:
        print("\n--- Step 2: KNN Smoothing SKIPPED ---")

    # Step 3: Compute object metadata
    print("\n--- Step 3: Computing Object Metadata ---")
    id_to_name = None
    if args.id_to_name_json and os.path.exists(args.id_to_name_json):
        with open(args.id_to_name_json) as f:
            id_to_name = {int(k): v for k, v in json.load(f).items()}
        print(f"Loaded {len(id_to_name)} custom object names")

    xyz_np = xyz.detach().cpu().numpy()
    objects_info = compute_object_info(xyz_np, object_ids, id_to_name)

    for obj_id, info in sorted(objects_info.items()):
        print(f"  Object {obj_id} ({info['name']}): {info['gaussian_count']} gaussians, "
              f"AABB size: {info['aabb_max'] - info['aabb_min']}")

    # Step 4: Export .gscene
    print("\n--- Step 4: Exporting .gscene ---")
    output_dir = args.output_dir or os.path.join(args.model_path, 'gscene_export')
    export_gscene(
        scene_gaussians, object_ids, objects_info, output_dir,
        scene_name=args.scene_name
    )

    # Save vote counts for debugging
    vote_path = os.path.join(output_dir, "vote_counts.npy")
    np.save(vote_path, vote_counts)
    print(f"Saved vote counts: {vote_path}")

    # Save object IDs array
    ids_path = os.path.join(output_dir, "object_ids.npy")
    np.save(ids_path, object_ids)
    print(f"Saved object IDs: {ids_path}")

    print("\n" + "=" * 60)
    print("Stage 3 complete!")
    print(f"Output directory: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
