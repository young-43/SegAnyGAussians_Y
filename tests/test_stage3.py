"""
Unit tests for Stage 3: Semantic Objectification & .gscene Export.

These tests validate core logic using synthetic data without requiring
GPU, CUDA, or actual 3DGS scene files.
"""

import os
import sys
import json
import tempfile
import unittest

import numpy as np
import torch

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from object_utils import compute_object_info, compute_local_coordinates
from export_gscene import (
    export_gscene_metadata,
    _numpy_to_list,
)


class TestObjectUtils(unittest.TestCase):
    """Tests for object_utils.py"""

    def setUp(self):
        # Create synthetic Gaussian positions: two clusters
        np.random.seed(42)
        cluster_a = np.random.randn(50, 3) + np.array([10, 0, 0])
        cluster_b = np.random.randn(30, 3) + np.array([0, 10, 0])
        background = np.random.randn(20, 3) * 5
        self.xyz = np.vstack([background, cluster_a, cluster_b]).astype(np.float32)
        self.object_ids = np.array(
            [0] * 20 + [1] * 50 + [2] * 30, dtype=np.int64
        )

    def test_compute_object_info_keys(self):
        info = compute_object_info(self.xyz, self.object_ids)
        self.assertIn(0, info)
        self.assertIn(1, info)
        self.assertIn(2, info)
        self.assertEqual(len(info), 3)

    def test_compute_object_info_counts(self):
        info = compute_object_info(self.xyz, self.object_ids)
        self.assertEqual(info[0]["gaussian_count"], 20)
        self.assertEqual(info[1]["gaussian_count"], 50)
        self.assertEqual(info[2]["gaussian_count"], 30)

    def test_compute_object_info_centroid(self):
        info = compute_object_info(self.xyz, self.object_ids)
        # Cluster A is centered near (10, 0, 0)
        centroid_1 = info[1]["centroid"]
        self.assertAlmostEqual(centroid_1[0], 10.0, delta=1.0)
        # Cluster B is centered near (0, 10, 0)
        centroid_2 = info[2]["centroid"]
        self.assertAlmostEqual(centroid_2[1], 10.0, delta=1.0)

    def test_compute_object_info_aabb(self):
        info = compute_object_info(self.xyz, self.object_ids)
        for obj_id in [0, 1, 2]:
            aabb_min = info[obj_id]["aabb_min"]
            aabb_max = info[obj_id]["aabb_max"]
            self.assertTrue(np.all(aabb_min <= aabb_max))

    def test_compute_object_info_names_default(self):
        info = compute_object_info(self.xyz, self.object_ids)
        self.assertEqual(info[0]["name"], "background")
        self.assertEqual(info[1]["name"], "object_1")
        self.assertEqual(info[2]["name"], "object_2")

    def test_compute_object_info_names_custom(self):
        names = {0: "bg", 1: "Lathe_01", 2: "Robot_Arm"}
        info = compute_object_info(self.xyz, self.object_ids, id_to_name=names)
        self.assertEqual(info[0]["name"], "bg")
        self.assertEqual(info[1]["name"], "Lathe_01")
        self.assertEqual(info[2]["name"], "Robot_Arm")

    def test_compute_object_info_local_to_world(self):
        info = compute_object_info(self.xyz, self.object_ids)
        for obj_id in [0, 1, 2]:
            T = info[obj_id]["local_to_world"]
            self.assertEqual(T.shape, (4, 4))
            # Identity rotation
            np.testing.assert_array_almost_equal(T[:3, :3], np.eye(3))
            # Translation = centroid
            np.testing.assert_array_almost_equal(
                T[:3, 3], info[obj_id]["centroid"]
            )
            self.assertAlmostEqual(T[3, 3], 1.0)

    def test_compute_object_info_indices(self):
        info = compute_object_info(self.xyz, self.object_ids)
        # Check that indices are correct
        for obj_id in [0, 1, 2]:
            indices = info[obj_id]["gaussian_indices"]
            for idx in indices:
                self.assertEqual(self.object_ids[idx], obj_id)

    def test_compute_local_coordinates(self):
        info = compute_object_info(self.xyz, self.object_ids)
        local_xyz = compute_local_coordinates(self.xyz, self.object_ids, info)

        # After subtracting centroid, the mean should be ~0 for each object
        for obj_id in [0, 1, 2]:
            mask = self.object_ids == obj_id
            local_mean = local_xyz[mask].mean(axis=0)
            np.testing.assert_array_almost_equal(local_mean, [0, 0, 0], decimal=5)


class TestExportGsceneMetadata(unittest.TestCase):
    """Tests for export_gscene.py metadata export."""

    def setUp(self):
        np.random.seed(42)
        self.xyz = np.random.randn(100, 3).astype(np.float32)
        self.object_ids = np.array([0] * 40 + [1] * 35 + [2] * 25, dtype=np.int64)
        self.objects_info = compute_object_info(self.xyz, self.object_ids)

    def test_export_gscene_metadata_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.gscene")
            export_gscene_metadata(self.object_ids, self.objects_info, path)
            self.assertTrue(os.path.exists(path))

    def test_export_gscene_metadata_valid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.gscene")
            export_gscene_metadata(self.object_ids, self.objects_info, path)
            with open(path) as f:
                data = json.load(f)
            self.assertIn("version", data)
            self.assertEqual(data["version"], "1.0")

    def test_export_gscene_metadata_counts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.gscene")
            export_gscene_metadata(self.object_ids, self.objects_info, path)
            with open(path) as f:
                data = json.load(f)
            self.assertEqual(data["total_gaussians"], 100)
            self.assertEqual(data["total_objects"], 3)
            self.assertEqual(len(data["gaussians"]), 100)
            self.assertEqual(len(data["objects"]), 3)

    def test_export_gscene_metadata_gaussian_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.gscene")
            export_gscene_metadata(self.object_ids, self.objects_info, path)
            with open(path) as f:
                data = json.load(f)
            g = data["gaussians"][0]
            self.assertIn("index", g)
            self.assertIn("semanticID", g)
            self.assertIn("objectID", g)

    def test_export_gscene_metadata_object_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.gscene")
            export_gscene_metadata(self.object_ids, self.objects_info, path)
            with open(path) as f:
                data = json.load(f)
            obj = data["objects"][0]
            self.assertIn("id", obj)
            self.assertIn("name", obj)
            self.assertIn("gaussian_count", obj)
            self.assertIn("gaussian_indices", obj)
            self.assertIn("centroid", obj)
            self.assertIn("aabb_min", obj)
            self.assertIn("aabb_max", obj)
            self.assertIn("local_to_world", obj)

    def test_export_gscene_metadata_local_to_world_shape(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.gscene")
            export_gscene_metadata(self.object_ids, self.objects_info, path)
            with open(path) as f:
                data = json.load(f)
            for obj in data["objects"]:
                T = obj["local_to_world"]
                self.assertEqual(len(T), 4)
                for row in T:
                    self.assertEqual(len(row), 4)

    def test_export_gscene_metadata_default_semantic_ids(self):
        """When semantic_ids is None, background=0 and objects=1."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.gscene")
            export_gscene_metadata(self.object_ids, self.objects_info, path)
            with open(path) as f:
                data = json.load(f)
            # First 40 are background (object_id=0, semantic=0)
            for g in data["gaussians"][:40]:
                self.assertEqual(g["semanticID"], 0)
            # Next are objects (semantic=1)
            for g in data["gaussians"][40:]:
                self.assertEqual(g["semanticID"], 1)

    def test_export_gscene_metadata_custom_semantic_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.gscene")
            sem_ids = np.array([3] * 40 + [5] * 35 + [7] * 25, dtype=np.int64)
            export_gscene_metadata(
                self.object_ids, self.objects_info, path, semantic_ids=sem_ids
            )
            with open(path) as f:
                data = json.load(f)
            self.assertEqual(data["gaussians"][0]["semanticID"], 3)
            self.assertEqual(data["gaussians"][45]["semanticID"], 5)
            self.assertEqual(data["gaussians"][80]["semanticID"], 7)


class TestNumpyToList(unittest.TestCase):
    """Tests for JSON serialization helper."""

    def test_array(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = _numpy_to_list(arr)
        self.assertEqual(result, [1.0, 2.0, 3.0])

    def test_nested_dict(self):
        d = {"a": np.array([1, 2]), "b": {"c": np.int64(5)}}
        result = _numpy_to_list(d)
        self.assertEqual(result, {"a": [1, 2], "b": {"c": 5}})

    def test_scalar_types(self):
        self.assertEqual(_numpy_to_list(np.int64(42)), 42)
        self.assertAlmostEqual(_numpy_to_list(np.float32(3.14)), 3.14, places=5)

    def test_regular_types_passthrough(self):
        self.assertEqual(_numpy_to_list("hello"), "hello")
        self.assertEqual(_numpy_to_list(42), 42)
        self.assertEqual(_numpy_to_list(3.14), 3.14)


class TestSemanticVotingProjection(unittest.TestCase):
    """Tests for semantic_voting projection logic with mock camera."""

    def test_project_gaussians_basic(self):
        """Test that projection produces valid UV coordinates."""
        from semantic_voting import project_gaussians_to_image

        # Create a simple camera-like object
        class MockCamera:
            def __init__(self):
                self.R = np.eye(3, dtype=np.float32)
                self.T = np.array([0, 0, 0], dtype=np.float32)
                self.FoVx = 1.0  # ~57 degrees
                self.FoVy = 0.75
                self.image_width = 800
                self.image_height = 600

        cam = MockCamera()
        # Points in front of camera at z > 0
        xyz = torch.tensor([
            [0.0, 0.0, 5.0],
            [1.0, 0.0, 5.0],
            [0.0, 1.0, 5.0],
        ], dtype=torch.float32)

        uv, valid = project_gaussians_to_image(xyz, cam)

        self.assertEqual(uv.shape, (3, 2))
        self.assertEqual(valid.shape, (3,))
        # Center point should project near image center
        self.assertTrue(valid[0].item())
        self.assertAlmostEqual(uv[0, 0].item(), 400.0, delta=5.0)
        self.assertAlmostEqual(uv[0, 1].item(), 300.0, delta=5.0)

    def test_project_behind_camera(self):
        """Points behind camera should be marked invalid."""
        from semantic_voting import project_gaussians_to_image

        class MockCamera:
            def __init__(self):
                self.R = np.eye(3, dtype=np.float32)
                self.T = np.array([0, 0, 0], dtype=np.float32)
                self.FoVx = 1.0
                self.FoVy = 0.75
                self.image_width = 800
                self.image_height = 600

        cam = MockCamera()
        xyz = torch.tensor([
            [0.0, 0.0, -5.0],  # Behind camera
        ], dtype=torch.float32)

        uv, valid = project_gaussians_to_image(xyz, cam)
        self.assertFalse(valid[0].item())


if __name__ == '__main__':
    unittest.main()
