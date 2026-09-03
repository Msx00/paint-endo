import unittest
import numpy as np

from reobsdiff.config import ReObsConfig
from reobsdiff.geometry.warp import warp_to_pose
from reobsdiff.geometry.reciprocal import reciprocal_corruption
from reobsdiff.leakage import assert_no_target_view
from reobsdiff.reobs.target_builder import build_reobservation


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.h, self.w = 64, 80
        yy, xx = np.mgrid[:self.h, :self.w]
        self.rgb = np.stack((xx / self.w, yy / self.h, np.ones_like(xx) * .5), -1).astype(np.float32)
        self.depth = np.full((self.h, self.w), 100.0, np.float32)
        self.K = np.array([[80, 0, self.w / 2], [0, 80, self.h / 2], [0, 0, 1]], np.float32)
        self.T = np.eye(4, dtype=np.float32)

    def test_identity_has_negligible_holes(self):
        result = warp_to_pose(self.rgb, self.depth, self.K, self.T, self.K, self.T)
        self.assertLess(1 - result["valid_mask"].mean(), 0.03)

    def test_translation_creates_border_holes(self):
        target = self.T.copy(); target[0, 3] = 5
        result = warp_to_pose(self.rgb, self.depth, self.K, self.T, self.K, target)
        holes = ~result["valid_mask"]
        self.assertGreater(holes.mean(), 0.02)
        self.assertGreater(holes[:, :8].mean() + holes[:, -8:].mean(), holes[:, 8:-8].mean())

    def test_reciprocal_mask_is_geometry_derived(self):
        target = self.T.copy(); target[0, 3] = 5
        result = reciprocal_corruption(self.rgb, self.depth, self.K, self.T, target,
                                       mask_mode="all_geometry", min_component=2, dilation=1)
        self.assertEqual(result["reciprocal_mask"].dtype, np.bool_)
        self.assertTrue(result["reciprocal_mask"].any())

    def test_reobservation_support(self):
        target = self.T.copy(); target[0, 3] = 5
        anchor = warp_to_pose(self.rgb, self.depth, self.K, self.T, self.K, target)
        obs = [{"rgb": self.rgb, "depth": self.depth, "K": self.K, "T": self.T,
                "valid_depth_mask": self.depth > 0, "delta_t": 1, "view": "left"}]
        value = build_reobservation(anchor["valid_mask"], obs, self.K, target, ReObsConfig())
        self.assertEqual(value["reobs_mask"].shape, (self.h, self.w))

    def test_leakage_fails_closed(self):
        with self.assertRaises(RuntimeError):
            assert_no_target_view({"target_image": "/x/endoscope1/L/frame.png"})


if __name__ == "__main__":
    unittest.main()
