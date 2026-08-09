"""CPU tests for the supplemental W25 EVUAV Runtime entry."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from tools.runtime.evuav_window.evuav_window_size_runtime_common import load_asset_lock
from tools.runtime.evuav_window.run_evuav_w25_runtime import (
    PARENT_ASSET_LOCK,
    resolve_w25_assets,
    run_preflight,
    stream_full_diagnostic,
)


class W25SupplementTests(unittest.TestCase):
    def test_stream_full_diagnostic_is_nonblocking_and_explicit(self) -> None:
        stream = np.asarray([0.1, 0.6, 0.9], dtype=np.float32)
        full = np.asarray([0.1, 0.4, 0.9], dtype=np.float32)
        result = stream_full_diagnostic(stream, full)
        self.assertFalse(result["allclose"])
        self.assertFalse(result["blocking"])
        self.assertAlmostEqual(result["binary_agreement"], 2.0 / 3.0)

    def test_w25_assets_and_cpu_preflight(self) -> None:
        self.assertFalse(torch.cuda.is_initialized())
        parent = load_asset_lock(PARENT_ASSET_LOCK)
        assets, addendum = resolve_w25_assets(parent)
        self.assertEqual(assets.variant.variant_id, "w25")
        self.assertEqual(assets.variant.window_ms, 25.0)
        self.assertEqual(
            assets.variant.schedule["expected_scheduled_updates"],
            320,
        )
        self.assertTrue(addendum["sha256"])
        with tempfile.TemporaryDirectory() as temporary:
            result = run_preflight(parent, Path(temporary))
            self.assertTrue(result["preflight_ok"])
            self.assertFalse(result["cuda_initialized"])
            self.assertFalse(result["gpu_inference_run"])
            self.assertEqual(result["dataset"]["split"], "test")
            self.assertEqual(result["model_probe"]["model_parameters"], 1858037)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
