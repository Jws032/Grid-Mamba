"""CPU tests for the unified EVUAV Runtime command entry."""

from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import torch

from tools.runtime.evuav_window.evuav_window_size_runtime_common import (
    EXPECTED_VARIANT_IDS,
    load_asset_lock,
)
from tools.runtime.evuav_window.run_evuav_window_size_runtime import (
    run_preflight,
    selected_variant_ids,
)


class UnifiedEntryTests(unittest.TestCase):
    def test_variant_selection_order(self) -> None:
        self.assertEqual(
            tuple(selected_variant_ids("all")),
            EXPECTED_VARIANT_IDS,
        )
        self.assertEqual(tuple(selected_variant_ids("w300")), ("w300",))

    def test_single_variant_preflight_is_cpu_only(self) -> None:
        self.assertFalse(torch.cuda.is_initialized())
        asset_lock = load_asset_lock()
        with tempfile.TemporaryDirectory() as temporary:
            payload = run_preflight(
                asset_lock=asset_lock,
                variant_ids=("w300",),
                output_root=Path(temporary),
            )
            self.assertTrue(payload["preflight_ok"])
            self.assertEqual(payload["variant_count"], 1)
            self.assertEqual(payload["dataset"]["split"], "test")
            self.assertFalse(
                payload["dataset"]["train_or_val_files_read"]
            )
            self.assertFalse(payload["cuda_initialized_after"])
            self.assertFalse(payload["gpu_inference_run"])
            self.assertTrue(
                (Path(temporary) / "w300" / "preflight.json").is_file()
            )
            self.assertTrue(
                (Path(temporary) / "preflight_w300.json").is_file()
            )
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
