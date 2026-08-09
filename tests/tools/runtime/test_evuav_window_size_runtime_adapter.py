"""CPU/static tests for the Grid Mamba EVUAV Runtime adapter."""

from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np
import torch

from tools.runtime.evuav_window.evuav_window_size_runtime_adapter import (
    adapter_lock_payload,
    compare_stream_and_full,
    load_evuav_sample_cpu,
    make_window_points,
    resolve_locked_variant_assets,
)
from tools.runtime.evuav_window.evuav_window_size_runtime_common import (
    EXPECTED_VARIANT_IDS,
    RuntimeProtocolError,
    fixed_window_ranges,
    load_asset_lock,
)


class LockedVariantAdapterTests(unittest.TestCase):
    def test_all_variant_configs_match_the_asset_lock(self) -> None:
        self.assertFalse(torch.cuda.is_initialized())
        asset_lock = load_asset_lock()
        for variant_id in EXPECTED_VARIANT_IDS:
            with self.subTest(variant_id=variant_id):
                assets = resolve_locked_variant_assets(
                    asset_lock,
                    variant_id,
                )
                self.assertEqual(assets.variant.variant_id, variant_id)
                self.assertEqual(
                    float(assets.cfg.window_size),
                    assets.variant.window_ms,
                )
                adapter = adapter_lock_payload(assets)
                self.assertEqual(adapter["precision"], "fp32")
                self.assertFalse(adapter["tf32"])
                self.assertFalse(adapter["event_downsampling"])
        self.assertFalse(torch.cuda.is_initialized())


class EVUAVSampleAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.asset_lock = load_asset_lock()
        cls.loaded = load_evuav_sample_cpu(cls.asset_lock.samples[0])

    def test_locked_test_sample_uses_two_timestamp_sources(self) -> None:
        loaded = self.loaded
        self.assertEqual(
            loaded.model_locations.shape,
            (loaded.identity.num_events, 3),
        )
        self.assertEqual(
            loaded.response_t_ms.shape,
            (loaded.identity.num_events,),
        )
        np.testing.assert_array_equal(
            loaded.model_locations[:, 2].astype(np.int64),
            np.floor(loaded.response_t_ms).astype(np.int64),
        )
        self.assertGreater(
            int(np.count_nonzero(
                loaded.model_locations[:, 2]
                != loaded.response_t_ms
            )),
            0,
        )
        self.assertEqual(loaded.labels.dtype, np.uint8)
        self.assertFalse(torch.cuda.is_initialized())

    def test_window_representation_is_float32_and_complete(self) -> None:
        loaded = self.loaded
        windows = fixed_window_ranges(
            loaded.response_t_ms,
            window_ms=300,
        )
        points = [make_window_points(loaded, window) for window in windows]
        self.assertEqual(len(points), 27)
        self.assertEqual(sum(point.shape[0] for point in points), loaded.identity.num_events)
        self.assertTrue(all(point.dtype == np.float32 for point in points))
        self.assertEqual(windows[-1].duration_ms, 200.0)

    def test_train_or_val_sample_identity_is_rejected_before_read(self) -> None:
        forbidden = replace(
            self.loaded.identity,
            relative_path="dataset/EV-UAV-dataset/val/test_000.npz",
        )
        with self.assertRaisesRegex(RuntimeProtocolError, "EVUAV test"):
            load_evuav_sample_cpu(forbidden, verify_sha256=False)

    def test_stream_full_comparison_contract(self) -> None:
        probability = np.asarray([0.1, 0.2, 0.9], dtype=np.float32)
        result = compare_stream_and_full(probability, probability.copy())
        self.assertTrue(result["allclose"])
        changed = probability.copy()
        changed[1] = 0.5
        with self.assertRaises(RuntimeProtocolError):
            compare_stream_and_full(probability, changed)


if __name__ == "__main__":
    unittest.main()
