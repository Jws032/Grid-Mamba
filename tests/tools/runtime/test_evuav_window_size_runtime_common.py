"""CPU-only tests for the EVUAV window-size Runtime protocol."""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.runtime.evuav_window.evuav_window_size_runtime_common import (
    EXPECTED_TEST_EVENTS,
    EXPECTED_TEST_SAMPLES,
    EXPECTED_VARIANT_IDS,
    PROTOCOL_ID,
    RuntimeExperimentStore,
    RuntimeProtocolError,
    RuntimeSample,
    SampleInference,
    fixed_window_ranges,
    load_asset_lock,
    schedule_specification,
    threshold_metrics,
    update_timings_from_windows,
    virtual_response_statistics,
)


class AssetLockTests(unittest.TestCase):
    def test_real_asset_lock_is_complete(self) -> None:
        lock = load_asset_lock()
        self.assertEqual(
            tuple(variant.variant_id for variant in lock.variants),
            EXPECTED_VARIANT_IDS,
        )
        self.assertEqual(len(lock.samples), EXPECTED_TEST_SAMPLES)
        self.assertEqual(
            sum(sample.num_events for sample in lock.samples),
            EXPECTED_TEST_EVENTS,
        )


class FixedWindowTests(unittest.TestCase):
    def test_all_locked_window_schedules(self) -> None:
        expected_counts = {
            50: 160,
            100: 80,
            200: 40,
            300: 27,
            400: 20,
            800: 10,
            1600: 5,
        }
        for window_ms, update_count in expected_counts.items():
            with self.subTest(window_ms=window_ms):
                specification = schedule_specification(window_ms)
                self.assertEqual(
                    specification["expected_scheduled_updates"],
                    update_count,
                )
                expected_final = 200.0 if window_ms == 300 else float(window_ms)
                self.assertEqual(
                    specification["final_window_duration_ms"],
                    expected_final,
                )
                self.assertEqual(
                    specification["has_partial_final_window"],
                    window_ms == 300,
                )

    def test_w300_boundary_partial_window_and_empty_ticks(self) -> None:
        times = np.asarray(
            [0.0, 299.999, 300.0, 610.0, 7_799.999],
            dtype=np.float64,
        )
        windows = fixed_window_ranges(times, window_ms=300)
        self.assertEqual(len(windows), 27)
        self.assertEqual((windows[0].event_start, windows[0].event_end), (0, 2))
        self.assertEqual((windows[1].event_start, windows[1].event_end), (2, 3))
        self.assertEqual((windows[2].event_start, windows[2].event_end), (3, 4))
        self.assertTrue(windows[3].is_empty)
        self.assertEqual(windows[-1].duration_ms, 200.0)
        self.assertTrue(windows[-1].is_empty)
        self.assertEqual(sum(window.input_events for window in windows), len(times))

    def test_empty_middle_windows_are_scheduled(self) -> None:
        times = np.asarray([10.0, 610.0], dtype=np.float64)
        windows = fixed_window_ranges(times, window_ms=100)
        self.assertEqual(len(windows), 80)
        self.assertEqual(sum(not window.is_empty for window in windows), 2)
        self.assertEqual(sum(window.is_empty for window in windows), 78)


class SchedulerAndMetricTests(unittest.TestCase):
    def test_serial_virtual_scheduler_without_backlog(self) -> None:
        times = np.asarray([10.0, 110.0], dtype=np.float64)
        windows = fixed_window_ranges(times, window_ms=100)
        updates = update_timings_from_windows(
            windows,
            processing_ms=[10.0] * len(windows),
            forward_ms=[5.0] * len(windows),
            event_t_ms=times,
        )
        runtime, trace = virtual_response_statistics(
            times,
            updates,
            sample_processing_ms=800.0,
            window_ms=100,
        )
        self.assertAlmostEqual(trace[0]["virtual_completion_ms"], 110.0)
        self.assertAlmostEqual(trace[1]["virtual_completion_ms"], 210.0)
        self.assertAlmostEqual(runtime["response_mean_ms"], 100.0)
        self.assertAlmostEqual(runtime["queue_mean_ms"], 0.0)
        self.assertAlmostEqual(runtime["real_time_factor"], 0.1)
        self.assertAlmostEqual(runtime["real_time_speedup"], 10.0)

    def test_serial_virtual_scheduler_with_backlog(self) -> None:
        times = np.asarray([0.0, 100.0], dtype=np.float64)
        windows = fixed_window_ranges(times, window_ms=100)
        processing = [150.0, 150.0] + [0.0] * (len(windows) - 2)
        updates = update_timings_from_windows(
            windows,
            processing_ms=processing,
            forward_ms=processing,
            event_t_ms=times,
        )
        runtime, trace = virtual_response_statistics(
            times,
            updates,
            sample_processing_ms=300.0,
            window_ms=100,
        )
        self.assertAlmostEqual(trace[0]["virtual_completion_ms"], 250.0)
        self.assertAlmostEqual(trace[1]["virtual_queue_ms"], 50.0)
        self.assertAlmostEqual(trace[1]["virtual_completion_ms"], 400.0)
        self.assertAlmostEqual(runtime["response_mean_ms"], 275.0)
        self.assertAlmostEqual(runtime["queue_mean_ms"], 25.0)

    def test_threshold_curve_and_smallest_threshold_tie(self) -> None:
        metrics = threshold_metrics(
            np.asarray([0, 1], dtype=np.uint8),
            np.asarray([0.1, 0.9], dtype=np.float64),
        )
        self.assertEqual(len(metrics["IoU"]), 101)
        best = int(np.argmax(np.asarray(metrics["IoU"])))
        self.assertEqual(best, 11)
        self.assertEqual(metrics["IoU"][best], 1.0)
        self.assertEqual(metrics["Fa"][best], 0.0)


class ExperimentStoreTests(unittest.TestCase):
    @staticmethod
    def _sample() -> RuntimeSample:
        return RuntimeSample(
            sample_index=0,
            file_name="synthetic.npz",
            relative_path="dataset/EV-UAV-dataset/test/synthetic.npz",
            source_sha256="a" * 64,
            size_bytes=128,
            num_events=4,
            t_min_ms=10.0,
            t_max_ms=7_999.0,
        ).validated()

    @staticmethod
    def _lock(sample: RuntimeSample) -> dict:
        return {
            "protocol_id": PROTOCOL_ID,
            "mode": "smoke",
            "variant_id": "w400",
            "window_ms": 400.0,
            "schedule": schedule_specification(400),
            "selected_samples": [
                {
                    "sample_index": sample.sample_index,
                    "relative_path": sample.relative_path,
                    "sha256": sample.source_sha256,
                    "num_events": sample.num_events,
                }
            ],
        }

    def test_atomic_resume_trace_and_summary(self) -> None:
        sample = self._sample()
        times = np.asarray([10.0, 410.0, 410.0, 7_999.0])
        labels = np.asarray([0, 1, 0, 1], dtype=np.uint8)
        probabilities = np.asarray([0.1, 0.9, 0.2, 0.8])
        windows = fixed_window_ranges(times, window_ms=400)
        updates = update_timings_from_windows(
            windows,
            processing_ms=[1.0] * len(windows),
            representation_ms=[0.1] * len(windows),
            h2d_ms=[0.1] * len(windows),
            forward_ms=[0.5] * len(windows),
            post_d2h_ms=[0.2] * len(windows),
            event_t_ms=times,
        )
        inference = SampleInference(
            processing_ms=20.0,
            peak_cuda_memory_mb=12.5,
            updates=updates,
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            store = RuntimeExperimentStore(
                output,
                self._lock(sample),
                (sample,),
                window_ms=400,
            )
            result_path = store.add_sample(
                sample,
                labels=labels,
                probabilities=probabilities,
                event_t_ms=times,
                inference=inference,
            )
            self.assertTrue(result_path.is_file())
            self.assertTrue(store.completed(sample))
            with gzip.open(
                store.trace_path(sample),
                "rt",
                encoding="utf-8",
            ) as handle:
                trace_rows = list(csv.DictReader(handle))
            self.assertEqual(len(trace_rows), 20)
            self.assertEqual(
                sum(row["is_empty"] == "True" for row in trace_rows),
                17,
            )

            resumed = RuntimeExperimentStore(
                output,
                self._lock(sample),
                (sample,),
                window_ms=400,
            )
            self.assertTrue(resumed.completed(sample))
            summary = resumed.finalize()
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["num_samples"], 1)
            self.assertEqual(summary["num_events"], 4)
            self.assertEqual(
                summary["target_test_oracle_best_iou"]["threshold"],
                0.21,
            )
            with (output / "threshold_curve.csv").open(
                "r",
                encoding="utf-8",
            ) as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 101)
            self.assertFalse((output / "predictions.txt").exists())
            self.assertFalse(list(output.rglob("*.npy")))
            self.assertFalse(list(output.rglob("*.npz")))

    def test_lock_mismatch_is_rejected(self) -> None:
        sample = self._sample()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            RuntimeExperimentStore(
                output,
                self._lock(sample),
                (sample,),
                window_ms=400,
            )
            changed = self._lock(sample)
            changed["variant_id"] = "w800"
            with self.assertRaises(RuntimeProtocolError):
                RuntimeExperimentStore(
                    output,
                    changed,
                    (sample,),
                    window_ms=400,
                )

    def test_wrong_update_count_is_rejected(self) -> None:
        sample = self._sample()
        times = np.asarray([10.0, 410.0, 410.0, 7_999.0])
        windows = fixed_window_ranges(times, window_ms=400)
        updates = update_timings_from_windows(
            windows,
            processing_ms=[1.0] * len(windows),
        )[:-1]
        inference = SampleInference(
            processing_ms=19.0,
            peak_cuda_memory_mb=0.0,
            updates=updates,
        )
        with tempfile.TemporaryDirectory() as temporary:
            store = RuntimeExperimentStore(
                temporary,
                self._lock(sample),
                (sample,),
                window_ms=400,
            )
            with self.assertRaises(RuntimeProtocolError):
                store.add_sample(
                    sample,
                    labels=np.asarray([0, 1, 0, 1]),
                    probabilities=np.asarray([0.1, 0.9, 0.2, 0.8]),
                    event_t_ms=times,
                    inference=inference,
                )


if __name__ == "__main__":
    unittest.main()
