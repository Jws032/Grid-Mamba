"""Tests for organized tool path resolution."""

from pathlib import Path
import unittest

from tools._paths import GRID_MAMBA_ROOT, WORKSPACE_ROOT, resolve_recorded_path


class RecordedPathResolutionTests(unittest.TestCase):
    def test_removed_dataset_aliases_resolve_to_shared_roots(self) -> None:
        cases = {
            "dataset/EV-UAV-dataset/test/test_000.npz": (
                WORKSPACE_ROOT / "datasets" / "EV-UAV" / "test" / "test_000.npz"
            ),
            "dataset/Ev-Flying-processed/test/test_0.npz": (
                WORKSPACE_ROOT / "datasets" / "EV-Flying" / "test" / "test_0.npz"
            ),
            "dataset/Ev-Flying/Train": (
                WORKSPACE_ROOT / "datasets" / "EV-Flying-raw" / "Train"
            ),
        }
        for recorded, expected in cases.items():
            with self.subTest(recorded=recorded):
                self.assertEqual(resolve_recorded_path(recorded), expected.resolve())

    def test_removed_full_run_aliases_resolve_to_canonical_baseline(self) -> None:
        expected = (
            GRID_MAMBA_ROOT
            / "experiments"
            / "runs"
            / "evuav"
            / "baseline"
            / "FULL_SC12"
            / "best_loss_seed37.pt"
        ).resolve()
        cases = (
            "save_model/grid_mamba/ablation_window_size/"
            "SC12_GS_G4_FINE_LOW_MID/best_loss_seed37.pt",
            "experiments/runs/evuav/window_size/formal/"
            "SC12_GS_G4_FINE_LOW_MID/best_loss_seed37.pt",
            "experiments/runs/evuav/baseline/"
            "SC12_GS_G4_FINE_LOW_MID/best_loss_seed37.pt",
        )
        for recorded in cases:
            with self.subTest(recorded=recorded):
                self.assertEqual(resolve_recorded_path(recorded), expected)

    def test_renamed_window_paths_resolve_to_short_canonical_directories(self) -> None:
        names = {
            25: "W025",
            50: "W050",
            100: "W100",
            200: "W200",
            300: "W300",
            800: "W800",
            1600: "W1600",
        }
        root = (
            GRID_MAMBA_ROOT
            / "experiments"
            / "runs"
            / "evuav"
            / "window_size"
            / "formal"
        )
        for window_ms, canonical_name in names.items():
            old_name = f"SC12_GS_G4_FINE_LOW_MID_W{window_ms}_FULL"
            expected = (root / canonical_name / "best_iou_seed37.pt").resolve()
            cases = (
                "save_model/grid_mamba/ablation_window_size/"
                f"{old_name}/best_iou_seed37.pt",
                "experiments/runs/evuav/window_size/formal/"
                f"{old_name}/best_iou_seed37.pt",
            )
            for recorded in cases:
                with self.subTest(recorded=recorded):
                    self.assertEqual(resolve_recorded_path(recorded), expected)


if __name__ == "__main__":
    unittest.main()
