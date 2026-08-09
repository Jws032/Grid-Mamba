"""Tests for organized tool path resolution."""

from pathlib import Path
import unittest

from tools._paths import WORKSPACE_ROOT, resolve_recorded_path


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


if __name__ == "__main__":
    unittest.main()
