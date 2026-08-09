from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from dataset.preprocessing.fred.filter_by_bbox_area import resolve_source_zip


class FredSourceZipResolutionTests(unittest.TestCase):
    def test_existing_recorded_path_takes_precedence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            recorded = Path(tmp_dir) / "recorded" / "3.zip"
            fallback = Path(tmp_dir) / "FRED" / "train" / "3.zip"
            recorded.parent.mkdir()
            fallback.parent.mkdir(parents=True)
            recorded.touch()
            fallback.touch()

            resolved = resolve_source_zip(
                {
                    "source_zip": str(recorded),
                    "sequence_id": "3",
                    "split": "val",
                    "original_split": "train",
                },
                Path(tmp_dir) / "FRED",
            )

            self.assertEqual(resolved, recorded)

    def test_remote_val_path_falls_back_to_raw_train_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            fred_root = Path(tmp_dir) / "FRED"
            expected = fred_root / "train" / "3.zip"
            expected.parent.mkdir(parents=True)
            expected.touch()

            resolved = resolve_source_zip(
                {
                    "source_zip": "/home/remote/datasets/FRED/train/3.zip",
                    "sequence_id": "3",
                    "split": "val",
                    "original_split": "train",
                },
                fred_root,
            )

            self.assertEqual(resolved, expected)

    def test_missing_source_reports_recorded_and_fallback_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            fred_root = Path(tmp_dir) / "FRED"
            recorded = "/home/remote/datasets/FRED/test/8.zip"

            with self.assertRaises(FileNotFoundError) as context:
                resolve_source_zip(
                    {
                        "source_zip": recorded,
                        "sequence_id": "8",
                        "split": "test",
                    },
                    fred_root,
                )

            message = str(context.exception)
            self.assertIn(recorded, message)
            self.assertIn(str(fred_root / "test" / "8.zip"), message)


if __name__ == "__main__":
    unittest.main()
