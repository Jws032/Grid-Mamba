#!/usr/bin/env python3
"""Protect the retained experiment registries after project cleanup."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]


class ExperimentEntrypointTests(unittest.TestCase):
    def run_probe(self, source: str) -> None:
        result = subprocess.run(
            [sys.executable, "-c", source],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(
            result.returncode,
            0,
            msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )

    def test_shared_engine_has_no_development_registry(self) -> None:
        self.run_probe(
            "from tools.experiments.core import run_ablation as r; "
            "assert r.EXPERIMENTS == {}; assert r.FULL_GRID_MAMBA == {}"
        )

    def test_ev_flying_registry_only_contains_ef45(self) -> None:
        self.run_probe(
            "from tools.experiments.ev_flying import run_ev_flying_ablation as r; "
            "assert set(r.EV_FLYING_EXPERIMENTS) == {'EF45'}"
        )

    def test_fred_registry_only_contains_formal_experiment(self) -> None:
        self.run_probe(
            "from tools.experiments.fred import run_fred_ablation as r; "
            "assert set(r.EXPERIMENTS) == {'FRED_SC12_GS_SCALED'}; "
            "assert r.DEFAULT_OUTPUT_ROOT == r.REPO_ROOT / 'experiments' / "
            "'runs' / 'fred' / 'ablation'"
        )

    def test_fred_entrypoint_supports_direct_file_execution(self) -> None:
        script = REPO_ROOT / "tools" / "experiments" / "fred" / "run_fred_ablation.py"
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(
            result.returncode,
            0,
            msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )

    def test_fixed50_registry_only_contains_full_sc12(self) -> None:
        self.run_probe(
            "from tools.experiments.evuav import run_fixed50_sc4_sc12 as r; "
            "assert set(r.EXPERIMENTS) == {'SC12_GS_G4_FINE_LOW_MID'}"
        )

    def test_window_registry_only_contains_formal_assets(self) -> None:
        self.run_probe(
            "from tools.experiments.evuav import run_window_size_curve as r; "
            "assert set(r.FORMAL_EXPERIMENTS) == {"
            "'SC12_GS_G4_FINE_LOW_MID_W25_FULL',"
            "'SC12_GS_G4_FINE_LOW_MID_W50_FULL',"
            "'SC12_GS_G4_FINE_LOW_MID_W100_FULL',"
            "'SC12_GS_G4_FINE_LOW_MID_W200_FULL',"
            "'SC12_GS_G4_FINE_LOW_MID_W300_FULL',"
            "'SC12_GS_G4_FINE_LOW_MID_W800_FULL',"
            "'SC12_GS_G4_FINE_LOW_MID_W1600_FULL'}"
        )

    def test_formal_windows_use_short_canonical_directories(self) -> None:
        self.run_probe(
            "from tools.experiments.evuav import run_window_size_curve as r; "
            "expected = {25: 'W025', 50: 'W050', 100: 'W100', "
            "200: 'W200', 300: 'W300', 800: 'W800', 1600: 'W1600'}; "
            "assert all(r.canonical_experiment_dir("
            "r.DEFAULT_OUTPUT_ROOT, r.formal_experiment_id(ms), False"
            ") == r.DEFAULT_OUTPUT_ROOT / name "
            "for ms, name in expected.items()); "
            "assert r.formal_experiment_id(25.0) == "
            "'SC12_GS_G4_FINE_LOW_MID_W25_FULL'"
        )

    def test_hlc2_uses_grouped_canonical_directories(self) -> None:
        self.run_probe(
            "from tools.experiments.evuav import run_hlc2_paper_ablation as r; "
            "assert r.canonical_dir(r.DEFAULT_OUTPUT_ROOT, 'MC01') == "
            "r.DEFAULT_OUTPUT_ROOT / 'MC' / 'MC01'; "
            "assert r.canonical_dir(r.DEFAULT_OUTPUT_ROOT, 'CP04') == "
            "r.DEFAULT_OUTPUT_ROOT / 'CP' / 'CP04'"
        )

    def test_swc_visualization_defaults_to_paper_source(self) -> None:
        self.run_probe(
            "from tools.analysis.evuav import "
            "visualize_evuav_swc_temporal as r; "
            "assert r.DEFAULT_OUTPUT_DIR.name == 'test_020_w09_15'; "
            "assert r.DEFAULT_START_WINDOW == 9; "
            "assert r.DEFAULT_NUM_WINDOWS == 7"
        )


if __name__ == "__main__":
    unittest.main()
