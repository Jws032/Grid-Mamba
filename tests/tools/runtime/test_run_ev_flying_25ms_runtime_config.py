#!/usr/bin/env python3
"""Validate the runtime-only EV-Flying Grid Mamba 25-ms training config."""

from __future__ import annotations

import unittest
from typing import Any

import yaml


from tools.experiments.ev_flying import run_ev_flying_ablation as runner


SOURCE_CONFIG = (
    runner.base.REPO_ROOT
    / "experiments"
    / "archive_pending"
    / "diagnostic_runs"
    / "ev_flying"
    / "EF53_50ms"
    / "train_config.yaml"
)

ALLOWED_DIFFS = {
    "DATA.root",
    "EXPERIMENT.checkpoint_selection",
    "EXPERIMENT.expected_updates_per_stream",
    "EXPERIMENT.group",
    "EXPERIMENT.id",
    "EXPERIMENT.name",
    "EXPERIMENT.output_frequency_hz",
    "EXPERIMENT.performance_metrics_status",
    "EXPERIMENT.purpose",
    "EXPERIMENT.source_config_sha256",
    "EXPERIMENT.source_experiment",
    "EXPERIMENT.stream_duration_ms",
    "EXPERIMENT.training_budget_epochs",
    "EXPERIMENT.training_memory_strategy",
    "EXPERIMENT.window_size_ms",
    "GRID_MAMBA.scale_strides",
    "GRID_MAMBA.use_stream_mamba_checkpoint",
    "GRID_MAMBA.window_size",
    "TEST.model_path",
    "TRAIN.epochs",
    "TRAIN.model_save_root",
    "TRAIN.train_window_backward_chunk_size",
}

# 核心模型收口后，生成配置不再携带已删除分支和从未生效的参数。
# 新增的两个 temporal_cell_diffusion 参数只是显式写出模型原有默认值。
MODEL_CORE_NORMALIZATION_DIFFS = {
    "GRID_MAMBA.sparse_conv_hidden_dim",
    "GRID_MAMBA.sparse_conv_mode",
    "GRID_MAMBA.sparse_conv_norm",
    "GRID_MAMBA.sparse_conv_spatial_dilations",
    "GRID_MAMBA.sparse_conv_time_dilations",
    "GRID_MAMBA.sparse_conv_use_se",
    "GRID_MAMBA.spatial_grid_size",
    "GRID_MAMBA.spatial_pool_use_score",
    "GRID_MAMBA.tau_t",
    "GRID_MAMBA.temporal_cell_diffusion_alpha_init",
    "GRID_MAMBA.temporal_cell_diffusion_gate_bias",
    "GRID_MAMBA.temporal_cell_diffusion_source",
    "GRID_MAMBA.temporal_context_diffusion_alpha_init",
    "GRID_MAMBA.temporal_context_diffusion_gate_bias",
    "GRID_MAMBA.temporal_token_diffusion_alpha_init",
    "GRID_MAMBA.temporal_token_diffusion_gate_bias",
    "GRID_MAMBA.time_bin_size",
    "GRID_MAMBA.ts_stream_norm_eps",
    "GRID_MAMBA.ts_stream_norm_min_count",
    "GRID_MAMBA.use_bidirectional_local_mamba",
    "GRID_MAMBA.use_global_density",
    "GRID_MAMBA.use_grid_sequence_conv",
    "GRID_MAMBA.use_grid_sequence_mha",
    "GRID_MAMBA.use_knn_spatial_encoder",
    "GRID_MAMBA.use_sparse_conv_encoder",
    "GRID_MAMBA.use_streaming_ts_embedding",
    "GRID_MAMBA.use_ts_embedding",
    "GRID_MAMBA.use_window",
}


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten(child, path))
        return result
    return {prefix: value}


class Runtime25msConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        runner.base.DEFAULT_CONFIG = runner.DEFAULT_CONFIG
        runner.base.DEFAULT_OUTPUT_ROOT = runner.DEFAULT_OUTPUT_ROOT
        runner.base.FULL_GRID_MAMBA = runner.EV_FLYING_SC4_FULL
        runner.base.EXPERIMENTS = runner.EV_FLYING_EXPERIMENTS
        cls.config = runner.base.build_train_config(
            runner.DEFAULT_CONFIG,
            runner.DEFAULT_OUTPUT_ROOT,
            "EF54_25ms",
            smoke=False,
            smoke_epochs=1,
            smoke_max_events=100000,
            smoke_train_batches=0,
            smoke_val_batches=0,
        )
        with SOURCE_CONFIG.open("r", encoding="utf-8") as handle:
            cls.source_config = yaml.safe_load(handle)

    def test_runtime_identity(self) -> None:
        grid = self.config["GRID_MAMBA"]
        train = self.config["TRAIN"]
        experiment = self.config["EXPERIMENT"]
        self.assertEqual(grid["window_size"], 25.0)
        self.assertEqual(
            grid["scale_strides"],
            [[84.0, 70.0, 25.0], [168.0, 140.0, 25.0], [296.0, 236.0, 25.0]],
        )
        self.assertEqual(train["epochs"], 1)
        self.assertEqual(grid["use_stream_mamba_checkpoint"], True)
        self.assertEqual(train["train_window_backward_chunk_size"], 8)
        self.assertEqual(train["model_save_root"], (
            "experiments/runs/ev_flying/ablation/EF54_25ms"
        ))
        self.assertEqual(experiment["expected_updates_per_stream"], 320)
        self.assertEqual(experiment["output_frequency_hz"], 40.0)
        self.assertEqual(experiment["purpose"], "runtime_only_checkpoint")
        self.assertEqual(
            experiment["performance_metrics_status"],
            "diagnostic_only",
        )

    def test_only_intended_fields_differ_from_locked_50ms_config(self) -> None:
        source = flatten(self.source_config)
        target = flatten(self.config)
        actual_diffs = {
            key
            for key in source.keys() | target.keys()
            if source.get(key, object()) != target.get(key, object())
        }
        self.assertEqual(
            actual_diffs,
            ALLOWED_DIFFS | MODEL_CORE_NORMALIZATION_DIFFS,
        )


if __name__ == "__main__":
    unittest.main()
