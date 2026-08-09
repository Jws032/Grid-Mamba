"""Canonical repository paths shared by reorganized tools."""

from pathlib import Path


GRID_MAMBA_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = GRID_MAMBA_ROOT.parent
EXPERIMENTS_ROOT = GRID_MAMBA_ROOT / "experiments"

_LEGACY_WINDOW_ROOT = Path("save_model/grid_mamba/ablation_window_size")
_CANONICAL_WINDOW_ROOT = (
    EXPERIMENTS_ROOT / "runs" / "evuav" / "window_size" / "formal"
)
_CANONICAL_EVUAV_FULL_RUN = (
    EXPERIMENTS_ROOT / "runs" / "evuav" / "baseline" / "FULL_SC12"
)
_EVUAV_FULL_EXPERIMENT_ID = "SC12_GS_G4_FINE_LOW_MID"
_RENAMED_WINDOW_RUNS = {
    "SC12_GS_G4_FINE_LOW_MID_W25_FULL": "W025",
    "SC12_GS_G4_FINE_LOW_MID_W50_FULL": "W050",
    "SC12_GS_G4_FINE_LOW_MID_W100_FULL": "W100",
    "SC12_GS_G4_FINE_LOW_MID_W200_FULL": "W200",
    "SC12_GS_G4_FINE_LOW_MID_W300_FULL": "W300",
    "SC12_GS_G4_FINE_LOW_MID_W800_FULL": "W800",
    "SC12_GS_G4_FINE_LOW_MID_W1600_FULL": "W1600",
}
_REMOVED_FULL_ALIASES = (
    _CANONICAL_WINDOW_ROOT / _EVUAV_FULL_EXPERIMENT_ID,
    EXPERIMENTS_ROOT
    / "runs"
    / "evuav"
    / "baseline"
    / _EVUAV_FULL_EXPERIMENT_ID,
)
_LEGACY_DATASET_ROOTS = {
    Path("dataset/EV-UAV-dataset"): WORKSPACE_ROOT / "datasets" / "EV-UAV",
    Path("dataset/Ev-Flying-processed"): WORKSPACE_ROOT
    / "datasets"
    / "EV-Flying",
    Path("dataset/Ev-Flying"): WORKSPACE_ROOT
    / "datasets"
    / "EV-Flying-raw",
}


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def resolve_recorded_path(recorded_path: str) -> Path:
    """Resolve a frozen manifest path against the organized workspace.

    Historical manifests remain byte-for-byte evidence of the original run.
    This resolver maps removed experiment and dataset roots without recreating
    compatibility links. Frozen manifests can therefore keep their original
    paths while active code reads the canonical organized assets.
    """

    recorded = Path(recorded_path)
    if not recorded.parts:
        raise ValueError("empty recorded path")
    if not recorded.is_absolute():
        if _is_within(recorded, _LEGACY_WINDOW_ROOT):
            relative = recorded.relative_to(_LEGACY_WINDOW_ROOT)
            if relative.parts and relative.parts[0] == _EVUAV_FULL_EXPERIMENT_ID:
                candidate = _CANONICAL_EVUAV_FULL_RUN.joinpath(*relative.parts[1:])
            elif relative.parts and relative.parts[0] in _RENAMED_WINDOW_RUNS:
                candidate = _CANONICAL_WINDOW_ROOT.joinpath(
                    _RENAMED_WINDOW_RUNS[relative.parts[0]],
                    *relative.parts[1:],
                )
            else:
                candidate = _CANONICAL_WINDOW_ROOT / relative
        else:
            candidate = GRID_MAMBA_ROOT / recorded
            for removed_alias in _REMOVED_FULL_ALIASES:
                if _is_within(candidate, removed_alias):
                    relative = candidate.relative_to(removed_alias)
                    candidate = _CANONICAL_EVUAV_FULL_RUN / relative
                    break
            for old_name, canonical_name in _RENAMED_WINDOW_RUNS.items():
                removed_alias = _CANONICAL_WINDOW_ROOT / old_name
                if _is_within(candidate, removed_alias):
                    relative = candidate.relative_to(removed_alias)
                    candidate = (
                        _CANONICAL_WINDOW_ROOT / canonical_name / relative
                    )
                    break
            for legacy_root, canonical_root in _LEGACY_DATASET_ROOTS.items():
                if _is_within(recorded, legacy_root):
                    relative = recorded.relative_to(legacy_root)
                    candidate = canonical_root / relative
                    break
    else:
        candidate = recorded

    resolved = candidate.resolve()
    allowed_roots = (
        GRID_MAMBA_ROOT.resolve(),
        (WORKSPACE_ROOT / "datasets").resolve(),
    )
    if not any(_is_within(resolved, root) for root in allowed_roots):
        raise ValueError(f"recorded path leaves allowed roots: {recorded_path}")
    return resolved
