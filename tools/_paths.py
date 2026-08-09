"""Canonical repository paths shared by reorganized tools."""

from pathlib import Path


GRID_MAMBA_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = GRID_MAMBA_ROOT.parent
EXPERIMENTS_ROOT = GRID_MAMBA_ROOT / "experiments"

_LEGACY_WINDOW_ROOT = Path("save_model/grid_mamba/ablation_window_size")
_CANONICAL_WINDOW_ROOT = (
    EXPERIMENTS_ROOT / "runs" / "evuav" / "window_size" / "formal"
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
            candidate = _CANONICAL_WINDOW_ROOT / relative
        else:
            candidate = GRID_MAMBA_ROOT / recorded
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
