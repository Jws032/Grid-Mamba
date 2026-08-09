#!/usr/bin/env python3
"""Repository-local entry for the Grid Mamba FRED Runtime adapter."""

from __future__ import annotations

import sys
from tools._paths import WORKSPACE_ROOT

WORKSPACE_TOOLS = WORKSPACE_ROOT / "tools"
if str(WORKSPACE_TOOLS) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_TOOLS))

from eval_grid_mamba_fred_event_count_runtime import main


if __name__ == "__main__":
    main()
