#!/usr/bin/env python3
"""Entry point: multi-GPU SAP evolution via openevolve_sap.core.scheduler."""
from __future__ import annotations

import multiprocessing as mp
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

# Spawn workers re-execute this file (__name__ == "__main__"); must not import scheduler.
if __name__ == "__main__":
    if mp.current_process().name == "MainProcess":
        from openevolve_sap.core.scheduler import main

        sys.exit(main())
