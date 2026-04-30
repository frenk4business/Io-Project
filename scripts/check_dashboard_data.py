"""
Validate that production dashboard data artifacts are present.

This script intentionally does not run ingest, preprocessing, model training,
or analysis pipelines. It only checks restored/committed dashboard-ready files.
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.getLogger("streamlit.runtime.caching.cache_data_api").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.caching.cache_data_api").disabled = True

from dashboard.data_loader import FILE_SPECS, REQUIRED_DASHBOARD_FILE_KEYS


def main() -> int:
    missing = []
    print("Io dashboard production data check")
    print("=" * 42)
    for key in REQUIRED_DASHBOARD_FILE_KEYS:
        spec = FILE_SPECS[key]
        exists = spec.path.exists()
        status = "PASS" if exists else "FAIL"
        print(f"{status} {key}: {spec.path}")
        print(f"     purpose: {spec.purpose}")
        if not exists:
            print(f"     restore: {spec.restore_policy}")
            if spec.regenerate_command:
                print(f"     regenerate: {spec.regenerate_command}")
            else:
                print("     regenerate: not available; restore from Git or production data artifact")
            missing.append(spec)

    print("=" * 42)
    if missing:
        print(f"FAIL: {len(missing)} required dashboard file(s) missing.")
        for spec in missing:
            print(f"- {spec.key}: {spec.path}")
        return 1

    print("PASS: all required production dashboard files are present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
