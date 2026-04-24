"""
analysis/
---------
Scientific Analysis layer for the Io hotspot project.

Modules in this package implement auditable statistical analyses of the
observed hotspot catalogue and its association with physical-proxy and
geological covariates. Each module:

  - Takes DataFrames as input (no disk I/O — orchestration lives in scripts/)
  - Returns numeric results as a dict or DataFrame
  - Persists artifacts to data/results/ when run through the entry-point
  - Uses neutral scientific phrasing — no "AI predicts" / "model discovers"

The analysis layer is additive to the existing Exploration/Teaching layer.
It exists so that any claim surfaced by the dashboard is auditable, reproducible,
and honest about the project's known limitations.

See docs/scientific_methods.md for the full methodology write-up.
"""

from __future__ import annotations
