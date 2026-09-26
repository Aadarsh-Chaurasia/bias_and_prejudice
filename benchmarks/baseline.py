"""
Loads the baseline implementation *verbatim* from notebooks/kaggle_mirror.ipynb (cells 2 and 4),
so the benchmark always compares against the notebook as the source of truth.
"""
from __future__ import annotations

import json
import types
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "kaggle_mirror.ipynb"


def load_baseline(notebook: Path = NOTEBOOK) -> types.ModuleType:
    nb = json.loads(notebook.read_text())
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    src = "\n\n".join("".join(c["source"]) for c in code
                      if "def normalize_pipeline" in "".join(c["source"])
                      or "def run_country_global_scan" in "".join(c["source"]))
    mod = types.ModuleType("kaggle_mirror_baseline")
    exec(compile(src, str(notebook), "exec"), mod.__dict__)
    return mod


def baseline_pool(mod: types.ModuleType, s1n, s2n, s3n, top_k: int = 7, n_threads: int = 4):
    """Cell 10 of the notebook, unchanged."""
    import polars as pl
    g = mod.generate_global_candidate_pool(s1n, s2n, s3n, top_k=top_k, n_threads=n_threads)
    geo = mod.run_multi_source_geo_blocking(s1n, s2n, s3n, top_k=top_k)
    return pl.from_pandas(g).join(pl.from_pandas(geo), on=['entity_A', 'entity_B'], how='full',
                                  coalesce=True).fill_null(0.0)
