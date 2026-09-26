"""
Baseline (notebook, verbatim) vs optimized candidate generation on the same data and seed.

  python -m benchmarks.run_benchmark                              # synthetic, 20k/45k/47k
  python -m benchmarks.run_benchmark --scale 5                    # synthetic x5
  python -m benchmarks.run_benchmark --data-dir data/sample       # real sample (generate_sample.py)
  python -m benchmarks.run_benchmark --impls optimized --opt "include_s2_s3=False,max_df_ratio=0.05"
  python -m benchmarks.run_benchmark --check-parity               # exactness checks vs notebook

Each implementation runs in its own subprocess, so peak RSS (self + children) and CPU time are
measured cleanly. Metrics: per-stage wall time, records/s, peak RAM, CPU utilization,
candidate pairs, candidate recall (S1->S2/S3 ground-truth pairs, like the notebook's
validate_blocking_recall), per-signal recall and reduction ratio.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import resource
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ------------------------------------------------------------------------------------------- data
def load_data(data_dir: str | None, scale: float, seed: int):
    if data_dir and Path(data_dir, "sample_source1.tsv").exists():
        rd = lambda f: pd.read_csv(Path(data_dir, f), sep="\t")  # noqa: E731
        return (rd("sample_source1.tsv"), rd("sample_source2.tsv"), rd("sample_source3.tsv"),
                rd("sample_ground_truth.tsv"), f"real:{data_dir}")
    from benchmarks.synthetic_data import make_synthetic
    s1, s2, s3, gt = make_synthetic(int(20_000 * scale), int(45_000 * scale), int(47_000 * scale), seed=seed)
    return s1, s2, s3, gt, f"synthetic x{scale} seed={seed}"


# ---------------------------------------------------------------------------------------- metrics
def true_pairs(gt: pd.DataFrame, available: set[str]) -> pl.DataFrame:
    g = pl.from_pandas(gt.dropna(subset=["matched_entity_ids"]).astype(str))
    g = (g.with_columns(pl.col("matched_entity_ids").str.split(","))
          .explode("matched_entity_ids")
          .with_columns(pl.col("matched_entity_ids").str.strip_chars(), pl.col("source1_entity_id").str.strip_chars())
          .filter(pl.col("matched_entity_ids") != ""))
    g = g.filter(pl.col("source1_entity_id").is_in(list(available)) & pl.col("matched_entity_ids").is_in(list(available)))
    return _canon(g, "source1_entity_id", "matched_entity_ids").unique()


def _canon(df: pl.DataFrame, a: str, b: str) -> pl.DataFrame:
    return df.select(pl.min_horizontal(a, b).alias("u"), pl.max_horizontal(a, b).alias("v"))


def recall_report(pool: pl.DataFrame, tp: pl.DataFrame, n1: int, n2: int, n3: int) -> dict:
    pool = pool.with_columns(pl.col("entity_A").cast(pl.String), pl.col("entity_B").cast(pl.String))
    canon = pool.select(pl.min_horizontal("entity_A", "entity_B").alias("u"),
                        pl.max_horizontal("entity_A", "entity_B").alias("v"),
                        *[pl.col(c) for c in ("bm25_text_score", "bm25_phonetic_score", "geo_ngram_score")])
    canon = canon.group_by("u", "v").agg(pl.all().max())
    hit = tp.join(canon, on=["u", "v"], how="left")
    found = hit.filter(pl.col("bm25_text_score").is_not_null())
    n_true = tp.height
    max_pairs = n1 * n2 + n1 * n3 + n2 * n3
    s1_pairs = canon.filter(pl.col("u").str.starts_with("S1-") | pl.col("v").str.starts_with("S1-")).height
    rep = {
        "candidate_pairs": canon.height,
        "candidate_pairs_s1_anchored": s1_pairs,
        "true_pairs": n_true,
        "recall": found.height / n_true if n_true else float("nan"),
        "reduction_ratio": 1 - canon.height / max_pairs,
        "pairs_per_true_match": canon.height / max(found.height, 1),
    }
    for c, nm in (("bm25_text_score", "text"), ("bm25_phonetic_score", "phonetic"), ("geo_ngram_score", "geo")):
        rep[f"recall_{nm}_only_signal"] = found.filter(pl.col(c) > 0).height / n_true if n_true else float("nan")
    return rep


class PeakRSS:
    """
    Samples PSS (proportional set size) of this process + children every 50 ms. Plain RSS summed
    over fork() workers counts copy-on-write pages shared with the parent several times.
    """

    def __init__(self) -> None:
        self.peak = 0
        self._stop = threading.Event()

    def _run(self) -> None:
        import psutil
        p = psutil.Process()
        while not self._stop.is_set():
            try:
                procs = [p] + p.children(recursive=True)
                rss = sum(getattr(q.memory_full_info(), "pss", 0) or q.memory_info().rss for q in procs)
                self.peak = max(self.peak, rss)
            except Exception:
                pass
            time.sleep(0.05)

    def __enter__(self):
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join()
        self.self_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def _cpu() -> float:
    s, c = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    return s.ru_utime + s.ru_stime + c.ru_utime + c.ru_stime


# ----------------------------------------------------------------------------------------- runners
def run_impl(impl: str, args) -> dict:
    s1, s2, s3, gt, label = load_data(args.data_dir, args.scale, args.seed)
    n_rec = len(s1) + len(s2) + len(s3)
    stages: dict[str, float] = {}
    cpu0, t_all = _cpu(), time.perf_counter()
    with PeakRSS() as mem:
        if impl == "baseline":
            import io, contextlib
            from benchmarks.baseline import load_baseline
            mod = load_baseline()
            quiet = contextlib.redirect_stderr(io.StringIO())
            with quiet, contextlib.redirect_stdout(io.StringIO()):
                t = time.perf_counter()
                s1n, s2n, s3n = (mod.normalize_pipeline(s) for s in (s1, s2, s3))
                stages["normalize"] = time.perf_counter() - t
                t = time.perf_counter()
                g = mod.generate_global_candidate_pool(s1n, s2n, s3n, top_k=args.top_k, n_threads=4)
                stages["bm25"] = time.perf_counter() - t
                t = time.perf_counter()
                geo = mod.run_multi_source_geo_blocking(s1n, s2n, s3n, top_k=args.top_k)
                stages["geo"] = time.perf_counter() - t
                t = time.perf_counter()
                pool = pl.from_pandas(g).join(pl.from_pandas(geo), on=["entity_A", "entity_B"], how="full",
                                              coalesce=True).fill_null(0.0)
                stages["merge"] = time.perf_counter() - t
        else:
            from src.fast_normalization import normalize_sources
            from src import fast_blocking as fb
            cfg = fb.BlockingConfig(top_k=args.top_k)
            for kv in filter(None, (args.opt or "").split(",")):
                k, v = kv.split("=")
                setattr(cfg, k.strip(), eval(v))
            t = time.perf_counter()
            s1n, s2n, s3n = normalize_sources([s1, s2, s3], joint_thresholds=args.joint_thresholds)
            stages["normalize"] = time.perf_counter() - t
            t = time.perf_counter()
            ent = fb.EntityTable.build([s1n, s2n, s3n])
            bm = fb.generate_global_candidate_pool(ent, cfg)
            stages["bm25"] = time.perf_counter() - t
            t = time.perf_counter()
            geo = fb.run_multi_source_geo_blocking(ent, cfg)
            stages["geo"] = time.perf_counter() - t
            t = time.perf_counter()
            pool = fb.merge_pools(bm, geo)
            n = len(ent)
            ids = pl.Series(ent.ids.astype(str))
            pool = pool.select((pl.col("key") // n).alias("a"), (pl.col("key") % n).alias("b"),
                               "bm25_text_score", "bm25_phonetic_score", "geo_ngram_score")
            pool = pool.with_columns(ids.gather(pool["a"]).alias("entity_A"),
                                     ids.gather(pool["b"]).alias("entity_B")).drop("a", "b")
            stages["merge"] = time.perf_counter() - t
    wall = time.perf_counter() - t_all
    cpu = _cpu() - cpu0
    avail = set(s1.entity_id.astype(str)) | set(s2.entity_id.astype(str)) | set(s3.entity_id.astype(str))
    rep = recall_report(pool, true_pairs(gt, avail), len(s1), len(s2), len(s3))
    ncores = os.cpu_count() or 1
    return {
        "impl": impl, "data": label, "records": n_rec, "sizes": [len(s1), len(s2), len(s3)],
        "stages_s": {k: round(v, 3) for k, v in stages.items()}, "runtime_s": round(wall, 3),
        "normalize_records_per_s": round(n_rec / stages["normalize"]),
        "bm25_queries_per_s": round((len(s1) * 2 + len(s2)) * 2 / stages["bm25"]),
        "peak_ram_mb": round(mem.peak / 2**20), "main_proc_maxrss_mb": round(mem.self_maxrss / 2**20), "cpu_time_s": round(cpu, 2),
        "cpu_utilization": round(cpu / (wall * ncores), 3), "cores": ncores,
        "candidates_per_s": round(rep["candidate_pairs"] / wall), **rep,
    }


def check_parity(args) -> None:
    """Exactness checks: normalization byte-identical; BM25 scores equal where both return a pair."""
    from benchmarks.baseline import load_baseline
    from src.fast_normalization import normalize_pipeline as fast_norm
    from src import fast_blocking as fb
    import contextlib, io, bm25s
    s1, s2, s3, gt, label = load_data(args.data_dir, args.scale, args.seed)
    mod = load_baseline()
    with contextlib.redirect_stdout(io.StringIO()):
        base = [mod.normalize_pipeline(s) for s in (s1, s2, s3)]
    fast = [fast_norm(s) for s in (s1, s2, s3)]
    cols = ["clean_name", "clean_address", "geo_block_key", "phonetic_hash", "search_document"]
    for i, (b, f) in enumerate(zip(base, fast), 1):
        for c in cols:
            diff = int((b[c].astype(str).to_numpy() != f[c].astype(str).to_numpy()).sum())
            print(f"normalize S{i}.{c:16s} mismatches: {diff}")

    # BM25: one country, S1 -> S2, text modality, exact score comparison
    s1n, s2n = base[0], base[1]
    cty = s1n["country"].str.lower().value_counts().index[0]
    q, d = s1n[s1n.country.str.lower() == cty], s2n[s2n.country.str.lower() == cty]
    with contextlib.redirect_stderr(io.StringIO()):
        r = bm25s.BM25(); r.index(bm25s.tokenize(d.search_document.tolist(), show_progress=False), show_progress=False)
        docs, scores = r.retrieve(bm25s.tokenize(q.search_document.tolist(), show_progress=False), k=args.top_k,
                                  show_progress=False)
    ref = {(i, int(j)): float(s) for i, (row_d, row_s) in enumerate(zip(docs, scores)) for j, s in zip(row_d, row_s) if s > 0}
    vocab = fb._Vocab()
    tq = fb._term_counts(pl.Series(q.search_document.tolist()), vocab, 10**9)
    td = fb._term_counts(pl.Series(d.search_document.tolist()), vocab, 10**9)
    V = len(vocab)
    WT, _ = fb._bm25_index(fb._pad_cols(td, V), 1.5, .75)
    m = fb.sp_matmul_topn(fb._pad_cols(tq, V), WT, top_n=args.top_k).tocoo()
    mine = {(int(a), int(b)): float(s) for a, b, s in zip(m.row, m.col, m.data)}
    common = ref.keys() & mine.keys()
    err = max((abs(ref[k] - mine[k]) for k in common), default=0.0)
    print(f"BM25 [{cty}] S1->S2: ref non-zero pairs={len(ref)} optimized={len(mine)} common={len(common)} "
          f"max|score diff|={err:.2e}")
    # Differing pairs must be tie-breaks: per query, the sorted top-k score lists must agree.
    ref_rows = [np.sort(r[r > 0])[::-1] for r in scores]
    csr = m.tocsr()
    bad = 0
    for i, rs in enumerate(ref_rows):
        ms = np.sort(csr.data[csr.indptr[i]:csr.indptr[i + 1]])[::-1]
        if len(ms) != len(rs) or (len(ms) and np.abs(ms - rs).max() > 1e-4):
            bad += 1
    print(f"  pairs only in ref: {len(ref.keys() - mine.keys())}; only in optimized: {len(mine.keys() - ref.keys())}")
    print(f"  queries whose top-k score lists differ: {bad} of {len(ref_rows)} (0 => differences are tie-breaks only)")
    print(f"  baseline zero-score padding pairs dropped: {int((scores == 0).sum())} of {scores.size}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "sample"))
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top-k", type=int, default=7)
    ap.add_argument("--impls", default="baseline,optimized")
    ap.add_argument("--opt", default="", help="BlockingConfig overrides, e.g. include_s2_s3=False")
    ap.add_argument("--joint-thresholds", type=lambda s: s.lower() in ("1", "true", "yes"), default=False)
    ap.add_argument("--check-parity", action="store_true")
    ap.add_argument("--_child", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    if args.check_parity:
        return check_parity(args)
    if args._child:
        print("@@RESULT@@" + json.dumps(run_impl(args._child, args)))
        return

    results = []
    for impl in args.impls.split(","):
        cmd = [sys.executable, "-m", "benchmarks.run_benchmark", "--_child", impl] + \
              [a for a in sys.argv[1:] if not a.startswith("--impls") and a not in args.impls.split(",")]
        out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        line = next((l for l in out.stdout.splitlines() if l.startswith("@@RESULT@@")), None)
        if line is None:
            print(out.stdout[-3000:], out.stderr[-3000:])
            raise SystemExit(f"{impl} failed")
        results.append(json.loads(line[len("@@RESULT@@"):]))
        print(json.dumps(results[-1], indent=1))

    if len(results) == 2:
        b, o = results
        rows = [("Runtime (s)", "runtime_s"), ("  normalize (s)", None), ("  bm25 (s)", None), ("  geo (s)", None),
                ("  merge (s)", None), ("Peak RAM (MB)", "peak_ram_mb"), ("Candidate pairs", "candidate_pairs"),
                ("Recall", "recall"), ("Reduction ratio", "reduction_ratio"), ("CPU utilization", "cpu_utilization"),
                ("Normalize rec/s", "normalize_records_per_s"), ("BM25 queries/s", "bm25_queries_per_s")]
        print("\n| Metric | Baseline | Optimized | Improvement |\n|---|---:|---:|---:|")
        for name, k in rows:
            if k is None:
                st = name.strip().split(" ")[0]
                bv, ov = b["stages_s"][st], o["stages_s"][st]
            else:
                bv, ov = b[k], o[k]
            imp = f"{bv / ov:.1f}x" if isinstance(bv, (int, float)) and ov and k not in ("recall", "reduction_ratio", "cpu_utilization", "normalize_records_per_s", "bm25_queries_per_s") else \
                  (f"{(ov - bv) * 100:+.2f} pp" if k in ("recall", "reduction_ratio", "cpu_utilization") else f"{ov / bv:.1f}x")
            fmt = (lambda v: f"{v:.4%}") if k in ("recall", "reduction_ratio") else (lambda v: f"{v:,}")
            print(f"| {name} | {fmt(bv)} | {fmt(ov)} | {imp} |")
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
