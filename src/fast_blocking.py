"""
Optimized candidate generation (replaces cell 4 of notebooks/kaggle_mirror.ipynb).

Key ideas (details and measurements in docs/blocking_optimization.md):

1. BM25 as sparse matrix multiplication.
   bm25s (numpy backend) scores every query by allocating a dense float array of size N_docs,
   filling it with np.add.at and arg-partitioning all N_docs entries. Cost per query is O(N_docs)
   no matter how selective the query is: 883k India S1 queries x 2M-doc index = ~1.8e12 element
   ops per index per modality. Here the index is a CSR matrix W^T (vocab x docs) holding exactly
   the bm25s "lucene" weights idf(t) * tf/(tf + k1*(1-b+b*dl/avgdl)), the queries are a CSR
   term-count matrix Q, and scores = Q @ W^T with per-row top-k done by sparse_dot_topn in C++
   across all cores. Cost per query = number of postings touched, output is O(k) per query,
   and only non-zero scores are returned (bm25s pads every query to k with random score-0 docs).

2. Tokenize once. Every document is tokenized once per modality with a Polars (Rust,
   multi-threaded) regex identical to bm25s' `(?u)\\b\\w\\w+\\b` + the same English stopword list,
   into one shared vocabulary per country. The notebook tokenized S2 twice (index + queries) and
   rebuilt a fresh vocabulary for every 50k-row query chunk (then bm25s converted ids -> strings
   -> index ids again).

3. Integer entity handles. Entities are int64 positions into one concatenated id array; pairs
   are int64 keys a*N+b. The pandas outer merges on Python-string columns are replaced by Polars
   joins on integers; ids are decoded to strings once at the end (optional).

4. Geo blocking in one sort-based pass. The notebook filtered the whole frame once per block
   (`s1_df[s1_df.geo_block_key == block]`, O(N x B)) and ran the pair loops 12/13/23 separately,
   vectorizing every document twice. Here rows are argsorted by block once, each block is
   char-n-gram counted once for all three sources, and per-target TF-IDF weights reproduce
   `TfidfVectorizer(analyzer='char_wb', ngram_range=(2,4)).fit(target)` exactly. Blocks are
   distributed over a fork process pool (the n-gram analyzer is Python, i.e. GIL-bound).
"""
from __future__ import annotations

import logging
import math
import multiprocessing as mp
import os
import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
import polars as pl
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer
from sparse_dot_topn import sp_matmul_topn

try:  # the exact list bm25s.tokenize(stopwords="english") uses
    from bm25s.stopwords import STOPWORDS_EN as _BM25S_STOPWORDS
except Exception:  # pragma: no cover - fallback keeps the module importable without bm25s
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as _BM25S_STOPWORDS

log = logging.getLogger(__name__)

TOKEN_PATTERN = r"\b\w\w+\b"  # Rust regex: \b and \w are Unicode-aware, same as Python (?u)
STOPWORDS = sorted(set(_BM25S_STOPWORDS))


@dataclass
class BlockingConfig:
    top_k: int = 7
    n_threads: int = field(default_factory=lambda: os.cpu_count() or 1)
    # Query S2 against S3 as well (the notebook does). The ground truth only contains S1->S2/S3
    # pairs, so for the leaderboard metric these candidates are pure overhead; set False to
    # skip ~45% of BM25 work and ~1/3 of geo work.
    include_s2_s3: bool = True
    # BM25 parameters (bm25s defaults: method="lucene", k1=1.5, b=0.75)
    k1: float = 1.5
    b: float = 0.75
    # Optional query-side pruning of very common tokens (df / N_target > max_df_ratio). Their
    # idf is ~0.1-1 so they barely change rankings but dominate postings traversed ("ltd",
    # "india", state names). None = exact bm25s scores. Measure recall before enabling.
    max_df_ratio: float | None = None
    query_chunk_rows: int = 250_000
    tokenize_chunk_rows: int = 1_000_000
    # Geo stage
    geo_ngram_range: tuple[int, int] = (2, 4)
    geo_n_jobs: int = field(default_factory=lambda: os.cpu_count() or 1)
    geo_blocks_per_task: int = 64
    # Output
    decode_ids: bool = True


# ---------------------------------------------------------------------------------------------
# Entity table
# ---------------------------------------------------------------------------------------------
@dataclass
class EntityTable:
    """All sources concatenated once. Row position == global integer handle."""
    ids: np.ndarray                 # object/str, entity_id per handle
    source: np.ndarray              # int8: 0=S1, 1=S2, 2=S3
    country: np.ndarray             # int32 codes into country_names, -1 = NaN country
    country_names: list[str]
    text: pl.Series                 # search_document
    phonetic: pl.Series             # phonetic_hash
    geo_key: np.ndarray             # object, geo_block_key

    @classmethod
    def build(cls, sources: Sequence[pd.DataFrame]) -> "EntityTable":
        def col(name: str) -> list[pd.Series]:
            return [s[name] for s in sources]

        def to_pl(series: list[pd.Series]) -> pl.Series:
            return pl.concat([pl.Series(s.fillna("").astype(str).to_numpy(dtype=object), dtype=pl.String)
                              if s.dtype == object else pl.from_pandas(s.fillna("")).cast(pl.String)
                              for s in series])

        ids = np.concatenate([s.to_numpy(dtype=object) for s in col('entity_id')])
        source = np.concatenate([np.full(len(s), i, np.int8) for i, s in enumerate(sources)])
        # notebook: s['country'].str.lower(), NaN rows excluded from the BM25 loop
        low = pd.concat([s.astype(object) for s in col('country')], ignore_index=True).str.lower()
        codes, uniq = pd.factorize(low, sort=True)
        return cls(ids=ids, source=source, country=codes.astype(np.int32), country_names=list(uniq),
                   text=to_pl(col('search_document')), phonetic=to_pl(col('phonetic_hash')),
                   geo_key=pd.concat(col('geo_block_key'), ignore_index=True).astype(object).to_numpy())

    def __len__(self) -> int:
        return len(self.ids)


# ---------------------------------------------------------------------------------------------
# Tokenization -> sparse term-count matrices over a shared vocabulary
# ---------------------------------------------------------------------------------------------
class _Vocab:
    def __init__(self) -> None:
        self.table = pl.DataFrame(schema={"t": pl.String, "tid": pl.UInt32})

    def __len__(self) -> int:
        return self.table.height

    def encode(self, tokens: pl.DataFrame) -> pl.DataFrame:
        # maintain_order: token ids (hence tie-breaking order in top-k) must not depend on thread timing
        new = tokens.select("t").unique(maintain_order=True).join(self.table, on="t", how="anti", maintain_order="left")
        if new.height:
            start = self.table.height
            new = new.with_columns(pl.int_range(start, start + new.height, dtype=pl.UInt32).alias("tid"))
            self.table = pl.concat([self.table, new])
        return tokens.join(self.table, on="t", how="inner")


def _term_counts(texts: pl.Series, vocab: _Vocab, chunk_rows: int) -> sp.csr_matrix:
    """CSR (n_docs x |vocab| at call time) of token counts; bm25s.tokenize semantics."""
    n = len(texts)
    rows, cols = [], []
    for start in range(0, n, chunk_rows):
        part = texts.slice(start, chunk_rows)
        toks = (
            pl.DataFrame({"t": part})
            .with_row_index("d")
            .with_columns(pl.col("t").str.to_lowercase().str.extract_all(TOKEN_PATTERN))
            .explode("t")
            .drop_nulls("t")
            .filter(~pl.col("t").is_in(STOPWORDS))
        )
        enc = vocab.encode(toks)
        rows.append(enc["d"].to_numpy().astype(np.int64) + start)
        cols.append(enc["tid"].to_numpy())
    r = np.concatenate(rows) if rows else np.empty(0, np.int64)
    c = np.concatenate(cols) if cols else np.empty(0, np.uint32)
    m = sp.csr_matrix((np.ones(len(r), np.float32), (r, c.astype(np.int64))), shape=(n, max(len(vocab), 1)))
    m.sum_duplicates()
    return m


def _pad_cols(m: sp.csr_matrix, n_cols: int) -> sp.csr_matrix:
    if m.shape[1] == n_cols:
        return m
    return sp.csr_matrix((m.data, m.indices, m.indptr), shape=(m.shape[0], n_cols))


def _bm25_index(tf: sp.csr_matrix, k1: float, b: float) -> tuple[sp.csr_matrix, np.ndarray]:
    """
    Returns (W^T as CSR vocab x docs, df) with bm25s 'lucene' weights:
      idf = ln(1 + (N - df + .5)/(df + .5)),  w = idf * tf / (tf + k1*(1 - b + b*dl/avgdl))
    dl counts duplicate tokens (len(token_ids) in bm25s); empty docs have dl=0.
    """
    n_docs = tf.shape[0]
    dl = np.asarray(tf.sum(axis=1)).ravel().astype(np.float32)
    avgdl = float(dl.mean()) if n_docs else 1.0
    df = np.bincount(tf.indices, minlength=tf.shape[1]).astype(np.float32)
    idf = np.zeros(tf.shape[1], np.float32)
    nz = df > 0
    idf[nz] = np.log1p((n_docs - df[nz] + 0.5) / (df[nz] + 0.5))
    row_of = np.repeat(np.arange(n_docs), np.diff(tf.indptr))
    t = tf.data
    w = idf[tf.indices] * (t / (t + k1 * ((1 - b) + b * dl[row_of] / max(avgdl, 1e-9))))
    W = sp.csr_matrix((w.astype(np.float32), tf.indices, tf.indptr), shape=tf.shape)
    return W.T.tocsr(), df


def _topk_pairs(Q: sp.csr_matrix, WT: sp.csr_matrix, q_handles: np.ndarray, t_handles: np.ndarray,
                cfg: BlockingConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    out_a, out_b, out_s = [], [], []
    k = min(cfg.top_k, WT.shape[1])
    if k == 0 or Q.shape[0] == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.float32)
    for start in range(0, Q.shape[0], cfg.query_chunk_rows):
        q = Q[start:start + cfg.query_chunk_rows]
        m = sp_matmul_topn(q, WT, top_n=k, n_threads=cfg.n_threads).tocoo()
        keep = m.data > 0
        out_a.append(q_handles[start + m.row[keep]])
        out_b.append(t_handles[m.col[keep]])
        out_s.append(m.data[keep].astype(np.float32))
    return np.concatenate(out_a), np.concatenate(out_b), np.concatenate(out_s)


def _pairs_frame(a: np.ndarray, b: np.ndarray, s: np.ndarray, n: int, col: str) -> pl.DataFrame:
    return pl.DataFrame({"key": a.astype(np.int64) * n + b.astype(np.int64), col: s})


def _scan_modality(ent: EntityTable, texts: pl.Series, groups: dict[int, np.ndarray],
                   col: str, cfg: BlockingConfig) -> pl.DataFrame:
    """One modality for one country. groups: source -> sorted global handles."""
    vocab = _Vocab()
    tf = {s: _term_counts(texts.gather(h), vocab, cfg.tokenize_chunk_rows) for s, h in groups.items() if len(h)}
    V = max(len(vocab), 1)
    tf = {s: _pad_cols(m, V) for s, m in tf.items()}

    jobs = [(0, 1), (0, 2)] + ([(1, 2)] if cfg.include_s2_s3 else [])
    frames = []
    for target in (1, 2):
        if target not in tf or not any(q in tf and t == target for q, t in jobs):
            continue
        WT, df = _bm25_index(tf[target], cfg.k1, cfg.b)
        drop = None
        if cfg.max_df_ratio is not None:
            drop = np.flatnonzero(df > cfg.max_df_ratio * tf[target].shape[0])
        for q, t in jobs:
            if t != target or q not in tf:
                continue
            Q = tf[q]
            if drop is not None and len(drop):
                Q = Q.copy()
                mask = np.isin(Q.indices, drop)
                Q.data[mask] = 0
                Q.eliminate_zeros()
            a, b, s = _topk_pairs(Q, WT, groups[q], groups[t], cfg)
            frames.append(_pairs_frame(a, b, s, len(ent), col))
        del WT
    if not frames:
        return pl.DataFrame(schema={"key": pl.Int64, col: pl.Float32})
    return pl.concat(frames)


def generate_global_candidate_pool(ent: EntityTable, cfg: BlockingConfig) -> pl.DataFrame:
    """Stages 1-3: BM25 text + phonetic per country. Returns key | bm25_text_score | bm25_phonetic_score."""
    log.info("BM25 candidate generation (threads=%d, top_k=%d)", cfg.n_threads, cfg.top_k)
    order = np.lexsort((ent.source, ent.country))          # one sort instead of 3 filters x countries
    cty, src = ent.country[order], ent.source[order]
    bounds = np.flatnonzero(np.diff(cty)) + 1
    per_country = []
    for seg in np.split(np.arange(len(order)), bounds):
        if not len(seg) or cty[seg[0]] < 0:                 # NaN country: excluded (as notebook)
            continue
        h = order[seg]
        groups = {s: np.sort(h[src[seg] == s]) for s in (0, 1, 2)}
        name = ent.country_names[cty[seg[0]]]
        t0 = time.perf_counter()
        text = _scan_modality(ent, ent.text, groups, "bm25_text_score", cfg)
        phon = _scan_modality(ent, ent.phonetic, groups, "bm25_phonetic_score", cfg)
        merged = text.join(phon, on="key", how="full", coalesce=True)
        per_country.append(merged)
        log.info("[%s] S1=%d S2=%d S3=%d -> %d pairs in %.2fs", name.upper(), len(groups[0]),
                 len(groups[1]), len(groups[2]), merged.height, time.perf_counter() - t0)
    if not per_country:
        return pl.DataFrame(schema={"key": pl.Int64, "bm25_text_score": pl.Float32,
                                    "bm25_phonetic_score": pl.Float32})
    return pl.concat(per_country).unique("key", keep="first").with_columns(_fill0(_BM25_COLS))


# ---------------------------------------------------------------------------------------------
# Geo blocking
# ---------------------------------------------------------------------------------------------
_GEO_DOCS: list[str] | None = None  # set only inside pool workers via initializer


def _geo_init(docs: list[str]) -> None:
    global _GEO_DOCS
    _GEO_DOCS = docs


def _l2_rows(m: sp.csr_matrix) -> sp.csr_matrix:
    sq = np.asarray(m.multiply(m).sum(axis=1)).ravel()
    inv = np.zeros_like(sq)
    nz = sq > 0
    inv[nz] = 1.0 / np.sqrt(sq[nz])
    return sp.csr_matrix(sp.diags(inv.astype(np.float32)) @ m)


def _geo_block(docs: list[str], handles: np.ndarray, src: np.ndarray, pairs: Sequence[tuple[int, int]],
               top_k: int, ngram_range: tuple[int, int]) -> list[np.ndarray]:
    """One block, all source pairs. Reproduces TfidfVectorizer(char_wb).fit(target)/transform(query)."""
    cv = CountVectorizer(analyzer="char_wb", ngram_range=ngram_range, dtype=np.float32)
    try:
        C = cv.fit_transform(docs).tocsr()
    except ValueError:  # empty vocabulary
        return []
    out = []
    for q, t in pairs:
        qi, ti = np.flatnonzero(src == q), np.flatnonzero(src == t)
        if not len(qi) or not len(ti):
            continue
        Ct, Cq = C[ti], C[qi]
        n = Ct.shape[0]
        df = np.bincount(Ct.indices, minlength=C.shape[1])
        idf = np.zeros(C.shape[1], np.float32)
        in_vocab = df > 0          # query n-grams unseen in the target are dropped by transform()
        idf[in_vocab] = np.log((1 + n) / (1 + df[in_vocab])) + 1
        D = sp.diags(idf)
        T = _l2_rows(sp.csr_matrix(Ct @ D))
        Qm = _l2_rows(sp.csr_matrix(Cq @ D))
        m = sp_matmul_topn(Qm, T.T.tocsr(), top_n=min(top_k, n), n_threads=1).tocoo()
        keep = m.data > 0
        out.append(np.stack([handles[qi[m.row[keep]]].astype(np.float64),
                             handles[ti[m.col[keep]]].astype(np.float64),
                             m.data[keep].astype(np.float64)]))
    return out


def _geo_task(args: tuple) -> np.ndarray:
    blocks, pairs, top_k, ngram_range = args
    res = []
    for handles, src in blocks:
        docs = [_GEO_DOCS[h] for h in handles]
        res.extend(_geo_block(docs, handles, src, pairs, top_k, ngram_range))
    return np.concatenate(res, axis=1) if res else np.empty((3, 0))


def run_multi_source_geo_blocking(ent: EntityTable, cfg: BlockingConfig) -> pl.DataFrame:
    """Stage 4: local char n-gram TF-IDF inside shared geo blocks. Returns key | geo_ngram_score."""
    t0 = time.perf_counter()
    pairs = [(0, 1), (0, 2)] + ([(1, 2)] if cfg.include_s2_s3 else [])
    valid = ~pd.Series(ent.geo_key).str.endswith("_unknown").to_numpy(dtype=bool)
    codes, _ = pd.factorize(ent.geo_key)
    idx = np.flatnonzero(valid)
    order = idx[np.argsort(codes[idx], kind="stable")]
    oc = codes[order]
    starts = np.r_[0, np.flatnonzero(np.diff(oc)) + 1]
    ends = np.r_[starts[1:], len(order)]

    blocks = []
    for s, e in zip(starts, ends):
        h = order[s:e]
        src = ent.source[h]
        present = set(np.unique(src).tolist())
        if any(q in present and t in present for q, t in pairs):
            blocks.append((h, src))
    log.info("Geo: %d shared blocks", len(blocks))
    if not blocks:
        return pl.DataFrame(schema={"key": pl.Int64, "geo_ngram_score": pl.Float32})

    # Balance tasks: biggest blocks first, round-robin over tasks.
    blocks.sort(key=lambda x: -len(x[0]))
    n_tasks = max(1, math.ceil(len(blocks) / cfg.geo_blocks_per_task))
    tasks = [([], pairs, cfg.top_k, cfg.geo_ngram_range) for _ in range(n_tasks)]
    for i, blk in enumerate(blocks):
        tasks[i % n_tasks][0].append(blk)

    docs = ent.text.to_list()
    if cfg.geo_n_jobs > 1 and n_tasks > 1:
        ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()
        with ctx.Pool(cfg.geo_n_jobs, initializer=_geo_init, initargs=(docs,)) as pool:
            parts = pool.map(_geo_task, tasks, chunksize=1)
    else:
        _geo_init(docs)
        try:
            parts = [_geo_task(t) for t in tasks]
        finally:
            _geo_init(None)  # type: ignore[arg-type]
    del docs
    allp = np.concatenate(parts, axis=1)
    a, b, s = allp[0].astype(np.int64), allp[1].astype(np.int64), allp[2].astype(np.float32)
    out = _pairs_frame(a, b, s, len(ent), "geo_ngram_score").unique("key", keep="first")
    log.info("Geo: %d pairs in %.2fs", out.height, time.perf_counter() - t0)
    return out


# ---------------------------------------------------------------------------------------------
# Orchestration + merge
# ---------------------------------------------------------------------------------------------
_BM25_COLS = ("bm25_text_score", "bm25_phonetic_score")
_SCORE_COLS = _BM25_COLS + ("geo_ngram_score",)


def _fill0(cols: Sequence[str]) -> list[pl.Expr]:
    # Only score columns: a frame-wide fill_null(0.0) would up-cast the int64 pair key to f64.
    return [pl.col(c).fill_null(0.0) for c in cols]


def merge_pools(bm25: pl.DataFrame, geo: pl.DataFrame) -> pl.DataFrame:
    """Outer union of BM25 and geo candidates on the int64 pair key (Polars hash join)."""
    return bm25.join(geo, on="key", how="full", coalesce=True).with_columns(_fill0(_SCORE_COLS))


def build_candidate_pool(s1: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame,
                         cfg: BlockingConfig | None = None) -> pl.DataFrame:
    """
    Full candidate pool: entity_A | entity_B | bm25_text_score | bm25_phonetic_score | geo_ngram_score
    (same schema as `final_master_pool` in the notebook, as a Polars frame; `.to_pandas()` if needed).
    With cfg.decode_ids=False, entity_A/entity_B are int64 handles into EntityTable.ids.
    """
    cfg = cfg or BlockingConfig()
    ent = EntityTable.build([s1, s2, s3])
    bm25 = generate_global_candidate_pool(ent, cfg)
    geo = run_multi_source_geo_blocking(ent, cfg)
    pool = merge_pools(bm25, geo)
    n = len(ent)
    pool = pool.with_columns((pl.col("key") // n).alias("entity_A"), (pl.col("key") % n).alias("entity_B")).drop("key")
    if cfg.decode_ids:
        ids = pl.Series(ent.ids.astype(str))
        pool = pool.with_columns(
            ids.gather(pool["entity_A"]).alias("entity_A"), ids.gather(pool["entity_B"]).alias("entity_B"))
    log.info("Final candidate pool: %d pairs", pool.height)
    return pool.select("entity_A", "entity_B", "bm25_text_score", "bm25_phonetic_score", "geo_ngram_score")
