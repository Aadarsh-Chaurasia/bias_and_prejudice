"""
Optimized drop-in replacement for `normalize_pipeline` (notebooks/kaggle_mirror.ipynb, cell 2).

Output columns and values are identical to the notebook version (verified by
benchmarks/run_benchmark.py --check-parity), except where an option explicitly says otherwise.

What changed vs. the notebook and why:
  * One pass per row instead of ~15 full-column passes (2x unidecode list-comps, 13 pandas
    `.str.replace` calls, 1 geo loop, 1 row-wise `DataFrame.apply`, 3 threshold list-comps).
  * `unidecode` is skipped for ASCII strings (`str.isascii()` is a C-level scan); it was called
    3x per address in the notebook (clean_address + once per comma chunk).
  * All regexes are pre-compiled at import time; the 9 suffix regexes only run when a cheap
    substring trigger is present.
  * Geo chunks are scanned right-to-left and the scan stops after 2 valid chunks (only
    chunk_1 / chunk_2 are ever used).
  * `geo_df.apply(assign_base_key, axis=1)` (builds a pandas Series per row) is replaced by
    plain tuple logic inside the same row pass.
  * Threshold passes use `pd.factorize` + `np.bincount` instead of `value_counts` + Python
    `re.search` on every key.
  * Row chunks are processed in a fork-based process pool (pure-Python work -> GIL-bound, so
    threads would not help). No Polars / BLAS threads are alive inside the workers.
  * `df.copy()` of the raw frame is gone; strings are stored as pyarrow strings (~3-5x less RAM
    than Python objects).
  * `joint_thresholds=True` (via `normalize_sources`) computes block-size thresholds on the union
    of S1+S2+S3 so the same address gets the same `geo_block_key` in every source. The notebook
    computes thresholds per source, so a city split into `<key>_<letter>` in S2 (5M rows) can stay
    unsplit in S1 (2.2M rows) and the two blocks never meet in geo blocking.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import re
from typing import Iterable, Sequence

import jellyfish
import numpy as np
import pandas as pd
from unidecode import unidecode

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------------------------
# Constants (identical to the notebook)
# ---------------------------------------------------------------------------------------------
SUFFIX_MAP = {
    r'\bprivate limited\b': 'ltd', r'\bpvt ltd\b': 'ltd', r'\bpvt\.?\s*ltd\.?\b': 'ltd',
    r'\bcorporation\b': 'corp', r'\bincorporated\b': 'inc', r'\bstreet\b': 'st',
    r'\broad\b': 'rd', r'\bavenue\b': 'ave', r'\bboulevard\b': 'blvd'
}
COMPILED_SUFFIXES = [(re.compile(p), r) for p, r in SUFFIX_MAP.items()]
# Every suffix pattern contains one of these literals; if none is present no pattern can match.
_SUFFIX_TRIGGERS = ("private", "pvt", "corporation", "incorporated", "street", "road",
                    "avenue", "boulevard")

GEO_STOPWORDS = frozenset({
    "drive", "dr", "street", "st", "road", "rd", "avenue", "ave", "lane", "ln",
    "boulevard", "blvd", "no", "number", "plot", "flat", "shop", "floor", "building",
    "bldg", "near", "opp", "opposite", "behind", "city", "town", "pradesh", "state",
    "ltd", "inc", "corp", "co", "llc", "room", "suite", "ste", "apt", "apartment",
    "marg", "nagar", "vihar", "puram", "gali", "bhavan", "mahal", "complex",
    "bagh", "chowk", "naka", "taluka", "zila", "mandal", "gram", "phase", "sector",
    "khand", "colony", "enclave", "extension", "ext", "cross", "main",
    "rue", "chemin", "place", "allee", "route", "impasse", "batiment",
    "immeuble", "etage", "cedex", "bp", "boite", "postale"
})
_COUNTRY_ALIASES = frozenset({"usa", "us", "india", "ind", "france"})

_RE_PUNCT = re.compile(r'[^\w\s]')
_RE_WS = re.compile(r'\s+')
_RE_PIN = re.compile(r'\b\d{5,6}\b(?=[^\w]*$|\s*[a-z]{2,}\s*$)')
_RE_PIN_ANY = re.compile(r'\b\d{5,6}\b')
_RE_ENDS_PIN = re.compile(r'\d{5,6}$')


def _is_missing(x) -> bool:
    # Equivalent to `not pd.notna(x)` for scalars, ~10x cheaper.
    return x is None or (isinstance(x, float) and x != x) or x is pd.NA or x is pd.NaT


def _ascii(s: str) -> str:
    return s if s.isascii() else unidecode(s)


def _clean_name(x) -> str:
    if _is_missing(x):
        return ""
    s = _RE_PUNCT.sub(' ', _ascii(str(x)).lower())
    if any(t in s for t in _SUFFIX_TRIGGERS):
        for pat, repl in COMPILED_SUFFIXES:
            s = pat.sub(repl, s)
    return _RE_WS.sub(' ', s).strip()


def _clean_address(x) -> str:
    if _is_missing(x):
        return ""
    return _RE_WS.sub(' ', _RE_PUNCT.sub(' ', _ascii(str(x)).lower())).strip()


def _geo_parts(addr, country) -> tuple[str, str | None, str | None, str | None]:
    """Returns (country_str, pin, chunk_1, chunk_2) exactly as the notebook's geo loop."""
    c_str = str(country).lower().strip() if not _is_missing(country) else ""
    r_addr = str(addr).lower().strip() if not _is_missing(addr) else ""

    m = _RE_PIN.search(r_addr)
    pin = m.group(0) if m else None

    found: list[str] = []
    # Right-to-left: only the last two valid chunks are ever used.
    for chunk in reversed(r_addr.split(',')):
        chunk = chunk.strip()
        if not chunk:
            continue
        chunk = _ascii(chunk)
        cleaned = _RE_PIN_ANY.sub('', chunk)
        cleaned = _RE_PUNCT.sub(' ', cleaned)
        cleaned = _RE_WS.sub(' ', cleaned).strip()
        if (len(cleaned) > 1 and cleaned not in GEO_STOPWORDS
                and cleaned != c_str and cleaned not in _COUNTRY_ALIASES):
            found.append(cleaned.replace(" ", "_"))
            if len(found) == 2:
                break
    c1 = found[0] if found else None
    c2 = found[1] if len(found) > 1 else None
    return c_str, pin, c1, c2


def _process_rows(args: tuple[list, list, list]) -> tuple[list, ...]:
    """Worker: one pass over a chunk of rows. Pure Python; runs in a child process."""
    names, addrs, countries = args
    n = len(names)
    clean_name = [""] * n
    clean_addr = [""] * n
    phon = [""] * n
    base_key = [""] * n
    c_strs = [""] * n
    c1s: list = [None] * n
    c2s: list = [None] * n
    metaphone = jellyfish.metaphone
    for i in range(n):
        cn = _clean_name(names[i])
        clean_name[i] = cn
        clean_addr[i] = _clean_address(addrs[i])
        phon[i] = metaphone(cn) if cn else ""
        c_str, pin, c1, c2 = _geo_parts(addrs[i], countries[i])
        c_strs[i], c1s[i], c2s[i] = c_str, c1, c2
        if pin:
            base_key[i] = f"{c_str}_{pin}"
        elif c1:
            base_key[i] = f"{c_str}_{c1}"
        else:
            base_key[i] = f"{c_str}_unknown"
    return clean_name, clean_addr, phon, base_key, c_strs, c1s, c2s


def _chunks(n: int, size: int) -> Iterable[tuple[int, int]]:
    for i in range(0, n, size):
        yield i, min(i + size, n)


def _row_pass(df: pd.DataFrame, n_jobs: int, chunk_rows: int) -> dict[str, list]:
    names = df['business_name'].tolist()
    addrs = df['business_address'].tolist()
    countries = df['country'].tolist()
    tasks = [(names[a:b], addrs[a:b], countries[a:b]) for a, b in _chunks(len(df), chunk_rows)]

    if n_jobs > 1 and len(tasks) > 1:
        ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()
        with ctx.Pool(processes=n_jobs) as pool:
            parts = pool.map(_process_rows, tasks, chunksize=1)
    else:
        parts = [_process_rows(t) for t in tasks]
    del tasks, names, addrs, countries

    keys = ("clean_name", "clean_address", "phonetic_hash", "base_key", "c_str", "chunk_1", "chunk_2")
    out: dict[str, list] = {k: [] for k in keys}
    for p in parts:
        for k, v in zip(keys, p):
            out[k].extend(v)
    return out


def _apply_thresholds(
    base_key: np.ndarray, c_str: np.ndarray, c1: np.ndarray, c2: np.ndarray,
    first_char: np.ndarray, upper_threshold: int, lower_threshold: int,
) -> np.ndarray:
    """Vectorized version of the notebook's 3 threshold passes (same semantics)."""
    keys = base_key.astype(object)

    def counts_of(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        codes, uniq = pd.factorize(arr)
        cnt = np.bincount(codes, minlength=len(uniq))
        return codes, uniq, cnt

    def pin_suffix(uniq) -> np.ndarray:
        return np.fromiter((bool(_RE_ENDS_PIN.search(k)) for k in uniq), bool, len(uniq))

    # Pass 1: massive non-PIN keys -> country_chunk2_chunk1 (only if chunk_2 exists)
    codes, uniq, cnt = counts_of(keys)
    split_key = (cnt > upper_threshold) & ~pin_suffix(uniq)
    rows = np.flatnonzero(split_key[codes] & (c2 != None))  # noqa: E711
    if len(rows):
        keys[rows] = [f"{c}_{b}_{a}" for c, a, b in zip(c_str[rows], c1[rows], c2[rows])]

    # Pass 2: still mega non-PIN keys -> append first letter of clean_name
    codes, uniq, cnt = counts_of(keys)
    split_key = (cnt > upper_threshold) & ~pin_suffix(uniq)
    rows = np.flatnonzero(split_key[codes])
    if len(rows):
        keys[rows] = [f"{k}_{f}" for k, f in zip(keys[rows], first_char[rows])]

    # Pass 3: tiny keys -> <country>_unknown
    codes, uniq, cnt = counts_of(keys)
    small = cnt < lower_threshold
    if small.any():
        repl = np.array([f"{k.split('_')[0]}_unknown" if s else k for k, s in zip(uniq, small)], dtype=object)
        keys = repl[codes]
    return keys


def _assemble(df: pd.DataFrame, cols: dict[str, list], geo_key: np.ndarray,
              string_dtype: str) -> pd.DataFrame:
    out = df.drop(columns=['business_name', 'business_address'])
    sd = pd.StringDtype(string_dtype) if string_dtype == "pyarrow" else object
    cn = pd.Series(cols["clean_name"], index=df.index, dtype=sd)
    ca = pd.Series(cols["clean_address"], index=df.index, dtype=sd)
    out['clean_name'] = cn
    out['clean_address'] = ca
    out['geo_block_key'] = pd.Series(geo_key, index=df.index, dtype=sd)
    out['phonetic_hash'] = pd.Series(cols["phonetic_hash"], index=df.index, dtype=sd)
    out['search_document'] = cn + " " + ca
    return out


def _prepare(df: pd.DataFrame, n_jobs: int | None, chunk_rows: int) -> dict:
    n_jobs = n_jobs or os.cpu_count() or 1
    cols = _row_pass(df, n_jobs, chunk_rows)
    return {
        "cols": cols,
        "base_key": np.array(cols.pop("base_key"), dtype=object),
        "c_str": np.array(cols.pop("c_str"), dtype=object),
        "c1": np.array(cols.pop("chunk_1"), dtype=object),
        "c2": np.array(cols.pop("chunk_2"), dtype=object),
        "first": np.array([n[0] if n else 'z' for n in cols["clean_name"]], dtype=object),
    }


def normalize_pipeline(
    df: pd.DataFrame,
    upper_threshold: int = 15000,
    lower_threshold: int = 50,
    n_jobs: int | None = None,
    chunk_rows: int = 100_000,
    string_dtype: str = "pyarrow",
) -> pd.DataFrame:
    """Drop-in for the notebook's `normalize_pipeline` (per-source thresholds, same output)."""
    log.info("Normalizing %d rows (n_jobs=%s)", len(df), n_jobs or os.cpu_count())
    p = _prepare(df, n_jobs, chunk_rows)
    geo_key = _apply_thresholds(p["base_key"], p["c_str"], p["c1"], p["c2"], p["first"],
                                upper_threshold, lower_threshold)
    return _assemble(df, p["cols"], geo_key, string_dtype)


def normalize_sources(
    sources: Sequence[pd.DataFrame],
    upper_threshold: int = 15000,
    lower_threshold: int = 50,
    joint_thresholds: bool = True,
    n_jobs: int | None = None,
    chunk_rows: int = 100_000,
    string_dtype: str = "pyarrow",
) -> list[pd.DataFrame]:
    """
    Normalize several sources. With joint_thresholds=True, block-size thresholds are computed on
    the union of all sources, so identical addresses get identical geo_block_key values in every
    source (recall fix, see module docstring). joint_thresholds=False == calling
    normalize_pipeline on each source independently (notebook behavior).
    """
    if not joint_thresholds:
        return [normalize_pipeline(df, upper_threshold, lower_threshold, n_jobs, chunk_rows, string_dtype)
                for df in sources]
    preps = [_prepare(df, n_jobs, chunk_rows) for df in sources]
    sizes = [len(df) for df in sources]
    cat = {k: np.concatenate([p[k] for p in preps]) for k in ("base_key", "c_str", "c1", "c2", "first")}
    geo_all = _apply_thresholds(cat["base_key"], cat["c_str"], cat["c1"], cat["c2"], cat["first"],
                                upper_threshold, lower_threshold)
    del cat
    bounds = np.cumsum([0] + sizes)
    return [_assemble(df, p["cols"], geo_all[bounds[i]:bounds[i + 1]], string_dtype)
            for i, (df, p) in enumerate(zip(sources, preps))]
