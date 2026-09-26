# Candidate-generation optimization: `notebooks/kaggle_mirror.ipynb`

Source of truth: cells 2 (`normalize_pipeline`), 4 (blocking) and 10 (merge) of
`notebooks/kaggle_mirror.ipynb`. The optimized code is in `src/fast_normalization.py` and
`src/fast_blocking.py`. The benchmark in `benchmarks/` loads the baseline **verbatim from the
notebook JSON**, so the comparison always runs the notebook's actual code.

> **About the numbers.** The real TSVs (`student_resource/`) are not in the repo, so every
> measurement below comes from a deterministic synthetic dataset with the same schema and noise
> patterns (`benchmarks/synthetic_data.py`, seed 42), run on a 4-core / 15 GB container. Full-scale
> numbers are **estimates** derived from those measurements plus the sizes in your logs. Run
> `python -m benchmarks.run_benchmark --data-dir data/sample` on the real sample to replace them.

---

## 1. Executive summary

| | Baseline (notebook) | Optimized (exact mode) | Optimized (recommended) |
|---|---:|---:|---:|
| Runtime, 112k records | 91.9 s | 9.3 s (**9.9x**) | 7.1 s (**12.9x**) |
| Runtime, 336k records | 337.0 s | 46.0 s (**7.3x**) | 29.4 s (**11.5x**) |
| BM25 stage (112k / 336k) | 71.8 s / 263.6 s | 3.6 s / 28.2 s | 2.0 s / 14.5 s |
| Candidate recall (S1→S2/S3 GT pairs) | 99.33 % | 99.31 % | 99.31 % |
| Candidate pairs | 1.38 M | 1.38 M | 0.65 M (**-53 %**) |
| Geo-signal recall | 47.2 % | 47.2 % | **74.4 %** |
| Output of normalization | – | byte-identical | byte-identical text, consistent geo keys |
| BM25 scores | – | equal to 3e-6; top-k score lists equal for 100 % of queries | same |

"Recommended" means `BlockingConfig(include_s2_s3=False)` plus `normalize_sources(..., joint_thresholds=True)`.
At 3x the size (336k records) the result holds: **337 s → 46 s** (exact mode, 7.3x) and **29 s**
(recommended, 11.5x), with recall again equal to tie-breaking. The BM25 speedup is a constant factor
(~10-20x), **not** an asymptotic one. Both versions must touch every posting of every query token, and
common tokens (`ltd`, state names, `india`) have posting lists that grow with N. The optimized
version removes the dense O(N)-per-query work and runs the rest in C++ on all cores. Changing the
asymptotics requires pruning common query tokens (`max_df_ratio`). That trades recall and is measured in §7.

The findings that matter most:

1. **BM25 retrieval is O(queries × index size).** `bm25s` (numpy backend) allocates
   `np.zeros(num_docs)` for **every query**, fills it with `np.add.at`, and arg-partitions all
   `num_docs` entries. For India that is ~1.8·10¹² element operations per index per modality,
   which is why the log stalls at "Processed S1 queries: 200000/883188". Replacing this with
   sparse-matrix × sparse-matrix top-k (`sparse_dot_topn`, C++, multi-threaded) yields **the
   same scores** at a cost proportional to the postings touched.
2. **The S2→S3 queries cannot improve the metric.** The ground truth is S1-anchored
   (`source1_entity_id → matched_entity_ids`). S2→S3 accounts for ~45 % of BM25 work and ~55 %
   of all candidate pairs. Skipping it left measured recall **unchanged** (99.3146 % both ways).
3. **Geo keys are inconsistent across sources.** `normalize_pipeline` applies the 15000/50 size
   thresholds **per source**. A city large enough to be split into `country_c2_c1` / `_<letter>`
   in S2 (5.0M rows) may stay unsplit in S1 (2.2M rows), so the two blocks never meet. Computing
   thresholds on the union raised the geo signal's own recall from 47 % to 74 %.
4. **Geo blocking is O(rows × blocks).** `s1_df[s1_df['geo_block_key'] == block]` scans the
   full frame once per block, three times (12/13/23). With PIN-level blocks at full scale
   (tens of thousands of blocks × 5M rows) that is ~10¹¹ string comparisons before any TF-IDF
   work. One argsort replaces all of it.
5. **Two silent bugs in the baseline:**
   (a) On **pandas ≥ 3.0**, `pd.DataFrame(geo_results)` converts a missing `pin` into `NaN`, which
   is truthy, so `if row['pin']` makes every PIN-less record `"<country>_nan"`. On a pandas-3
   run, ~25 % of rows land in one garbage block per country. Pandas 2.x (Kaggle) is unaffected.
   The `.pyc` files in `src/__pycache__` are built by CPython 3.14, which suggests local runs may
   be on pandas 3. The optimized code does not depend on the pandas version.
   (b) `bm25s.retrieve` always returns exactly `k` documents, padding with **arbitrary score-0
   documents** when fewer than `k` match (including empty queries). The baseline keeps them
   (`valid_mask = cand_indices >= 0` never filters them). They are random pairs. The optimized
   code drops them.
6. **Memory explosion comes from Python-string pair tables**, not from the indexes.
   `pd.merge(..., how='outer')` on object `entity_A/entity_B` columns, then `pl.from_pandas`
   (copies every string into Arrow), then `.to_pandas()` (creates new Python `str` objects for
   every pair). The optimized pool is `int64 key + 3×float32` = 20 B/pair, and ids are decoded once
   at the end, or never.

---

## 2. Current architecture (traced from the notebook)

```text
read_csv x3 (object columns)                              S1 2.2M | S2 5.0M | S3 5.3M
 └─ normalize_pipeline(df)  – called once per source, independently
     out = df.copy()
     clean_name    = [unidecode(x).lower()]            1 Python pass
     clean_address = [unidecode(x).lower()]            1 Python pass
     .str.replace([^\w\s]) x2                          2 passes (pandas object .str = Python loop)
     .str.replace(suffix_i) x9                         9 passes over clean_name
     .str.replace(\s+).str.strip() x2                  2 passes
     geo loop: per row: lower, re.search(PIN), split(','),
               per chunk: unidecode, 3x re.sub, set lookups    1 pass, list of 12.5M dicts
     pd.DataFrame(geo_results)                         12.5M-row frame of dicts
     geo_df.apply(assign_base_key, axis=1)             row-wise apply: builds a Series per row
     value_counts + list-comp with re.search  x3       3 passes (thresholds)
     jellyfish.metaphone(clean_name)                   1 pass
     search_document = clean_name + " " + clean_address
 └─ generate_global_candidate_pool
     countries = union(str.lower())                    3 passes
     for country:                                      (India, US, France, ...)
        s{1,2,3}[s.country.str.lower() == country]     3 full passes per country
        for modality in (text, phonetic):              run_country_global_scan
            bm25s.tokenize(S2) -> BM25().index        S2 tokenized #1
            bm25s.tokenize(S3) -> BM25().index        S3 tokenized #1
            for 50k chunk of S1:  bm25s.tokenize(chunk) (new vocab per chunk)
                retrieve(S2 index, k=7) -> DataFrame
                retrieve(S3 index, k=7) -> DataFrame
            for 50k chunk of S2:  bm25s.tokenize(chunk)    S2 tokenized #2
                retrieve(S3 index, k=7) -> DataFrame
            pd.concat(pairs)
        pd.merge(text, phonetic, outer).fillna(0)     object-key hash join
     pd.concat(countries).drop_duplicates()
 └─ run_multi_source_geo_blocking
     for (X,Y) in (12, 13, 23):                         3 sweeps
        valid_blocks = keys(X) ∩ keys(Y) − *_unknown
        for block:  X[X.key == block], Y[Y.key == block]   O(N) filter per block
                    TfidfVectorizer(char_wb,2-4).fit_transform(Y_b); transform(X_b)
                    sp_matmul_topn(X_b, Y_b.T, top_n=7)
     concat + drop_duplicates
 └─ pl.from_pandas x2 -> full join -> fill_null -> to_pandas
```

## 3. Bottleneck analysis (ranked)

| # | Where | Why it is slow | Fix | Measured speedup |
|---|---|---|---|---:|
| 1 | `_query_and_extract → bm25s.retrieve` | dense `np.zeros(N)` + `np.add.at` + argpartition(N) **per query**. The numpy backend's `n_threads` uses a ThreadPool, but `np.add.at` holds the GIL for most of its runtime (baseline CPU utilization 0.41 on 4 cores) | sparse Q·Wᵀ with per-row top-k in C++ (`sp_matmul_topn`, `n_threads=cpu_count`) | 19.7x (1x data), 9.4x (3x data). Constant factor: see §8 |
| 2 | S2→S3 retrieval, geo_23 | work that cannot contribute to S1-anchored recall | `include_s2_s3=False` | -45 % BM25 time, -53 % pairs |
| 3 | `execute_local_geo_index` block filter | `df[df.key == block]` → O(N·B) | one argsort + contiguous slices | (dominant at full scale; B≈10⁴) |
| 4 | geo TF-IDF | each doc char-n-gram analyzed twice (fit in one pair, transform in another); serial Python | one `CountVectorizer` per block for all 3 sources + exact per-target IDF; fork pool over blocks | 5.0x |
| 5 | normalization | ~15 full passes, `unidecode` 3×/address, 9 suffix regexes/row, row-wise `apply` | one pass per row, ASCII fast path, trigger-gated suffixes, early-exit geo scan, multiprocessing | 2.0x at 4 cores (scales with cores) |
| 6 | tokenization | S2 tokenized twice per modality; fresh vocab for each 50k chunk; bm25s then converts ids → strings → index ids | Polars `str.extract_all` (Rust, parallel), one shared vocab per country, tokenized once | included in #1 |
| 7 | pair tables | object-string keys through `pd.concat`/`pd.merge`/`pl.from_pandas`/`to_pandas` | int64 pair key, Polars joins, decode once | 5.2x on merge; ~5-10x RAM at full scale |
| 8 | country filtering | `str.lower()` + boolean filter 3× per country + 3× for the set | factorize once + `lexsort` | minor |
| 9 | per-source thresholds | recall bug (§1.3) | `joint_thresholds=True` | geo recall 47 → 74 % |

## 4. Complexity analysis

Notation: N₁,N₂,N₃ records per source in a country, L ≈ tokens/doc (~12), P(q) = Σ over query
tokens of the token's posting-list length in the target, C = char n-grams/doc (~150), B = geo
blocks, bᵢ = size of block i.

| Stage | Baseline time | Optimized time | Space (optimized) |
|---|---|---|---|
| Normalization | O(N·(15 passes + 3 unidecode + 9 regex)) single-core | O(N·(1 pass + ≤1 unidecode + gated regex)) / cores | O(N) arrow strings |
| Tokenize | 2·N₂ + N₃ + N₁ per modality (Python regex, lists of Python ints) | N₁+N₂+N₃ once per modality (Rust, parallel) | O(N·L) CSR, int32/float32 |
| Index build | O(N·L) Python loop per doc (`Counter`) | O(N·L) vectorized numpy | O(N₂L + N₃L) |
| Query | **O(Q·N_target) + O(Σ P(q))** (dense) | **O(Σ P(q) + Q·k·log k)** | O(Q·k) output, per-thread accumulator O(N_target) |
| Pair merge | object hashing, O(pairs) with ~60 B/str key | int64 hashing, 20 B/pair | O(pairs) |
| Geo slicing | **O(N·B)** | O(N log N) once | O(N) |
| Geo TF-IDF | Σ 2·(bᵢ·C) analysis + matmul | Σ bᵢ·C analysis + matmul, parallel | O(max bᵢ·C) |

With Q = 883k, N_target = 2M: the dense term alone is 1.77·10¹² per index per modality. It does
not depend on k or on how selective the query is, so tuning `chunk_size`/`top_k` cannot fix it.

The postings term Σ P(q) is **also** ~O(Q·N) whenever queries contain tokens with df ∝ N: a
query holding `ltd` (~40 % of Indian docs) and a state name (~10-15 %) touches ~1M postings in a
2M-doc index. Both implementations pay this. The optimized version pays it at ~1-2 ns per posting in C++
on all cores, while `np.add.at` costs roughly an order of magnitude more per element under the GIL. Measured
optimized BM25 time grew 7.7x for 3x data (3x queries × ~2.6x postings), and the synthetic
vocabulary is small, which makes this worst-case. Only query-side pruning of high-df tokens
(`max_df_ratio`) changes the exponent.

## 5. Memory analysis (full data, 12.5M records; estimates)

Assumptions: CPython `str` ≈ 49 B + len; object column = 8 B pointer + object; arrow string =
len + 8 B offset; avg lengths: id 12, name 25, address 60, key 20, phonetic 12, search doc 86.

| Component | Current RAM | Optimized RAM | How |
|---|---:|---:|---|
| Raw datasets (4 object cols) | ~4.1 GB | ~4.1 GB (freed after normalize) → 0 | `del s1,s2,s3` after `normalize_sources`; or read with `pl.read_csv` (~1.5 GB) |
| Normalized data | ~6.1 GB (+`df.copy()`, 12.5M dicts transient ≈ +4 GB peak) | ~3.1 GB arrow (1.8 GB if `clean_address` dropped) | pyarrow strings, no geo dict list |
| BM25 indexes (per country×modality) | ~0.4 GB + ~2 GB transient Python token lists | ~0.4 GB (float32/int32 CSR) | vectorized build |
| Token arrays | lists of Python ints, rebuilt per chunk | ~0.5 GB CSR for India (all 3 sources) | tokenize once |
| Candidate pairs (66M/modality incl. S2→S3) | ~8-10 GB peak in `pd.merge` + ~11 GB at `to_pandas()` | ~1.8 GB (0.8 GB with `include_s2_s3=False`) | int64 key + float32 |
| Geo matrices | per block, small; but full-frame boolean masks per block | per block, bounded by block size | slicing |
| **Total peak** | **~20-25 GB** | **~6-8 GB** (≈4 GB if raw frames are released) | |

Measured at 112k records the peak is ~0.8 GB for both. That is interpreter and library
baseline, since the data itself is tiny. PSS is used so forked workers' shared pages are not
double-counted.

## 6. Redundant computation analysis

**6.1 Country lowercasing and filtering (3 × #countries full scans + 3 for the set)**

CURRENT:
```python
countries = set(s1['country'].str.lower().dropna()) | ...
for country in sorted(countries):
    s1_sub = s1[s1['country'].str.lower() == country]
```
OPTIMIZED (`EntityTable.build`, `generate_global_candidate_pool`):
```python
codes, names = pd.factorize(all_countries.str.lower(), sort=True)   # once
order = np.lexsort((source, codes))                                  # once
for seg in np.split(np.arange(len(order)), np.flatnonzero(np.diff(codes[order])) + 1): ...
```
Faster because it does one lowercase and one sort, and every country is a contiguous slice with no boolean masks.

**6.2 Double tokenization of S2 and per-chunk vocabularies**

CURRENT: `bm25s.tokenize(s2_text)` for the index, then `bm25s.tokenize(s2_text[i:j])` again
for S2→S3 queries. Every 50k S1 chunk builds its own vocab, and `retrieve` then converts ids back
to strings and looks them up in the index vocab.

OPTIMIZED (`_scan_modality`):
```python
vocab = _Vocab()
tf = {s: _term_counts(texts.gather(h), vocab, chunk) for s, h in groups.items()}  # once per source
```
One shared vocab per country means query columns *are* index rows, so no remapping is needed.

**6.3 Dense per-query scoring** (see §8).

**6.4 Geo block filtering and double vectorization**

CURRENT:
```python
for block in valid_blocks:
    s1_sub = s1_df[s1_df['geo_block_key'] == block]    # O(N) per block, per pair sweep
    s2_matrix = vectorizer.fit_transform(s2_sub[...])  # S2 analyzed here and again in geo_23
```
OPTIMIZED (`run_multi_source_geo_blocking`):
```python
order = idx[np.argsort(codes[idx], kind="stable")]     # once
for s, e in zip(starts, ends): h = order[s:e]           # O(1) slice
C = CountVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit_transform(block_docs)  # once for S1+S2+S3
# per target t: idf_t from df over target rows only; zero for n-grams absent in t
```
This reproduces `TfidfVectorizer.fit(target).transform(query)`: smooth idf `ln((1+n)/(1+df))+1`,
query n-grams unseen in the target are dropped **before** L2 normalization, same as `transform()`.

**6.5 Normalization passes**

CURRENT: `unidecode` on the full name, the full address, and again on every comma chunk. 13
`.str.replace` passes (pandas object `.str` is a Python loop, not vectorized).

OPTIMIZED (`_process_rows`): a single loop per row:
```python
def _ascii(s): return s if s.isascii() else unidecode(s)     # C-speed check, skip ~all rows
s = _RE_PUNCT.sub(' ', _ascii(name).lower())
if any(t in s for t in _SUFFIX_TRIGGERS):                     # 9 regexes only when possible
    for pat, repl in COMPILED_SUFFIXES: s = pat.sub(repl, s)
for chunk in reversed(addr.split(',')): ...; if len(found) == 2: break   # only chunk_1/2 used
```
Exactness is kept: the suffix substitutions still run in the same order on the same
un-collapsed whitespace, `unidecode` maps characters independently, and the right-to-left scan
yields the same last two valid chunks.

**6.6 Row-wise `apply` and threshold regexes**

CURRENT: `geo_df.apply(assign_base_key, axis=1)` builds a Series per row. Three `value_counts` +
list comprehensions each run `re.search(r'\d{5,6}$', k)` on **every row**.

OPTIMIZED (`_apply_thresholds`): base key computed inside the row pass. Thresholds use
`pd.factorize` + `np.bincount`, and the PIN-suffix regex runs once per **unique** key.

**6.7 Pandas ↔ Polars round trips**

CURRENT: `pl.from_pandas(global)`, `pl.from_pandas(geo)` (copy all strings), then `.to_pandas()`
(re-create 2 Python strings per pair).

OPTIMIZED: pairs are Polars frames with an int64 key from the start. `merge_pools` is an
integer hash join, and `decode_ids` does one gather per column at the end.

## 7. Blocking analysis (mathematical)

Definitions used by `benchmarks/run_benchmark.py`:
- **Recall** = |GT pairs ∩ candidates| / |GT pairs|, over S1→S2/S3 GT pairs whose two ids exist
  in the evaluated data (identical to the notebook's `validate_blocking_recall`).
- **Reduction ratio** = 1 − candidates / (N₁N₂ + N₁N₃ + N₂N₃).
- **Signal recall** = recall of each generator alone (text, phonetic, geo), to show what each
  generator contributes.
- **Cost**: records/s, queries/s, candidates/s, peak PSS, CPU utilization = CPU time / (wall × cores).

Measured (synthetic, 20k/45k/47k, top_k=7):

| Config | Pairs | Recall | Text | Phonetic | Geo | Runtime |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 1,381,663 | 99.33 % | 98.99 % | 49.54 % | 47.22 % | 91.9 s |
| optimized, exact | 1,380,805 | 99.31 % | 99.00 % | 49.37 % | 47.22 % | 9.5 s |
| + joint thresholds | 1,417,786 | 99.31 % | 99.00 % | 49.43 % | **74.44 %** | 10.0 s |
| + include_s2_s3=False | **617,552** | 99.31 % | 99.00 % | 49.37 % | 47.22 % | 6.1 s |
| + both (recommended) | 653,537 | 99.31 % | 99.00 % | 49.36 % | 74.44 % | 7.1 s |
| + max_df_ratio=0.3 | 658,034 | 99.26 % | 98.96 % | 43.05 % | 74.44 % | 6.5 s |
| + max_df_ratio=0.2 | 661,151 | 99.21 % | 98.95 % | 38.28 % | 74.44 % | 6.7 s |
| + max_df_ratio=0.1 | 670,694 | 99.07 % | 98.75 % | 33.71 % | 74.44 % | 6.3 s |
| + max_df_ratio=0.05 | 672,919 | 98.08 % | 97.40 % | 30.74 % | 74.44 % | 6.2 s |
| + max_df_ratio=0.01 | 441,527 | **88.79 %** | 81.56 % | 0.00 % | 47.22 % | 4.7 s |

Recall@k (recommended config: `include_s2_s3=False`, joint thresholds):

| top_k | Pairs | Recall | Text | Phonetic |
|---:|---:|---:|---:|---:|
| 5 | 467,814 | 99.11 % | 98.78 % | 40.89 % |
| 7 (current) | 653,537 | 99.31 % | 99.00 % | 49.36 % |
| 10 | 931,604 | 99.47 % | 99.15 % | 59.18 % |
| 15 | 1,397,995 | 99.57 % | 99.22 % | 69.12 % |

Pairs grow linearly with k while recall gains shrink: +0.16 pp for +43 % pairs (7→10),
+0.10 pp for +50 % pairs (10→15). BM25 time barely changes with k in the optimized version.
The cost of a larger k lands on the downstream matcher, so pick k from what the matcher can
afford.

Reading the table:
- Text BM25 carries almost all of the recall, so the phonetic and geo generators are safety nets.
  Their *marginal* recall is what matters (baseline: 99.33 % union vs 98.99 % text alone, a
  +0.34 pp union gain).
- Recall differences of 0.01 pp between baseline and exact mode are tie-breaks at the k-th score
  (verified: 0 of 9,890 queries have different top-k score lists).
- **Do not enable `max_df_ratio` without measuring on real data.** The synthetic vocabulary is
  small, so "common" tokens carry identity. Phonetic codes are especially short and common, which
  explains the collapse at 0.01.

## 8. BM25 analysis

**Is tokenize-once possible?** Yes. Tokenization is deterministic per document, so each
document is tokenized once per modality (`_term_counts`) into a CSR over one per-country vocab.

**Can indexes be reused?** The S3 index serves both S1→S3 and S2→S3 (as before). Indexes stay
**per target source** because IDF and `top_k` are per target in the baseline. A merged S2+S3
index would change both.

**Exactness.** `bm25s` defaults are `method="lucene", k1=1.5, b=0.75`:
`w(t,d) = ln(1 + (N − df + .5)/(df + .5)) · tf / (tf + k1·(1 − b + b·dl/avgdl))`, where `dl` counts
duplicate tokens after stopword removal (empty docs have dl = 0). Query score = Σ over query
tokens **with multiplicity** of w(t,d). `_bm25_index` stores exactly W, and Q holds query term
counts, so `Q @ Wᵀ` is the bm25s score. The parity check prints the max score difference (2.9e-6, float32
rounding) and 0 queries with different top-k score lists.

**chunk_size = 50000.** Chunking the baseline only bounded memory, and it forced a vocabulary
rebuild each time. `sp_matmul_topn` output is O(chunk × k). The optimized `query_chunk_rows = 250_000`
exists only for progress logging and a bounded result buffer, and the value barely affects speed.

**top_k = 7.** See the recall@k table in §7. Candidate count scales linearly with k, while recall
saturates. Choose k from the recall-at-k curve on real data, not a fixed constant.

**n_threads = 4.** The baseline hard-codes 4, and its numpy backend does not scale anyway (GIL).
The optimized default is `os.cpu_count()` for `sp_matmul_topn` (native threads, no GIL), and
Polars uses its own pool for tokenization. These phases never overlap, so nothing is oversubscribed.

**Alternatives considered:**

| Method | Speed | Memory | Recall vs BM25 | Verdict |
|---|---|---|---|---|
| BM25 via sparse top-k matmul (this) | O(postings touched) | O(nnz) | identical | **replace bm25s backend** |
| bm25s `backend="numba"` | ~5-10x over numpy, still O(N) per query | same | identical | fallback if `sparse_dot_topn` unavailable |
| TF-IDF word cosine | same cost as above | same | slightly lower for long addresses | no gain |
| Char n-gram TF-IDF global | 10-15x more nnz per doc (~150 vs ~12) | 15x | catches typos BM25 misses | only **inside blocks** (as now) |
| MinHash/LSH on name shingles | O(N·bands) | O(N·bands) | tunable; weak on short strings (names ~25 chars) | not worth it: BM25 already sub-linear with sparse matmul |
| ANN/HNSW/FAISS on embeddings | fast queries, **slow build** (12.5M × encoder) | ~12.5M × 384 × 4 B ≈ 19 GB | good for semantic variants, weak for codes/PINs | future supplement for the misses, not a replacement |
| Exact hashing (e.g. name+PIN) | O(N) | O(N) | high precision, low recall | good as a cheap Stage-1 that **short-circuits** easy matches |
| Sorted neighbourhood on phonetic | O(N log N) | O(N) | similar to phonetic BM25 on single-token names | could replace phonetic BM25 if it proves low-value on real data |

## 9. Geographic blocking analysis

- **Block sizes.** Upper 15000 / lower 50 are applied per source (bug §1.3). PIN blocks are
  never split (`not re.search(r'\d{5,6}$', k)`), so a dense PIN can exceed 15000. Cost per block is
  ~ b_q · avg postings in the n-gram space. The matmul uses `top_n` with a per-thread accumulator
  of size b_t, which is fine up to ~10⁵.
- **Adaptive subdivision.** The notebook's second split (`_<first letter of name>`) is a recall
  risk: a typo or a dropped leading word in the first letter ("Shree Ganesh" vs "Ganesh")
  separates true matches. Because geo is a safety net for BM25 misses (typos), splitting by first
  letter is exactly the wrong axis. Better options: leave oversized blocks unsplit (the sparse top-k
  handles 10⁴-10⁵ rows) or split by `chunk_2`. This needs real-data measurement; the code keeps the
  notebook's behaviour.
- **Small blocks (<50)** become `_unknown` and are **excluded** from geo. They still go through
  BM25. With joint thresholds more of them survive, which is where the 47 → 74 % geo-signal gain
  comes from.
- **Vectorizer.** Fitting per block (per-block IDF) is intentional: IDF inside a city
  down-weights the city's own n-grams. The optimized code keeps per-target per-block IDF exactly,
  but analyzes each document once per block instead of twice.
- **(2,4) char_wb.** This is standard for typo tolerance. (3,3) alone would roughly halve nnz,
  but it needs a recall measurement before adopting.
- **`sp_matmul_topn`** was already the right primitive. The baseline's cost was the O(N·B)
  filtering and the Python analyzer, not the matmul.

## 10. Parallelism analysis

| Phase | Nature | Parallelism used | Why not something else |
|---|---|---|---|
| Normalization | pure Python (regex, unidecode, metaphone) → GIL-bound | `multiprocessing` fork pool over 100k-row chunks, `n_jobs=cpu_count` | threads would serialize on the GIL |
| Tokenization | Polars regex | Polars' Rust thread pool | – |
| Index build | numpy vectorized | single thread, memory-bound | fast already |
| BM25 query | C++ `sp_matmul_topn` | `n_threads=cpu_count` native threads | processes would duplicate the index |
| Countries | each country already saturates cores | **none** (sequential) | nesting country × thread pools would oversubscribe and multiply RAM |
| Geo blocks | Python analyzer + small matmuls | fork pool over block batches (largest-first round-robin), `sp_matmul_topn(n_threads=1)` inside | nested threads inside workers would oversubscribe |

Fork safety: the fork pools run while no Polars or BLAS work is executing, and the children only
use numpy, scipy and sklearn. The documents reach workers through the pool initializer. With
`fork`, that is inherited memory, not pickling. The notebook's pandarallel + bm25s combination
hit a deadlock, which the `_query_and_extract` cell in `Aa_EDA` "fixed" by forcing `n_threads=1`.
Measured CPU utilization rises from 0.41 to 0.72 on 4 cores. The remainder is the serial
merge/sort phases and pool start-up at this small size.

## 11. Optimized architecture

```text
Raw TSV (pandas or Polars)
   ↓
normalize_sources([s1,s2,s3], joint_thresholds=True)   one row pass, multiprocess, arrow strings
   ↓
EntityTable: concat once → int handles, country codes, text/phonetic as Polars Series
   ↓
Country partition (one lexsort)
   ↓  per country, per modality (text, phonetic):
Tokenize once (Polars, shared vocab) → CSR term counts for S1, S2, S3
   ↓
BM25 weights Wᵀ for S2 and S3 (vectorized, exact bm25s-lucene)
   ↓
Sparse top-k: S1→S2, S1→S3 [, S2→S3 optional]  (sp_matmul_topn, all cores, drop score-0)
   ↓
Geo: argsort by block → per block one char-n-gram count → exact per-target TF-IDF → top-k
   ↓
Union on int64 pair key (Polars) → scores float32
   ↓
Decode ids once (or keep int handles for feature building / LightGBM)
```

Suggested cascade for the matching stage (not in scope of this change): Stage-1 exact
`(clean_name, pin)` hash hits → Stage-2 this candidate pool → Stage-3 fuzzy features
(rapidfuzz, Jaro-Winkler on names, address token overlap) only on the pool → LightGBM.
The candidate pool is already ~30 pairs per S1 record, so a separate cheap lexical pre-filter
before BM25 would not reduce work meaningfully. BM25 via sparse matmul **is** the cheap stage now.

## 12. Exact code changes

| Notebook function | Replacement | File |
|---|---|---|
| `normalize_pipeline` | `normalize_pipeline` (same signature + `n_jobs`, `chunk_rows`, `string_dtype`), `normalize_sources` | `src/fast_normalization.py` |
| `run_country_global_scan`, `_query_and_extract` | `_term_counts`, `_bm25_index`, `_topk_pairs`, `_scan_modality` | `src/fast_blocking.py` |
| `generate_global_candidate_pool` | `generate_global_candidate_pool(ent, cfg)` | `src/fast_blocking.py` |
| `execute_local_geo_index`, `run_multi_source_geo_blocking`, `extract_top_k_pairs` | `run_multi_source_geo_blocking(ent, cfg)`, `_geo_block`, `_geo_task` | `src/fast_blocking.py` |
| cell 10 merge | `merge_pools`, `build_candidate_pool` | `src/fast_blocking.py` |

Notebook usage (Kaggle: paste both files into cells, or add the repo as a dataset):
```python
from src.fast_normalization import normalize_sources
from src.fast_blocking import BlockingConfig, build_candidate_pool

s1n, s2n, s3n = normalize_sources([s1, s2, s3], joint_thresholds=True)
del s1, s2, s3
cfg = BlockingConfig(top_k=7, include_s2_s3=False)       # exact bm25s scores, all cores
final_master_pool = build_candidate_pool(s1n, s2n, s3n, cfg)          # Polars DataFrame
# final_master_pool.to_pandas() only if downstream code needs pandas
```

## 13. Benchmark framework

`benchmarks/run_benchmark.py`:
- runs each implementation in a **separate subprocess** (clean peak memory and CPU accounting),
- uses the baseline code **exec'd from the notebook JSON** (`benchmarks/baseline.py`),
- uses the same data and seed for both (`--data-dir data/sample` for the real sample from
  `generate_sample.py`, otherwise synthetic `--scale`/`--seed`),
- reports per-stage wall time, records/s, BM25 queries/s, candidates/s, peak PSS, CPU utilization,
  candidate pairs, recall, per-signal recall, reduction ratio, and a baseline-vs-optimized table,
- `--check-parity`: normalization column-by-column mismatch counts and BM25 score/ranking
  equivalence against real `bm25s`.

```bash
python -m benchmarks.run_benchmark --check-parity
python -m benchmarks.run_benchmark                                   # table below
python -m benchmarks.run_benchmark --impls optimized --joint-thresholds true --opt "include_s2_s3=False"
python -m benchmarks.run_benchmark --data-dir data/sample            # real data
```

Result (synthetic 112k records, 4 cores, seed 42):

| Metric | Baseline | Optimized | Improvement |
|---|---:|---:|---:|
| Runtime (s) | 91.855 | 9.296 | 9.9x |
|   normalize (s) | 3.228 | 1.582 | 2.0x |
|   bm25 (s) | 71.804 | 3.648 | 19.7x |
|   geo (s) | 15.131 | 3.038 | 5.0x |
|   merge (s) | 0.9 | 0.173 | 5.2x |
| Peak RAM (MB, PSS) | 819 | 850 | ≈ |
| Candidate pairs | 1,381,663 | 1,380,814 | ≈ |
| Recall | 99.3285% | 99.3146% | -0.01 pp (tie-breaks) |
| Reduction ratio | 99.9651% | 99.9651% | ≈ |
| CPU utilization | 0.413 | 0.716 | +30 pp |
| Normalize rec/s | 34,696 | 70,785 | 2.0x |
| BM25 queries/s | 2,368 | 46,606 | 19.7x |

Result at 3x scale (60k/135k/141k = 336k records, same seed):

| Metric | Baseline | Optimized (exact) | Improvement |
|---|---:|---:|---:|
| Runtime (s) | 337.0 | 46.0 | 7.3x |
|   normalize (s) | 9.8 | 6.3 | 1.6x |
|   bm25 (s) | 263.6 | 28.2 | 9.4x |
|   geo (s) | 60.0 | 10.2 | 5.9x |
|   merge (s) | 2.8 | 0.6 | 4.7x |
| Peak RAM (MB, PSS) | 1,931 | 1,359 | 1.4x less |
| Candidate pairs | 4,309,806 | 4,307,003 | ≈ |
| Recall | 98.5522% | 98.5429% | -0.01 pp (tie-breaks) |
| CPU utilization | 0.53 | 0.84 | +31 pp |

Recommended config at 3x (`include_s2_s3=False`, joint thresholds): **29.4 s (11.5x)**,
1,984,752 pairs (-54 %), recall 98.55 %, BM25 14.5 s. With `max_df_ratio` at 3x: 0.3 → BM25 10.4 s,
recall 98.50 %; 0.2 → 8.4 s, 98.22 %; 0.1 → 5.7 s, 97.68 %.

## 14. Validation methodology

For every optimization: *could it remove true matches?*

| Change | Can remove true matches? | Evidence / safeguard |
|---|---|---|
| One-pass normalization | No | `--check-parity`: 0 mismatches in all 5 output columns, all 3 sources |
| Sparse-matmul BM25 | Only by tie-breaking at rank k (both implementations pick arbitrarily among equal scores) | max score diff 2.9e-6; 0/9,890 queries with different top-k score lists |
| Drop score-0 padding | Only by chance: the padded docs are arbitrary | the baseline returns them for empty/no-overlap queries; recall unchanged in the benchmark |
| Geo exact TF-IDF (float32) | Tie/rounding at rank k | geo-signal recall identical (47.2205 %) |
| `include_s2_s3=False` | No for the S1-anchored metric. Yes if you later use S2–S3 edges for transitive closure | recall identical; flag defaults to True |
| `joint_thresholds=True` | Could split a block differently. Net effect in the benchmark is positive | geo recall 47 → 74 %; overall unchanged; flag defaults to per-source in `normalize_pipeline` |
| `max_df_ratio` | **Yes** | -1.3 pp at 0.05, -10.5 pp at 0.01 on synthetic. Off by default; only enable with a real-data curve |
| `top_k` changes | Yes when lowered | see the top-k table; choose from the recall@k curve |

Fallback: keep the notebook functions (unchanged in `src/blocking.py`, `src/normalization.py`
and the notebook) and run `--check-parity` on each new data drop.

## 15. Expected performance at full scale (estimates)

These are extrapolations. Validate them with `--data-dir` on the real sample, then on one country.

- **Normalization** (12.5M rows): measured 54-70k rows/s on 4 cores → **~3-4 min** on 4 cores,
  ~1.5 min on 8. Baseline: ~34k rows/s single-threaded plus the row-wise `apply` → ~6-8 min, with a
  ~4 GB transient peak from the list of 12.5M dicts.
- **BM25**: both scale ~ Q × postings touched. Using the measured ratio (9-20x) and your log (the
  baseline had not finished India S1→S2 text after several chunks), the baseline is a **many-hours**
  job for India alone. The optimized version runs the same postings ~10x cheaper, with 1.8x fewer
  queries when `include_s2_s3=False`. Expect **~15-25x less wall time** for this stage. With
  `max_df_ratio=0.3` (−0.05 pp recall on synthetic) expect about another 1.4x.
- **Geo**: the O(N·B) filter disappears (≈ 10⁴ blocks × 12.5M comparisons ≈ 10¹¹ baseline string
  compares), analysis runs once per document and scales with cores → **>10x**.
- **Merge / RAM**: pool of ~40M (S1-only) to ~90M pairs at 20 B/pair = 0.8-1.8 GB instead of
  ~20 GB peak across `pd.merge` → `from_pandas` → `to_pandas`.

## 16. Migration plan

| Phase | Change | Files | Benefit | Risk | Validation |
|---|---|---|---|---|---|
| 1 Safe | `fast_normalization.normalize_pipeline` (per-source), sort-based geo slicing, int64 pair keys | `src/fast_normalization.py`, `src/fast_blocking.py` | 2x normalize, removes O(N·B), removes the merge RAM blow-up | none (byte-identical) | `--check-parity` |
| 2 Memory | arrow strings, `del` raw frames, `decode_ids=False` until submission | same | ~3x less RAM | none | peak PSS column |
| 3 Candidate gen | sparse-matmul BM25, tokenize once, drop score-0 padding | `src/fast_blocking.py` | 9-20x BM25 (constant factor) | tie-break-level differences | parity + recall |
| 4 Blocking | `joint_thresholds=True`; evaluate removing the first-letter split | `normalize_sources` | +geo recall | block membership changes | per-signal recall |
| 5 Parallelism | `n_jobs`/`n_threads`/`geo_n_jobs` = cores | config | linear in cores for normalize and geo | fork on macOS: set `n_jobs=1` if a library complains | CPU utilization |
| 6 Algorithmic | `include_s2_s3=False`; tune `top_k` by recall@k; optional `max_df_ratio` from a real-data curve | config | -50 % pairs, -45 % BM25 | measured per flag | recall table |

## 17. Final optimized implementation

See `src/fast_normalization.py` and `src/fast_blocking.py` (type hints, `logging`, no hidden
global state except the pool-initializer document list, which is reset after serial use,
deterministic ordering, explicit `del` of large intermediates).

Determinism: vocabulary ids are assigned in first-seen order (`unique(maintain_order=True)`),
so top-k tie-breaking does not depend on thread timing. The pool is bit-identical across repeated
runs and across `n_threads`/`geo_n_jobs` settings (checked with 1 vs 4 threads, twice each).
