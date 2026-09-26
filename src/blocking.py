import time
import gc
import numpy as np
import pandas as pd
import bm25s
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn


def _query_and_extract(
    retriever: bm25s.BM25, query_tokens, query_ids: np.ndarray, 
    target_ids: np.ndarray, score_col: str, top_k: int, n_threads: int
) -> pd.DataFrame:
    """Helper to query a BM25 index and extract non-empty matches with explicit threading."""
    # Explicitly pass n_threads to force the backend to utilize all cores
    results, scores = retriever.retrieve(query_tokens, k=top_k, n_threads=n_threads, show_progress=False)
    
    cand_indices = results.flatten()
    flat_scores = scores.flatten()
    
    query_indices = np.repeat(np.arange(len(query_ids)), top_k)
    valid_mask = cand_indices >= 0
    
    if not np.any(valid_mask):
        return pd.DataFrame(columns=['entity_A', 'entity_B', score_col])
        
    return pd.DataFrame({
        'entity_A': query_ids[query_indices[valid_mask]],
        'entity_B': target_ids[cand_indices[valid_mask]],
        score_col: flat_scores[valid_mask]
    })


def run_country_global_scan(
    s1_sub: pd.DataFrame, s2_sub: pd.DataFrame, s3_sub: pd.DataFrame,
    country_str: str, modality: str, top_k: int = 7, chunk_size: int = 50000, n_threads: int = 4
) -> pd.DataFrame:
    """Hybrid Architecture: Tokenize all at once, index once, but query in safe CPU chunks."""
    print(f"\n[{country_str.upper()} | {modality.upper()}] S1: {len(s1_sub)} | S2: {len(s2_sub)} | S3: {len(s3_sub)}")
    t0 = time.time()
    score_col = f'bm25_{modality}_score'
    
    # 1. Text Selection
    if modality == 'text':
        s1_text = s1_sub['search_document'].fillna("").tolist()
        s2_text = s2_sub['search_document'].fillna("").tolist()
        s3_text = s3_sub['search_document'].fillna("").tolist()
    else:
        s1_text = s1_sub['phonetic_hash'].fillna("").tolist()
        s2_text = s2_sub['phonetic_hash'].fillna("").tolist()
        s3_text = s3_sub['phonetic_hash'].fillna("").tolist()
        
    pairs = []
    
    # 2. Tokenize Once
    print("  -> Pre-tokenizing texts to RAM...")
    s2_tokens = bm25s.tokenize(s2_text) if len(s2_sub) > 0 else None
    s3_tokens = bm25s.tokenize(s3_text) if len(s3_sub) > 0 else None
    s1_tokens_all = bm25s.tokenize(s1_text) if len(s1_sub) > 0 else None
    
    # 3. Build Indices Once
    retriever_s2, retriever_s3 = None, None
    if s2_tokens is not None:
        print("  -> Building S2 Index...")
        retriever_s2 = bm25s.BM25()
        retriever_s2.index(s2_tokens)
    if s3_tokens is not None:
        print("  -> Building S3 Index...")
        retriever_s3 = bm25s.BM25()
        retriever_s3.index(s3_tokens)
        
    s1_ids = s1_sub['entity_id'].values
    s2_ids = s2_sub['entity_id'].values
    s3_ids = s3_sub['entity_id'].values

    # 4. CPU-Friendly Chunked Querying
    if s1_tokens_all is not None:
        print(f"  -> Querying S1 across targets in {chunk_size} row batches...")
        for i in range(0, len(s1_ids), chunk_size):
            chunk_end = min(i + chunk_size, len(s1_ids))
            
            # Sub-slice the pre-tokenized object directly to prevent stalling
            chunk_tokens = bm25s.tokenization.Tokenized(
                vocab=s1_tokens_all.vocab,
                ids=s1_tokens_all.ids[i:chunk_end]
            )
            
            if retriever_s2 is not None:
                pairs.append(_query_and_extract(retriever_s2, chunk_tokens, s1_ids[i:chunk_end], s2_ids, score_col, top_k, n_threads))
            if retriever_s3 is not None:
                pairs.append(_query_and_extract(retriever_s3, chunk_tokens, s1_ids[i:chunk_end], s3_ids, score_col, top_k, n_threads))
            
            print(f"     ...Processed S1 queries: {chunk_end}/{len(s1_ids)}")

    if s2_tokens is not None and retriever_s3 is not None:
        print(f"  -> Querying S2 against S3 in {chunk_size} row batches...")
        for i in range(0, len(s2_ids), chunk_size):
            chunk_end = min(i + chunk_size, len(s2_ids))
            chunk_tokens = bm25s.tokenization.Tokenized(
                vocab=s2_tokens.vocab,
                ids=s2_tokens.ids[i:chunk_end]
            )
            pairs.append(_query_and_extract(retriever_s3, chunk_tokens, s2_ids[i:chunk_end], s3_ids, score_col, top_k, n_threads))
            print(f"     ...Processed S2 queries: {chunk_end}/{len(s2_ids)}")

    # Extreme RAM wipe
    del s1_tokens_all, s2_tokens, s3_tokens
    del retriever_s2, retriever_s3
    del s1_text, s2_text, s3_text
    gc.collect()
    
    merged = pd.concat(pairs, ignore_index=True) if pairs else pd.DataFrame(columns=['entity_A', 'entity_B', score_col])
    print(f"  -> Generated {len(merged)} candidate pairs in {time.time() - t0:.2f}s.")
    return merged


def generate_global_candidate_pool(
    s1: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame, 
    top_k: int = 7, chunk_size: int = 50000, n_threads: int = 4
) -> pd.DataFrame:
    """Executes global candidate extraction with hybrid batch chunking."""
    print("=" * 40)
    print("STAGES 1-3: GLOBAL BM25 CANDIDATE EXTRACTION")
    print("=" * 40)
    
    countries = set(s1['country'].str.lower().dropna()) | \
                set(s2['country'].str.lower().dropna()) | \
                set(s3['country'].str.lower().dropna())
                
    country_candidates = []
    for country in sorted(countries):
        s1_sub = s1[s1['country'].str.lower() == country]
        s2_sub = s2[s2['country'].str.lower() == country]
        s3_sub = s3[s3['country'].str.lower() == country]
        
        if s1_sub.empty and s2_sub.empty and s3_sub.empty:
            continue
            
        text_df = run_country_global_scan(s1_sub, s2_sub, s3_sub, country, 'text', top_k, chunk_size, n_threads)
        phonetic_df = run_country_global_scan(s1_sub, s2_sub, s3_sub, country, 'phonetic', top_k, chunk_size, n_threads)
        
        print(f"  -> Merging text & phonetic candidates for {country.upper()}...")
        country_pairs = pd.merge(text_df, phonetic_df, on=['entity_A', 'entity_B'], how='outer').fillna(0.0)
        country_candidates.append(country_pairs)
        
    if not country_candidates:
        return pd.DataFrame(columns=['entity_A', 'entity_B', 'bm25_text_score', 'bm25_phonetic_score'])
        
    master_global = pd.concat(country_candidates, ignore_index=True)
    master_global = master_global.drop_duplicates(subset=['entity_A', 'entity_B'])
    
    print(f"\nTotal Unified Global BM25 Pairs: {len(master_global)}")
    return master_global


def extract_top_k_pairs(s1_ids: np.ndarray, s2_ids: np.ndarray, matches_matrix, score_col: str) -> pd.DataFrame:
    """Rapid sparse matrix extraction."""
    nonzeros = matches_matrix.nonzero()
    return pd.DataFrame({
        'entity_A': s1_ids[nonzeros[0]],
        'entity_B': s2_ids[nonzeros[1]],
        score_col: matches_matrix.data
    })


def execute_local_geo_index(s1_df: pd.DataFrame, s2_df: pd.DataFrame, top_k: int = 7) -> pd.DataFrame:
    """Executes fuzzy Character N-Grams strictly within valid local geographic buckets."""
    valid_blocks = {b for b in (set(s1_df['geo_block_key'].unique()) & set(s2_df['geo_block_key'].unique())) if not b.endswith('_unknown')}
    
    if not valid_blocks:
        return pd.DataFrame(columns=['entity_A', 'entity_B', 'geo_ngram_score'])
        
    print(f"  -> Sweeping {len(valid_blocks)} shared geographic blocks...")
    all_geo_pairs = []
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=1)
    
    for block in valid_blocks:
        s1_sub = s1_df[s1_df['geo_block_key'] == block]
        s2_sub = s2_df[s2_df['geo_block_key'] == block]
        
        s2_matrix = vectorizer.fit_transform(s2_sub['search_document'].fillna(""))
        s1_matrix = vectorizer.transform(s1_sub['search_document'].fillna(""))
        matches_matrix = sp_matmul_topn(s1_matrix, s2_matrix.T.tocsr(), top_n=min(top_k, len(s2_sub)))
        
        pairs = extract_top_k_pairs(s1_sub['entity_id'].values, s2_sub['entity_id'].values, matches_matrix, 'geo_ngram_score')
        all_geo_pairs.append(pairs)
        
    return pd.concat(all_geo_pairs, ignore_index=True) if all_geo_pairs else pd.DataFrame(columns=['entity_A', 'entity_B', 'geo_ngram_score'])


def run_multi_source_geo_blocking(s1: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame, top_k: int = 7) -> pd.DataFrame:
    print("\n" + "=" * 40)
    print("STAGE 4: INDEX 3 (LOCAL GEO BLOCKING)")
    print("=" * 40)
    
    geo_12 = execute_local_geo_index(s1, s2, top_k)
    geo_13 = execute_local_geo_index(s1, s3, top_k)
    geo_23 = execute_local_geo_index(s2, s3, top_k)
    
    master_geo = pd.concat([geo_12, geo_13, geo_23], ignore_index=True).drop_duplicates(subset=['entity_A', 'entity_B'])
    print(f"\nTotal Local Geo Pairs Generated: {len(master_geo)}")
    return master_geo