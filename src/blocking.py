import pandas as pd
import numpy as np
import bm25s
import time
import gc
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn


def _query_and_extract(
    retriever: bm25s.BM25,
    query_tokens,
    query_ids: np.ndarray,
    target_ids: np.ndarray,
    score_col: str,
    top_k: int
) -> pd.DataFrame:
    """Helper to query a BM25 index with pre-tokenized chunks and extract non-empty matches."""
    results, scores = retriever.retrieve(query_tokens, k=top_k, show_progress=False)
    
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
    s1_sub: pd.DataFrame,
    s2_sub: pd.DataFrame,
    s3_sub: pd.DataFrame,
    country_str: str,
    modality: str,
    top_k: int = 15,
    chunk_size: int = 100000,
    n_threads: int = 16
) -> pd.DataFrame:
    """
    Builds S2 and S3 inverted indices once in memory and executes S1->S2, S1->S3,
    and S2->S3 sweeps without redundant tokenization or index re-builds.
    """
    print(f"\n[{country_str.upper()} | {modality.upper()}] S1: {len(s1_sub)} | S2: {len(s2_sub)} | S3: {len(s3_sub)}")
    t0 = time.time()
    
    score_col = f'bm25_{modality}_score'
    
    # 1. Select text representations
    if modality == 'text':
        s1_text = (s1_sub['clean_name'] + " " + s1_sub['clean_address']).fillna("").tolist()
        s2_text = (s2_sub['clean_name'] + " " + s2_sub['clean_address']).fillna("").tolist()
        s3_text = (s3_sub['clean_name'] + " " + s3_sub['clean_address']).fillna("").tolist()
    elif modality == 'phonetic':
        s1_text = s1_sub['phonetic_hash'].fillna("").tolist()
        s2_text = s2_sub['phonetic_hash'].fillna("").tolist()
        s3_text = s3_sub['phonetic_hash'].fillna("").tolist()
    else:
        raise ValueError("modality must be 'text' or 'phonetic'")
        
    pairs = []
    
    # 2. Build S2 and S3 Inverted Indices ONCE
    has_s2 = len(s2_sub) > 0
    has_s3 = len(s3_sub) > 0
    
    retriever_s2 = None
    if has_s2:
        print("  -> Building Target S2 Index...")
        s2_tokens = bm25s.tokenize(s2_text)
        retriever_s2 = bm25s.BM25()
        retriever_s2.index(s2_tokens)
        del s2_tokens
        
    retriever_s3 = None
    if has_s3:
        print("  -> Building Target S3 Index...")
        s3_tokens = bm25s.tokenize(s3_text)
        retriever_s3 = bm25s.BM25()
        retriever_s3.index(s3_tokens)
        del s3_tokens
        
    s2_ids = s2_sub['entity_id'].values
    s3_ids = s3_sub['entity_id'].values

    # 3. Query S1 against S2 and S3 simultaneously
    if len(s1_sub) > 0 and (has_s2 or has_s3):
        print(f"  -> Querying S1 across target indices in chunks of {chunk_size}...")
        s1_ids = s1_sub['entity_id'].values
        
        for i in range(0, len(s1_sub), chunk_size):
            chunk_end = min(i + chunk_size, len(s1_sub))
            chunk_ids = s1_ids[i:chunk_end]
            
            # Tokenize this S1 chunk once for both target retrievers
            chunk_tokens = bm25s.tokenize(s1_text[i:chunk_end])
            
            if retriever_s2 is not None:
                pairs.append(_query_and_extract(retriever_s2, chunk_tokens, chunk_ids, s2_ids, score_col, top_k))
                
            if retriever_s3 is not None:
                pairs.append(_query_and_extract(retriever_s3, chunk_tokens, chunk_ids, s3_ids, score_col, top_k))
                
            del chunk_tokens

    # 4. Query S2 against S3 (Re-using the already indexed S3 retriever)
    if has_s2 and has_s3:
        print(f"  -> Querying S2 against existing S3 Index in chunks of {chunk_size}...")
        for i in range(0, len(s2_sub), chunk_size):
            chunk_end = min(i + chunk_size, len(s2_sub))
            chunk_ids = s2_ids[i:chunk_end]
            chunk_tokens = bm25s.tokenize(s2_text[i:chunk_end])
            
            pairs.append(_query_and_extract(retriever_s3, chunk_tokens, chunk_ids, s3_ids, score_col, top_k))
            del chunk_tokens

    # Clean up index objects for this country/modality
    del retriever_s2
    del retriever_s3
    gc.collect()
    
    merged = pd.concat(pairs, ignore_index=True) if pairs else pd.DataFrame(columns=['entity_A', 'entity_B', score_col])
    print(f"  -> Generated {len(merged)} candidate pairs in {time.time() - t0:.2f}s.")
    return merged


def generate_global_candidate_pool(
    s1: pd.DataFrame, 
    s2: pd.DataFrame, 
    s3: pd.DataFrame, 
    top_k: int = 15,
    chunk_size: int = 100000,
    n_threads: int = 16
) -> pd.DataFrame:
    """
    Executes global BM25 text and phonetic candidate extraction across all sources
    with deduplicated index construction and shared tokenization.
    """
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
            
        # Text pass
        text_df = run_country_global_scan(
            s1_sub, s2_sub, s3_sub, country, 'text', 
            top_k=top_k, chunk_size=chunk_size, n_threads=n_threads
        )
        
        # Phonetic pass
        phonetic_df = run_country_global_scan(
            s1_sub, s2_sub, s3_sub, country, 'phonetic', 
            top_k=top_k, chunk_size=chunk_size, n_threads=n_threads
        )
        
        # Merge modalities for this country
        print(f"  -> Merging text & phonetic candidates for {country.upper()}...")
        country_pairs = pd.merge(
            text_df, phonetic_df, 
            on=['entity_A', 'entity_B'], 
            how='outer'
        ).fillna(0.0)
        
        country_candidates.append(country_pairs)
        
    if not country_candidates:
        return pd.DataFrame(columns=['entity_A', 'entity_B', 'bm25_text_score', 'bm25_phonetic_score'])
        
    master_global = pd.concat(country_candidates, ignore_index=True)
    master_global = master_global.drop_duplicates(subset=['entity_A', 'entity_B'])
    
    print(f"\nTotal Unified Global BM25 Pairs: {len(master_global)}")
    return master_global


def extract_top_k_pairs(s1_ids: np.ndarray, s2_ids: np.ndarray, matches_matrix, score_col: str) -> pd.DataFrame:
    """Helper function to rapidly extract sparse matrix results into a Pandas DataFrame."""
    nonzeros = matches_matrix.nonzero()
    s1_indices = nonzeros[0]
    s2_indices = nonzeros[1]
    scores = matches_matrix.data
    
    return pd.DataFrame({
        'entity_A': s1_ids[s1_indices],
        'entity_B': s2_ids[s2_indices],
        score_col: scores
    })


def execute_local_geo_index(s1_df: pd.DataFrame, s2_df: pd.DataFrame, top_k: int = 15) -> pd.DataFrame:
    """
    Executes fuzzy Character N-Grams strictly within local geographic buckets.
    Bypasses _unknown buckets to prevent dimensionality explosions.
    """
    valid_blocks = set(s1_df['geo_block_key'].unique()) & set(s2_df['geo_block_key'].unique())
    valid_blocks = {b for b in valid_blocks if not b.endswith('_unknown')}
    
    if not valid_blocks:
        return pd.DataFrame(columns=['entity_A', 'entity_B', 'geo_ngram_score'])
        
    print(f"  -> Sweeping {len(valid_blocks)} shared geographic blocks...")
    
    all_geo_pairs = []
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=1)
    
    for block in valid_blocks:
        s1_sub = s1_df[s1_df['geo_block_key'] == block]
        s2_sub = s2_df[s2_df['geo_block_key'] == block]
        
        s2_matrix = vectorizer.fit_transform((s2_sub['clean_name'] + " " + s2_sub['clean_address']).fillna(""))
        s1_matrix = vectorizer.transform((s1_sub['clean_name'] + " " + s1_sub['clean_address']).fillna(""))
        
        matches_matrix = sp_matmul_topn(s1_matrix, s2_matrix.T.tocsr(), top_n=min(top_k, len(s2_sub)))
        
        pairs = extract_top_k_pairs(
            s1_sub['entity_id'].values, 
            s2_sub['entity_id'].values, 
            matches_matrix, 
            'geo_ngram_score'
        )
        all_geo_pairs.append(pairs)
        
    if all_geo_pairs:
        return pd.concat(all_geo_pairs, ignore_index=True)
        
    return pd.DataFrame(columns=['entity_A', 'entity_B', 'geo_ngram_score'])


def run_multi_source_geo_blocking(s1: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame, top_k: int = 15) -> pd.DataFrame:
    """Executes Index 3 across all dataset combinations."""
    print("\n" + "=" * 40)
    print("STAGE 4: INDEX 3 (LOCAL GEO BLOCKING)")
    print("=" * 40)
    
    print("\n[S1 vs S2]")
    geo_12 = execute_local_geo_index(s1, s2, top_k)
    print("\n[S1 vs S3]")
    geo_13 = execute_local_geo_index(s1, s3, top_k)
    print("\n[S2 vs S3]")
    geo_23 = execute_local_geo_index(s2, s3, top_k)
    
    master_geo = pd.concat([geo_12, geo_13, geo_23], ignore_index=True)
    master_geo = master_geo.drop_duplicates(subset=['entity_A', 'entity_B'])
    
    print(f"\nTotal Local Geo Pairs Generated: {len(master_geo)}")
    return master_geo