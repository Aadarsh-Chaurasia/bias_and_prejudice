import pandas as pd
import numpy as np
import bm25s
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

import os
import shutil
import gc

def execute_global_index(s1_df: pd.DataFrame, s2_df: pd.DataFrame, country_str: str, index_type: str, top_k: int = 15) -> pd.DataFrame:
    """Executes global scan using Memory Mapping (mmap) to protect 16GB RAM limits."""
    s1_sub = s1_df[s1_df['country'].str.lower() == country_str].copy()
    s2_sub = s2_df[s2_df['country'].str.lower() == country_str].copy()
    
    if s1_sub.empty or s2_sub.empty:
        return pd.DataFrame()
        
    print(f"\n[{country_str.upper()} | {index_type.upper()}] S1: {len(s1_sub)} | S2: {len(s2_sub)}")
    t0 = time.time()
    
    if index_type == 'text':
        s2_text = (s2_sub['clean_name'] + " " + s2_sub['clean_address']).fillna("").tolist()
        s1_text = (s1_sub['clean_name'] + " " + s1_sub['clean_address']).fillna("").tolist()
        score_col = 'bm25_text_score'
    else:
        s2_text = s2_sub['phonetic_hash'].fillna("").tolist()
        s1_text = s1_sub['phonetic_hash'].fillna("").tolist()
        score_col = 'bm25_phonetic_score'

    # 1. Build and Save the Index to Disk
    print("  -> Tokenizing S2 (Target) text...")
    s2_tokens = bm25s.tokenize(s2_text)
    
    print("  -> Building BM25 Search Engine...")
    retriever = bm25s.BM25()
    retriever.index(s2_tokens)
    
    index_dir = f"temp_index_{country_str}_{index_type}"
    retriever.save(index_dir)
    
    # 2. Aggressive RAM Clearing
    print("  -> Clearing RAM...")
    del retriever
    del s2_tokens
    del s2_text
    gc.collect()
    
    # 3. Load Index via mmap (Zero RAM footprint)
    print("  -> Loading Index via mmap...")
    retriever_mmap = bm25s.BM25.load(index_dir, mmap=True)
    
    # 4. Chunked Querying 
    chunk_size = 50000
    print(f"  -> Querying in chunks of {chunk_size}...")
    
    all_pairs = []
    for i in range(0, len(s1_sub), chunk_size):
        chunk_end = min(i + chunk_size, len(s1_sub))
        
        # Tokenize only this specific chunk
        s1_chunk_tokens = bm25s.tokenize(s1_text[i:chunk_end])
        
        # Query utilizing M4 cores, disabling the internal progress bar
        results, scores = retriever_mmap.retrieve(s1_chunk_tokens, k=top_k, n_threads=8, show_progress=False)
        
        cand_indices = results.flatten()
        flat_scores = scores.flatten()
        
        query_indices = np.repeat(np.arange(i, chunk_end), top_k)
        valid_mask = cand_indices >= 0
        
        chunk_pairs = pd.DataFrame({
            's1_entity_id': s1_sub['entity_id'].values[query_indices[valid_mask]],
            's2_entity_id': s2_sub['entity_id'].values[cand_indices[valid_mask]],
            score_col: flat_scores[valid_mask]
        })
        all_pairs.append(chunk_pairs)
        print(f"     ...Processed {chunk_end}/{len(s1_sub)}")

    # Cleanup temp files
    shutil.rmtree(index_dir)
    
    pairs_df = pd.concat(all_pairs, ignore_index=True) if all_pairs else pd.DataFrame()
    print(f"  -> Found {len(pairs_df)} pairs total in {time.time() - t0:.2f}s.")
    return pairs_df


def run_multi_index_generation(s1_df: pd.DataFrame, s2_df: pd.DataFrame, top_k: int = 15) -> pd.DataFrame:
    """Iterates through all countries, running Index 1 and Index 2, then unions the results."""
    all_text_pairs = []
    all_phonetic_pairs = []
    
    countries = set(s1_df['country'].str.lower().dropna()) | set(s2_df['country'].str.lower().dropna())
    
    for country in countries:
        text_df = execute_global_index(s1_df, s2_df, country, 'text', top_k)
        if not text_df.empty:
            all_text_pairs.append(text_df)
            
        phonetic_df = execute_global_index(s1_df, s2_df, country, 'phonetic', top_k)
        if not phonetic_df.empty:
            all_phonetic_pairs.append(phonetic_df)
            
    final_text = pd.concat(all_text_pairs, ignore_index=True) if all_text_pairs else pd.DataFrame()
    final_phonetic = pd.concat(all_phonetic_pairs, ignore_index=True) if all_phonetic_pairs else pd.DataFrame()
    
    print("\nMerging global indices...")
    merged_pairs = pd.merge(
        final_text, 
        final_phonetic, 
        on=['s1_entity_id', 's2_entity_id'], 
        how='outer'
    ).fillna(0.0)
    
    return merged_pairs

def extract_top_k_pairs(s1_ids: np.ndarray, s2_ids: np.ndarray, matches_matrix, score_col: str) -> pd.DataFrame:
    """Helper function to rapidly extract sparse matrix results into a Pandas DataFrame."""
    nonzeros = matches_matrix.nonzero()
    s1_indices = nonzeros[0]
    s2_indices = nonzeros[1]
    scores = matches_matrix.data
    
    return pd.DataFrame({
        's1_entity_id': s1_ids[s1_indices],
        's2_entity_id': s2_ids[s2_indices],
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
        return pd.DataFrame()
        
    print(f"  -> Sweeping {len(valid_blocks)} shared geographic blocks...")
    
    all_geo_pairs = []
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=1)
    
    for block in valid_blocks:
        s1_sub = s1_df[s1_df['geo_block_key'] == block]
        s2_sub = s2_df[s2_df['geo_block_key'] == block]
        
        # Fit on S2, transform S1
        s2_matrix = vectorizer.fit_transform((s2_sub['clean_name'] + " " + s2_sub['clean_address']).fillna(""))
        s1_matrix = vectorizer.transform((s1_sub['clean_name'] + " " + s1_sub['clean_address']).fillna(""))
        
        # Safe matrix math due to capped bucket sizes
        matches_matrix = sp_matmul_topn(s1_matrix, s2_matrix.T.tocsr(), top_n=min(top_k, len(s2_sub)))
        
        pairs = extract_top_k_pairs(
            s1_sub['entity_id'].values, 
            s2_sub['entity_id'].values, 
            matches_matrix, 
            'geo_ngram_score'
        )
        all_geo_pairs.append(pairs)
        
    if all_geo_pairs:
        final_geo_df = pd.concat(all_geo_pairs, ignore_index=True)
        final_geo_df = final_geo_df.rename(columns={
            's1_entity_id': 'entity_A',
            's2_entity_id': 'entity_B'
        })
        return final_geo_df
        
    return pd.DataFrame(columns=['entity_A', 'entity_B', 'geo_ngram_score'])

def run_multi_source_geo_blocking(s1, s2, s3, top_k=15):
    """Executes Index 3 across all dataset combinations."""
    print("\n" + "="*40)
    print("STAGE 4: INDEX 3 (LOCAL GEO BLOCKING)")
    print("="*40)
    
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