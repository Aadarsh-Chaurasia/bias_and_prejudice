import pandas as pd
import numpy as np
import bm25s
import time

def execute_global_index(s1_df: pd.DataFrame, s2_df: pd.DataFrame, country_str: str, index_type: str, top_k: int = 15) -> pd.DataFrame:
    """Executes a hyper-fast global scan using a BM25 Inverted Index."""
    s1_sub = s1_df[s1_df['country'].str.lower() == country_str].copy()
    s2_sub = s2_df[s2_df['country'].str.lower() == country_str].copy()
    
    if s1_sub.empty or s2_sub.empty:
        return pd.DataFrame()
        
    print(f"\n[{country_str.upper()} | {index_type.upper()}] S1: {len(s1_sub)} | S2: {len(s2_sub)}")
    t0 = time.time()
    
    # Define text targets based on index type
    if index_type == 'text':
        s2_text = (s2_sub['clean_name'] + " " + s2_sub['clean_address']).fillna("").tolist()
        s1_text = (s1_sub['clean_name'] + " " + s1_sub['clean_address']).fillna("").tolist()
        score_col = 'bm25_text_score'
    elif index_type == 'phonetic':
        s2_text = s2_sub['phonetic_hash'].fillna("").tolist()
        s1_text = s1_sub['phonetic_hash'].fillna("").tolist()
        score_col = 'bm25_phonetic_score'
    else:
        raise ValueError("index_type must be 'text' or 'phonetic'")

    # 1. Tokenize (Multithreaded in C/Rust)
    print("  -> Tokenizing text...")
    s2_tokens = bm25s.tokenize(s2_text)
    s1_tokens = bm25s.tokenize(s1_text)
    
    # 2. Build the Inverted Index
    print("  -> Building BM25 Search Engine...")
    retriever = bm25s.BM25()
    retriever.index(s2_tokens)
    
    # 3. Query the Index (Bypasses the 1.7 Trillion cross-products entirely)
    print(f"  -> Querying Top {top_k} matches...")
    # Results shape: (n_queries, top_k). Scores shape: (n_queries, top_k)
    results, scores = retriever.retrieve(s1_tokens, k=top_k)
    
    # 4. Fast Numpy Extraction
    print("  -> Mapping IDs...")
    # Flatten the arrays
    cand_indices = results.flatten()
    flat_scores = scores.flatten()
    
    # Repeat S1 queries to match the flattened candidates
    query_indices = np.repeat(np.arange(len(s1_sub)), top_k)
    
    # Filter out empty padded slots (if bm25 couldn't find K matches, it returns -1)
    valid_mask = cand_indices >= 0
    
    s1_ids = s1_sub['entity_id'].values[query_indices[valid_mask]]
    s2_ids = s2_sub['entity_id'].values[cand_indices[valid_mask]]
    valid_scores = flat_scores[valid_mask]
    
    pairs_df = pd.DataFrame({
        's1_entity_id': s1_ids,
        's2_entity_id': s2_ids,
        score_col: valid_scores
    })
    
    print(f"  -> Found {len(pairs_df)} pairs total in {time.time() - t0:.2f}s.")
    return pairs_df

def run_multi_index_generation(s1_df: pd.DataFrame, s2_df: pd.DataFrame, top_k: int = 15) -> pd.DataFrame:
    """Iterates through all countries, running Index 1 and Index 2, then unions the results."""
    all_text_pairs = []
    all_phonetic_pairs = []
    
    countries = set(s1_df['country'].str.lower().dropna()) | set(s2_df['country'].str.lower().dropna())
    
    for country in countries:
        # Index 1: Global Lexical Search
        text_df = execute_global_index(s1_df, s2_df, country, 'text', top_k)
        if not text_df.empty:
            all_text_pairs.append(text_df)
            
        # Index 2: Global Phonetic Search
        phonetic_df = execute_global_index(s1_df, s2_df, country, 'phonetic', top_k)
        if not phonetic_df.empty:
            all_phonetic_pairs.append(phonetic_df)
            
    final_text = pd.concat(all_text_pairs, ignore_index=True) if all_text_pairs else pd.DataFrame()
    final_phonetic = pd.concat(all_phonetic_pairs, ignore_index=True) if all_phonetic_pairs else pd.DataFrame()
    
    # Union the indices
    print("\nMerging global indices...")
    merged_pairs = pd.merge(
        final_text, 
        final_phonetic, 
        on=['s1_entity_id', 's2_entity_id'], 
        how='outer'
    ).fillna(0.0)
    
    return merged_pairs