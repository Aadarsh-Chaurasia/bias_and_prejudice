import re
import pandas as pd
import jellyfish
import gc
from unidecode import unidecode
from pandarallel import pandarallel

# Initialize pandarallel to max out Mac's 8 CPU cores
pandarallel.initialize(nb_workers=8, progress_bar=True)

SUFFIX_MAP = {
    r'\bprivate limited\b': 'ltd', r'\bpvt ltd\b': 'ltd', r'\bpvt\.?\s*ltd\.?\b': 'ltd',
    r'\bcorporation\b': 'corp', r'\bincorporated\b': 'inc', r'\bstreet\b': 'st',
    r'\broad\b': 'rd', r'\bavenue\b': 'ave', r'\bboulevard\b': 'blvd'
}
COMPILED_SUFFIXES = [(re.compile(p), r) for p, r in SUFFIX_MAP.items()]

GEO_STOPWORDS = {
    "drive", "dr", "street", "st", "road", "rd", "avenue", "ave", "lane", "ln",
    "boulevard", "blvd", "no", "number", "plot", "flat", "shop", "floor", "building",
    "bldg", "near", "opp", "opposite", "behind", "city", "town", "pradesh", "state",
    "ltd", "inc", "corp", "co", "llc", "room", "suite", "ste", "apt", "apartment",
    "marg", "nagar", "vihar", "puram", "gali", "bhavan", "mahal", "complex", 
    "bagh", "chowk", "naka", "taluka", "zila", "mandal", "gram", "phase", "sector",
    "khand", "colony", "enclave", "extension", "ext", "cross", "main",
    "rue", "chemin", "place", "allee", "route", "impasse", "batiment", 
    "immeuble", "etage", "cedex", "bp", "boite", "postale"
}

def clean_text(text: str) -> str:
    if pd.isna(text): return ""
    text = unidecode(str(text)).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    for pattern, replacement in COMPILED_SUFFIXES:
        text = pattern.sub(replacement, text)
    return re.sub(r'\s+', ' ', text).strip()

def extract_geo_layers(address: str, country: str):
    if pd.isna(address) or pd.isna(country):
        return {"country": str(country).lower(), "pin": None, "chunk_1": None, "chunk_2": None}
    
    country_str = str(country).lower().strip()
    raw_addr = str(address).lower().strip()
    
    pin_match = re.search(r'\b\d{5,6}\b(?=[^\w]*$|\s*[a-z]{2,}\s*$)', raw_addr)
    pin = pin_match.group(0) if pin_match else None

    chunks = [c.strip() for c in raw_addr.split(',') if c.strip()]
    valid_chunks = []
    
    for chunk in chunks:
        chunk = unidecode(chunk)
        cleaned = re.sub(r'\b\d{5,6}\b', '', chunk)
        cleaned = re.sub(r'[^\w\s]', ' ', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        if len(cleaned) > 1 and cleaned not in GEO_STOPWORDS and cleaned not in {country_str, "usa", "us", "india", "ind", "france"}:
            valid_chunks.append(cleaned.replace(" ", "_"))
            
    return {"country": country_str, "pin": pin, "chunk_1": valid_chunks[-1] if len(valid_chunks) >= 1 else None, "chunk_2": valid_chunks[-2] if len(valid_chunks) >= 2 else None}

def apply_layered_blocking(df: pd.DataFrame, upper_threshold: int = 15000, lower_threshold: int = 50) -> pd.DataFrame:
    geo_data = df.parallel_apply(lambda row: extract_geo_layers(row['business_address'], row['country']), axis=1)
    geo_df = pd.DataFrame(geo_data.tolist(), index=df.index)
    
    def assign_base_key(row):
        if row['pin']: return f"{row['country']}_{row['pin']}"
        elif row['chunk_1']: return f"{row['country']}_{row['chunk_1']}"
        return f"{row['country']}_unknown"
        
    df['geo_block_key'] = geo_df.apply(assign_base_key, axis=1)
    
    key_counts = df['geo_block_key'].value_counts()
    massive_keys = set(key_counts[key_counts > upper_threshold].index)
    
    def apply_layer_3_split(idx, current_key):
        if current_key in massive_keys and not re.search(r'\d{5,6}$', current_key):
            chunk_2 = geo_df.at[idx, 'chunk_2']
            if chunk_2: return f"{geo_df.at[idx, 'country']}_{chunk_2}_{geo_df.at[idx, 'chunk_1']}"
        return current_key
        
    df['geo_block_key'] = [apply_layer_3_split(idx, k) for idx, k in zip(df.index, df['geo_block_key'])]
    
    post_layer3_counts = df['geo_block_key'].value_counts()
    mega_keys = set(post_layer3_counts[post_layer3_counts > upper_threshold].index)
    
    def apply_layer_4_split(row):
        key = row['geo_block_key']
        if key in mega_keys and not re.search(r'\d{5,6}$', key):
            name = str(row['clean_name']).strip()
            return f"{key}_{name[0] if name else 'z'}"
        return key
        
    df['geo_block_key'] = df.apply(apply_layer_4_split, axis=1)
    
    final_counts = df['geo_block_key'].value_counts()
    valid_keys = set(final_counts[final_counts >= lower_threshold].index)
    df['geo_block_key'] = df['geo_block_key'].apply(lambda k: k if k in valid_keys else f"{k.split('_')[0]}_unknown")
    return df

def normalize_pipeline(df: pd.DataFrame, upper_threshold: int = 15000, lower_threshold: int = 50) -> pd.DataFrame:
    print(f"Normalizing {len(df)} rows across 4 CPU cores (Pandarallel)...")
    out = df.copy()
    
    out['clean_name'] = out['business_name'].parallel_apply(clean_text)
    out['clean_address'] = out['business_address'].parallel_apply(clean_text)
    
    out = apply_layered_blocking(out, upper_threshold=upper_threshold, lower_threshold=lower_threshold)
    
    out['phonetic_hash'] = out['clean_name'].parallel_apply(lambda x: jellyfish.metaphone(x) if x else "")
    out['search_document'] = out['clean_name'] + " " + out['clean_address']
    
    out = out.drop(columns=['business_name', 'business_address'])
    gc.collect()
    return out