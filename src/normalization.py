import re
import pandas as pd
import jellyfish
from unidecode import unidecode

# 1. Legal and address suffix mappings
SUFFIX_MAP = {
    r'\bprivate limited\b': 'ltd',
    r'\bpvt ltd\b': 'ltd',
    r'\bpvt\.?\s*ltd\.?\b': 'ltd',
    r'\bcorporation\b': 'corp',
    r'\bincorporated\b': 'inc',
    r'\bstreet\b': 'st',
    r'\broad\b': 'rd',
    r'\bavenue\b': 'ave',
    r'\bboulevard\b': 'blvd'
}
COMPILED_SUFFIXES = [(re.compile(p), r) for p, r in SUFFIX_MAP.items()]

# 2. Multilingual non-geographic address noise tokens
GEO_STOPWORDS = {
    # English (US/India Structural)
    "drive", "dr", "street", "st", "road", "rd", "avenue", "ave", "lane", "ln",
    "boulevard", "blvd", "no", "number", "plot", "flat", "shop", "floor", "building",
    "bldg", "near", "opp", "opposite", "behind", "city", "town", "pradesh", "state",
    "ltd", "inc", "corp", "co", "llc", "room", "suite", "ste", "apt", "apartment",
    
    # Transliterated Indian (Spatial & Administrative)
    "marg", "nagar", "vihar", "puram", "gali", "bhavan", "mahal", "complex", 
    "bagh", "chowk", "naka", "taluka", "zila", "mandal", "gram", "phase", "sector",
    "khand", "colony", "enclave", "extension", "ext", "cross", "main",
    
    # French (For the test set zero-shot generalization)
    "rue", "chemin", "place", "allee", "route", "impasse", "batiment", 
    "immeuble", "etage", "cedex", "bp", "boite", "postale"
}

def clean_text(text: str) -> str:
    """Universal text cleaner. Transliterates all global scripts to ASCII."""
    if pd.isna(text):
        return ""
    
    # Transliterate non-Latin scripts (Devanagari, Cyrillic, Accents) to ASCII
    text = unidecode(str(text))
    text = text.lower()
    
    # Strip punctuation and apply suffixes
    text = re.sub(r'[^\w\s]', ' ', text)
    for pattern, replacement in COMPILED_SUFFIXES:
        text = pattern.sub(replacement, text)
        
    return re.sub(r'\s+', ' ', text).strip()

def extract_geo_layers(address: str, country: str):
    """
    Returns a dictionary containing Layer 1 (PIN) and the trailing chunks for Layer 2/3.
    """
    if pd.isna(address) or pd.isna(country):
        return {"country": str(country).lower(), "pin": None, "chunk_1": None, "chunk_2": None}
    
    country_str = str(country).lower().strip()
    raw_addr = str(address).lower().strip()
    
    # LAYER 1: Strict Postal Code Regex near string end
    pin_match = re.search(r'\b\d{5,6}\b(?=[^\w]*$|\s*[a-z]{2,}\s*$)', raw_addr)
    pin = pin_match.group(0) if pin_match else None

    # Layers 2/3: Comma split processing
    chunks = [c.strip() for c in raw_addr.split(',') if c.strip()]
    valid_chunks = []
    
    for chunk in chunks:
        # Pass the chunk through unidecode to match the clean_text logic
        chunk = unidecode(chunk)
        cleaned = re.sub(r'\b\d{5,6}\b', '', chunk)
        cleaned = re.sub(r'[^\w\s]', ' ', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        if len(cleaned) > 1 and cleaned not in GEO_STOPWORDS and cleaned not in {country_str, "usa", "us", "india", "ind", "france"}:
            valid_chunks.append(cleaned.replace(" ", "_"))
            
    chunk_1 = valid_chunks[-1] if len(valid_chunks) >= 1 else None
    chunk_2 = valid_chunks[-2] if len(valid_chunks) >= 2 else None
    
    return {"country": country_str, "pin": pin, "chunk_1": chunk_1, "chunk_2": chunk_2}

def apply_layered_blocking(df: pd.DataFrame, upper_threshold: int = 15000, lower_threshold: int = 50) -> pd.DataFrame:
    """
    Executes a 4-Layer spatial blocking strategy to balance RAM constraints and recall.
    """
    # LAYER 1 & 2: Extract base geographic chunks and PIN codes
    geo_data = df.apply(lambda row: extract_geo_layers(row['business_address'], row['country']), axis=1)
    geo_df = pd.DataFrame(geo_data.tolist())
    
    def assign_base_key(row):
        if row['pin']: return f"{row['country']}_{row['pin']}"
        elif row['chunk_1']: return f"{row['country']}_{row['chunk_1']}"
        return f"{row['country']}_unknown"
        
    df['geo_block_key'] = geo_df.apply(assign_base_key, axis=1)
    
    # LAYER 3: Sub-cluster massive state-level blocks into cities
    key_counts = df['geo_block_key'].value_counts()
    massive_keys = set(key_counts[key_counts > upper_threshold].index)
    
    def apply_layer_3_split(idx, current_key):
        if current_key in massive_keys and not re.search(r'\d{5,6}$', current_key):
            chunk_2 = geo_df.at[idx, 'chunk_2']
            if chunk_2:
                return f"{geo_df.at[idx, 'country']}_{chunk_2}_{geo_df.at[idx, 'chunk_1']}"
        return current_key
        
    df['geo_block_key'] = [apply_layer_3_split(i, k) for i, k in enumerate(df['geo_block_key'])]
    
    # LAYER 4: The Alphabetic Failsafe for Mega-Cities
    post_layer3_counts = df['geo_block_key'].value_counts()
    mega_keys = set(post_layer3_counts[post_layer3_counts > upper_threshold].index)
    
    def apply_layer_4_split(row):
        key = row['geo_block_key']
        if key in mega_keys and not re.search(r'\d{5,6}$', key):
            name = str(row['clean_name']).strip()
            # Grab the first character, default to 'z' if the name string is empty
            first_char = name[0] if name else 'z'
            return f"{key}_{first_char}"
        return key
        
    df['geo_block_key'] = df.apply(apply_layer_4_split, axis=1)
    
    # DUST SWEEPING: Enforce the lower threshold to prevent Python loop bottlenecks
    final_counts = df['geo_block_key'].value_counts()
    valid_keys = set(final_counts[final_counts >= lower_threshold].index)
    
    df['geo_block_key'] = df['geo_block_key'].apply(
        lambda k: k if k in valid_keys else f"{k.split('_')[0]}_unknown"
    )
    
    return df

def normalize_pipeline(df: pd.DataFrame, upper_threshold: int = 15000, lower_threshold: int = 50) -> pd.DataFrame:
    """
    Ingests a raw dataframe and outputs the fully normalized version with search indices.
    """
    out = df.copy()
    
    out['clean_name'] = out['business_name'].apply(clean_text)
    out['clean_address'] = out['business_address'].apply(clean_text)
    
    out = apply_layered_blocking(out, upper_threshold=upper_threshold, lower_threshold=lower_threshold)
    
    out['phonetic_hash'] = out['clean_name'].apply(
        lambda x: jellyfish.metaphone(x) if x else ""
    )
    
    out['search_document'] = out['clean_name'] + " " + out['clean_address']
    return out