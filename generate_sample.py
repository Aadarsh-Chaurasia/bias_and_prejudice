import pandas as pd
import os
import argparse

def create_deterministic_sample(
    s1_path: str, 
    s2_path: str, 
    s3_path: str, 
    gt_path: str, 
    output_dir: str = "data/sample",
    gt_sample_size: int = 1500, 
    source_noise_size: int = 8500
):
    """
    Creates a mathematically sound sample dataset for local Entity Resolution testing.
    Ensures all sampled ground truth pairs physically exist in the source files.
    """
    print(f"Loading full datasets from {os.path.dirname(s1_path)}...")
    s1_full = pd.read_csv(s1_path, sep='\t')
    s2_full = pd.read_csv(s2_path, sep='\t')
    s3_full = pd.read_csv(s3_path, sep='\t')
    gt_full = pd.read_csv(gt_path, sep='\t')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Sample the Ground Truth (The Anchors)
    print(f"Sampling {gt_sample_size} true clusters from Ground Truth...")
    gt_valid = gt_full.dropna(subset=['matched_entity_ids'])
    gt_sample = gt_valid.sample(n=gt_sample_size, random_state=42)
    
    # Explode the comma-separated IDs in-memory to get all valid anchor IDs
    s1_anchors_set = set(gt_sample['source1_entity_id'])
    s2_s3_anchors_set = set(
        gt_sample['matched_entity_ids'].str.split(',').explode().str.strip()
    )
    all_anchored_ids = s1_anchors_set.union(s2_s3_anchors_set)
    
    # 2. Extract Anchor Rows from Sources
    print("Extracting anchored rows from source datasets...")
    s1_anchors = s1_full[s1_full['entity_id'].isin(all_anchored_ids)]
    s2_anchors = s2_full[s2_full['entity_id'].isin(all_anchored_ids)]
    s3_anchors = s3_full[s3_full['entity_id'].isin(all_anchored_ids)]
    
    # 3. Add Random Noise (Negative Samples) to simulate real-world blocking
    print(f"Injecting ~{source_noise_size} random noise records into each source...")
    s1_noise = s1_full[~s1_full['entity_id'].isin(all_anchored_ids)].sample(n=min(source_noise_size, len(s1_full)), random_state=42)
    s2_noise = s2_full[~s2_full['entity_id'].isin(all_anchored_ids)].sample(n=min(source_noise_size, len(s2_full)), random_state=42)
    s3_noise = s3_full[~s3_full['entity_id'].isin(all_anchored_ids)].sample(n=min(source_noise_size, len(s3_full)), random_state=42)
    
    # 4. Combine and Shuffle
    s1_sample = pd.concat([s1_anchors, s1_noise]).sample(frac=1, random_state=42).reset_index(drop=True)
    s2_sample = pd.concat([s2_anchors, s2_noise]).sample(frac=1, random_state=42).reset_index(drop=True)
    s3_sample = pd.concat([s3_anchors, s3_noise]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 5. Export
    print(f"Saving files to {output_dir}/ ...")
    s1_sample.to_csv(os.path.join(output_dir, "sample_source1.tsv"), sep='\t', index=False)
    s2_sample.to_csv(os.path.join(output_dir, "sample_source2.tsv"), sep='\t', index=False)
    s3_sample.to_csv(os.path.join(output_dir, "sample_source3.tsv"), sep='\t', index=False)
    gt_sample.to_csv(os.path.join(output_dir, "sample_ground_truth.tsv"), sep='\t', index=False)
    
    print("\n=== Sample Generation Complete ===")
    print(f"S1 Sample Size: {len(s1_sample)}")
    print(f"S2 Sample Size: {len(s2_sample)}")
    print(f"S3 Sample Size: {len(s3_sample)}")
    print(f"Ground Truth Size: {len(gt_sample)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate deterministic data samples.")
    parser.add_argument("--train_dir", type=str, default="../student_resource/dataset/train", help="Directory containing the full training data")
    parser.add_argument("--output_dir", type=str, default="data/sample", help="Directory to save the sample data")
    parser.add_argument("--gt_size", type=int, default=1500, help="Number of ground truth pairs to sample")
    parser.add_argument("--noise_size", type=int, default=8500, help="Number of random noise rows to inject per source")
    
    args = parser.parse_args()
    
    s1_file = os.path.join(args.train_dir, "train_source1.tsv")
    s2_file = os.path.join(args.train_dir, "train_source2.tsv")
    s3_file = os.path.join(args.train_dir, "train_source3.tsv")
    gt_file = os.path.join(args.train_dir, "train_ground_truth.tsv")
    
    create_deterministic_sample(
        s1_path=s1_file,
        s2_path=s2_file,
        s3_path=s3_file,
        gt_path=gt_file,
        output_dir=args.output_dir,
        gt_sample_size=args.gt_size,
        source_noise_size=args.noise_size
    )