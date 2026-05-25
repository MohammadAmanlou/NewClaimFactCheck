import os
import pandas as pd
from collections import Counter
from src.dataset_registry import DatasetRegistry
from src.label_utils import normalize_label

def test_dataset_loading_and_labels():
    print("Testing DatasetRegistry and Label Normalization...")
    
    # 1. Initialize registry
    yaml_path = "datasets.yaml"
    if not os.path.exists(yaml_path):
        print(f"Error: {yaml_path} not found.")
        return

    registry = DatasetRegistry(yaml_path)
    
    # 2. Extract specific dataset
    dataset_name = "combined_biased"
    if dataset_name not in registry.definitions:
        print(f"Error: {dataset_name} not found in {yaml_path}.")
        return

    print(f"\nLoading '{dataset_name}' dataset...")
    # DatasetRegistry.load requires name, data_dir, output_date_format
    from pathlib import Path
    claims, is_test = registry.load(dataset_name, Path("data"), "%Y-%m-%d")
    total_claims = len(claims)
    print(f"Successfully loaded {total_claims} claims.")
    
    if total_claims == 0:
        print("Dataset is empty!")
        return

    # 3. Test mapping functionality via loop
    mapped_counts = Counter()
    unmapped = []
    
    for claim in claims:
        # Note: dataset_registry maps "factcheck_label_norm" -> "label"
        raw_val = claim.get("label") 
        canonical_label = normalize_label(raw_val)
        
        if canonical_label:
            mapped_counts[canonical_label] += 1
        else:
            unmapped.append(str(raw_val))

    # 4. Display Coverage Metrics
    mapped_total = sum(mapped_counts.values())
    coverage = (mapped_total / total_claims) * 100

    print("\n--- Canonical Label Distribution ---")
    for label, count in mapped_counts.items():
        print(f"{label}: {count} ({count/total_claims*100:.2f}%)")

    print("\n--- Overall Coverage ---")
    print(f"Mapped successfully: {mapped_total}")
    print(f"Unmapped (Discarded): {len(unmapped)}")
    print(f"Coverage Rate: {coverage:.2f}%")

    # 5. Display what is still slipping through
    print("\n--- Top 10 Most Frequent Unmapped Values ---")
    unmapped_counter = Counter(unmapped)
    for val, count in unmapped_counter.most_common(10):
        # Safely encode and decode string prints just in case
        safe_val = str(val).encode('ascii', 'replace').decode()
        print(f"{safe_val}: {count}")

if __name__ == "__main__":
    test_dataset_loading_and_labels()
