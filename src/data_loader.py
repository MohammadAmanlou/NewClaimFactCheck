import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple
from datasets import load_dataset
from .config import Config

def load_claimreview_plus(config: Config) -> List[Dict[str, Any]]:
    print("Loading ClaimReview Plus from HuggingFace...")
    dataset = load_dataset("MAI-Lab/ClaimReview2024plus", split="test", token=config.hf_token)
    claims = []
    for idx, item in enumerate(dataset):
        claims.append({
            "claim_id": idx,
            "claim": item.get("claim", ""),
            "claim_date": item.get("claimDate", ""),
            "label": item.get("rating", ""),
            "speaker": item.get("claimant", None),
            "source": "claimreview_plus"
        })
    print(f"✓ Loaded {len(claims)} claims from ClaimReview Plus")
    return claims

def load_factors(config: Config) -> List[Dict[str, Any]]:
    csv_path = Path("Datasets/FACTor/FACTors.csv")  # adjust if needed
    print(f"Loading FACTors from {csv_path}...")
    df = pd.read_csv(csv_path)

    excluded_labels = ['other', 'unverifiable']
    original_count = len(df)
    df = df[~df['normalised_rating'].isin(excluded_labels)]
    filtered_count = original_count - len(df)
    print(f"  Filtered out {filtered_count} claims with labels: {excluded_labels}")

    claims = []
    for _, row in df.iterrows():
        try:
            date_str = str(row['date_published']).replace('T00:00:00', '')
            date_obj = datetime.fromisoformat(date_str)
            formatted_date = date_obj.strftime(config.date_format)
        except:
            formatted_date = ""

        claims.append({
            "claim_id": row['claim_id'],
            "claim": row['claim'],
            "claim_date": formatted_date,
            "label": str(row['normalised_rating']).capitalize() if pd.notna(row['normalised_rating']) else "",
            "speaker": row.get('author', None),
            "source": "factors"
        })
    print(f"✓ Loaded {len(claims)} claims from FACTors (after filtering)")
    return claims

def load_fever_json(file_path: Path, is_test: bool = False) -> List[Dict[str, Any]]:
    print(f"Loading FEVER dataset from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    claims = []
    for item in data:
        claim_dict = {
            "claim_id": item.get("claim_id", len(claims)),
            "claim": item["claim"],
            "claim_date": item.get("claim_date", ""),
            "speaker": item.get("speaker", None),
            "source": file_path.stem
        }
        if not is_test:
            claim_dict["label"] = item.get("label", "")
        if is_test:
            claim_dict["original_data"] = item
        claims.append(claim_dict)
    print(f"✓ Loaded {len(claims)} claims from {file_path.name}")
    return claims

def load_dataset(dataset_name: str, config: Config) -> Tuple[List[Dict[str, Any]], bool]:
    base_path = Path("Datasets")
    if dataset_name == "claimreview_plus":
        return load_claimreview_plus(config), False
    elif dataset_name == "factors":
        return load_factors(config), False
    elif dataset_name == "fever_train":
        return load_fever_json(base_path / "AVeriTeC_FEVER" / "train.json", is_test=False), False
    elif dataset_name == "fever_dev":
        return load_fever_json(base_path / "AVeriTeC_FEVER" / "data_dev.json", is_test=False), False
    elif dataset_name == "fever_test_2023_2024":
        return load_fever_json(base_path / "AVeriTeC_FEVER" / "test_2023_2024.json", is_test=True), True
    elif dataset_name == "fever_test_2025":
        return load_fever_json(base_path / "AVeriTeC_FEVER" / "test_2025.json", is_test=True), True
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")