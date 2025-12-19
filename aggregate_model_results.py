"""
Aggregate model predictions for FEVER test datasets with seen/unseen classification.

This script aggregates predictions from multiple models based on their knowledge cutoff dates:
- claude-sonet-3.5: April 1, 2024
- gpt4o-mini: December 31, 2023
- grok-2-1212: December 12, 2024
- llama3.1-70b-instruct: March 31, 2024
- qwen2.5-72b-instruct: September 30, 2024
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from collections import defaultdict

# Model knowledge cutoff dates
MODEL_CUTOFFS = {
    "anthropic.claude-3-5-sonnet-20241022-v2:0": datetime(2024, 4, 1),
    "gpt-4o-mini": datetime(2023, 12, 31),
    "grok-2-1212": datetime(2024, 12, 12),
    "llama-3.1-70b-instruct": datetime(2024, 3, 31),
    "qwen2.5-72b-instruct": datetime(2024, 9, 30)
}

# Normalize model names for directory matching
MODEL_DIR_MAP = {
    "anthropic.claude-3-5-sonnet-20241022-v2:0": "anthropic_claude-3-5-sonnet-20241022-v2_0",
    "gpt-4o-mini": "gpt-4o-mini",
    "grok-2-1212": "grok-2-1212",
    "llama-3.1-70b-instruct": "cf_llama-3_1-70b-instruct",
    "qwen2.5-72b-instruct": "qwen2_5-72b-instruct"
}

DATE_FORMAT = "%d-%m-%Y"


def parse_claim_date(date_str: str) -> datetime:
    """Parse claim date string to datetime object."""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, DATE_FORMAT)
    except:
        try:
            return datetime.fromisoformat(date_str.replace('T00:00:00', ''))
        except:
            return None


def determine_subset(claim_date: datetime, cutoff_date: datetime) -> str:
    """Determine if claim is in seen or unseen subset based on cutoff date."""
    if claim_date is None:
        return "unknown"
    return "seen" if claim_date < cutoff_date else "unseen"


def load_model_predictions(model_name: str, dataset_name: str) -> Dict[str, Any]:
    """Load predictions for a specific model and dataset."""
    results_dir = Path("results")
    model_dir_name = MODEL_DIR_MAP.get(model_name, model_name.replace(":", "_").replace(".", "_"))
    prediction_file = results_dir / model_dir_name / dataset_name / "predictions_with_labels.json"
    
    if not prediction_file.exists():
        print(f"⚠ Warning: Prediction file not found: {prediction_file}")
        return {}
    
    with open(prediction_file, 'r', encoding='utf-8') as f:
        predictions = json.load(f)
    
    # Index by claim_id
    predictions_dict = {pred.get("claim_id"): pred for pred in predictions}
    return predictions_dict


def aggregate_dataset(dataset_name: str) -> List[Dict[str, Any]]:
    """Aggregate predictions from all models for a specific dataset."""
    print(f"\n{'='*80}")
    print(f"Processing dataset: {dataset_name}")
    print(f"{'='*80}\n")
    
    # Load original dataset to get claim IDs and dates
    dataset_path = Path("Datasets/AVeriTeC_FEVER") / f"{dataset_name.replace('fever_', '')}.json"
    with open(dataset_path, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    print(f"Loaded {len(original_data)} claims from original dataset")
    
    # Aggregate predictions
    aggregated = []
    
    for claim_data in original_data:
        claim_id = claim_data.get("claim_id")
        claim_date_str = claim_data.get("claim_date", "")
        claim_date = parse_claim_date(claim_date_str)
        
        aggregated_claim = {
            "claim_id": claim_id,
            "claim": claim_data.get("claim", ""),
            "claim_date": claim_date_str,
            "speaker": claim_data.get("speaker", None),
            "original_claim_url": claim_data.get("original_claim_url", None),
            "reporting_source": claim_data.get("reporting_source", None),
            "location_ISO_code": claim_data.get("location_ISO_code", None),
            "model_predictions": {}
        }
        
        # Load predictions from each model
        for model_name, cutoff_date in MODEL_CUTOFFS.items():
            model_preds = load_model_predictions(model_name, dataset_name)
            
            if claim_id in model_preds:
                pred_data = model_preds[claim_id]
                subset = determine_subset(claim_date, cutoff_date)
                
                aggregated_claim["model_predictions"][model_name] = {
                    "prediction": pred_data.get("prediction", None),
                    "prediction_error": pred_data.get("prediction_error", None),
                    "subset": subset,
                    "knowledge_cutoff": cutoff_date.strftime("%Y-%m-%d")
                }
        
        aggregated.append(aggregated_claim)
    
    # Print statistics
    print(f"\n📊 Statistics for {dataset_name}:")
    print(f"  Total claims: {len(aggregated)}")
    
    for model_name in MODEL_CUTOFFS.keys():
        successful = sum(1 for c in aggregated if model_name in c["model_predictions"] and c["model_predictions"][model_name]["prediction"] is not None)
        failed = sum(1 for c in aggregated if model_name in c["model_predictions"] and c["model_predictions"][model_name]["prediction"] is None)
        seen = sum(1 for c in aggregated if model_name in c["model_predictions"] and c["model_predictions"][model_name]["subset"] == "seen")
        unseen = sum(1 for c in aggregated if model_name in c["model_predictions"] and c["model_predictions"][model_name]["subset"] == "unseen")
        unknown = sum(1 for c in aggregated if model_name in c["model_predictions"] and c["model_predictions"][model_name]["subset"] == "unknown")
        
        print(f"\n  {model_name}:")
        print(f"    Successful predictions: {successful}")
        print(f"    Failed predictions: {failed}")
        print(f"    Seen subset: {seen}")
        print(f"    Unseen subset: {unseen}")
        if unknown > 0:
            print(f"    Unknown date: {unknown}")
    
    return aggregated


def main():
    """Main execution function."""
    print("="*80)
    print("AGGREGATING MODEL PREDICTIONS FOR FEVER TEST DATASETS")
    print("="*80)
    
    datasets = ["fever_test_2023_2024", "fever_test_2025"]
    
    for dataset in datasets:
        aggregated_data = aggregate_dataset(dataset)
        
        # Save aggregated results
        output_file = Path("results") / f"{dataset}_aggregated_predictions.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(aggregated_data, f, indent=4, ensure_ascii=False)
        
        print(f"\n✓ Saved aggregated predictions to: {output_file}")
    
    print("\n" + "="*80)
    print("AGGREGATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
