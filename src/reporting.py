import json
import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from .config import Config

def save_test_predictions(results: List[Dict], output_file: Path, logger: logging.Logger):
    output_data = []
    for result in results:
        if "original_data" in result:
            item = result["original_data"].copy()
        else:
            item = {
                "claim": result["claim"],
                "claim_id": result["claim_id"],
                "claim_date": result.get("claim_date", ""),
                "speaker": result.get("speaker", None),
                "original_claim_url": result.get("original_claim_url", None),
                "reporting_source": result.get("reporting_source", None),
                "location_ISO_code": result.get("location_ISO_code", None)
            }
        item["prediction"] = result.get("prediction", None)
        if result.get("prediction_error"):
            item["prediction_error"] = result["prediction_error"]
        output_data.append(item)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    logger.info(f"Test predictions saved to {output_file}")
    print(f"✓ Test predictions saved to {output_file}")

def save_results_with_metrics(results: List[Dict], metrics: Dict, output_dir: Path, split_name: str, logger: logging.Logger):
    # CSV
    pred_csv = output_dir / f"{split_name}_predictions.csv"
    pd.DataFrame(results).to_csv(pred_csv, index=False, encoding='utf-8')
    logger.info(f"Predictions saved to {pred_csv}")

    # JSON
    pred_json = output_dir / f"{split_name}_predictions.json"
    with open(pred_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    logger.info(f"Predictions JSON saved to {pred_json}")

    # Metrics
    metrics_file = output_dir / f"{split_name}_metrics.json"
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Metrics saved to {metrics_file}")

    print(f"✓ {split_name.capitalize()} results saved to {output_dir}")

def generate_summary_report(
    seen_metrics: dict,
    unseen_metrics: dict,
    comparison: dict,
    config: Config,
    output_dir: Path,
    logger: logging.Logger,
) -> None:
    summary = {
        "experiment_info": {
            "dataset": config.dataset,
            "model": config.model,
            "temporal_split_date": config.temporal_split,
            "canonical_labels": config.labels,
            "balanced_sampling": comparison.get("sampling_info", {}).get("enabled", False),
            "sampling_seed": comparison.get("sampling_info", {}).get("seed"),
            "prompt_method": comparison.get("prompt_method"),   # already saved
            "timestamp": datetime.now().isoformat(),
        },
        "seen_data": seen_metrics,
        "unseen_data": unseen_metrics,
        "comparison": comparison,
    }

    summary_file = output_dir / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
    logger.info("Summary report saved to %s", summary_file)
    print(f"✓ Summary report saved to {summary_file}")

    # Console output
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    print(f"Dataset: {config.dataset}")
    print(f"Model: {config.model}")
    # Show prompt method if available
    method = comparison.get("prompt_method")
    if method:
        print(f"Prompt Method: {method}")
    print(f"Temporal Split: {config.temporal_split}")

    si = comparison.get("sampling_info", {})
    if si.get("enabled"):
        print("\nBALANCED SAMPLING:")
        print(f"  Seed: {si['seed']}")
        if "original_seen_count" in si:
            print(f"  Original Seen: {si['original_seen_count']} -> Sampled: {si['sampled_seen_count']}")

    print("\nSEEN DATA PERFORMANCE:")
    print(f"  Accuracy: {seen_metrics.get('accuracy', 'N/A')}")
    print(f"  Macro F1: {seen_metrics.get('macro_f1', 'N/A')}")
    print(f"  Samples: {seen_metrics.get('total_samples', 'N/A')}")
    print("\nUNSEEN DATA PERFORMANCE:")
    print(f"  Accuracy: {unseen_metrics.get('accuracy', 'N/A')}")
    print(f"  Macro F1: {unseen_metrics.get('macro_f1', 'N/A')}")
    print(f"  Samples: {unseen_metrics.get('total_samples', 'N/A')}")

    if "nei_prediction_rate" in unseen_metrics:
        print("\nNEI METRICS (post‑split):")
        print(f"  NEI Prediction Rate: {unseen_metrics.get('nei_prediction_rate', 'N/A')}")
        print(f"  False NEI Rate: {unseen_metrics.get('false_nei_rate', 'N/A')}")

    if isinstance(comparison.get("statistical_tests"), dict):
        print("\nSTATISTICAL SIGNIFICANCE:")
        print(f"  Accuracy Difference: {comparison['accuracy_difference']:.4f}")
        print(f"  T-test p-value: {comparison['statistical_tests']['t_test']['p_value']:.4f}")
        print(f"  Chi-square p-value: {comparison['statistical_tests']['chi_square_test']['p_value']:.4f}")
        print(f"  Interpretation: {comparison['statistical_tests']['t_test']['interpretation']}")
    print("=" * 80)