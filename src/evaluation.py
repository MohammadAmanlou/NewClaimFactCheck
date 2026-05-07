"""
Evaluation module for fact-checking pipeline.

Provides functions to compute accuracy, macro‑averaged metrics,
per‑label metrics, and statistical comparison between seen/unseen splits.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from scipy import stats
from .label_utils import normalize_label


# ---------------------------------------------------------------------------
# Helper functions – clean, single‑purpose utilities
# ---------------------------------------------------------------------------

def _get_valid_predictions(
    results: List[Dict],
) -> List[Dict]:
    """Return results where both a valid prediction and a normalisable label exist."""
    return [
        r for r in results
        if r.get("prediction") and r.get("label")
        and normalize_label(r["label"])
    ]


def _build_confusion_matrix(
    valid_results: List[Dict],
) -> Tuple[Dict[str, int], Dict[str, Dict[str, int]], int]:
    """
    Compute label counts, confusion dict, and total correct predictions.

    Returns:
        label_counts: {label: count}
        confusion: {true_label: {pred_label: count}}
        correct: total correct predictions
    """
    label_counts: Dict[str, int] = defaultdict(int)
    confusion: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    correct = 0

    for result in valid_results:
        true_label = normalize_label(result["label"])
        pred_label = result["prediction"]
        label_counts[true_label] += 1
        confusion[true_label][pred_label] += 1
        if true_label == pred_label:
            correct += 1

    return label_counts, confusion, correct


def _compute_per_label(
    label: str,
    label_counts: Dict[str, int],
    confusion: Dict[str, Dict[str, int]],
    canonical_labels: List[str],
) -> Optional[Dict[str, float]]:
    """Return precision/recall/f1 for a single label, or None if unsupported."""
    if label_counts[label] == 0:
        return None

    tp = confusion[label][label]
    fn = label_counts[label] - tp
    fp = sum(confusion[other][label] for other in canonical_labels if other != label)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "support": label_counts[label],
    }


def _macro_averages(per_label: Dict[str, Dict[str, float]]) -> Tuple[float, float, float]:
    """Compute macro averages from per‑label metrics."""
    if not per_label:
        return 0.0, 0.0, 0.0
    precisions = [m["precision"] for m in per_label.values()]
    recalls = [m["recall"] for m in per_label.values()]
    f1s = [m["f1"] for m in per_label.values()]
    return (
        round(sum(precisions) / len(precisions), 4),
        round(sum(recalls) / len(recalls), 4),
        round(sum(f1s) / len(f1s), 4),
    )


# ---------------------------------------------------------------------------
# Statistical tests – reusable, pure functions
# ---------------------------------------------------------------------------

def _extract_binary_correct(
    results: List[Dict],
) -> List[int]:
    """Return list of 1 (correct) or 0 (incorrect) for every valid prediction."""
    return [
        1 if r.get("prediction") == normalize_label(r.get("label", "")) else 0
        for r in results
        if r.get("prediction") and r.get("label")
    ]


def _run_statistical_tests(
    seen_correct: List[int],
    unseen_correct: List[int],
) -> Dict[str, Any]:
    """Perform t‑test and chi‑square test on two binary accuracy lists."""
    if len(seen_correct) <= 1 or len(unseen_correct) <= 1:
        raise ValueError("Need more than one sample per group for statistical tests.")

    # t‑test
    t_stat, p_value = stats.ttest_ind(seen_correct, unseen_correct)

    # chi‑square
    seen_success = sum(seen_correct)
    seen_total = len(seen_correct)
    unseen_success = sum(unseen_correct)
    unseen_total = len(unseen_correct)
    contingency = [
        [seen_success, seen_total - seen_success],
        [unseen_success, unseen_total - unseen_success],
    ]
    chi2, chi_p, dof, _ = stats.chi2_contingency(contingency)

    return {
        "t_test": {
            "t_statistic": round(t_stat, 4),
            "p_value": round(p_value, 4),
            "significant_at_0.05": p_value < 0.05,
            "interpretation": "Significant difference" if p_value < 0.05 else "No significant difference",
        },
        "chi_square_test": {
            "chi2_statistic": round(chi2, 4),
            "p_value": round(chi_p, 4),
            "degrees_of_freedom": dof,
            "significant_at_0.05": chi_p < 0.05,
            "interpretation": "Significant difference" if chi_p < 0.05 else "No significant difference",
        },
    }


# ---------------------------------------------------------------------------
# Public API – identical to original, but internally decomposed
# ---------------------------------------------------------------------------

def calculate_metrics(
    results: List[Dict[str, Any]],
    canonical_labels: List[str],
    logger: logging.Logger,
) -> Dict[str, Any]:
    """
    Calculate accuracy, macro‑averaged metrics, and per‑label breakdown.

    Args:
        results: List of dicts with keys 'prediction' and 'label'.
        canonical_labels: Allowed labels.
        logger: Logger instance.

    Returns:
        Dict with keys: accuracy, macro_precision, macro_recall, macro_f1,
        total_samples, correct_predictions, per_label_metrics.
        If no valid results, returns {"error": "No valid results"}.
    """
    valid = _get_valid_predictions(results)
    if not valid:
        logger.warning("No valid results for metric calculation")
        return {"error": "No valid results"}

    label_counts, confusion, correct = _build_confusion_matrix(valid)
    total = len(valid)
    accuracy = correct / total if total > 0 else 0.0

    # Per‑label metrics
    per_label = {}
    for label in canonical_labels:
        m = _compute_per_label(label, label_counts, confusion, canonical_labels)
        if m is not None:
            per_label[label] = m

    macro_prec, macro_rec, macro_f1 = _macro_averages(per_label)

    metrics = {
        "accuracy": round(accuracy, 4),
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
        "total_samples": total,
        "correct_predictions": correct,
        "per_label_metrics": per_label,
    }

    logger.info(f"Metrics: Accuracy={metrics['accuracy']:.4f}, Macro-F1={metrics['macro_f1']:.4f}")
    return metrics


def compare_seen_unseen(
    seen_results: List[Dict],
    unseen_results: List[Dict],
    canonical_labels: List[str],
    logger: logging.Logger,
) -> Dict[str, Any]:
    """
    Compare performance on seen vs unseen splits with statistical tests.

    Args:
        seen_results: Predictions from pre‑split data.
        unseen_results: Predictions from post‑split data.
        canonical_labels: Allowed labels.
        logger: Logger instance.

    Returns:
        Dict with seen_metrics, unseen_metrics, statistical_tests,
        accuracy_difference. If insufficient data for tests, a placeholder
        message is stored in statistical_tests.
    """
    logger.info("=" * 80)
    logger.info("Comparing seen vs unseen performance...")

    seen_metrics = calculate_metrics(seen_results, canonical_labels, logger)
    unseen_metrics = calculate_metrics(unseen_results, canonical_labels, logger)

    seen_correct = _extract_binary_correct(seen_results)
    unseen_correct = _extract_binary_correct(unseen_results)

    # Try to run statistical tests only if enough data
    try:
        tests = _run_statistical_tests(seen_correct, unseen_correct)
    except Exception as e:  # e.g., ValueError, or if lists empty
        logger.warning(f"Statistical tests skipped: {e}")
        tests = "Insufficient data for statistical testing"

    accuracy_diff = round(unseen_metrics.get("accuracy", 0.0) - seen_metrics.get("accuracy", 0.0), 4)

    comparison = {
        "seen_metrics": seen_metrics,
        "unseen_metrics": unseen_metrics,
        "statistical_tests": tests,
        "accuracy_difference": accuracy_diff,
    }

    # Log results
    logger.info(f"Seen accuracy: {seen_metrics.get('accuracy', 'N/A')}")
    logger.info(f"Unseen accuracy: {unseen_metrics.get('accuracy', 'N/A')}")
    logger.info(f"Difference: {accuracy_diff:.4f}")
    if isinstance(tests, dict):
        t = tests["t_test"]
        logger.info(f"T-test p-value: {t['p_value']:.4f} ({'significant' if t['significant_at_0.05'] else 'not significant'})")
        c = tests["chi_square_test"]
        logger.info(f"Chi-square p-value: {c['p_value']:.4f} ({'significant' if c['significant_at_0.05'] else 'not significant'})")

    return comparison