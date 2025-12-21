#!/usr/bin/env python3
"""
Production-Level Fact-Checking Model Inference Pipeline

This script provides a unified pipeline for running fact-checking inference across multiple datasets:
- ClaimReview Plus (HuggingFace)
- FACTors (CSV format)
- AVeriTeC FEVER (Train/Dev/Test JSON format)

Features:
- Command-line interface for all configuration parameters
- Temporal split analysis (seen vs. unseen data)
- Statistical significance testing
- Balanced sampling for large datasets
- Comprehensive error handling and logging
- Production-ready code structure

Usage Examples:
    # Basic usage with defaults
    python run_fact_check_inference.py --dataset factors

    # Custom model and API
    python run_fact_check_inference.py --dataset factors --model gpt-4o-mini --api-key YOUR_KEY

    # With balanced sampling
    python run_fact_check_inference.py --dataset factors --balanced-sampling --seed 42

    # FEVER test data
    python run_fact_check_inference.py --dataset fever_test_2025

    # Custom split date and labels
    python run_fact_check_inference.py --dataset factors --split-date 2024-01-01 --labels "false,true,misleading,partially true"

Author: Generated for Fact-Checking Research
Version: 1.0.0
"""

import os
import re
import json
import logging
import time
import random
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from collections import defaultdict

# Third-party imports
try:
    import requests
    import pandas as pd
    from tqdm import tqdm
    from datasets import load_dataset
    from scipy import stats
except ImportError as e:
    print(f"Error: Missing required package. Please install dependencies:")
    print("  pip install requests tqdm datasets pandas scipy")
    sys.exit(1)


# ==================== Constants ====================
VERSION = "1.0.0"
DEFAULT_API_URL = "https://api.avalai.ir/v1/chat/completions"
DEFAULT_MODEL = "anthropic.claude-3-5-sonnet-20241022-v2:0"
DEFAULT_LABELS = ["false", "misleading", "partially true", "true"]
DEFAULT_DATE_FORMAT = "%d-%m-%Y"
DEFAULT_SPLIT_DATE = "2024-04-01"


# ==================== Logging Setup ====================
def setup_logger(log_file: Path, level: int = logging.INFO) -> logging.Logger:
    """
    Configure logging to write to both file and console.
    
    Args:
        log_file: Path to the log file
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(f"FactCheck-{log_file.stem}")
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers = []
    
    # File handler with detailed formatting
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(level)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger


# ==================== Label Normalization ====================
def normalize_label(label: str, canonical_labels: List[str]) -> Optional[str]:
    """
    Normalize a label to match one of the canonical labels.
    
    Priority order:
    1. Check for exact canonical label match (case-insensitive)
    2. Check for partial match with canonical labels
    3. Filter out unverifiable/invalid labels
    
    Args:
        label: The label to normalize
        canonical_labels: List of valid canonical labels
        
    Returns:
        Normalized label or None if not mappable
    """
    if not label:
        return None
    
    label_lower = label.lower().strip()
    
    # STEP 1: Check for exact match with canonical labels FIRST
    for canonical in canonical_labels:
        if label_lower == canonical.lower():
            return canonical.lower()
    
    # STEP 2: Check if label contains any canonical label
    for canonical in canonical_labels:
        if canonical.lower() in label_lower:
            return canonical.lower()
    
    # STEP 3: NOW filter out unverifiable/invalid labels
    unverifiable_keywords = [
        'unverifiable', 'not enough', 'cannot be determined',
        'unproven', 'unknown', 'lacks context', 'needs context',
        'no evidence', 'unsubstantiated', 'inconclusive',
        'not yet rated', 'in dispute', 'disputed', 'opinion',
        'satire', 'scam', 'legend', 'outdated', 'miscaptioned',
        'misinformation', 'other', 'mixture', 'half true',
        'mostly true', 'mostly false', 'half-true', 'pants on fire'
    ]
    
    for keyword in unverifiable_keywords:
        if keyword in label_lower:
            return None
    
    # STEP 4: Common label mappings
    label_mappings = {
        'correct': 'true',
        'accurate': 'true',
        'factual': 'true',
        'verified': 'true',
        'confirmed': 'true',
        'incorrect': 'false',
        'inaccurate': 'false',
        'wrong': 'false',
        'fake': 'false',
        'debunked': 'false',
        'refuted': 'false',
        'partly true': 'partially true',
        'partial': 'partially true',
        'mixed': 'partially true',
        'cherry picks': 'misleading',
        'out of context': 'misleading',
        'lacks context': 'misleading',
        'exaggerated': 'misleading',
        'distorts': 'misleading',
        'spins': 'misleading',
    }
    
    for key, value in label_mappings.items():
        if key in label_lower and value.lower() in [c.lower() for c in canonical_labels]:
            return value.lower()
    
    return None


# ==================== API Functions ====================
def ask_model(
    claim: str,
    date: str,
    model_name: str,
    api_key: str,
    api_url: str,
    labels: List[str],
    logger: logging.Logger,
    max_retries: int = 5,
    initial_retry_delay: float = 0.5
) -> Optional[str]:
    """
    Send a fact-checking request to the model API with retry logic.
    
    Args:
        claim: The claim to fact-check
        date: The date of the claim
        model_name: Name of the model to use
        api_key: API authentication key
        api_url: API endpoint URL
        labels: List of valid labels for classification
        logger: Logger instance
        max_retries: Maximum number of retry attempts
        initial_retry_delay: Initial delay between retries
        
    Returns:
        Normalized label prediction or None if failed
    """
    labels_str = ", ".join(labels)
    
    system_prompt = f"""You are a fact-checking assistant. Your task is to verify claims and classify them into one of these categories: {labels_str}.

Guidelines:
1. Analyze the claim carefully based on factual accuracy
2. Consider the context and date of the claim
3. Respond with ONLY the label, nothing else
4. If uncertain, choose the most likely category based on available information"""

    user_prompt = f"""Claim: {claim}
Date: {date}

Classify this claim using one of these labels: {labels_str}

Respond with only the label."""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 50
    }
    
    retry_delay = initial_retry_delay
    
    for attempt in range(max_retries):
        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"].strip().lower()
                
                # Extract label from response
                normalized = normalize_label(content, labels)
                if normalized:
                    return normalized
                
                # Try to find label in response
                for label in labels:
                    if label.lower() in content.lower():
                        return label.lower()
                
                logger.warning(f"Could not extract valid label from response: {content}")
                return None
                
            elif response.status_code == 429:
                logger.warning(f"Rate limited (attempt {attempt + 1}/{max_retries}). Waiting {retry_delay}s...")
                time.sleep(retry_delay)
                retry_delay *= 2
                continue
                
            else:
                logger.error(f"API error {response.status_code}: {response.text[:200]}")
                return None
                
        except requests.exceptions.Timeout:
            logger.warning(f"Request timeout (attempt {attempt + 1}/{max_retries})")
            time.sleep(retry_delay)
            retry_delay *= 2
            
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return None
    
    logger.error(f"All {max_retries} retry attempts failed")
    return None


# ==================== Dataset Loaders ====================
def load_claimreview_plus(hf_token: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Load ClaimReview Plus dataset from HuggingFace.
    
    Args:
        hf_token: HuggingFace API token
        
    Returns:
        List of standardized claim dictionaries
    """
    print("Loading ClaimReview Plus dataset from HuggingFace...")
    
    token = hf_token or os.environ.get("HF_TOKEN")
    dataset = load_dataset("Webis/claimreview-2025", token=token)
    df = dataset["train"].to_pandas()
    
    claims = []
    for _, row in df.iterrows():
        claims.append({
            "claim_id": len(claims),
            "claim": row['claim'],
            "claim_date": row.get('date', ''),
            "label": row.get('rating', None),
            "speaker": row.get('claimant', None),
            "source": "claimreview_plus"
        })
    
    print(f"✓ Loaded {len(claims)} claims from ClaimReview Plus")
    return claims


def load_factors(
    csv_path: str = "Datasets/factors.csv",
    date_format: str = DEFAULT_DATE_FORMAT,
    canonical_labels: List[str] = DEFAULT_LABELS
) -> List[Dict[str, Any]]:
    """
    Load FACTors dataset from CSV file, filtering out invalid labels.
    
    Args:
        csv_path: Path to the CSV file
        date_format: Date format string
        canonical_labels: List of valid canonical labels
        
    Returns:
        List of standardized claim dictionaries
    """
    print(f"Loading FACTors dataset from {csv_path}...")
    
    df = pd.read_csv(csv_path)
    print(f"  Raw dataset size: {len(df)} claims")
    
    claims = []
    skipped_count = 0
    
    for _, row in df.iterrows():
        raw_label = str(row.get('normalised_rating', '')).lower().strip()
        
        # Filter out 'other' and 'unverifiable' labels at load time
        if raw_label in ['other', 'unverifiable', '']:
            skipped_count += 1
            continue
        
        # Also check if label can be normalized
        if pd.notna(row.get('normalised_rating')):
            normalized = normalize_label(str(row['normalised_rating']), canonical_labels)
            if not normalized:
                skipped_count += 1
                continue
        
        # Parse date
        try:
            date_obj = datetime.fromisoformat(str(row['date_published']).replace('T00:00:00', ''))
            formatted_date = date_obj.strftime(date_format)
        except:
            formatted_date = ""
        
        claims.append({
            "claim_id": row.get('claim_id', len(claims)),
            "claim": row['claim'],
            "claim_date": formatted_date,
            "label": str(row['normalised_rating']).capitalize() if pd.notna(row.get('normalised_rating')) else "",
            "speaker": row.get('author', None),
            "source": "factors"
        })
    
    print(f"  Skipped {skipped_count} claims with invalid/unverifiable labels")
    print(f"✓ Loaded {len(claims)} claims from FACTors (after filtering)")
    return claims


def load_fever_json(file_path: Path, is_test: bool = False) -> List[Dict[str, Any]]:
    """
    Load FEVER dataset from JSON file.
    
    Args:
        file_path: Path to the JSON file
        is_test: Whether this is test data (no labels)
        
    Returns:
        List of standardized claim dictionaries
    """
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


def load_dataset_by_name(
    dataset_name: str,
    base_path: str = "Datasets",
    hf_token: Optional[str] = None,
    date_format: str = DEFAULT_DATE_FORMAT,
    canonical_labels: List[str] = DEFAULT_LABELS
) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Load dataset based on configuration name.
    
    Args:
        dataset_name: Name of the dataset to load
        base_path: Base path for dataset files
        hf_token: HuggingFace token (for claimreview_plus)
        date_format: Date format string
        canonical_labels: List of valid canonical labels
        
    Returns:
        Tuple of (claims list, is_test_data boolean)
    """
    base_path = Path(base_path)
    
    if dataset_name == "claimreview_plus":
        return load_claimreview_plus(hf_token), False
    
    elif dataset_name == "factors":
        return load_factors(
            csv_path=str(base_path / "factors.csv"),
            date_format=date_format,
            canonical_labels=canonical_labels
        ), False
    
    elif dataset_name == "fever_train":
        return load_fever_json(base_path / "AVeriTeC_FEVER" / "train.json", is_test=False), False
    
    elif dataset_name == "fever_dev":
        return load_fever_json(base_path / "AVeriTeC_FEVER" / "data_dev.json", is_test=False), False
    
    elif dataset_name == "fever_test_2023_2024":
        return load_fever_json(base_path / "AVeriTeC_FEVER" / "test_2023_2024.json", is_test=True), True
    
    elif dataset_name == "fever_test_2025":
        return load_fever_json(base_path / "AVeriTeC_FEVER" / "test_2025.json", is_test=True), True
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                        f"Available: claimreview_plus, factors, fever_train, fever_dev, "
                        f"fever_test_2023_2024, fever_test_2025")


# ==================== Processing Functions ====================
def split_by_date(
    claims: List[Dict[str, Any]],
    split_date: str,
    date_format: str = DEFAULT_DATE_FORMAT
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split claims into seen (before split date) and unseen (after split date).
    
    Args:
        claims: List of claim dictionaries
        split_date: Split date in YYYY-MM-DD format
        date_format: Format of dates in claims
        
    Returns:
        Tuple of (seen_claims, unseen_claims)
    """
    split_dt = datetime.strptime(split_date, "%Y-%m-%d")
    seen = []
    unseen = []
    no_date = []
    
    for claim in claims:
        date_str = claim.get("claim_date", "")
        if not date_str:
            no_date.append(claim)
            continue
        
        try:
            claim_dt = datetime.strptime(date_str, date_format)
        except:
            try:
                claim_dt = datetime.fromisoformat(date_str.replace('T00:00:00', ''))
            except:
                no_date.append(claim)
                continue
        
        if claim_dt < split_dt:
            seen.append(claim)
        else:
            unseen.append(claim)
    
    print(f"✓ Split complete: {len(seen)} seen, {len(unseen)} unseen, {len(no_date)} without date")
    
    # Add no_date claims to seen by default
    seen.extend(no_date)
    
    return seen, unseen


def process_claims(
    claims: List[Dict[str, Any]],
    is_test: bool,
    logger: logging.Logger,
    model_name: str,
    api_key: str,
    api_url: str,
    canonical_labels: List[str],
    max_samples: Optional[int] = None,
    sleep_between_calls: float = 0.1,
    display_every_n: int = 10,
    max_retries: int = 5,
    initial_retry_delay: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Process claims through the model and collect predictions.
    
    Args:
        claims: List of claim dictionaries
        is_test: Whether this is test data (no ground truth labels)
        logger: Logger instance
        model_name: Name of the model to use
        api_key: API authentication key
        api_url: API endpoint URL
        canonical_labels: List of valid canonical labels
        max_samples: Maximum samples to process (None for all)
        sleep_between_calls: Sleep time between API calls
        display_every_n: Display progress every N claims
        max_retries: Maximum retry attempts for API calls
        initial_retry_delay: Initial retry delay
        
    Returns:
        List of claims with predictions added
    """
    results = []
    successful = 0
    failed = 0
    correct = 0
    incorrect = 0
    
    # Apply sampling if configured
    if max_samples and len(claims) > max_samples:
        claims = claims[:max_samples]
        logger.info(f"Processing {max_samples} samples (limited for testing)")
    
    logger.info(f"Processing {len(claims)} claims...")
    
    # Create progress bar
    desc = "Processing claims"
    pbar = tqdm(claims, desc=desc)
    
    for idx, claim_dict in enumerate(pbar):
        claim = claim_dict["claim"]
        date = claim_dict.get("claim_date", "")
        claim_id = claim_dict.get("claim_id", idx)
        
        logger.info("=" * 80)
        logger.info(f"Processing claim #{idx} (ID: {claim_id})")
        logger.info(f"Claim: {claim[:150]}...")
        logger.info(f"Date: {date}")
        
        # Get model prediction
        prediction = ask_model(
            claim=claim,
            date=date,
            model_name=model_name,
            api_key=api_key,
            api_url=api_url,
            labels=canonical_labels,
            logger=logger,
            max_retries=max_retries,
            initial_retry_delay=initial_retry_delay
        )
        
        # Add prediction to result
        result = claim_dict.copy()
        
        if prediction:
            result["prediction"] = prediction
            result["prediction_error"] = None
            logger.info(f"Prediction: {prediction}")
            successful += 1
            
            # For labeled data, check if prediction is correct
            if not is_test and claim_dict.get("label"):
                true_label = normalize_label(claim_dict["label"], canonical_labels)
                if true_label and prediction == true_label:
                    correct += 1
                    logger.info("✓ Correct prediction")
                elif true_label:
                    incorrect += 1
                    logger.info(f"✗ Incorrect prediction (True: {true_label}, Pred: {prediction})")
            
            # Display progress
            if (idx + 1) % display_every_n == 0:
                print(f"\n[Claim #{idx + 1}] {claim[:100]}...")
                print(f"  → Prediction: {prediction}")
                if not is_test and claim_dict.get("label"):
                    true_label = normalize_label(claim_dict["label"], canonical_labels)
                    if true_label:
                        print(f"  → Ground Truth: {true_label}")
                        print(f"  → {'✓ Correct' if prediction == true_label else '✗ Incorrect'}")
        else:
            result["prediction"] = None
            result["prediction_error"] = "Failed to get valid prediction"
            logger.warning("Failed to get valid prediction")
            failed += 1
        
        results.append(result)
        
        # Update progress bar
        if is_test:
            pbar.set_description(f"Processing [✓{successful} ✗{failed}]")
        else:
            pbar.set_description(f"Processing [✓{correct} ✗{incorrect} F:{failed}]")
        
        # Rate limiting
        time.sleep(sleep_between_calls)
    
    logger.info("=" * 80)
    if is_test:
        logger.info(f"Processing complete: {successful} successful, {failed} failed")
    else:
        logger.info(f"Processing complete: {successful} predictions ({correct} correct, {incorrect} incorrect), {failed} failed")
    
    return results


# ==================== Metrics & Evaluation ====================
def calculate_metrics(
    results: List[Dict[str, Any]],
    canonical_labels: List[str],
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Calculate performance metrics for predictions with ground truth.
    
    Args:
        results: List of results with predictions and labels
        canonical_labels: List of valid canonical labels
        logger: Logger instance
        
    Returns:
        Dictionary of metrics
    """
    # Filter valid predictions
    valid_results = [
        r for r in results 
        if r.get("prediction") and r.get("label") and 
        normalize_label(r["label"], canonical_labels)
    ]
    
    if not valid_results:
        logger.warning("No valid results for metric calculation")
        return {"error": "No valid results"}
    
    # Normalize labels and collect predictions
    correct = 0
    total = len(valid_results)
    label_counts = defaultdict(int)
    confusion = defaultdict(lambda: defaultdict(int))
    
    for result in valid_results:
        true_label = normalize_label(result["label"], canonical_labels)
        pred_label = result["prediction"]
        
        label_counts[true_label] += 1
        confusion[true_label][pred_label] += 1
        
        if true_label == pred_label:
            correct += 1
    
    accuracy = correct / total if total > 0 else 0.0
    
    # Calculate per-label metrics
    per_label_metrics = {}
    for label in canonical_labels:
        label_lower = label.lower()
        if label_counts[label_lower] == 0:
            continue
        
        tp = confusion[label_lower][label_lower]
        fn = label_counts[label_lower] - tp
        fp = sum(confusion[other][label_lower] for other in canonical_labels if other.lower() != label_lower)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        per_label_metrics[label_lower] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": label_counts[label_lower]
        }
    
    # Calculate macro averages
    precisions = [m["precision"] for m in per_label_metrics.values()]
    recalls = [m["recall"] for m in per_label_metrics.values()]
    f1s = [m["f1"] for m in per_label_metrics.values()]
    
    metrics = {
        "accuracy": round(accuracy, 4),
        "macro_precision": round(sum(precisions) / len(precisions), 4) if precisions else 0.0,
        "macro_recall": round(sum(recalls) / len(recalls), 4) if recalls else 0.0,
        "macro_f1": round(sum(f1s) / len(f1s), 4) if f1s else 0.0,
        "total_samples": total,
        "correct_predictions": correct,
        "per_label_metrics": per_label_metrics
    }
    
    logger.info(f"Metrics: Accuracy={metrics['accuracy']:.4f}, Macro-F1={metrics['macro_f1']:.4f}")
    
    return metrics


def compare_seen_unseen(
    seen_results: List[Dict],
    unseen_results: List[Dict],
    canonical_labels: List[str],
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Compare performance on seen vs unseen data with statistical testing.
    
    Args:
        seen_results: Results from seen data
        unseen_results: Results from unseen data
        canonical_labels: List of valid canonical labels
        logger: Logger instance
        
    Returns:
        Dictionary with comparison statistics
    """
    logger.info("=" * 80)
    logger.info("Comparing seen vs unseen performance...")
    
    # Calculate metrics for each split
    seen_metrics = calculate_metrics(seen_results, canonical_labels, logger)
    unseen_metrics = calculate_metrics(unseen_results, canonical_labels, logger)
    
    # Extract accuracies for statistical test
    seen_correct = [
        1 if r.get("prediction") == normalize_label(r.get("label", ""), canonical_labels) else 0 
        for r in seen_results 
        if r.get("prediction") and r.get("label")
    ]
    
    unseen_correct = [
        1 if r.get("prediction") == normalize_label(r.get("label", ""), canonical_labels) else 0 
        for r in unseen_results 
        if r.get("prediction") and r.get("label")
    ]
    
    # Perform statistical significance test
    if len(seen_correct) > 1 and len(unseen_correct) > 1:
        t_stat, p_value = stats.ttest_ind(seen_correct, unseen_correct)
        
        # Chi-square test
        seen_success = sum(seen_correct)
        seen_total = len(seen_correct)
        unseen_success = sum(unseen_correct)
        unseen_total = len(unseen_correct)
        
        contingency_table = [
            [seen_success, seen_total - seen_success],
            [unseen_success, unseen_total - unseen_success]
        ]
        chi2, chi_p_value, dof, expected = stats.chi2_contingency(contingency_table)
        
        comparison = {
            "seen_metrics": seen_metrics,
            "unseen_metrics": unseen_metrics,
            "statistical_tests": {
                "t_test": {
                    "t_statistic": round(t_stat, 4),
                    "p_value": round(p_value, 4),
                    "significant_at_0.05": p_value < 0.05,
                    "interpretation": "Significant difference" if p_value < 0.05 else "No significant difference"
                },
                "chi_square_test": {
                    "chi2_statistic": round(chi2, 4),
                    "p_value": round(chi_p_value, 4),
                    "degrees_of_freedom": dof,
                    "significant_at_0.05": chi_p_value < 0.05,
                    "interpretation": "Significant difference" if chi_p_value < 0.05 else "No significant difference"
                }
            },
            "accuracy_difference": round(unseen_metrics.get("accuracy", 0) - seen_metrics.get("accuracy", 0), 4)
        }
        
        logger.info(f"Seen accuracy: {seen_metrics.get('accuracy', 'N/A')}")
        logger.info(f"Unseen accuracy: {unseen_metrics.get('accuracy', 'N/A')}")
        logger.info(f"Difference: {comparison['accuracy_difference']:.4f}")
        logger.info(f"T-test p-value: {p_value:.4f} ({'significant' if p_value < 0.05 else 'not significant'})")
    else:
        comparison = {
            "seen_metrics": seen_metrics,
            "unseen_metrics": unseen_metrics,
            "statistical_tests": "Insufficient data for statistical testing",
            "accuracy_difference": round(unseen_metrics.get("accuracy", 0) - seen_metrics.get("accuracy", 0), 4)
        }
        logger.warning("Insufficient data for statistical testing")
    
    return comparison


# ==================== Output Functions ====================
def save_test_predictions(
    results: List[Dict[str, Any]],
    output_file: Path,
    logger: logging.Logger
):
    """
    Save test predictions in the original format with prediction field added.
    
    Args:
        results: List of results with predictions
        output_file: Path to save the output JSON
        logger: Logger instance
    """
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
            }
        
        item["prediction"] = result.get("prediction", None)
        if result.get("prediction_error"):
            item["prediction_error"] = result["prediction_error"]
        
        output_data.append(item)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    
    logger.info(f"Test predictions saved to {output_file}")
    print(f"✓ Test predictions saved to {output_file}")


def save_results_with_metrics(
    results: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    output_dir: Path,
    split_name: str,
    logger: logging.Logger
):
    """
    Save results and metrics for labeled data.
    
    Args:
        results: List of results with predictions
        metrics: Calculated metrics
        output_dir: Directory to save outputs
        split_name: Name of the data split
        logger: Logger instance
    """
    # Save predictions as CSV
    predictions_file = output_dir / f"{split_name}_predictions.csv"
    df_results = pd.DataFrame(results)
    df_results.to_csv(predictions_file, index=False, encoding='utf-8')
    logger.info(f"Predictions saved to {predictions_file}")
    
    # Save predictions as JSON
    predictions_json = output_dir / f"{split_name}_predictions.json"
    with open(predictions_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    logger.info(f"Predictions (JSON) saved to {predictions_json}")
    
    # Save metrics
    metrics_file = output_dir / f"{split_name}_metrics.json"
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Metrics saved to {metrics_file}")
    
    print(f"✓ {split_name.capitalize()} results saved to {output_dir}")


def generate_summary_report(
    seen_metrics: Dict,
    unseen_metrics: Dict,
    comparison: Dict,
    output_dir: Path,
    config: argparse.Namespace,
    logger: logging.Logger
):
    """
    Generate a comprehensive summary report.
    
    Args:
        seen_metrics: Metrics for seen data
        unseen_metrics: Metrics for unseen data
        comparison: Comparison statistics
        output_dir: Directory to save the report
        config: Configuration namespace
        logger: Logger instance
    """
    summary = {
        "experiment_info": {
            "dataset": config.dataset,
            "model": config.model,
            "temporal_split_date": config.split_date,
            "canonical_labels": config.labels,
            "balanced_sampling": config.balanced_sampling,
            "sampling_seed": config.seed if config.balanced_sampling else None,
            "timestamp": datetime.now().isoformat()
        },
        "seen_data": seen_metrics,
        "unseen_data": unseen_metrics,
        "comparison": comparison
    }
    
    if "sampling_info" in comparison:
        summary["experiment_info"]["sampling_info"] = comparison["sampling_info"]
    
    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4)
    
    logger.info(f"Summary report saved to {summary_file}")
    print(f"✓ Summary report saved to {summary_file}")
    
    # Print summary to console
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    print(f"Dataset: {config.dataset}")
    print(f"Model: {config.model}")
    print(f"Temporal Split: {config.split_date}")
    
    if comparison.get("sampling_info", {}).get("enabled"):
        si = comparison["sampling_info"]
        print(f"\nBALANCED SAMPLING:")
        print(f"  Seed: {si['seed']}")
        print(f"  Original Seen: {si['original_seen_count']} -> Sampled: {si['sampled_seen_count']}")
    
    print("\nSEEN DATA PERFORMANCE:")
    acc = seen_metrics.get('accuracy', 'N/A')
    print(f"  Accuracy: {acc if acc == 'N/A' else f'{acc:.4f}'}")
    f1 = seen_metrics.get('macro_f1', 'N/A')
    print(f"  Macro F1: {f1 if f1 == 'N/A' else f'{f1:.4f}'}")
    print(f"  Samples: {seen_metrics.get('total_samples', 'N/A')}")
    
    print("\nUNSEEN DATA PERFORMANCE:")
    acc = unseen_metrics.get('accuracy', 'N/A')
    print(f"  Accuracy: {acc if acc == 'N/A' else f'{acc:.4f}'}")
    f1 = unseen_metrics.get('macro_f1', 'N/A')
    print(f"  Macro F1: {f1 if f1 == 'N/A' else f'{f1:.4f}'}")
    print(f"  Samples: {unseen_metrics.get('total_samples', 'N/A')}")
    
    if isinstance(comparison.get("statistical_tests"), dict):
        print("\nSTATISTICAL SIGNIFICANCE:")
        print(f"  Accuracy Difference: {comparison['accuracy_difference']:.4f}")
        print(f"  T-test p-value: {comparison['statistical_tests']['t_test']['p_value']:.4f}")
        print(f"  Chi-square p-value: {comparison['statistical_tests']['chi_square_test']['p_value']:.4f}")
        print(f"  Interpretation: {comparison['statistical_tests']['t_test']['interpretation']}")
    print("=" * 80)


# ==================== Argument Parser ====================
def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed argument namespace
    """
    parser = argparse.ArgumentParser(
        description="Production-Level Fact-Checking Model Inference Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_fact_check_inference.py --dataset factors
  python run_fact_check_inference.py --dataset factors --balanced-sampling --seed 42
  python run_fact_check_inference.py --dataset fever_test_2025 --model gpt-4o-mini
  python run_fact_check_inference.py --dataset factors --split-date 2024-01-01 --labels "false,true,misleading,partially true"
        """
    )
    
    # Dataset configuration
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        required=True,
        choices=["claimreview_plus", "factors", "fever_train", "fever_dev", 
                 "fever_test_2023_2024", "fever_test_2025"],
        help="Dataset to use for inference"
    )
    
    parser.add_argument(
        "--data-path",
        type=str,
        default="Datasets",
        help="Base path for dataset files (default: Datasets)"
    )
    
    # Model configuration
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model name to use (default: {DEFAULT_MODEL})"
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key (default: uses AVALAI_API_KEY environment variable)"
    )
    
    parser.add_argument(
        "--api-url",
        type=str,
        default=DEFAULT_API_URL,
        help=f"API endpoint URL (default: {DEFAULT_API_URL})"
    )
    
    # Label configuration
    parser.add_argument(
        "--labels",
        type=str,
        default=",".join(DEFAULT_LABELS),
        help=f"Comma-separated canonical labels (default: {','.join(DEFAULT_LABELS)})"
    )
    
    # Temporal split configuration
    parser.add_argument(
        "--split-date",
        type=str,
        default=DEFAULT_SPLIT_DATE,
        help=f"Temporal split date in YYYY-MM-DD format (default: {DEFAULT_SPLIT_DATE})"
    )
    
    parser.add_argument(
        "--date-format",
        type=str,
        default=DEFAULT_DATE_FORMAT,
        help=f"Date format in dataset (default: {DEFAULT_DATE_FORMAT})"
    )
    
    # Sampling configuration
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to process (default: all)"
    )
    
    parser.add_argument(
        "--balanced-sampling",
        action="store_true",
        help="Enable balanced sampling (sample seen data to match unseen size)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for balanced sampling (default: 42)"
    )
    
    # API rate limiting
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum retry attempts for API calls (default: 5)"
    )
    
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=0.5,
        help="Initial retry delay in seconds (default: 0.5)"
    )
    
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.1,
        help="Sleep between API calls in seconds (default: 0.1)"
    )
    
    # Output configuration
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Output directory (default: results/<model>/<dataset>)"
    )
    
    parser.add_argument(
        "--display-every",
        type=int,
        default=10,
        help="Display progress every N claims (default: 10)"
    )
    
    # HuggingFace configuration
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace API token (default: uses HF_TOKEN environment variable)"
    )
    
    # Utility options
    parser.add_argument(
        "--version", "-v",
        action="version",
        version=f"%(prog)s {VERSION}"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load data and show configuration without processing"
    )
    
    args = parser.parse_args()
    
    # Post-process arguments
    args.labels = [label.strip().lower() for label in args.labels.split(",")]
    
    # Set API key from environment if not provided
    if not args.api_key:
        args.api_key = os.environ.get("AVALAI_API_KEY", "")
        if not args.api_key:
            parser.error("API key required. Set --api-key or AVALAI_API_KEY environment variable.")
    
    # Set HuggingFace token
    if not args.hf_token:
        args.hf_token = os.environ.get("HF_TOKEN")
    
    # Set output directory
    if not args.output_dir:
        model_safe = args.model.replace(":", "_").replace(".", "_")
        args.output_dir = Path("results") / model_safe / args.dataset
    else:
        args.output_dir = Path(args.output_dir)
    
    return args


# ==================== Main Function ====================
def main():
    """Main entry point for the fact-checking inference pipeline."""
    
    # Parse arguments
    config = parse_arguments()
    
    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logger
    log_file = config.output_dir / "main_run_log.txt"
    logger = setup_logger(log_file)
    
    # Log configuration
    logger.info("=" * 80)
    logger.info("STARTING FACT-CHECK INFERENCE PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Dataset: {config.dataset}")
    logger.info(f"Model: {config.model}")
    logger.info(f"Output Directory: {config.output_dir}")
    logger.info(f"Canonical Labels: {config.labels}")
    logger.info(f"Temporal Split Date: {config.split_date}")
    logger.info(f"Balanced Sampling: {config.balanced_sampling}")
    if config.balanced_sampling:
        logger.info(f"Sampling Seed: {config.seed}")
    
    print(f"\n{'=' * 80}")
    print(f"Starting inference pipeline for {config.dataset}")
    print(f"{'=' * 80}")
    print(f"Model: {config.model}")
    print(f"Output: {config.output_dir}")
    print(f"Labels: {config.labels}")
    print(f"{'=' * 80}\n")
    
    # Load dataset
    claims, is_test = load_dataset_by_name(
        dataset_name=config.dataset,
        base_path=config.data_path,
        hf_token=config.hf_token,
        date_format=config.date_format,
        canonical_labels=config.labels
    )
    logger.info(f"Loaded {len(claims)} claims (is_test={is_test})")
    
    # Dry run - show config and exit
    if config.dry_run:
        print(f"\n[DRY RUN] Configuration validated. Would process {len(claims)} claims.")
        print(f"Output would be saved to: {config.output_dir}")
        return 0
    
    # Branch based on test vs labeled data
    if is_test:
        # ==================== TEST DATA PIPELINE ====================
        logger.info("Processing TEST data (no labels available)")
        
        results = process_claims(
            claims=claims,
            is_test=True,
            logger=logger,
            model_name=config.model,
            api_key=config.api_key,
            api_url=config.api_url,
            canonical_labels=config.labels,
            max_samples=config.max_samples,
            sleep_between_calls=config.sleep,
            display_every_n=config.display_every,
            max_retries=config.max_retries,
            initial_retry_delay=config.retry_delay
        )
        
        # Save predictions
        output_file = config.output_dir / "predictions_with_labels.json"
        save_test_predictions(results, output_file, logger)
        
        # Generate test summary
        successful_predictions = sum(1 for r in results if r.get("prediction"))
        failed_predictions = len(results) - successful_predictions
        
        test_summary = {
            "dataset": config.dataset,
            "model": config.model,
            "total_claims": len(results),
            "successful_predictions": successful_predictions,
            "failed_predictions": failed_predictions,
            "success_rate": round(successful_predictions / len(results), 4) if results else 0,
            "timestamp": datetime.now().isoformat()
        }
        
        summary_file = config.output_dir / "test_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(test_summary, f, indent=4)
        
        print("\n" + "=" * 80)
        print("TEST DATA SUMMARY")
        print("=" * 80)
        print(f"Total Claims: {test_summary['total_claims']}")
        print(f"Successful Predictions: {test_summary['successful_predictions']}")
        print(f"Failed Predictions: {test_summary['failed_predictions']}")
        print(f"Success Rate: {test_summary['success_rate']:.4f}")
        print("=" * 80)
        
        logger.info("Test data processing complete")
    
    else:
        # ==================== LABELED DATA PIPELINE ====================
        logger.info("Processing LABELED data with temporal split analysis")
        
        # Split data by date
        seen_claims, unseen_claims = split_by_date(
            claims, config.split_date, config.date_format
        )
        
        # Apply balanced sampling if enabled
        sampling_info = None
        if config.balanced_sampling and len(seen_claims) > len(unseen_claims):
            original_seen_count = len(seen_claims)
            target_size = len(unseen_claims)
            
            random.seed(config.seed)
            sampled_seen_claims = random.sample(seen_claims, target_size)
            
            sampling_info = {
                "enabled": True,
                "seed": config.seed,
                "original_seen_count": original_seen_count,
                "sampled_seen_count": target_size,
                "unseen_count": len(unseen_claims),
                "sampled_claim_ids": [c.get("claim_id") for c in sampled_seen_claims],
                "timestamp": datetime.now().isoformat()
            }
            
            sampling_file = config.output_dir / "sampling_info.json"
            with open(sampling_file, 'w', encoding='utf-8') as f:
                json.dump(sampling_info, f, indent=4)
            
            print(f"\n📊 BALANCED SAMPLING ENABLED:")
            print(f"   Original seen claims: {original_seen_count}")
            print(f"   Sampled seen claims: {target_size}")
            print(f"   Unseen claims: {len(unseen_claims)}")
            print(f"   Sampling seed: {config.seed}")
            print(f"   Sampling info saved to: {sampling_file}\n")
            
            logger.info(f"Balanced sampling: {original_seen_count} -> {target_size} seen (seed={config.seed})")
            seen_claims = sampled_seen_claims
        else:
            sampling_info = {"enabled": False}
            if config.balanced_sampling:
                print(f"\n⚠ Balanced sampling requested but seen ({len(seen_claims)}) <= unseen ({len(unseen_claims)})\n")
        
        # Create subdirectories
        seen_dir = config.output_dir / "seen"
        unseen_dir = config.output_dir / "unseen"
        seen_dir.mkdir(exist_ok=True)
        unseen_dir.mkdir(exist_ok=True)
        
        # Process seen data
        logger.info("=" * 80)
        logger.info("PROCESSING SEEN DATA")
        logger.info("=" * 80)
        seen_logger = setup_logger(seen_dir / "seen_run_log.txt")
        seen_results = process_claims(
            claims=seen_claims,
            is_test=False,
            logger=seen_logger,
            model_name=config.model,
            api_key=config.api_key,
            api_url=config.api_url,
            canonical_labels=config.labels,
            max_samples=config.max_samples,
            sleep_between_calls=config.sleep,
            display_every_n=config.display_every,
            max_retries=config.max_retries,
            initial_retry_delay=config.retry_delay
        )
        seen_metrics = calculate_metrics(seen_results, config.labels, seen_logger)
        save_results_with_metrics(seen_results, seen_metrics, seen_dir, "seen", seen_logger)
        
        # Process unseen data
        logger.info("=" * 80)
        logger.info("PROCESSING UNSEEN DATA")
        logger.info("=" * 80)
        unseen_logger = setup_logger(unseen_dir / "unseen_run_log.txt")
        unseen_results = process_claims(
            claims=unseen_claims,
            is_test=False,
            logger=unseen_logger,
            model_name=config.model,
            api_key=config.api_key,
            api_url=config.api_url,
            canonical_labels=config.labels,
            max_samples=config.max_samples,
            sleep_between_calls=config.sleep,
            display_every_n=config.display_every,
            max_retries=config.max_retries,
            initial_retry_delay=config.retry_delay
        )
        unseen_metrics = calculate_metrics(unseen_results, config.labels, unseen_logger)
        save_results_with_metrics(unseen_results, unseen_metrics, unseen_dir, "unseen", unseen_logger)
        
        # Compare seen vs unseen
        comparison = compare_seen_unseen(seen_results, unseen_results, config.labels, logger)
        
        if sampling_info:
            comparison["sampling_info"] = sampling_info
        
        # Generate summary report
        generate_summary_report(seen_metrics, unseen_metrics, comparison, config.output_dir, config, logger)
        
        logger.info("Labeled data processing complete")
    
    logger.info("=" * 80)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 80)
    
    print(f"\n✓ All outputs saved to: {config.output_dir}")
    print(f"✓ Log file: {log_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
