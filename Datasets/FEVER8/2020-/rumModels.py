#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FEVER8 Evaluator (AvalAI, OpenAI-compatible)
- Loads local JSON dataset
- Filters claims (2024+)
- Calls model for label prediction
- Computes metrics (Accuracy, Precision/Recall/F1 per label + Macro)
- Saves predictions.csv, metrics.json, plots (confusion matrix, per-label recall, distribution)
- Writes a concise run_log.txt

Notes:
- Set AVALAI_API_KEY via environment variable for safety.
- Endpoint is assumed OpenAI-compatible: POST {AVALAI_BASE_URL} with chat 'messages'
"""

import os
import sys
import csv
import json
import time
import math
import subprocess
from datetime import datetime
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional

# قبلی:
# AVALAI_BASE_URL = "https://api.avalai.ir/v1"

# جدید:

# (اگر خواستی از Responses API استفاده کنی: AVALAI_RESPONSES_URL = f"{AVALAI_BASE_URL}/responses")

MODEL_NAME = "gpt-4o-mini"  # نام درست مدل

# --------------------- USER CONFIG (defaults; can be overridden by CLI) ---------------------
AVALAI_BASE_URL = "https://api.avalai.ir/v1"
AVALAI_CHAT_URL = f"{AVALAI_BASE_URL}/chat/completions"   # مسیر درست   # OpenAI-compatible base endpoint
AVALAI_API_KEY  = "aa-hXsC7ulscaBbPcU6663EvyHVCiyty0HKu2ar4UUXCVU0W89Y"  # Prefer env var
MODEL_NAME      = "gpt-4o-mini"
DEFAULT_TEST_JSON = "data_dev.json"
MAX_CLAIMS      = -1               # -1 => use all filtered claims
SLEEP_BETWEEN   = 0.2              # seconds between API calls
RESULTS_DIR     = "gpt-4o-mini"    # output folder
REQUEST_TIMEOUT = 45               # seconds

# --------------------------------- Package management ---------------------------------
def ensure_package(pkg: str, import_name: Optional[str] = None, version: Optional[str] = None) -> None:
    try:
        __import__(import_name or pkg)
        return
    except ImportError:
        pass
    cmd = [sys.executable, "-m", "pip", "install", pkg] + ([version] if version else [])
    try:
        subprocess.check_call(cmd)
    except Exception:
        print(f"[WARN] Could not auto-install {pkg}. Please install it manually.", file=sys.stderr)

# Minimal deps
ensure_package("requests")
ensure_package("matplotlib")
ensure_package("numpy")
ensure_package("scikit-learn")

# Imports that depend on ensured packages
import requests  # type: ignore
import numpy as np  # type: ignore
import matplotlib
matplotlib.use("Agg")  # headless / no-GUI backend for servers
import matplotlib.pyplot as plt  # type: ignore
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support  # type: ignore

# --------------------------------- I/O utilities ---------------------------------
def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

# -------------------------- Dataset loading & filtering --------------------------
def load_local_test_json(path: str) -> List[Dict[str, str]]:
    """Load local test JSON file with items containing 'claim', 'label', optional 'claim_date'."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    claims: List[Dict[str, str]] = []
    for x in data:
        claim = x.get("claim") or x.get("normalized_claim") or ""
        label = x.get("label") or ""
        cdate = x.get("claim_date") or x.get("date") or ""
        if not claim or not label:
            # Skip invalid rows but continue
            continue
        claims.append({"claim": claim, "label": label, "claim_date": cdate})
    return claims

def is_2024_or_after(date_str: str) -> bool:
    """Return True if date_str parses and is >= 2024-01-01."""
    if not date_str:
        return False
    fmts = ("%d-%m-%Y", "%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%Y.%m.%d", "%d.%m.%Y", "%b %d, %Y")
    for fmt in fmts:
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt >= datetime(2020, 1, 1)
        except ValueError:
            continue
    return False

def load_claims_2024p(path: str) -> Tuple[List[Dict[str, str]], str]:
    """Load claims, keep those with date >= 2024-01-01."""
    data = load_local_test_json(path)
    filtered = [d for d in data if is_2024_or_after(d.get("claim_date", ""))]
    return filtered, "local_test_json"

# ------------------------------- Labels & normalization -------------------------------
CANON_LABELS = ["Supported", "Refuted", "Not Enough Evidence", "Conflicting Evidence/Cherry-picking"]

LABEL_NORMALIZATION = {
    "supported": "Supported",
    "refuted": "Refuted",
    "not enough evidence": "Not Enough Evidence",
    "conflicting evidence": "Conflicting Evidence/Cherry-picking",
    "conflicting evidence/cherry-picking": "Conflicting Evidence/Cherry-picking",
}

import re
def normalize_label(label: Optional[str]) -> Optional[str]:
    if not label:
        return None
    s = str(label).strip().lower()

    # نگاشت مستقیم
    direct = {
        "supported": "Supported",
        "refuted": "Refuted",
        "not enough evidence": "Not Enough Evidence",
        "conflicting evidence/cherry-picking": "Conflicting Evidence/Cherry-picking",
        "conflicting evidence": "Conflicting Evidence/Cherry-picking",  # اگر مدل کوتاه گفت
        "cherry-picking": "Conflicting Evidence/Cherry-picking",
    }
    if s in direct:
        return direct[s]

    # حذف کاراکترهای غیرحرفی/عددی برای مچ بهتر
    compact = re.sub(r"[^a-z]+", "", s)

    if compact in {"supported"}:
        return "Supported"
    if compact in {"refuted"}:
        return "Refuted"
    if compact in {"notenoughevidence","insufficientevidence","notenoughdata"}:
        return "Not Enough Evidence"
    # انواع ناقص CE را هم بپوشان
    if compact.startswith("conflic") or "cherrypick" in compact:
        return "Conflicting Evidence/Cherry-picking"

    return None


# ------------------------------- Model calling -------------------------------
def ask_model(model: str, claim: str) -> Optional[str]:
    if not AVALAI_API_KEY:
        print("[ERROR] AVALAI_API_KEY is not set.", file=sys.stderr)
        return None

    user_prompt = (
        "You are a fact-checking classifier.\n"
        "Choose ONE label strictly from:\n"
        "Supported | Refuted | Not Enough Evidence | Conflicting Evidence/Cherry-picking\n\n"
        f"Claim:\n{claim}\n\n"
        "Reply strictly as JSON: {\"label\": \"...\"} with no extra text."
    )

    try:
        resp = requests.post(
            AVALAI_CHAT_URL,
            headers={
                "Authorization": f"Bearer {AVALAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "You classify claims into the given label set."},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.0,
                "top_p": 0.0,
                "max_tokens": 50,  # <-- قبلاً 10 بود
                "response_format": {"type": "json_object"},  # JSON mode
            },
            timeout=REQUEST_TIMEOUT,
        )
        if resp.status_code != 200:
            print(f"[WARN] Non-200 response: {resp.status_code} - {resp.text[:300]}", file=sys.stderr)
            return None

        data = resp.json()
        txt = (data.get("choices", [{}])[0].get("message", {}).get("content", "")).strip()

        import json as _json, re
        label_raw = None
        try:
            label_raw = _json.loads(txt).get("label")
        except Exception:
            m = re.search(r'\"label\"\s*:\s*\"([^\"]+)\"', txt)
            if m:
                label_raw = m.group(1)

        return normalize_label(label_raw)

    except Exception as e:
        print(f"[ERROR] ask_model failed: {e}", file=sys.stderr)
        return None


# ------------------------------- Metrics & plotting -------------------------------
def compute_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    """
    Compute Accuracy, per-label precision/recall/f1/support and macro P/R/F1.
    Includes numeric confusion matrix.
    """
    labels = CANON_LABELS
    if len(y_true) and len(y_pred):
        cm = confusion_matrix(y_true, y_pred, labels=labels)
    else:
        cm = np.zeros((len(labels), len(labels)), dtype=int)

    acc = (cm.trace() / cm.sum()) if cm.sum() > 0 else 0.0

    if any(p in labels for p in y_pred):
        prec, rec, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, zero_division=0
        )
    else:
        # No valid predictions
        k = len(labels)
        prec = np.zeros(k)
        rec = np.zeros(k)
        f1 = np.zeros(k)
        support = np.zeros(k, dtype=int)

    per_label = {
        lab: {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, lab in enumerate(labels)
    }
    macro = {
        "precision": float(np.mean(prec)) if len(prec) else 0.0,
        "recall": float(np.mean(rec)) if len(rec) else 0.0,
        "f1": float(np.mean(f1)) if len(f1) else 0.0,
    }
    return {
        "accuracy": float(acc),
        "macro": macro,
        "per_label": per_label,
        "confusion_matrix": {"labels": labels, "matrix": cm.tolist()},
    }

def draw_confusion_matrix(cm: np.ndarray, labels: List[str], out_png: str) -> None:
    """Plot and save confusion matrix image."""
    fig = plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)

    # Annotate cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                horizontalalignment="center", verticalalignment="center"
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_bar(values_by_label: Dict[str, float], title: str, out_png: str) -> None:
    """Simple bar plot for dict label->value."""
    fig = plt.figure(figsize=(7, 4))
    names = list(values_by_label.keys())
    vals = list(values_by_label.values())
    xpos = np.arange(len(names))
    plt.bar(xpos, vals)
    plt.xticks(xpos, names, rotation=45, ha="right")
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

# -------------------------------------- Main --------------------------------------
def main(
    input_json: str = DEFAULT_TEST_JSON,
    results_dir: str = RESULTS_DIR,
    model_name: str = MODEL_NAME,
    max_claims: int = MAX_CLAIMS,
    sleep_between: float = SLEEP_BETWEEN,
) -> None:
    os.makedirs(results_dir, exist_ok=True)

    # Load and filter claims
    try:
        claims, source = load_claims_2024p(input_json)
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}", file=sys.stderr)
        sys.exit(1)

    if max_claims and max_claims > 0:
        claims = claims[:max_claims]

    # Predict
    y_true: List[str] = []
    y_pred: List[str] = []
    rows: List[List[Any]] = []
    invalid = 0
    t0 = time.time()

    for idx, item in enumerate(claims):
        gold = normalize_label(item.get("label"))
        claim_text = (item.get("claim") or "").strip()
        if not gold or not claim_text:
            continue

        pred = ask_model(model_name, claim_text)
        if pred is None:
            invalid += 1
            pred = "__INVALID__"

        y_true.append(gold)
        y_pred.append(pred)
        rows.append([idx, claim_text, item.get("claim_date", ""), gold, pred])
        if sleep_between > 0:
            time.sleep(sleep_between)

    t1 = time.time()

    # Save predictions CSV
    csv_path = os.path.join(results_dir, "predictions.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "claim", "claim_date", "gold", "pred"])
        w.writerows(rows)

    # Filter to valid preds for metrics
    valid_mask = [p in CANON_LABELS for p in y_pred]
    y_true_valid = [a for a, m in zip(y_true, valid_mask) if m]
    y_pred_valid = [b for b, m in zip(y_pred, valid_mask) if m]

    # Compute & save metrics
    metrics = compute_metrics(y_true_valid, y_pred_valid)
    metrics["n_total"] = len(y_true)
    metrics["n_valid"] = len(y_true_valid)
    metrics["invalid_outputs"] = int(invalid)
    metrics["runtime_sec"] = float(t1 - t0)
    metrics["model"] = model_name
    metrics["data_source"] = source
    write_json(os.path.join(results_dir, "metrics.json"), metrics)

    # Plots
    cm = np.array(metrics["confusion_matrix"]["matrix"])
    labels = metrics["confusion_matrix"]["labels"]
    draw_confusion_matrix(cm, labels, os.path.join(results_dir, "confusion_matrix.png"))

    # Per-label accuracy (recall)
    per_label_acc = {lab: metrics["per_label"][lab]["recall"] for lab in labels}
    plot_bar(per_label_acc, "Per-label Accuracy (Recall)", os.path.join(results_dir, "per_label_accuracy.png"))

    # Label distribution (gold)
    gold_counts = Counter(y_true)
    gold_dist = {k: gold_counts.get(k, 0) for k in labels}
    plot_bar(gold_dist, "Gold Label Distribution", os.path.join(results_dir, "label_distribution.png"))

    # Log
    log_lines: List[str] = []
    log_lines.append(f"Model: {model_name}")
    log_lines.append(f"Data source: {source}")
    log_lines.append(f"Claims processed (total/valid): {metrics['n_total']} / {metrics['n_valid']}")
    log_lines.append(f"Invalid outputs: {metrics['invalid_outputs']}")
    log_lines.append(f"Overall Accuracy: {metrics['accuracy']:.3f}")
    log_lines.append(
        f"Macro P/R/F1: {metrics['macro']['precision']:.3f} / "
        f"{metrics['macro']['recall']:.3f} / {metrics['macro']['f1']:.3f}"
    )
    log_lines.append("Per-label metrics:")
    for lab in labels:
        d = metrics["per_label"][lab]
        log_lines.append(f"  - {lab}: P={d['precision']:.3f} R={d['recall']:.3f} F1={d['f1']:.3f} (n={d['support']})")
    log_lines.append(f"Runtime (sec): {metrics['runtime_sec']:.1f}")

    write_text(os.path.join(results_dir, "run_log.txt"), "\n".join(log_lines))

    # Console summary
    print("\n".join(log_lines))
    print(f"\nSaved artifacts under: {results_dir}/")

# --------------------------------- CLI entrypoint ---------------------------------
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="FEVER8 Evaluator (AvalAI, OpenAI-compatible)")
    p.add_argument("--input", default=DEFAULT_TEST_JSON, help="Path to local JSON (default: data_dev.json)")
    p.add_argument("--out", default=RESULTS_DIR, help="Results directory (default: 'GPT-4o mini')")
    p.add_argument("--model", default=MODEL_NAME, help="Model name (default: 'GPT-4o mini')")
    p.add_argument("--max-claims", type=int, default=MAX_CLAIMS, help="-1 for all (default: -1)")
    p.add_argument("--sleep", type=float, default=SLEEP_BETWEEN, help="Seconds between API calls (default: 0.2)")
    args = p.parse_args()

    main(
        input_json=args.input,
        results_dir=args.out,
        model_name=args.model,
        max_claims=args.max_claims,
        sleep_between=args.sleep,
    )
