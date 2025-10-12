#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FEVER8 / AVeriTeC-style Evaluator — updated to work with your dataset schema
----------------------------------------------------------------------------
Works with items shaped like:
[
  {
    "claim": "Kenyan counties allocated KSh361.05 billion (64%) to recurrent expenditure.",
    "claim_id": 0,
    "claim_date": "06-10-2024",
    "speaker": "Anne Waiguru",
    "original_claim_url": "https://...",
    "reporting_source": "...",
    "location_ISO_code": "KE"
  },
  ...
]

What’s new vs. your previous script:
- Loader accepts JSON array or JSONL, with your exact fields. If a gold `label` exists, we use it; otherwise we still run predictions and skip metrics.
- Flexible **date parsing** (DD-MM-YYYY by default) + optional `--start-date/--end-date` filter.
- Safer **API key handling**: read AVALAI_API_KEY from env (no hardcoded secrets).
- Predictions CSV/JSONL now include your metadata (claim_id, speaker, location_ISO_code, URL, ...).
- Metrics are computed only if gold labels are present; otherwise a minimal metrics.json is written.
- Small CLI polish and clearer logs.

Run examples:
  python fever8_evaluator_updated_for_AVeriTeC_like_dataset.py \
      --input your_dataset.json \
      --out runs/gpt-4o-mini --model gpt-4o-mini --start-date 2024-01-01

  # If your dates are definitely day-first (e.g., 06-10-2024 => 6 Oct 2024):
  python fever8_evaluator_updated_for_AVeriTeC_like_dataset.py --input your.json --dayfirst

Environment:
  export AVALAI_API_KEY=...   # REQUIRED
  export AVALAI_BASE_URL=https://api.avalai.ir/v1   # optional (default below)

Notes:
- Endpoint uses OpenAI-compatible Chat Completions with JSON mode.
- Label set is: Supported | Refuted | Not Enough Evidence | Conflicting Evidence/Cherry-picking
"""

from __future__ import annotations
import os
import sys
import csv
import json
import time
import math
import subprocess
from datetime import datetime, date
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional

# --------------------- USER CONFIG (overridable by env/CLI) ---------------------
DEFAULT_MODEL       = "gpt-4o-mini"
AVALAI_BASE_URL     = "https://api.avalai.ir/v1"
AVALAI_CHAT_URL     = f"{AVALAI_BASE_URL}/chat/completions"
AVALAI_API_KEY      = "aa-hXsC7ulscaBbPcU6663EvyHVCiyty0HKu2ar4UUXCVU0W89Y"
DEFAULT_TEST_JSON   = "test_2025.json"
MAX_CLAIMS_DEFAULT  = -1               # -1 => use all filtered claims
SLEEP_BETWEEN_DEF   = 0.2              # seconds between API calls
RESULTS_DIR_DEFAULT = "gpt-4o-mini"    # output folder
REQUEST_TIMEOUT     = 45               # seconds

# --------------------------------- Package management ---------------------------------
def _ensure_package(pkg: str, import_name: Optional[str] = None) -> None:
    try:
        __import__(import_name or pkg)
        return
    except ImportError:
        pass
    cmd = [sys.executable, "-m", "pip", "install", pkg]
    try:
        subprocess.check_call(cmd)
    except Exception:
        print(f"[WARN] Could not auto-install {pkg}. Please install it manually.", file=sys.stderr)

for _p, _imp in [("requests", None), ("matplotlib", None), ("numpy", None), ("scikit-learn", None)]:
    _ensure_package(_p, _imp)

import requests  # type: ignore
import numpy as np  # type: ignore
import matplotlib
matplotlib.use("Agg")  # headless backend
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
DATE_FORMATS = (
    "%d-%m-%Y", "%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%Y.%m.%d", "%d.%m.%Y",
    "%b %d, %Y", "%d %b %Y", "%B %d, %Y", "%d %B %Y",
)

def parse_date_flexible(s: str, dayfirst: bool = True) -> Optional[date]:
    if not s:
        return None
    s = s.strip()
    # Try explicit formats first
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    # Heuristic for DD-MM-YYYY vs MM-DD-YYYY
    parts = s.replace("/", "-").replace(".", "-").split("-")
    if len(parts) == 3 and all(p.isdigit() for p in parts):
        a, b, c = parts
        if len(c) == 4:
            d1, m1 = int(a), int(b)
            if dayfirst:
                # interpret as DD-MM-YYYY
                if 1 <= d1 <= 31 and 1 <= m1 <= 12:
                    try:
                        return date(int(c), m1, d1)
                    except ValueError:
                        return None
            else:
                # interpret as MM-DD-YYYY
                if 1 <= d1 <= 12 and 1 <= m1 <= 31:
                    try:
                        return date(int(c), d1, m1)
                    except ValueError:
                        return None
    return None

SchemaItem = Dict[str, Any]

def load_json_any(path: str) -> List[SchemaItem]:
    """Load a JSON array or JSONL file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    if path.lower().endswith(".jsonl"):
        rows: List[SchemaItem] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows
    # else assume JSON array
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        if isinstance(data, list):
            return data
        raise ValueError("Expected a JSON array; got an object")

def load_claims_for_schema(path: str, start_date: Optional[date], end_date: Optional[date], dayfirst: bool) -> List[SchemaItem]:
    """
    Load items with your schema, keep essential fields, and apply optional date filter.
    - Required: `claim`
    - Optional: `label`, `claim_id`, `claim_date`, `speaker`, `original_claim_url`, `reporting_source`, `location_ISO_code`
    """
    raw = load_json_any(path)
    items: List[SchemaItem] = []
    for x in raw:
        claim = (x.get("claim") or x.get("normalized_claim") or "").strip()
        if not claim:
            continue
        item: SchemaItem = {
            "claim": claim,
            "claim_id": x.get("claim_id"),
            "claim_date": x.get("claim_date") or x.get("date") or "",
            "speaker": x.get("speaker"),
            "original_claim_url": x.get("original_claim_url"),
            "reporting_source": x.get("reporting_source"),
            "location_ISO_code": x.get("location_ISO_code"),
            "label": x.get("label") or "",
        }
        # Date filter (if provided)
        if start_date or end_date:
            dt = parse_date_flexible(item["claim_date"], dayfirst=dayfirst)
            if dt is None:
                # If date can't be parsed, drop it when filtering is requested
                continue
            if start_date and dt < start_date:
                continue
            if end_date and dt > end_date:
                continue
        items.append(item)
    return items

# ------------------------------- Labels & normalization -------------------------------
CANON_LABELS = [
    "Supported",
    "Refuted",
    "Not Enough Evidence",
    "Conflicting Evidence/Cherry-picking",
]

import re

def normalize_label(label: Optional[str]) -> Optional[str]:
    if label is None:
        return None
    s = str(label).strip().lower()
    direct = {
        "supported": "Supported",
        "refuted": "Refuted",
        "not enough evidence": "Not Enough Evidence",
        "conflicting evidence/cherry-picking": "Conflicting Evidence/Cherry-picking",
        "conflicting evidence": "Conflicting Evidence/Cherry-picking",
        "cherry-picking": "Conflicting Evidence/Cherry-picking",
    }
    if s in direct:
        return direct[s]
    compact = re.sub(r"[^a-z]+", "", s)
    if compact == "supported":
        return "Supported"
    if compact == "refuted":
        return "Refuted"
    if compact in {"notenoughevidence", "insufficientevidence", "notenoughdata"}:
        return "Not Enough Evidence"
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
                "max_tokens": 50,
                "response_format": {"type": "json_object"},
            },
            timeout=REQUEST_TIMEOUT,
        )
        if resp.status_code != 200:
            print(f"[WARN] Non-200 response: {resp.status_code} - {resp.text[:300]}", file=sys.stderr)
            return None

        data = resp.json()
        txt = (data.get("choices", [{}])[0].get("message", {}).get("content", "")).strip()

        label_raw = None
        try:
            label_raw = json.loads(txt).get("label")
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
        k = len(labels)
        prec = np.zeros(k); rec = np.zeros(k); f1 = np.zeros(k); support = np.zeros(k, dtype=int)

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
    fig = plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"), ha="center", va="center")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_bar(values_by_label: Dict[str, float], title: str, out_png: str) -> None:
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
    results_dir: str = RESULTS_DIR_DEFAULT,
    model_name: str = DEFAULT_MODEL,
    max_claims: int = MAX_CLAIMS_DEFAULT,
    sleep_between: float = SLEEP_BETWEEN_DEF,
    start_date_str: Optional[str] = None,
    end_date_str: Optional[str] = None,
    dayfirst: bool = True,
) -> None:
    os.makedirs(results_dir, exist_ok=True)

    # Parse date range if provided
    start_date = parse_date_flexible(start_date_str, dayfirst=dayfirst) if start_date_str else None
    end_date   = parse_date_flexible(end_date_str,   dayfirst=dayfirst) if end_date_str   else None

    # Load and (optionally) filter claims
    try:
        claims = load_claims_for_schema(input_json, start_date, end_date, dayfirst)
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
        gold = normalize_label(item.get("label")) if item.get("label") else None
        claim_text = (item.get("claim") or "").strip()
        if not claim_text:
            continue

        pred = ask_model(model_name, claim_text)
        if pred is None:
            invalid += 1
            pred = "__INVALID__"

        # Collect for metrics if gold present
        if gold:
            y_true.append(gold)
            y_pred.append(pred)

        rows.append([
            item.get("claim_id", idx),
            claim_text,
            item.get("claim_date", ""),
            item.get("speaker", ""),
            item.get("location_ISO_code", ""),
            item.get("original_claim_url", ""),
            item.get("reporting_source", ""),
            gold or "",
            pred,
        ])
        if sleep_between > 0:
            time.sleep(sleep_between)

    t1 = time.time()

    # Save predictions CSV
    csv_path = os.path.join(results_dir, "predictions.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "claim_id", "claim", "claim_date", "speaker", "location_ISO_code",
            "original_claim_url", "reporting_source", "gold", "pred"
        ])
        w.writerows(rows)

    # Also save predictions as JSONL for convenience
    jsonl_path = os.path.join(results_dir, "predictions.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in rows:
            obj = {
                "claim_id": r[0],
                "claim": r[1],
                "claim_date": r[2],
                "speaker": r[3],
                "location_ISO_code": r[4],
                "original_claim_url": r[5],
                "reporting_source": r[6],
                "gold": r[7],
                "pred": r[8],
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # Compute & save metrics (only if gold labels exist)
    metrics: Dict[str, Any]
    have_gold = len(y_true) > 0
    if have_gold:
        valid_mask = [p in CANON_LABELS for p in y_pred]
        y_true_valid = [a for a, m in zip(y_true, valid_mask) if m]
        y_pred_valid = [b for b, m in zip(y_pred, valid_mask) if m]
        metrics = compute_metrics(y_true_valid, y_pred_valid)
        metrics["n_total_with_gold"] = len(y_true)
        metrics["n_valid_preds"] = len(y_pred_valid)
    else:
        metrics = {
            "note": "No gold labels found; metrics skipped.",
            "confusion_matrix": {"labels": CANON_LABELS, "matrix": [[0]*4 for _ in range(4)]},
            "accuracy": 0.0,
            "macro": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "per_label": {lab: {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0} for lab in CANON_LABELS}
        }

    metrics["invalid_outputs"] = int(invalid)
    metrics["runtime_sec"] = float(t1 - t0)
    metrics["model"] = model_name
    metrics["data_source"] = "local_json/jsonl"
    metrics["date_filter"] = {
        "start_date": start_date_str,
        "end_date": end_date_str,
        "dayfirst": dayfirst,
    }

    write_json(os.path.join(results_dir, "metrics.json"), metrics)

    # Plots (only if we have gold)
    if have_gold:
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

    # Lightweight dataset summary
    summary = {
        "n_items": len(claims),
        "by_country": dict(Counter([c.get("location_ISO_code", "") or "" for c in claims])),
        "by_speaker_top10": dict(Counter([c.get("speaker", "") or "" for c in claims]).most_common(10)),
        "date_span": {
            "min": min((parse_date_flexible(c.get("claim_date", ""), True) for c in claims if c.get("claim_date")), default=None),
            "max": max((parse_date_flexible(c.get("claim_date", ""), True) for c in claims if c.get("claim_date")), default=None),
        },
    }
    # Convert dates to ISO strings if present
    for k in ("min", "max"):
        if isinstance(summary["date_span"][k], date):
            summary["date_span"][k] = summary["date_span"][k].isoformat()
    write_json(os.path.join(results_dir, "dataset_summary.json"), summary)

    # Run log
    log_lines: List[str] = []
    log_lines.append(f"Model: {model_name}")
    log_lines.append(f"Data source: local_json/jsonl")
    log_lines.append(f"Items processed: {len(claims)}")
    log_lines.append(f"Invalid API outputs: {metrics['invalid_outputs']}")
    if have_gold:
        log_lines.append(f"Overall Accuracy: {metrics['accuracy']:.3f}")
        log_lines.append(
            f"Macro P/R/F1: {metrics['macro']['precision']:.3f} / "
            f"{metrics['macro']['recall']:.3f} / {metrics['macro']['f1']:.3f}"
        )
        log_lines.append("Per-label metrics:")
        for lab in CANON_LABELS:
            d = metrics["per_label"][lab]
            log_lines.append(f"  - {lab}: P={d['precision']:.3f} R={d['recall']:.3f} F1={d['f1']:.3f} (n={d['support']})")
    else:
        log_lines.append("No gold labels present — metrics skipped.")
    log_lines.append(f"Runtime (sec): {metrics['runtime_sec']:.1f}")

    write_text(os.path.join(results_dir, "run_log.txt"), "\n".join(log_lines))
    print("\n".join(log_lines))
    print(f"\nSaved artifacts under: {results_dir}/")

# --------------------------------- CLI entrypoint ---------------------------------
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="FEVER8 Evaluator (AvalAI/OpenAI-compatible) — updated for your schema")
    p.add_argument("--input", default=DEFAULT_TEST_JSON, help="Path to local JSON or JSONL (default: data_dev.json)")
    p.add_argument("--out", default=RESULTS_DIR_DEFAULT, help="Results directory (default: gpt-4o-mini)")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Model name (default: gpt-4o-mini)")
    p.add_argument("--max-claims", type=int, default=MAX_CLAIMS_DEFAULT, help="-1 for all (default: -1)")
    p.add_argument("--sleep", type=float, default=SLEEP_BETWEEN_DEF, help="Seconds between API calls (default: 0.2)")
    p.add_argument("--start-date", default=None, help="Filter: start date (e.g., 2024-01-01 or 01-01-2024)")
    p.add_argument("--end-date", default=None, help="Filter: end date (e.g., 2024-12-31 or 31-12-2024)")
    p.add_argument("--dayfirst", action="store_true", help="Interpret ambiguous numeric dates as DD-MM-YYYY (default: False)")

    args = p.parse_args()

    main(
        input_json=args.input,
        results_dir=args.out,
        model_name=args.model,
        max_claims=args.max_claims,
        sleep_between=args.sleep,
        start_date_str=args.start_date,
        end_date_str=args.end_date,
        dayfirst=args.dayfirst or True,  # default True for your snippet (DD-MM-YYYY)
    )
