from pathlib import Path
import json
import ast
import re
import pandas as pd


INPUT_FILE = Path("data/claimreviewplus/crp_with_bias.csv")
OUTPUT_FILE = Path("data/claimreviewplus/crp_bias_all.csv")


def normalize_label(x):
    if pd.isna(x):
        return ""

    s = str(x).strip().lower()

    if s in ["supported", "support", "true"]:
        return "Supported"

    if s in ["refuted", "refute", "false"]:
        return "Refuted"

    if s in ["misleading", "partly false", "partly true", "cherry-picking", "cherry picking"]:
        return "Misleading"

    if s in ["not enough information", "not enough evidence", "nei", "unknown", "unverifiable"]:
        return "Not Enough Information"

    return str(x).strip()


def extract_review_date(review_text):
    """
    Some rows have empty date, so we try to recover reviewDate from the review column.
    """
    if pd.isna(review_text) or str(review_text).strip() == "":
        return ""

    s = str(review_text).strip()

    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], dict):
            return str(obj[0].get("reviewDate", "")).strip()
    except Exception:
        pass

    m = re.search(r"reviewDate['\"]?\s*:\s*['\"]([^'\"]+)['\"]", s)
    if m:
        return m.group(1).strip()

    return ""


def normalize_date(date_value, review_value=""):
    raw = str(date_value).strip() if not pd.isna(date_value) else ""

    if raw == "":
        raw = extract_review_date(review_value)

    if raw == "":
        return ""

    dt = pd.to_datetime(raw, errors="coerce", utc=True)
    if pd.isna(dt):
        return ""

    return dt.strftime("%Y-%m-%d")


def clean_bias_string(x):
    if pd.isna(x):
        return ""

    s = str(x).strip()

    if s == "" or s.lower() in ["nan", "none", "null", "[]", "{}"]:
        return ""

    try:
        obj = json.loads(s)
        if isinstance(obj, list) and len(obj) == 0:
            return ""
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return s


def main():
    print("Looking for input file:")
    print(INPUT_FILE.resolve())

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    df = pd.read_csv(
        INPUT_FILE,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )

    print("\nRaw rows:", len(df))
    print("Raw columns:")
    print(df.columns.tolist())

    required = [
        "id",
        "text",
        "date",
        "author",
        "label",
        "review",
        "all_detected_biases",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        print("\nAvailable columns:")
        for c in df.columns:
            print("-", c)
        raise ValueError(f"Missing required columns: {missing}")

    out = pd.DataFrame()

    out["claim_id"] = "crp_" + df["id"].astype(str).str.strip()
    out["claim"] = df["text"].astype(str).str.strip()

    out["claim_date"] = [
        normalize_date(d, r)
        for d, r in zip(df["date"], df["review"])
    ]

    out["label"] = df["label"].apply(normalize_label)
    out["raw_label"] = df["label"].astype(str).str.strip()
    out["source"] = "claimreview2024plus"
    out["author"] = df["author"].astype(str).str.strip()
    out["detected_biases"] = df["all_detected_biases"].apply(clean_bias_string)

    if "cognitive_bias" in df.columns:
        out["bias_label"] = df["cognitive_bias"].astype(str).str.strip()

    if "bias_confidence" in df.columns:
        out["bias_confidence"] = df["bias_confidence"].astype(str).str.strip()

    if "confidence_level" in df.columns:
        out["bias_confidence_level"] = df["confidence_level"].astype(str).str.strip()

    print("\nBefore filtering:")
    print("Rows:", len(out))

    print("\nLabel distribution:")
    print(out["label"].value_counts(dropna=False))

    print("\nMissing dates:", int((out["claim_date"] == "").sum()))
    print("Missing claims:", int((out["claim"] == "").sum()))

    has_bias = out["detected_biases"].astype(str).str.strip() != ""
    print("\nRows with detected biases:", int(has_bias.sum()))
    print("Bias coverage:", round(has_bias.mean() * 100, 2), "%")

    valid_labels = [
        "Supported",
        "Refuted",
        "Misleading",
        "Not Enough Information",
    ]

    out = out[out["claim"].astype(str).str.strip() != ""]
    out = out[out["claim_date"].astype(str).str.strip() != ""]
    out = out[out["label"].isin(valid_labels)]

    before_dedup = len(out)
    out = out.drop_duplicates(subset=["claim_id"], keep="first")
    after_dedup = len(out)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print("\nAfter filtering:")
    print("Rows:", len(out))
    print("Dropped duplicates:", before_dedup - after_dedup)

    print("\nFinal label distribution:")
    print(out["label"].value_counts(dropna=False))

    final_has_bias = out["detected_biases"].astype(str).str.strip() != ""
    print("\nFinal rows with detected biases:", int(final_has_bias.sum()))
    print("Final bias coverage:", round(final_has_bias.mean() * 100, 2), "%")

    print("\nSaved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()