from pathlib import Path
import json
import pandas as pd


INPUT_FILE = Path("data/tracer/tracer_with_bias.csv")
OUTPUT_FILE = Path("data/tracer/tracer_bias_all.csv")


def normalize_label(x):
    if pd.isna(x):
        return ""

    s = str(x).strip().lower()

    if s in ["supported", "support", "true"]:
        return "Supported"

    if s in ["half-true", "half true", "half_true", "partly true", "partially true"]:
        return "Half-True"

    if s in ["refuted", "refute", "false"]:
        return "Refuted"

    return str(x).strip()


def normalize_date(x):
    if pd.isna(x) or str(x).strip() == "":
        return ""

    dt = pd.to_datetime(str(x).strip(), errors="coerce")
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


def read_csv_file(path: Path) -> pd.DataFrame:
    """
    Read the merged TRACER bias CSV file.
    This version is for comma-separated CSV files, not TSV files.
    """

    try:
        df = pd.read_csv(
            path,
            dtype=str,
            keep_default_na=False,
            encoding="utf-8-sig",
        )
        return df

    except Exception as e:
        print("\nNormal CSV read failed:")
        print(type(e).__name__, e)
        print("\nRetrying with safer CSV settings...")

        df = pd.read_csv(
            path,
            dtype=str,
            keep_default_na=False,
            encoding="utf-8-sig",
            quotechar='"',
            doublequote=True,
            escapechar="\\",
            engine="python",
            on_bad_lines="warn",
        )
        return df


def main():
    print("Looking for input file:")
    print(INPUT_FILE.resolve())

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    df = read_csv_file(INPUT_FILE)

    print("\nRaw rows:", len(df))
    print("Raw columns:")
    print(df.columns.tolist())

    required = [
        "source_dataset",
        "source_split",
        "source_id",
        "claim_text",
        "date_norm",
        "factcheck_label_norm",
        "all_detected_biases",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        print("\nAvailable columns:")
        for c in df.columns:
            print("-", c)
        raise ValueError(f"Missing required columns: {missing}")

    out = pd.DataFrame()

    # If _source_file exists, use it in claim_id to avoid duplicated ids across splits.
    if "_source_file" in df.columns:
        file_part = (
            df["_source_file"]
            .astype(str)
            .str.replace(".json", "", regex=False)
            .str.strip()
        )
    else:
        file_part = df["source_split"].astype(str).str.strip()

    out["claim_id"] = (
        df["source_dataset"].astype(str).str.strip()
        + "_"
        + file_part
        + "_"
        + df["source_id"].astype(str).str.strip()
    )

    out["claim"] = df["claim_text"].astype(str).str.strip()
    out["claim_date"] = df["date_norm"].apply(normalize_date)
    out["label"] = df["factcheck_label_norm"].apply(normalize_label)
    out["raw_label"] = df["factcheck_label_norm"]
    out["split"] = df["source_split"].astype(str).str.strip()
    out["source"] = df["source_dataset"].astype(str).str.strip()
    out["detected_biases"] = df["all_detected_biases"].apply(clean_bias_string)

    if "speaker" in df.columns:
        out["speaker"] = df["speaker"].astype(str).str.strip()

    print("\nBefore filtering:")
    print("Rows:", len(out))

    print("\nLabel distribution:")
    print(out["label"].value_counts(dropna=False))

    print("\nMissing dates:", int((out["claim_date"] == "").sum()))
    print("Missing claims:", int((out["claim"] == "").sum()))

    has_bias = out["detected_biases"].astype(str).str.strip() != ""
    print("\nRows with detected biases:", int(has_bias.sum()))
    print("Bias coverage:", round(has_bias.mean() * 100, 2), "%")

    # Keep only usable rows.
    out = out[out["claim"].astype(str).str.strip() != ""]
    out = out[out["claim_date"].astype(str).str.strip() != ""]
    out = out[out["label"].isin(["Supported", "Half-True", "Refuted"])]

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

    print("\nFinal split distribution:")
    print(out["split"].value_counts(dropna=False))

    final_has_bias = out["detected_biases"].astype(str).str.strip() != ""
    print("\nFinal rows with detected biases:", int(final_has_bias.sum()))
    print("Final bias coverage:", round(final_has_bias.mean() * 100, 2), "%")

    print("\nSaved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()