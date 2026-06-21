import json
from pathlib import Path
import pandas as pd


INPUT_DIR = Path("data/tracer/TRACER/dataset")
OUTPUT_DIR = Path("data/tracer")
OUTPUT_FILE = OUTPUT_DIR / "tracer_all.csv"


def load_json_or_jsonl(path: Path):
    text = path.read_text(encoding="utf-8").strip()

    if not text:
        return []

    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            for key in ["data", "examples", "claims", "items"]:
                if key in obj and isinstance(obj[key], list):
                    return obj[key]
            return [obj]
    except json.JSONDecodeError:
        pass

    rows = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def first_existing(d, keys, default=""):
    for key in keys:
        if key in d and d[key] is not None:
            return d[key]
    return default


def normalize_label(value):
    if value is None:
        return ""

    s = str(value).strip()
    low = s.lower().replace("_", " ").replace("-", " ")

    if low in ["true"]:
        return "True"

    if low in [
        "half true",
        "halftrue",
        "half",
        "partly true",
        "partially true",
        "mostly true",
        "barely true",
    ]:
        return "Half-True"

    if low in [
        "false",
        "mostly false",
        "pants on fire",
        "pantsfire",
        "pants on fire!",
    ]:
        return "False"

    if s in ["True", "Half-True", "False"]:
        return s

    return s


def normalize_date(value):
    if value is None or str(value).strip() == "":
        return ""

    dt = pd.to_datetime(str(value).strip(), errors="coerce")
    if pd.isna(dt):
        return ""

    return dt.strftime("%Y-%m-%d")


def convert_split(split_name: str, filename: str):
    path = INPUT_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    data = load_json_or_jsonl(path)
    rows = []

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue

        claim = str(first_existing(item, ["claim", "statement", "text"], "")).strip()

        raw_label = first_existing(
            item,
            ["veracity", "label", "rating", "verdict", "gold_label"],
            "",
        )
        label = normalize_label(raw_label)

        raw_date = first_existing(
            item,
            ["date", "claim_date", "statement_date", "published_date"],
            "",
        )
        claim_date = normalize_date(raw_date)

        claim_id = first_existing(
            item,
            ["example_id", "claim_id", "id", "uid"],
            f"{split_name}_{i}",
        )

        speaker = first_existing(item, ["speaker"], "")

        # فقط ستون‌های لازم برای pipeline را نگه می‌داریم.
        # ruling/evidence را عمداً حذف می‌کنیم چون خیلی بلند و noisy هستند.
        rows.append(
            {
                "claim_id": claim_id,
                "claim": claim,
                "claim_date": claim_date,
                "label": label,
                "raw_label": raw_label,
                "split": split_name,
                "source": "TRACER_POLITIFACT_HIDDEN",
                "speaker": speaker,
                "detected_biases": "",
            }
        )

    return rows


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []
    all_rows.extend(convert_split("train", "train.json"))
    all_rows.extend(convert_split("dev", "dev.json"))
    all_rows.extend(convert_split("test", "test.json"))

    df = pd.DataFrame(all_rows)

    print("\nBefore filtering")
    print("Rows:", len(df))
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nRaw label distribution:")
    print(df["raw_label"].value_counts(dropna=False))
    print("\nNormalized label distribution:")
    print(df["label"].value_counts(dropna=False))
    print("\nSplit distribution:")
    print(df["split"].value_counts(dropna=False))
    print("\nMissing dates:", df["claim_date"].eq("").sum())
    print("Missing claims:", df["claim"].eq("").sum())

    # فقط ردیف‌های قابل استفاده
    df = df[df["claim"].astype(str).str.strip() != ""]
    df = df[df["label"].isin(["True", "Half-True", "False"])]

    # claim_id را string نگه می‌داریم که بعداً مشکل type ندهد
    df["claim_id"] = df["claim_id"].astype(str)

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print("\nAfter filtering")
    print("Rows:", len(df))
    print("\nFinal label distribution:")
    print(df["label"].value_counts(dropna=False))
    print("\nFinal split distribution:")
    print(df["split"].value_counts(dropna=False))
    print("\nSaved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()