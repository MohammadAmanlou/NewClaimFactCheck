"""
Generic dataset loader that reads a YAML registry and maps any local CSV/JSON
(and special HuggingFace datasets) into the standard internal schema:

{
    "claim_id": int,
    "claim": str,
    "claim_date": str (formatted with config.date_format),
    "label": str (optional, not present for test data),
    "speaker": str | None,
    "source": str
}
"""

import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from datasets import load_dataset as hf_load_dataset


class DatasetRegistry:
    def __init__(self, registry_path: Path):
        with open(registry_path, "r", encoding="utf-8") as f:
            self.definitions = yaml.safe_load(f)

    def load(self, name: str, data_dir: Path, output_date_format: str, hf_token: Optional[str] = None) -> Tuple[List[Dict[str, Any]], bool]:
        """Return (list of standardized claim dicts, is_test)."""
        if name not in self.definitions:
            raise ValueError(f"Unknown dataset: {name}")

        spec = self.definitions[name]
        dtype = spec["type"]

        if dtype == "hf":
            return self._load_hf(spec, hf_token), False
        elif dtype == "csv":
            return self._load_csv(spec, data_dir, output_date_format), spec.get("is_test", False)
        elif dtype == "json":
            return self._load_json(spec, data_dir, output_date_format), spec.get("is_test", False)
        else:
            raise ValueError(f"Unsupported dataset type: {dtype}")

    def _load_hf(self, spec: dict, token: Optional[str]) -> List[Dict[str, Any]]:
        print(f"Loading HuggingFace dataset: {spec['source']}...")
        dataset = hf_load_dataset(spec["source"], split=spec.get("split", "test"), token=token)
        claims = []
        for idx, item in enumerate(dataset):
            claims.append({
                "claim_id": idx,
                "claim": item.get("claim", ""),
                "claim_date": item.get("claimDate", ""),
                "label": item.get("rating", ""),
                "speaker": item.get("claimant"),
                "source": "claimreview_plus"
            })
        print(f"✓ Loaded {len(claims)} claims")
        return claims

    def _load_csv(self, spec: dict, data_dir: Path, output_date_format: str) -> List[Dict[str, Any]]:
        file_path = data_dir / spec["path"]
        print(f"Loading CSV: {file_path}...")
        df = pd.read_csv(file_path)

        mapping = spec["mapping"]
        # Filter unwanted labels
        if "filter" in spec and "exclude_labels" in spec["filter"]:
            label_col = mapping.get("label", spec["filter"].get("label_column", "label"))
            excluded = spec["filter"]["exclude_labels"]
            original_count = len(df)
            df = df[~df[label_col].isin(excluded)]
            print(f"  Filtered out {original_count - len(df)} rows with labels {excluded}")

        claims = []
        date_fmt_in = spec.get("date_format", "%Y-%m-%d")
        is_test = spec.get("is_test", False)

        for _, row in df.iterrows():
            # Parse date
            raw_date = row.get(mapping.get("claim_date", ""))
            formatted_date = ""
            if raw_date and pd.notna(raw_date):
                try:
                    if date_fmt_in == "%Y-%m-%dT%H:%M:%S":
                        # handle ISO with time
                        raw_str = str(raw_date).replace("T00:00:00", "")
                        dt = datetime.fromisoformat(raw_str)
                    else:
                        dt = datetime.strptime(str(raw_date), date_fmt_in)
                    formatted_date = dt.strftime(output_date_format)
                except:
                    pass  # keep empty

            claim_dict = {
                "claim_id": row.get(mapping["claim_id"], len(claims)),
                "claim": str(row[mapping["claim"]]),
                "claim_date": formatted_date,
                "speaker": row.get(mapping.get("speaker")) if "speaker" in mapping else None,
                "source": file_path.stem,
            }
            if not is_test and "label" in mapping:
                claim_dict["label"] = str(row[mapping["label"]]) if pd.notna(row[mapping["label"]]) else ""
            claims.append(claim_dict)
        print(f"✓ Loaded {len(claims)} claims")
        return claims

    def _load_json(self, spec: dict, data_dir: Path, output_date_format: str) -> List[Dict[str, Any]]:
        file_path = data_dir / spec["path"]
        print(f"Loading JSON: {file_path}...")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        mapping = spec["mapping"]
        date_fmt_in = spec.get("date_format", output_date_format)
        is_test = spec.get("is_test", False)

        claims = []
        for idx, item in enumerate(data):
            # Parse date
            raw_date = item.get(mapping.get("claim_date", ""))
            formatted_date = ""
            if raw_date:
                try:
                    dt = datetime.strptime(str(raw_date), date_fmt_in)
                    formatted_date = dt.strftime(output_date_format)
                except:
                    pass

            claim_dict = {
                "claim_id": item.get(mapping["claim_id"], idx),
                "claim": item[mapping["claim"]],
                "claim_date": formatted_date,
                "speaker": item.get(mapping.get("speaker")) if "speaker" in mapping else None,
                "source": file_path.stem,
            }
            if not is_test and "label" in mapping:
                claim_dict["label"] = item.get(mapping["label"], "")
            if is_test:
                claim_dict["original_data"] = item  # preserve original for test output
            claims.append(claim_dict)
        print(f"✓ Loaded {len(claims)} claims")
        return claims