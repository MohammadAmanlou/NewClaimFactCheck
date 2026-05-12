"""
Registry for custom dataset filter functions.
"""
from typing import Callable, List, Dict, Any
import pandas as pd

_CSV_FILTERS = {}
_JSON_FILTERS = {}

def register_csv_filter(name: str):
    """Decorator to register a custom CSV (Pandas DataFrame) filter."""
    def decorator(fn: Callable[[pd.DataFrame], pd.DataFrame]):
        _CSV_FILTERS[name] = fn
        return fn
    return decorator

def register_json_filter(name: str):
    """Decorator to register a custom JSON List[dict] filter."""
    def decorator(fn: Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]):
        _JSON_FILTERS[name] = fn
        return fn
    return decorator

def apply_csv_filter(name: str, df: pd.DataFrame) -> pd.DataFrame:
    if name not in _CSV_FILTERS:
        raise ValueError(f"Unknown custom CSV filter '{name}'")
    return _CSV_FILTERS[name](df)

def apply_json_filter(name: str, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if name not in _JSON_FILTERS:
        raise ValueError(f"Unknown custom JSON filter '{name}'")
    return _JSON_FILTERS[name](data)

# ====== Dataset Specific Filters ======

@register_csv_filter("crp_no_image")
def crp_no_image(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out rows that have an image."""
    if 'image' in df.columns:
        original_count = len(df)
        df = df[df['image'].isna() | (df['image'] == '')]
        print(f"  Applied 'crp_no_image' custom filter: filtered out {original_count - len(df)} rows with images.")
    return df
