"""
Thin wrapper around the DatasetRegistry for backward compatibility.
"""

from typing import List, Dict, Any, Tuple
from .config import Config
from .dataset_registry import DatasetRegistry

DEFAULT_REGISTRY = "datasets.yaml"

def load_dataset(name: str, config: Config) -> Tuple[List[Dict[str, Any]], bool]:
    registry_path = getattr(config, "datasets_config", DEFAULT_REGISTRY)
    registry = DatasetRegistry(registry_path)
    return registry.load(name, config.data_dir, config.date_format, config.hf_token)