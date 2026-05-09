import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

@dataclass
class Config:
    dataset: str
    model: str
    api_url: str
    labels: List[str]
    temporal_split: str
    date_format: str
    max_retries: int
    initial_retry_delay: float
    sleep_between_calls: float
    max_samples: Optional[int]
    balanced_sampling: bool
    sampling_seed: int
    display_every_n: int
    output_root: Path
    api_key: str = ""
    hf_token: str = ""
    prompt_methods: List[str] = field(default_factory=lambda: ["naive"])
    data_dir: Path = Path("data")
    datasets_config: Path = Path("datasets.yaml")

    @classmethod
    def from_yaml(cls, path: str = "config.yaml") -> "Config":
        with open(path) as f:
            cfg = yaml.safe_load(f)

        api_key = os.getenv(cfg.get("api_key_env", "AVALAI_API_KEY"), "")
        hf_token = os.getenv(cfg.get("hf_token_env", "HF_TOKEN"), "")

        rate = cfg.get("rate_limit", {})
        samp = cfg.get("sampling", {})
        prompt_methods = cfg.get("prompt_methods", ["naive"])
        

        return cls(
            dataset=cfg["dataset"],
            model=cfg["model"],
            api_url=cfg["api_url"],
            labels=cfg["labels"],
            temporal_split=cfg["temporal_split"],
            date_format=cfg.get("date_format", "%d-%m-%Y"),
            max_retries=rate.get("max_retries", 5),
            initial_retry_delay=rate.get("initial_retry_delay", 0.5),
            sleep_between_calls=rate.get("sleep_between_calls", 0.1),
            max_samples=samp.get("max_samples"),
            balanced_sampling=samp.get("balanced", False),
            sampling_seed=samp.get("seed", 42),
            display_every_n=cfg.get("display_every_n", 10),
            output_root=Path(cfg.get("output_root", "results")),
            api_key=api_key,
            hf_token=hf_token,
            prompt_methods=prompt_methods,
            datasets_config=Path(cfg.get("datasets_config", "datasets.yaml"))
        )