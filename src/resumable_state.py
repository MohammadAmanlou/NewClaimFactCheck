import json
import logging
import threading
from pathlib import Path
from typing import Set, Dict, Any

logger = logging.getLogger(__name__)

class ResumableState:
    """
    Manages the resumable state of the experiment using an append-only JSONL file.
    This provides a lightweight, Kaggle-friendly way to resume progress.
    """
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.lock = threading.Lock()
        
        # Ensure file exists securely
        if not self.file_path.exists():
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            self.file_path.touch()
            
        logger.info("Initialized ResumableState at %s", self.file_path)

    def load_completed_ids(self, prompt_method: str) -> Set[str]:
        """
        Reads the state file and identifies which claims have already been processed
        for the specified prompt method.
        """
        completed = set()
        with self.lock:
            if not self.file_path.exists():
                return completed
                
            with open(self.file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        if record.get("prompt_method") == prompt_method and "claim_id" in record:
                            completed.add(str(record["claim_id"]))
                    except json.JSONDecodeError:
                        logger.warning("Corrupted line in state file skipped.")
        
        logger.info("Loaded %d completed items for method '%s'", len(completed), prompt_method)
        return completed

    def save_result(self, record: Dict[str, Any]) -> None:
        """
        Thread-safely appends a processed record to the JSONL state file.
        Uses ensure_ascii=False to support multilingual strings perfectly.
        """
        with self.lock:
            with open(self.file_path, "a", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False)
                f.write("\n")
