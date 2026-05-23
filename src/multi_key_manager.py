import threading
import time
import logging
from collections import deque
from typing import Set, Dict, List

logger = logging.getLogger(__name__)

class DailyQuotaExhaustedException(Exception):
    """Raised when all provided API keys have hit their Daily limits."""
    pass

class MultiKeyManager:
    """
    A thread-safe API Key manager that enforces Requests Per Minute (RPM)
    and Requests Per Day (RPD) limits across multiple Google API keys.
    """
    def __init__(self, key_file_path: str, rpm_limit: int = 15, rpd_limit: int = 1450):
        self.rpm_limit = rpm_limit
        self.rpd_limit = rpd_limit
        
        # Load unique keys (handling any duplicates silently)
        with open(key_file_path, 'r', encoding='utf-8') as f:
            raw_keys = [line.strip() for line in f if line.strip()]
        
        self.keys: List[str] = list(set(raw_keys))
        if not self.keys:
            raise ValueError(f"No API keys found in {key_file_path}")
            
        logger.info("Initialized MultiKeyManager with %d unique keys.", len(self.keys))
        
        self.lock = threading.Lock()
        
        # Tracking states
        # Deque of timestamps (maxlen=rpm_limit) for sliding window
        self.rpm_tracking: Dict[str, deque] = {k: deque(maxlen=self.rpm_limit) for k in self.keys}
        self.rpd_tracking: Dict[str, int] = {k: 0 for k in self.keys}
        
        self.exhausted_keys: Set[str] = set()
        self._round_robin_index = 0

    def get_available_key(self) -> str:
        """
        Fetches an available API key dynamically.
        Blocks and waits if keys are temporarily limited by RPM.
        Raises DailyQuotaExhaustedException if all keys hit the daily limit.
        """
        while True:
            key = self._try_get_key()
            if key is not None:
                return key
            
            # If no keys are currently available due to RPM limits, wait briefly
            time.sleep(0.5)

    def _try_get_key(self) -> str | None:
        """Helper to iterate through keys under the lock and find an available one."""
        with self.lock:
            if len(self.exhausted_keys) == len(self.keys):
                raise DailyQuotaExhaustedException("All API keys have exhausted their daily quota.")
            
            now = time.time()
            for _ in range(len(self.keys)):
                key = self.keys[self._round_robin_index]
                self._round_robin_index = (self._round_robin_index + 1) % len(self.keys)
                
                if key in self.exhausted_keys:
                    continue
                    
                self._clean_expired_timestamps(key, now)
                
                if len(self.rpm_tracking[key]) < self.rpm_limit:
                    if self.rpd_tracking[key] < self.rpd_limit:
                        return self._utilize_key(key, now)
                    else:
                        self.exhausted_keys.add(key)
                        
            return None

    def _clean_expired_timestamps(self, key: str, now: float) -> None:
        """Removes timestamps older than 61 seconds for the given key."""
        while self.rpm_tracking[key] and now - self.rpm_tracking[key][0] >= 61.0:
            self.rpm_tracking[key].popleft()

    def _utilize_key(self, key: str, now: float) -> str:
        """Records usage of a key and checks if it just hit the daily limit."""
        self.rpm_tracking[key].append(now)
        self.rpd_tracking[key] += 1
        
        if self.rpd_tracking[key] >= self.rpd_limit:
            self.exhausted_keys.add(key)
            logger.warning("Key starting with '%s...' has exhausted its daily limit.", key[:8])
            
        return key

    def get_stats(self) -> dict:
        """Return the current metrics of the Key Manager."""
        with self.lock:
            return {
                "total_keys": len(self.keys),
                "active_keys": len(self.keys) - len(self.exhausted_keys),
                "exhausted_keys": len(self.exhausted_keys)
            }
