import logging
from typing import Optional, Dict, Any

from .config import Config
from .api_client import construct_payload, extract_prediction_from_response, send_api_request
from .prompts import BUILDERS
from .multi_key_manager import MultiKeyManager

class DistributedApiClient:
    """
    An API client wrapper that handles key rotation and rate limits via a MultiKeyManager.
    Designed for clean dependency injection and multithreaded environments.
    """

    def __init__(self, config: Config, key_manager: MultiKeyManager, logger: logging.Logger):
        self.config = config
        self.key_manager = key_manager
        self.logger = logger

    def evaluate_claim(self, claim_data: Dict[str, Any], prompt_method: str) -> Optional[str]:
        """
        Generates a prompt for a claim and evaluates it against the LLM securely.
        """
        prompt = self._build_prompt(claim_data, prompt_method)
        if not prompt:
            return None

        payload = construct_payload(self.config.model, prompt)
        headers = self._get_auth_headers()
        
        response = send_api_request(
            url=self.config.api_url,
            headers=headers,
            payload=payload,
            max_retries=getattr(self.config, 'max_retries', 5),
            initial_delay=getattr(self.config, 'initial_retry_delay', 0.5),
            logger=self.logger,
        )

        if not response:
            self.logger.warning("Failed response for claim_id: %s", claim_data.get("claim_id"))
            return None

        return extract_prediction_from_response(response, self.logger)

    def _build_prompt(self, claim_data: Dict[str, Any], prompt_method: str) -> Optional[str]:
        """Constructs the prompt dynamically from registered builders."""
        builder = BUILDERS.get(prompt_method)
        if not builder:
            self.logger.error("Invalid prompt method: %s", prompt_method)
            return None

        kwargs = {
            "claim": claim_data.get("claim", ""),
            "date": claim_data.get("claim_date", ""),
            "labels": self.config.labels,
        }
        kwargs.update(claim_data)
        
        return builder(**kwargs)

    def _get_auth_headers(self) -> Dict[str, str]:
        """Fetches a safely rate-limited API key and constructs authorization headers."""
        api_key = self.key_manager.get_available_key()
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
