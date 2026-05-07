"""
API client for querying the fact‑checking model via Avalai (OpenAI‑compatible).
"""

import re
import json
import time
import logging
from typing import Optional, Dict, Any
import requests

from .config import Config
from .label_utils import normalize_label  


def build_prompt(claim: str, date: str, labels: list[str]) -> str:
    """Create the prompt that instructs the model to classify the claim."""
    labels_str = ", ".join(labels)
    return (
        f"You are a fact-checking expert. Analyze the following claim and classify "
        f"it into one of these categories: {labels_str}.\n\n"
        f"Claim: {claim}\n"
        f"Date: {date}\n\n"
        "Respond ONLY with a JSON object in this exact format:\n"
        '{"label": "your_classification"}\n\n'
        "Choose the label that best represents the claim's veracity."
    )


def construct_payload(
    model_name: str,
    prompt: str,
    temperature: float = 0.0,       
    top_p: float = 1.0,             
    max_tokens: int = 50,
) -> Dict[str, Any]:
    return {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }


def send_api_request(
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    max_retries: int,
    initial_delay: float,
    logger: logging.Logger,
) -> Optional[requests.Response]:
    """
    Send the request with exponential backoff on rate limiting and network errors.

    Returns the successful Response object or None after exhaustion.
    """
    retry_delay = initial_delay
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            logger.info("API Response Status: %d", response.status_code)

            if response.status_code == 429:
                if attempt < max_retries:
                    logger.info(
                        "Rate limited. Retry %d/%d after %.1fs",
                        attempt + 1, max_retries, retry_delay,
                    )
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                logger.error("Max retries reached for rate limiting")
                return None

            if response.status_code == 200:
                return response

            logger.error("API error %d: %s", response.status_code, response.text)
            return None

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            if attempt < max_retries:
                logger.warning("Network error: %s. Retrying...", e)
                time.sleep(retry_delay)
                retry_delay *= 2
                continue
            logger.error("Network error after %d retries: %s", max_retries, e)
            return None
        except Exception as e:
            logger.error("Unexpected error: %s", e)
            return None

    return None


def extract_prediction_from_response(
    response: requests.Response,
    logger: logging.Logger,
) -> Optional[str]:
    """Parse the model's JSON response and return the normalized label."""
    try:
        result = response.json()
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        logger.info("Raw model response: %s", content)
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        logger.error("Failed to decode JSON response: %s", e)
        return None

    json_match = re.search(r'\{.*?\}', content, re.DOTALL)
    if not json_match:
        logger.warning("Could not find JSON in response: %s", content)
        return None

    try:
        parsed = json.loads(json_match.group(0))
    except json.JSONDecodeError:
        logger.warning("Invalid JSON extracted: %s", json_match.group(0))
        return None

    raw_label = parsed.get("label", "")
    normalized = normalize_label(raw_label)
    logger.info("Extracted label: %s -> Normalized: %s", raw_label, normalized)
    return normalized


def ask_model(
    claim: str,
    date: str,
    config: Config,
    logger: logging.Logger,
) -> Optional[str]:
    """
    Query the fact-checking model for a prediction.

    Args:
        claim: Claim text to fact-check.
        date: Claim date.
        config: Configuration object containing model, API key, etc.
        logger: Logger for this invocation.

    Returns:
        Normalized label string or None if the query failed.
    """
    logger.info("Asking model: %s", config.model)
    logger.info("Claim: %.150s...", claim)

    prompt = build_prompt(claim, date, config.labels)
    payload = construct_payload(config.model, prompt, temperature=0.0, top_p=1.0,)

    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
    }

    response = send_api_request(
        url=config.api_url,
        headers=headers,
        payload=payload,
        max_retries=config.max_retries,
        initial_delay=config.initial_retry_delay,
        logger=logger,
    )

    if response is None:
        return None

    return extract_prediction_from_response(response, logger)