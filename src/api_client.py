import re
import json
import time
import logging
import requests
from typing import List, Optional
from .config import Config
from .label_utils import normalize_label

def ask_model(claim: str, date: str, config: Config, logger: logging.Logger) -> Optional[str]:
    """
    Query the language model for fact-checking prediction with retry logic.
    
    Args:
        claim: Claim text to fact-check
        date: Claim date
        model_name: Name of the model to use
        api_key: API authentication key
        api_url: API endpoint URL
        labels: List of valid labels
        logger: Logger instance
        max_retries: Maximum number of retry attempts
        
    Returns:
        Predicted label or None if failed
    """
    
    labels = config.labels
    prompt = f"""You are a fact-checking expert. Analyze the following claim and classify it into one of these categories: {', '.join(labels)}.

Claim: {claim}
Date: {date}

Respond ONLY with a JSON object in this exact format:
{{"label": "your_classification"}}

Choose the label that best represents the claim's veracity."""

    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": config.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 50
    }

    logger.info(f"Asking model: {config.model}")
    logger.info(f"Claim: {claim[:150]}...")

    retry_delay = config.initial_retry_delay
    for attempt in range(config.max_retries + 1):
        try:
            response = requests.post(config.api_url, json=payload, headers=headers, timeout=30)
            logger.info(f"API Response Status: {response.status_code}")

            if response.status_code == 429:
                if attempt < config.max_retries:
                    logger.info(f"Retry attempt {attempt + 1}/{config.max_retries} after {retry_delay}s delay...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                else:
                    logger.error("Max retries reached for rate limit")
                    return None

            if response.status_code == 200:
                result = response.json()
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                logger.info(f"Raw model response: {content}")

                json_match = re.search(r'\{.*?\}', content, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group(0))
                    raw_label = parsed.get("label", "")
                    normalized = normalize_label(raw_label, config.labels)
                    logger.info(f"Extracted label: {raw_label} -> Normalized: {normalized}")
                    return normalized

                logger.warning(f"Could not parse JSON from response: {content}")
                return None

            logger.error(f"API error: {response.status_code} - {response.text}")
            return None

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            if attempt < config.max_retries:
                logger.warning(f"Network error: {e}. Retrying...")
                time.sleep(retry_delay)
                retry_delay *= 2
                continue
            logger.error(f"Network error after {config.max_retries} retries: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return None

    return None