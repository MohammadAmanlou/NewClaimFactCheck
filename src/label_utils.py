from typing import List, Optional

def normalize_label(label: str, canonical_labels: List[str]) -> Optional[str]:
    """
    Normalize a label to match canonical format.
    
    Args:
        label: Input label to normalize
        canonical_labels: List of valid canonical labels
        
    Returns:
        Normalized label or None if unverifiable/invalid
    """
    
    if not label or not isinstance(label, str):
        return None

    label_lower = label.lower().strip()

    # Direct match
    for canonical in canonical_labels:
        if label_lower == canonical.lower():
            return canonical

    # Partial match
    for canonical in canonical_labels:
        if canonical.lower() in label_lower or label_lower in canonical.lower():
            return canonical

    # Unverifiable keywords
    unverifiable_keywords = ["not enough", "unverifiable", "inconclusive", "unclear"]
    if any(keyword in label_lower for keyword in unverifiable_keywords):
        return None

    # Common mappings (example)
    mappings = {
        "true": "Supported",
        "false": "Refuted",
        "mostly true": "Supported",
        "mostly false": "Refuted",
        "half true": "Partially true",
        "mixture": "Misleading",
        "outdated": "Misleading",
        "cherry picking": "Misleading",
        "correct attribution": "Supported",
        "incorrect attribution": "Refuted",
    }
    for key, value in mappings.items():
        if key in label_lower and value in canonical_labels:
            return value

    return None