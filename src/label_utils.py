"""
Label normalisation according to a standard fact‑checking taxonomy.

Maps dozens of raw label strings into one of the four canonical labels:
  Supported, Refuted, Not Enough Information, Misleading.
"""

import re
from typing import Optional

CANONICAL_LABELS = [
    "Supported",
    "Refuted",
    "Not Enough Information",
    "Misleading",
]

def normalize_label(label: Optional[str]) -> Optional[str]:
    if label is None:
        return None

    s = str(label).strip().lower()
    compact = re.sub(r"[^a-z]+", "", s)

    if s in {
        "not enough information",
        "not enough evidence",
        "insufficient information",
        "insufficient evidence",
    }:
        return "Not Enough Information"

    mapping = {
        "supported": "Supported",
        "support": "Supported",
        "true": "Supported",

        "refuted": "Refuted",
        "refute": "Refuted",
        "false": "Refuted",

        "misleading": "Misleading",
        "partlytrue": "Misleading",
        "partiallytrue": "Misleading",
        "halftrue": "Misleading",
        "cherrypicking": "Misleading",
        "cherrypicked": "Misleading",
        "conflictingevidence": "Misleading",

        "notenoughinformation": "Not Enough Information",
        "notenoughevidence": "Not Enough Information",
        "insufficientinformation": "Not Enough Information",
        "insufficientevidence": "Not Enough Information",
        "nei": "Not Enough Information",
        "unknown": "Not Enough Information",
        "unverifiable": "Not Enough Information",
    }

    return mapping.get(compact)