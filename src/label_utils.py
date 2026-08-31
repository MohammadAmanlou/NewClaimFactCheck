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
    
    # Clean up JSON-like array strings and typical punctuation (e.g. "['false']" -> "false")
    clean_s = re.sub(r"[\"\'\[\]]+", "", s).strip()

    # Preserve Half-True for three/four-class datasets.
    if clean_s in {
        "half-true",
        "half true",
        "halftrue",
    }:
        return "Half-True"

    if clean_s in {
        "not enough information",
        "not enough evidence",
        "insufficient information",
        "insufficient evidence",
        "not enough experts",
        "unverifiable",
        "unproven",
        "neoveriteľné",
        "other",
        "unverifiable"
    }:
        return "Not Enough Information"

    mapping = {
        # --- SUPPORTED ---
        "supported": "Supported",
        "support": "Supported",
        "true": "Supported",
        "pravda": "Supported",
        "prawda": "Supported",
        "verdadeiro": "Supported",
        "doğru": "Supported",
        "correct": "Supported",
        "mostly true": "Supported",
        "mostly-true": "Supported",
        "correct-attribution": "Supported",
        "istina": "Supported",

        # --- REFUTED ---
        "refuted": "Refuted",
        "refute": "Refuted",
        "false": "Refuted",
        "falso": "Refuted",
        "faux": "Refuted",
        "fals": "Refuted",
        "falsch": "Refuted",
        "fałsz": "Refuted",
        "yanlış": "Refuted",
        "yanlis": "Refuted",
        "errado": "Refuted",
        "nepravda": "Refuted",
        "neistina": "Refuted",
        "fake": "Refuted",
        "disinformation": "Refuted",
        "untrue": "Refuted",
        "incorrect": "Refuted",
        "نادرست": "Refuted",
        "خطأ": "Refuted",
        "زائف": "Refuted",
        "زائف, fake": "Refuted",
        "false, falso": "Refuted",
        "錯誤": "Refuted",
        "ψευδές": "Refuted",
        "rrenë": "Refuted",
        "pants on fire": "Refuted",
        "fourpinocchios": "Refuted",
        "pants-on-fire": "Refuted",
        "mostly false": "Refuted",
        "mostly-false": "Refuted",
        "notizia falsa": "Refuted",
        "salah": "Refuted",
        "salah -": "Refuted",
        "salah false context": "Refuted",
        "salah fabricated content": "Refuted",
        "me kos": "Refuted",
        "pimenta na língua": "Refuted",

        # --- MISLEADING ---
        "misleading": "Misleading",
        "partlytrue": "Misleading",
        "partiallytrue": "Misleading",
        "partially true": "Misleading",
        "halftrue": "Misleading",
        "half-true": "Misleading",
        "half true": "Misleading",
        "cherrypicking": "Misleading",
        "cherrypicked": "Misleading",
        "conflictingevidence": "Misleading",
        "engañoso": "Misleading",
        "enganoso": "Misleading",
        "impreciso": "Misleading",
        "zavádzajúce": "Misleading",
        "keliru": "Misleading",
        "مضلل": "Misleading",
        "misleading, مضلل": "Misleading",
        "mixture": "Misleading",
        "mixed": "Misleading",
        "altered": "Misleading",
        "miscaptioned": "Misleading",
        "missing context": "Misleading",
        "needs context": "Misleading",
        "needscontext": "Misleading",
        "fuori contesto": "Misleading",
        "fuoricontesto": "Misleading",
        "verdadeiro, mas": "Misleading",
        "labeled-satire": "Misleading",
        "salah misleading content": "Misleading",
        "salah manipulated content": "Misleading",
        "sesat": "Misleading",
        "部分錯誤": "Misleading",
        "trunchiat": "Misleading",
        "conflictingevidence": "Misleading",
    }

    if clean_s in mapping:
        return mapping[clean_s]

    # Fallback to compact (letters only) lookup for older normalized formats
    compact = re.sub(r"[^a-z]+", "", clean_s)
    if compact in mapping:
        return mapping[compact]
        
    return None
