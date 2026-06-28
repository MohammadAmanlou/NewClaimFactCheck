"""
Prompt templates for different fact-checking strategies.
Each builder takes (claim, date, labels) and returns a formatted prompt string.
"""

from .bias_registry import BiasRegistry


# -------------------------
# Label-set helpers
# -------------------------

def _label_set(labels: list[str] | None) -> set[str]:
    return set(labels or [])


def _is_binary_supported_refuted(labels: list[str] | None) -> bool:
    return _label_set(labels) == {"Supported", "Refuted"}


def _is_true_half_false(labels: list[str] | None) -> bool:
    return _label_set(labels) == {"True", "Half-True", "False"}


def _is_supported_half_refuted(labels: list[str] | None) -> bool:
    return _label_set(labels) == {"Supported", "Half-True", "Refuted"}


def _is_fact5_four_class(labels: list[str] | None) -> bool:
    return _label_set(labels) == {
        "Supported",
        "Half-True",
        "Refuted",
        "Not Enough Information",
    }


def _is_generic_four_class(labels: list[str] | None) -> bool:
    return _label_set(labels) == {
        "Supported",
        "Refuted",
        "Misleading",
        "Not Enough Information",
    }


def _format_date(date: str | None) -> str:
    if date:
        return f"\nDate: {date}\n"
    return ""


def _safe_bias_definitions(detected_biases: str = "") -> str:
    """
    Format bias definitions if possible.
    If a bias label is unknown or parsing fails, avoid crashing the whole pipeline.
    """
    if not detected_biases:
        return ""

    try:
        return BiasRegistry.format_definitions(detected_biases)
    except Exception:
        return ""


# -------------------------
# Naive prompt
# -------------------------

def build_naive(
    claim: str,
    date: str = "",
    labels: list[str] | None = None,
    **kwargs,
) -> str:
    labels = labels or ["Supported", "Refuted", "Misleading", "Not Enough Information"]
    labels_str = ", ".join(labels)
    date_str = _format_date(date)

    return f"""
You are a fact-checking classifier.

Choose exactly one label:
{labels_str}.

Claim:
{claim}
{date_str}
Return only valid JSON:
{{"label": "..."}}
""".strip()


# -------------------------
# Strong baseline prompt
# -------------------------

def build_prompt_strong_baseline_short(
    claim: str,
    date: str = "",
    labels: list[str] | None = None,
    **kwargs,
) -> str:
    date_str = _format_date(date)

    if _is_binary_supported_refuted(labels):
        return f"""
You are a strict fact-checking classifier.

Choose exactly one label:
Supported, Refuted.

Definitions:
- Supported: The central factual claim is accurate or substantially supported.
- Refuted: The central factual claim is inaccurate, unsupported, or contradicted by available facts.

Rules:
- Focus on the central factual assertion.
- Check entities, relations, dates, numbers, units, comparisons, quantifiers, causality, and negation.
- If the central factual assertion is accurate, choose Supported.
- If the central factual assertion is false or contradicted, choose Refuted.
- Always choose the single best label.

Claim:
{claim}
{date_str}
Return only valid JSON:
{{"label": "...", "brief_reason": "one short sentence"}}
""".strip()

    if _is_supported_half_refuted(labels):
        return f"""
You are a strict fact-checking classifier.

Choose exactly one label:
Supported, Half-True, Refuted.

Definitions:
- Supported: The central factual claim is accurate or substantially supported.
- Half-True: The claim contains some truth but leaves out important context, exaggerates, simplifies, or creates a partly misleading impression.
- Refuted: The central factual claim is inaccurate, unsupported, or contradicted by available facts.

Rules:
- Do not choose Half-True only because the claim is complex.
- Use Half-True when the claim is partly correct but missing important context, overstated, or contextually distorted.
- If the central factual assertion is false, choose Refuted.
- If the central factual assertion is accurate, choose Supported.
- Always choose the single best label.

Claim:
{claim}
{date_str}
Return only valid JSON:
{{"label": "...", "brief_reason": "one short sentence"}}
""".strip()

    if _is_true_half_false(labels):
        return f"""
You are a strict fact-checking classifier.

Choose exactly one label:
True, Half-True, False.

Definitions:
- True: The central factual claim is accurate or substantially supported.
- Half-True: The claim contains some truth but leaves out important context, exaggerates, simplifies, or creates a partly misleading impression.
- False: The central factual claim is inaccurate, unsupported, or contradicted by available facts.

Rules:
- Do not choose Half-True only because the claim is complex.
- Use Half-True when the claim is partly correct but missing important context, overstated, or contextually distorted.
- If the central factual assertion is false, choose False.
- If the central factual assertion is accurate, choose True.
- Always choose the single best label.

Claim:
{claim}
{date_str}
Return only valid JSON:
{{"label": "...", "brief_reason": "one short sentence"}}
""".strip()

    if _is_fact5_four_class(labels):
        return f"""
You are a strict fact-checking classifier.

Choose exactly one label:
Supported, Half-True, Refuted, Not Enough Information.

Definitions:
- Supported: The central factual claim is accurate or substantially supported.
- Half-True: The claim contains some truth but leaves out important context, exaggerates, simplifies, or creates a partly misleading impression.
- Refuted: The central factual claim is inaccurate, unsupported, or contradicted by available facts.
- Not Enough Information: The claim is too vague, unverifiable, underspecified, ambiguous, or impossible to judge.

Rules:
- Do not choose Not Enough Information only because no external evidence is provided.
- Use Not Enough Information only when the claim is genuinely unverifiable, too ambiguous, or lacks enough factual specificity to judge.
- Do not choose Half-True only because the claim is complex.
- Use Half-True when the claim is partly correct but missing important context, overstated, or contextually distorted.
- If the central factual assertion is false, choose Refuted.
- If the central factual assertion is accurate, choose Supported.
- Always choose the single best label.

Claim:
{claim}
{date_str}
Return only valid JSON:
{{"label": "...", "brief_reason": "one short sentence"}}
""".strip()

    # Generic 4-class fallback used by older datasets.
    return f"""
You are a strict fact-checking classifier.

Choose exactly one label:
Supported, Refuted, Misleading, Not Enough Information.

Definitions:
- Supported: The main factual claim is accurate.
- Refuted: The central factual claim is false.
- Misleading: The claim contains some truth but distorts context, exaggerates, cherry-picks facts, omits important information, or creates a false impression.
- Not Enough Information: The claim is too vague, underspecified, ambiguous, or impossible to judge.

Rules:
- Do not choose Not Enough Information only because no external evidence is provided.
- Use Not Enough Information only as a last resort.
- If the claim is partly true but contextually distorted, choose Misleading.
- If the central factual assertion is false, choose Refuted.
- If the central factual assertion is accurate, choose Supported.
- Always choose the single best label.

Claim:
{claim}
{date_str}
Return only valid JSON:
{{"label": "...", "brief_reason": "one short sentence"}}
""".strip()


# -------------------------
# Cognitive-bias-aware prompt
# -------------------------

def build_prompt_cognitive_bias_aware(
    claim: str,
    date: str = "",
    labels: list[str] | None = None,
    detected_biases: str = "",
    **kwargs,
) -> str:
    date_str = _format_date(date)
    bias_definitions_str = _safe_bias_definitions(detected_biases)
    bias_block = detected_biases if detected_biases else "No detected bias signals were provided."

    if _is_binary_supported_refuted(labels):
        return f"""
You are a careful fact-checking classifier.

Choose exactly one label:
Supported, Refuted.

Definitions:
- Supported: The central factual claim is accurate or substantially supported.
- Refuted: The central factual claim is inaccurate, unsupported, or contradicted by available facts.

Detected bias signals with confidence scores:
{bias_block}
{bias_definitions_str}

The detected bias signals above are externally identified bias indicators with confidence scores.
Use them as part of your verification process. A higher confidence score means that the corresponding bias is more likely to be present in the claim.
However, the final label must still be selected according to the factual meaning of the claim and the label definitions.

Apply this cognitive verification protocol internally:
1. Identify the central factual assertion.
2. Check whether the claim is time-sensitive.
3. Check entities, relations, dates, numbers, units, comparisons, quantifiers, causality, and negation.
4. Use the provided bias signals to inspect possible distortions such as missing context, cherry-picking, exaggeration, emotional framing, causal overclaiming, temporal mismatch, misleading comparison, or unsupported generalization.
5. If the central factual assertion is accurate, choose Supported.
6. If the central factual assertion is false or contradicted, choose Refuted.
7. Choose the single best final label.

Claim:
{claim}
{date_str}
Return only valid JSON:
{{"label": "...", "brief_reason": "one short sentence"}}
""".strip()

    if _is_supported_half_refuted(labels):
        return f"""
You are a careful fact-checking classifier.

Choose exactly one label:
Supported, Half-True, Refuted.

Definitions:
- Supported: The central factual claim is accurate or substantially supported.
- Half-True: The claim contains some truth but leaves out important context, exaggerates, simplifies, or creates a partly misleading impression.
- Refuted: The central factual claim is inaccurate, unsupported, or contradicted by available facts.

Rules:
- Do not choose Half-True only because the claim is complex.
- Use Half-True when the claim is partly correct but contextually distorted, missing important context, exaggerated, or overstated.
- If the central factual assertion is false, choose Refuted.
- If the central factual assertion is accurate, choose Supported.
- Always choose the single best label.

Detected bias signals with confidence scores:
{bias_block}
{bias_definitions_str}

The detected bias signals above are externally identified bias indicators with confidence scores.
Use them as part of your verification process. A higher confidence score means that the corresponding bias is more likely to be present in the claim.
However, the final label must still be selected according to the factual meaning of the claim and the label definitions.

Apply this cognitive verification protocol internally:
1. Identify the central factual assertion.
2. Check whether the claim is time-sensitive.
3. Check entities, relations, dates, numbers, quantifiers, causality, comparisons, and negation.
4. Use the provided bias signals to inspect possible distortions such as missing context, cherry-picking, exaggeration, emotional framing, causal overclaiming, temporal mismatch, misleading comparison, or unsupported generalization.
5. Before choosing Half-True, consider whether Supported or Refuted is more appropriate.
6. Choose the single best final label.

Claim:
{claim}
{date_str}
Return only valid JSON:
{{"label": "...", "brief_reason": "one short sentence"}}
""".strip()

    if _is_true_half_false(labels):
        return f"""
You are a careful fact-checking classifier.

Choose exactly one label:
True, Half-True, False.

Definitions:
- True: The central factual claim is accurate or substantially supported.
- Half-True: The claim contains some truth but leaves out important context, exaggerates, simplifies, or creates a partly misleading impression.
- False: The central factual claim is inaccurate, unsupported, or contradicted by available facts.

Rules:
- Do not choose Half-True only because the claim is complex.
- Use Half-True when the claim is partly correct but contextually distorted, missing important context, exaggerated, or overstated.
- If the central factual assertion is false, choose False.
- If the central factual assertion is accurate, choose True.
- Always choose the single best label.

Detected bias signals with confidence scores:
{bias_block}
{bias_definitions_str}

The detected bias signals above are externally identified bias indicators with confidence scores.
Use them as part of your verification process. A higher confidence score means that the corresponding bias is more likely to be present in the claim.
However, the final label must still be selected according to the factual meaning of the claim and the label definitions.

Apply this cognitive verification protocol internally:
1. Identify the central factual assertion.
2. Check whether the claim is time-sensitive.
3. Check entities, relations, dates, numbers, quantifiers, causality, comparisons, and negation.
4. Use the provided bias signals to inspect possible distortions such as missing context, cherry-picking, exaggeration, emotional framing, causal overclaiming, temporal mismatch, misleading comparison, or unsupported generalization.
5. Before choosing Half-True, consider whether True or False is more appropriate.
6. Choose the single best final label.

Claim:
{claim}
{date_str}
Return only valid JSON:
{{"label": "...", "brief_reason": "one short sentence"}}
""".strip()

    if _is_fact5_four_class(labels):
        return f"""
You are a careful fact-checking classifier.

Choose exactly one label:
Supported, Half-True, Refuted, Not Enough Information.

Definitions:
- Supported: The central factual claim is accurate or substantially supported.
- Half-True: The claim contains some truth but leaves out important context, exaggerates, simplifies, or creates a partly misleading impression.
- Refuted: The central factual claim is inaccurate, unsupported, or contradicted by available facts.
- Not Enough Information: The claim is too vague, unverifiable, underspecified, ambiguous, or impossible to judge.

Rules:
- Do not choose Not Enough Information only because no external evidence is provided.
- Use Not Enough Information only when the claim is genuinely unverifiable, too ambiguous, or lacks enough factual specificity to judge.
- Do not choose Half-True only because the claim is complex.
- Use Half-True when the claim is partly correct but contextually distorted, missing important context, exaggerated, or overstated.
- If the central factual assertion is false, choose Refuted.
- If the central factual assertion is accurate, choose Supported.
- Always choose the single best label.

Detected bias signals with confidence scores:
{bias_block}
{bias_definitions_str}

The detected bias signals above are externally identified bias indicators with confidence scores.
Use them as part of your verification process. A higher confidence score means that the corresponding bias is more likely to be present in the claim.
However, the final label must still be selected according to the factual meaning of the claim and the label definitions.

Apply this cognitive verification protocol internally:
1. Identify the central factual assertion.
2. Check whether the claim is time-sensitive.
3. Check entities, relations, dates, numbers, quantifiers, comparisons, causality, and negation.
4. Use the provided bias signals to inspect possible distortions such as missing context, cherry-picking, exaggeration, emotional framing, causal overclaiming, temporal mismatch, misleading comparison, or unsupported generalization.
5. Before choosing Half-True, consider whether Supported or Refuted is more appropriate.
6. Before choosing Not Enough Information, consider whether Refuted or Half-True is more appropriate.
7. Choose the single best final label.

Claim:
{claim}
{date_str}
Return only valid JSON:
{{"label": "...", "brief_reason": "one short sentence"}}
""".strip()

    # Generic 4-class fallback used by older datasets.
    return f"""
You are a careful fact-checking classifier.

Choose exactly one label:
Supported, Refuted, Misleading, Not Enough Information.

Definitions:
- Supported: The main factual claim is accurate.
- Refuted: The central factual claim is false.
- Misleading: The claim contains some truth but distorts context, exaggerates, cherry-picks facts, omits important information, or creates a false impression.
- Not Enough Information: The claim is too vague, underspecified, ambiguous, or impossible to judge.

Rules:
- Do not choose Not Enough Information only because no external evidence is provided.
- Use Not Enough Information only as a last resort.
- If the claim is partly true but contextually distorted, choose Misleading.
- If the central factual assertion is false, choose Refuted.
- If the central factual assertion is accurate, choose Supported.
- Always choose the single best label.

Detected bias signals with confidence scores:
{bias_block}
{bias_definitions_str}

The detected bias signals above are externally identified bias indicators with confidence scores.
Use them as part of your verification process. A higher confidence score means that the corresponding bias is more likely to be present in the claim.
However, the final label must still be selected according to the factual meaning of the claim and the label definitions.

Apply this cognitive verification protocol internally:
1. Identify the central factual assertion.
2. Check whether the claim is time-sensitive.
3. Check entities, relations, dates, numbers, quantifiers, comparisons, causality, and negation.
4. Use the provided bias signals to inspect possible distortions such as missing context, cherry-picking, exaggeration, emotional framing, causal overclaiming, temporal mismatch, misleading comparison, or unsupported generalization.
5. Before choosing Not Enough Information, consider whether Refuted or Misleading is more appropriate.
6. Choose the single best final label.

Claim:
{claim}
{date_str}
Return only valid JSON:
{{"label": "...", "brief_reason": "one short sentence"}}
""".strip()


BUILDERS = {
    "naive": build_naive,
    "strong_baseline_short": build_prompt_strong_baseline_short,
    "cognitive_bias_aware": build_prompt_cognitive_bias_aware,
}
