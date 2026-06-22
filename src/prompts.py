"""
Prompt templates for different fact-checking strategies.
Each builder takes (claim, date, labels) and returns a formatted prompt string.
"""

from .bias_registry import BiasRegistry


def _is_binary_supported_refuted(labels: list[str] | None) -> bool:
    return set(labels or []) == {"Supported", "Refuted"}


def _is_tracer_three_class(labels: list[str] | None) -> bool:
    return set(labels or []) == {"True", "Half-True", "False"}


def _is_supported_half_refuted(labels: list[str] | None) -> bool:
    return set(labels or []) == {"Supported", "Half-True", "Refuted"}


def _format_date(date: str | None) -> str:
    if date:
        return f"\nDate: {date}\n"
    return ""


def build_naive(
    claim: str,
    date: str = "",
    labels: list[str] | None = None,
    **kwargs
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
Return only JSON:
{{"label": "..."}}
""".strip()


def build_prompt_strong_baseline_short(
    claim: str,
    date: str = "",
    labels: list[str] | None = None,
    **kwargs
) -> str:
    date_str = _format_date(date)

    if _is_binary_supported_refuted(labels):
        return f"""
You are a scientific claim verification classifier.

Choose exactly one label:
Supported, Refuted.

Definitions:
- Supported: The claim is scientifically feasible or accurate.
- Refuted: The claim is scientifically infeasible, false, or contradicted by known scientific constraints.

Rules:
- Focus on the central scientific assertion.
- Check materials, mechanisms, experimental conditions, numbers, units, comparisons, and causal claims.
- If the claim describes a feasible or accurate scientific result, choose Supported.
- If the claim describes an infeasible or false scientific result, choose Refuted.
- Always choose the single best label.

Claim:
{claim}
{date_str}
Return only JSON:
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
- Refuted: The central factual claim is inaccurate, unsupported, or contradicted by the available facts.

Rules:
- Do not choose Half-True only because the claim is complex.
- Use Half-True when the claim is partly correct but missing important context or overstated.
- If the central factual assertion is false, choose Refuted.
- If the central factual assertion is accurate, choose Supported.
- Always choose the single best label.

Claim:
{claim}
{date_str}
Return only JSON:
{{"label": "...", "brief_reason": "one short sentence"}}
""".strip()

    if _is_tracer_three_class(labels):
        return f"""
You are a strict fact-checking classifier.

Choose exactly one label:
True, Half-True, False.

Definitions:
- True: The central factual claim is accurate or substantially supported.
- Half-True: The claim contains some truth but leaves out important context, exaggerates, simplifies, or creates a partly misleading impression.
- False: The central factual claim is inaccurate, unsupported, or contradicted by the available facts.

Rules:
- Do not choose Half-True only because the claim is complex.
- Use Half-True when the claim is partly correct but missing important context or overstated.
- If the central factual assertion is false, choose False.
- If the central factual assertion is accurate, choose True.
- Always choose the single best label.

Claim:
{claim}
{date_str}
Return only JSON:
{{"label": "...", "brief_reason": "one short sentence"}}
""".strip()

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

Claim:
{claim}
{date_str}
Return only JSON:
{{"label": "...", "brief_reason": "one short sentence"}}
""".strip()


def build_prompt_cognitive_bias_aware(
    claim: str,
    date: str = "",
    labels: list[str] | None = None,
    detected_biases: str = "",
    **kwargs
) -> str:
    date_str = _format_date(date)

    bias_definitions_str = ""
    if detected_biases:
        try:
            bias_definitions_str = BiasRegistry.format_definitions(detected_biases)
        except Exception:
            bias_definitions_str = ""

    if _is_binary_supported_refuted(labels):
        return f"""
You are a careful scientific claim verification classifier.

Choose exactly one label:
Supported, Refuted.

Definitions:
- Supported: The claim is scientifically feasible or accurate.
- Refuted: The claim is scientifically infeasible, false, or contradicted by known scientific constraints.

Detected bias signals with confidence scores:
{detected_biases if detected_biases else "No detected bias signals were provided."}
{bias_definitions_str}

The detected bias signals above are externally identified bias indicators with confidence scores.
Use them as part of your verification process. A higher confidence score means that the corresponding bias is more likely to be present in the claim.
However, the final label must still be selected according to the factual meaning of the claim and the label definitions.

Apply this cognitive verification protocol internally:
1. Identify the central scientific assertion.
2. Check whether the claim is time-sensitive.
3. Check materials, entities, mechanisms, relations, dates, numbers, units, comparisons, quantifiers, and negation.
4. Use the provided bias signals to inspect risks such as missing context, cherry-picking, exaggeration, causal overclaiming, temporal mismatch, misleading comparison, or unsupported generalization.
5. Decide whether the claim is more likely Supported or Refuted.

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
- Refuted: The central factual claim is inaccurate, unsupported, or contradicted by the available facts.

Rules:
- Do not choose Half-True only because the claim is complex.
- Use Half-True when the claim is partly correct but contextually distorted, missing important context, exaggerated, or overstated.
- If the central factual assertion is false, choose Refuted.
- If the central factual assertion is accurate, choose Supported.
- Always choose the single best label.

Detected bias signals with confidence scores:
{detected_biases if detected_biases else "No detected bias signals were provided."}
{bias_definitions_str}

The detected bias signals above are externally identified bias indicators with confidence scores.
Use them as part of your verification process. A higher confidence score means that the corresponding bias is more likely to be present in the claim.
However, the final label must still be selected according to the factual meaning of the claim and the label definitions.

Apply this cognitive verification protocol internally:
1. Identify the central factual assertion.
2. Check whether the claim is time-sensitive.
3. Check entities, relations, dates, numbers, quantifiers, and negation.
4. Use the provided bias signals to inspect possible distortions such as missing context, cherry-picking, exaggeration, emotional framing, causal overclaiming, temporal mismatch, misleading comparison, or unsupported generalization.
5. Before choosing Half-True, consider whether Supported or Refuted is more appropriate.
6. Choose the single best final label.

Claim:
{claim}
{date_str}
Return only valid JSON:
{{"label": "...", "brief_reason": "one short sentence"}}
""".strip()

    if _is_tracer_three_class(labels):
        return f"""
You are a careful fact-checking classifier.

Choose exactly one label:
True, Half-True, False.

Definitions:
- True: The central factual claim is accurate or substantially supported.
- Half-True: The claim contains some truth but leaves out important context, exaggerates, simplifies, or creates a partly misleading impression.
- False: The central factual claim is inaccurate, unsupported, or contradicted by the available facts.

Rules:
- Do not choose Half-True only because the claim is complex.
- Use Half-True when the claim is partly correct but contextually distorted, missing important context, exaggerated, or overstated.
- If the central factual assertion is false, choose False.
- If the central factual assertion is accurate, choose True.
- Always choose the single best label.

Detected bias signals with confidence scores:
{detected_biases if detected_biases else "No detected bias signals were provided."}
{bias_definitions_str}

The detected bias signals above are externally identified bias indicators with confidence scores.
Use them as part of your verification process. A higher confidence score means that the corresponding bias is more likely to be present in the claim.
However, the final label must still be selected according to the factual meaning of the claim and the label definitions.

Apply this cognitive verification protocol internally:
1. Identify the central factual assertion.
2. Check whether the claim is time-sensitive.
3. Check entities, relations, dates, numbers, quantifiers, and negation.
4. Use the provided bias signals to inspect possible distortions such as missing context, cherry-picking, exaggeration, emotional framing, causal overclaiming, temporal mismatch, misleading comparison, or unsupported generalization.
5. Before choosing Half-True, consider whether True or False is more appropriate.
6. Choose the single best final label.

Claim:
{claim}
{date_str}
Return only valid JSON:
{{"label": "...", "brief_reason": "one short sentence"}}
""".strip()

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

Detected bias signals with confidence scores:
{detected_biases if detected_biases else "No detected bias signals were provided."}
{bias_definitions_str}

The detected bias signals above are externally identified bias indicators with confidence scores.
Use them as part of your verification process. A higher confidence score means that the corresponding bias is more likely to be present in the claim.
However, the final label must still be selected according to the factual meaning of the claim and the label definitions.

Apply this cognitive verification protocol internally:
1. Identify the central factual assertion.
2. Check whether the claim is time-sensitive.
3. Check entities, relations, dates, numbers, quantifiers, and negation.
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
