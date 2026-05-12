"""
Prompt templates for different fact-checking strategies.
Each builder takes (claim, date, labels) and returns a formatted prompt string.
"""

from .bias_registry import BiasRegistry

def build_naive(claim: str, date: str, labels: list[str], **kwargs) -> str:
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

def build_prompt_strong_baseline_short(claim: str, **kwargs) -> str:
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

Return only JSON:
{{"label": "...", "brief_reason": "one short sentence"}}

Claim:
{claim}
""".strip()


def build_prompt_cognitive_bias_aware(
    claim: str,
    detected_biases: str = "",
    **kwargs
) -> str:
    
    # Delegate logical generation of dictionaries to the new dataclass registry
    bias_definitions_str = BiasRegistry.format_definitions(detected_biases)

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

Detected bias signals:
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

Return only valid JSON:
{{
  "label": "...",
  "brief_reason": "one short sentence"
}}

Claim:
{claim}
""".strip()

# Mapping from method strings to builder functions
BUILDERS = {
    "naive": build_naive,
    "strong_baseline_short": build_prompt_strong_baseline_short,
    "cognitive_bias_aware": build_prompt_cognitive_bias_aware,
}