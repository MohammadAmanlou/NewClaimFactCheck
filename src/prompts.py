"""
Prompt templates for different fact-checking strategies.
Each builder takes (claim, date, labels) and returns a formatted prompt string.
"""

def build_naive(claim: str, date: str, labels: list[str]) -> str:
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

def build_definition_rich(claim: str, date: str, labels: list[str]) -> str:
    return f"""
You are a fact-checking classifier.

Your task is to assign the most appropriate label to the claim.

Choose exactly one label:
- Supported
- Refuted
- Misleading
- Not Enough Information

Definitions:
- Supported: The main factual claim is accurate.
- Refuted: The main factual claim is false or contradicted by reliable knowledge.
- Misleading: The claim contains some truth but distorts context, exaggerates, omits important information, or creates a false impression.
- Not Enough Information: The claim is too vague, underspecified, or cannot be judged even after considering general knowledge and context.

Important rule:
Do NOT choose Not Enough Information only because no external evidence is provided.
Choose Not Enough Information only when the claim itself is not specific enough or cannot reasonably be judged.

Return only JSON:
{{"label": "...", "brief_reason": "one short sentence"}}

Claim:
{claim}
Date: {date}
""".strip()

def build_anti_nei(claim: str, date: str, labels: list[str]) -> str:
    return f"""
You are a strict fact-checking classifier.

Choose exactly one label:
Supported, Refuted, Misleading, Not Enough Information.

Before choosing Not Enough Information, consider whether the claim is better classified as:
- Refuted: the central factual assertion is false.
- Misleading: the claim is partially true but missing context or distorting the facts.
- Supported: the central factual assertion is accurate.

Use Not Enough Information only as a last resort, when the claim is too vague or impossible to assess.

Return only JSON:
{{"label": "...", "brief_reason": "one short sentence"}}

Claim:
{claim}
Date: {date}
""".strip()

def build_cvp(claim: str, date: str, labels: list[str]) -> str:
    return f"""
You are a careful fact-checking assistant.

Apply this cognitive verification protocol before deciding:

1. Identify the central factual assertion.
2. Check whether the claim is time-sensitive.
3. Identify the main entities and relations.
4. Decide whether the claim is mainly true, false, partially true but misleading, or too underspecified.
5. Before selecting Not Enough Information, explicitly consider whether Refuted or Misleading is more appropriate.
6. Choose exactly one final label.

Labels:
- Supported: The main factual claim is accurate.
- Refuted: The central factual claim is false.
- Misleading: The claim contains some truth but distorts context, exaggerates, omits important information, or creates a false impression.
- Not Enough Information: The claim is too vague, underspecified, or cannot be judged.

Return only JSON:
{{
  "label": "...",
  "risk_type": "temporal/entity/label-boundary/evidence/none",
  "brief_reason": "one short sentence"
}}

Claim:
{claim}
Date: {date}
""".strip()

# Mapping from method strings to builder functions
BUILDERS = {
    "naive": build_naive,
    "definition_rich": build_definition_rich,
    "anti_nei": build_anti_nei,
    "cvp": build_cvp,
}