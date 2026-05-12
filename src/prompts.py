"""
Prompt templates for different fact-checking strategies.
Each builder takes (claim, date, labels) and returns a formatted prompt string.
"""

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

BIAS_DEFINITIONS = {
    'Ambiguity effect': 'The tendency to avoid options for which missing information makes the probability seem "unknown".',
    'Attentional bias': 'The tendency of our perception to be affected by our recurring thoughts.',
    'Bandwagon effect': 'The tendency to do (or believe) things because many other people do (or believe) the same.',
    'Belief bias': 'An effect where someone\'s evaluation of the logical strength of an argument is biased by the believability of the conclusion.',
    'Anchoring effect': 'The tendency to rely too heavily, or "anchor", on one trait or piece of information when making decisions.',
    'Bias blind spot': 'The tendency to see oneself as less biased than other people.',
    'Cheerleader effect': 'The tendency for people to appear more attractive in a group than in isolation.',
    'Choice-supportive bias': 'The tendency to remember one\'s choices as better than they actually were.',
    'clustering illusion': 'The tendency to overestimate the importance of small runs, streaks, or clusters in large samples of random data.',
    'comfort zone effect': 'The tendency to perform activities within a familiar boundary, avoiding risks.',
    'confirmation bias': 'The tendency to search for, interpret, focus on and remember information in a way that confirms one\'s preconceptions.',
    'contrast effect': 'The enhancement or reduction of a certain stimulus\' perception when compared with a recently observed, contrasting object.',
    'curse of knowledge': 'When better-informed people find it extremely difficult to think about problems from the perspective of lesser-informed people.',
    'decoy effect': 'Preferences for either option A or B change in favor of option B when option C is presented.',
    'Distinction bias': 'The tendency to view two options as more dissimilar when evaluating them simultaneously than separately.',
    'Duration neglect': 'The psychological principle that the length of an experience has little effect on the memory of that event.',
    'empathy gap': 'The tendency to underestimate the influence or strength of feelings, in either oneself or others.',
    'Endowment effect': 'The fact that people often demand much more to give up an object than they would be willing to pay to acquire it.',
    'framing effect': 'Drawing different conclusions from the same information, depending on how that information is presented.',
    'Frequency Illusion': 'The illusion in which a word, a name, or other thing that has recently come to one\'s attention suddenly seems to appear with improbable frequency.',
    'hard-easy effect': 'Based on a specific level of task difficulty, the confidence in judgments is too conservative and not conservative enough.',
    'hindsight bias': 'The tendency to see past events as being predictable at the time those events happened.',
    'current moment bias': 'The tendency to prefer payoffs that are closer to the present time over those that are further in the future.',
    'Identifiable Victim Effect': 'The tendency to respond more strongly to a single identified person at risk than to a large group.',
    'IKEA effect': 'The tendency for people to place a disproportionately high value on objects that they partially assembled themselves.',
    'illusion of control': 'The tendency to overestimate one\'s degree of influence over other external events.',
    'Illusion of Validity': 'Belief that our judgments are accurate, especially when available information is consistent or inter-correlated.',
    'illusory correlation': 'Inaccurately perceiving a relationship between two unrelated events.',
    'impact bias': 'The tendency to overestimate the length or the intensity of the impact of future feeling states.',
    'information bias': 'The tendency to seek information even when it cannot affect action.',
    'Jumping to Conclusions': 'Drawing a conclusion without taking the needed time to reason through the evidence.',
    'Just-world hypothesis': 'The tendency for people to believe that the world is just and therefore people "get what they deserve".',
    'Less-is-better effect': 'The tendency to prefer a smaller set to a larger set judged separately, but not jointly.',
    'loss aversion': 'The disutility of giving up an object is greater than the utility associated with acquiring it.',
    'Mere exposure effect': 'The tendency to express undue liking for things merely because of familiarity with them.',
    'Positivity and Negativity Effect': 'The tendency to recall positive/negative information more explicitly depending on age or disposition.',
    'Negativity bias': 'Psychological phenomenon by which humans have a greater recall of unpleasant memories compared with positive memories.',
    'Neglect of probability': 'The tendency to completely disregard probability when making a decision under uncertainty.',
    'Normalcy bias': 'The refusal to plan for, or react to, a disaster which has never happened before.',
    'Omission bias': 'The tendency to judge harmful actions as worse, or less moral, than equally harmful omissions (inactions).',
    'Optimism bias': 'The tendency to be over-optimistic, overestimating favorable and pleasing outcomes.',
    'Ostrich Effect': 'Ignoring an obvious negative situation.',
    'Outcome bias': 'The tendency to judge a decision by its eventual outcome instead of based on the quality of the decision at the time it was made.',
    'Overconfidence effect': 'Excessive confidence in one\'s own answers to questions.',
    'Pessimism bias': 'The tendency to overestimate the likelihood of negative things happening to them.',
    'Planning Fallacy': 'The tendency to underestimate task-completion times.',
    'Positive Outcome Bias': 'The tendency to overestimate the probability of good things happening.',
    'Pro-Innovation Bias': 'The tendency to have an excessive optimism towards an invention or innovation\'s usefulness.',
    'Pseudocertainty effect': 'The tendency to make risk-averse choices if the expected outcome is positive, but risk-seeking choices to avoid negative outcomes.',
    'Reactance': 'The urge to do the opposite of what someone wants you to do out of a need to resist a perceived attempt to constrain your freedom.',
    'Reactive Devaluation': 'Devaluing proposals only because they purportedly originated with an adversary.',
    'recency illusion': 'The belief that things that have just recently come to one\'s attention have in fact only recently originated.',
    'Risk Compensation': 'The tendency to take greater risks when perceived safety increases.',
    'Selective Attention': 'The tendency to direct attention to specific stimuli while ignoring others.',
    'Social Comparison Bias': 'The tendency to favor potential candidates who don\'t compete with one\'s own particular strengths.',
    'Stereotyping': 'Expecting a member of a group to have certain characteristics without having actual information about that individual.',
    'subadditivity effect': 'The tendency to judge probability of the whole to be less than the probabilities of the parts.',
    'Subjective validation': 'Perception that something is true if a subject\'s belief demands it to be true.',
    'Survivorship bias': 'Concentrating on things that "survived" some process and inadvertently overlooking those that didn\'t.',
    'Time-saving bias': 'Underestimations of the time that could be saved when increasing from a relatively low speed.',
    'Unit Bias': 'The tendency to want to finish a given unit of a task or an item.',
    'Whole only effect': 'A preference to evaluate or consider a whole rather than a part.',
    'Zero-risk bias': 'Preference for reducing a small risk to zero over a greater reduction in a larger risk.',
    'Default effect': 'When given a choice between several options, the tendency to favor the default one.',
    'Exaggerated expectation bias': 'The tendency to expect or predict more extreme outcomes than those that actually happen.',
    'Forer effect': 'Individuals give high accuracy ratings to descriptions of their personality that supposedly are tailored specifically for them.',
    'sunk cost fallacy': 'The phenomenon whereby a person is reluctant to abandon a strategy or course of action because they have invested heavily in it.',
    'Essentialism': 'Categorizing people and things according to their essential nature, in spite of variations.',
    'Post-purchase rationalization': 'The tendency to persuade oneself through rational argument that a purchase was a good value.',
    'Semmelweis reflex': 'The tendency to reject new evidence that contradicts a paradigm.',
    'Availability heuristic': 'A mental shortcut that relies on immediate examples that come to a mind when evaluating a specific topic.',
    'Backfire effect.': 'The reaction to disconfirming evidence by strengthening one\'s previous beliefs.'
}


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
    
    # Generate definitions only for biases present in detected_biases
    bias_definitions_str = ""
    if detected_biases:
        detected_biases_lower = detected_biases.lower()
        found_biases = []
        for bias_name, bias_def in BIAS_DEFINITIONS.items():
            if bias_name.lower().replace('.', '') in detected_biases_lower.replace('.', ''):
                found_biases.append(f"- {bias_name}: {bias_def}")
                
        if found_biases:
            bias_definitions_str = "\nDefinitions of detected cognitive biases:\n" + "\n".join(found_biases) + "\n"

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