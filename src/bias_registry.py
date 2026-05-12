"""
Registry for cognitive biases, maintaining definitions as datalasses.
"""

from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class CognitiveBias:
    name: str
    definition: str

class BiasRegistry:
    """Registry of cognitive biases and their definitions."""
    _biases: List[CognitiveBias] = [
        CognitiveBias('Ambiguity effect', 'The tendency to avoid options for which missing information makes the probability seem "unknown".'),
        CognitiveBias('Attentional bias', 'The tendency of our perception to be affected by our recurring thoughts.'),
        CognitiveBias('Bandwagon effect', 'The tendency to do (or believe) things because many other people do (or believe) the same.'),
        CognitiveBias('Belief bias', "An effect where someone's evaluation of the logical strength of an argument is biased by the believability of the conclusion."),
        CognitiveBias('Anchoring effect', 'The tendency to rely too heavily, or "anchor", on one trait or piece of information when making decisions.'),
        CognitiveBias('Bias blind spot', 'The tendency to see oneself as less biased than other people.'),
        CognitiveBias('Cheerleader effect', 'The tendency for people to appear more attractive in a group than in isolation.'),
        CognitiveBias('Choice-supportive bias', "The tendency to remember one's choices as better than they actually were."),
        CognitiveBias('clustering illusion', 'The tendency to overestimate the importance of small runs, streaks, or clusters in large samples of random data.'),
        CognitiveBias('comfort zone effect', 'The tendency to perform activities within a familiar boundary, avoiding risks.'),
        CognitiveBias('confirmation bias', "The tendency to search for, interpret, focus on and remember information in a way that confirms one's preconceptions."),
        CognitiveBias('contrast effect', "The enhancement or reduction of a certain stimulus' perception when compared with a recently observed, contrasting object."),
        CognitiveBias('curse of knowledge', 'When better-informed people find it extremely difficult to think about problems from the perspective of lesser-informed people.'),
        CognitiveBias('decoy effect', 'Preferences for either option A or B change in favor of option B when option C is presented.'),
        CognitiveBias('Distinction bias', 'The tendency to view two options as more dissimilar when evaluating them simultaneously than separately.'),
        CognitiveBias('Duration neglect', 'The psychological principle that the length of an experience has little effect on the memory of that event.'),
        CognitiveBias('empathy gap', 'The tendency to underestimate the influence or strength of feelings, in either oneself or others.'),
        CognitiveBias('Endowment effect', 'The fact that people often demand much more to give up an object than they would be willing to pay to acquire it.'),
        CognitiveBias('framing effect', 'Drawing different conclusions from the same information, depending on how that information is presented.'),
        CognitiveBias('Frequency Illusion', "The illusion in which a word, a name, or other thing that has recently come to one's attention suddenly seems to appear with improbable frequency."),
        CognitiveBias('hard-easy effect', 'Based on a specific level of task difficulty, the confidence in judgments is too conservative and not conservative enough.'),
        CognitiveBias('hindsight bias', 'The tendency to see past events as being predictable at the time those events happened.'),
        CognitiveBias('current moment bias', 'The tendency to prefer payoffs that are closer to the present time over those that are further in the future.'),
        CognitiveBias('Identifiable Victim Effect', 'The tendency to respond more strongly to a single identified person at risk than to a large group.'),
        CognitiveBias('IKEA effect', 'The tendency for people to place a disproportionately high value on objects that they partially assembled themselves.'),
        CognitiveBias('illusion of control', "The tendency to overestimate one's degree of influence over other external events."),
        CognitiveBias('Illusion of Validity', 'Belief that our judgments are accurate, especially when available information is consistent or inter-correlated.'),
        CognitiveBias('illusory correlation', 'Inaccurately perceiving a relationship between two unrelated events.'),
        CognitiveBias('impact bias', 'The tendency to overestimate the length or the intensity of the impact of future feeling states.'),
        CognitiveBias('information bias', 'The tendency to seek information even when it cannot affect action.'),
        CognitiveBias('Jumping to Conclusions', 'Drawing a conclusion without taking the needed time to reason through the evidence.'),
        CognitiveBias('Just-world hypothesis', 'The tendency for people to believe that the world is just and therefore people "get what they deserve".'),
        CognitiveBias('Less-is-better effect', 'The tendency to prefer a smaller set to a larger set judged separately, but not jointly.'),
        CognitiveBias('loss aversion', 'The disutility of giving up an object is greater than the utility associated with acquiring it.'),
        CognitiveBias('Mere exposure effect', 'The tendency to express undue liking for things merely because of familiarity with them.'),
        CognitiveBias('Positivity and Negativity Effect', 'The tendency to recall positive/negative information more explicitly depending on age or disposition.'),
        CognitiveBias('Negativity bias', 'Psychological phenomenon by which humans have a greater recall of unpleasant memories compared with positive memories.'),
        CognitiveBias('Neglect of probability', 'The tendency to completely disregard probability when making a decision under uncertainty.'),
        CognitiveBias('Normalcy bias', 'The refusal to plan for, or react to, a disaster which has never happened before.'),
        CognitiveBias('Omission bias', 'The tendency to judge harmful actions as worse, or less moral, than equally harmful omissions (inactions).'),
        CognitiveBias('Optimism bias', 'The tendency to be over-optimistic, overestimating favorable and pleasing outcomes.'),
        CognitiveBias('Ostrich Effect', 'Ignoring an obvious negative situation.'),
        CognitiveBias('Outcome bias', 'The tendency to judge a decision by its eventual outcome instead of based on the quality of the decision at the time it was made.'),
        CognitiveBias('Overconfidence effect', "Excessive confidence in one's own answers to questions."),
        CognitiveBias('Pessimism bias', 'The tendency to overestimate the likelihood of negative things happening to them.'),
        CognitiveBias('Planning Fallacy', 'The tendency to underestimate task-completion times.'),
        CognitiveBias('Positive Outcome Bias', 'The tendency to overestimate the probability of good things happening.'),
        CognitiveBias('Pro-Innovation Bias', "The tendency to have an excessive optimism towards an invention or innovation's usefulness."),
        CognitiveBias('Pseudocertainty effect', 'The tendency to make risk-averse choices if the expected outcome is positive, but risk-seeking choices to avoid negative outcomes.'),
        CognitiveBias('Reactance', 'The urge to do the opposite of what someone wants you to do out of a need to resist a perceived attempt to constrain your freedom.'),
        CognitiveBias('Reactive Devaluation', 'Devaluing proposals only because they purportedly originated with an adversary.'),
        CognitiveBias('recency illusion', "The belief that things that have just recently come to one's attention have in fact only recently originated."),
        CognitiveBias('Risk Compensation', 'The tendency to take greater risks when perceived safety increases.'),
        CognitiveBias('Selective Attention', 'The tendency to direct attention to specific stimuli while ignoring others.'),
        CognitiveBias('Social Comparison Bias', "The tendency to favor potential candidates who don't compete with one's own particular strengths."),
        CognitiveBias('Stereotyping', 'Expecting a member of a group to have certain characteristics without having actual information about that individual.'),
        CognitiveBias('subadditivity effect', 'The tendency to judge probability of the whole to be less than the probabilities of the parts.'),
        CognitiveBias('Subjective validation', "Perception that something is true if a subject's belief demands it to be true."),
        CognitiveBias('Survivorship bias', 'Concentrating on things that "survived" some process and inadvertently overlooking those that didn\'t.'),
        CognitiveBias('Time-saving bias', 'Underestimations of the time that could be saved when increasing from a relatively low speed.'),
        CognitiveBias('Unit Bias', 'The tendency to want to finish a given unit of a task or an item.'),
        CognitiveBias('Whole only effect', 'A preference to evaluate or consider a whole rather than a part.'),
        CognitiveBias('Zero-risk bias', 'Preference for reducing a small risk to zero over a greater reduction in a larger risk.'),
        CognitiveBias('Default effect', 'When given a choice between several options, the tendency to favor the default one.'),
        CognitiveBias('Exaggerated expectation bias', 'The tendency to expect or predict more extreme outcomes than those that actually happen.'),
        CognitiveBias('Forer effect', 'Individuals give high accuracy ratings to descriptions of their personality that supposedly are tailored specifically for them.'),
        CognitiveBias('sunk cost fallacy', 'The phenomenon whereby a person is reluctant to abandon a strategy or course of action because they have invested heavily in it.'),
        CognitiveBias('Essentialism', 'Categorizing people and things according to their essential nature, in spite of variations.'),
        CognitiveBias('Post-purchase rationalization', 'The tendency to persuade oneself through rational argument that a purchase was a good value.'),
        CognitiveBias('Semmelweis reflex', 'The tendency to reject new evidence that contradicts a paradigm.'),
        CognitiveBias('Availability heuristic', 'A mental shortcut that relies on immediate examples that come to a mind when evaluating a specific topic.'),
        CognitiveBias('Backfire effect.', "The reaction to disconfirming evidence by strengthening one's previous beliefs."),
    ]

    @classmethod
    def format_definitions(cls, detected_biases: str) -> str:
        """Return formatted definitions for any biases found in the input string."""
        if not detected_biases:
            return ""

        detected_lower = detected_biases.lower().replace(".", "")
        found_defs = [
            f"- {bias.name}: {bias.definition}"
            for bias in cls._biases
            if bias.name.lower().replace(".", "") in detected_lower
        ]

        if found_defs:
            return "\nDefinitions of detected cognitive biases:\n" + "\n".join(found_defs) + "\n"
        return ""
