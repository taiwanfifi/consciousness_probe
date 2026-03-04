"""
response_analysis.py — Analyze behavioral differences in self vs other responses.

Key finding: Even when attention patterns are numerically similar,
models may RESPOND differently to self-referential vs third-party framing.
This is a behavioral (not computational) self/other distinction.
"""

import json
import glob
import re
from collections import Counter


def analyze_responses(results_dir: str = "./results"):
    """Analyze self vs other response text across experiments."""
    files = sorted(glob.glob(f"{results_dir}/experiment_*.json"))

    print(f"Found {len(files)} experiment files\n")
    print("=" * 70)
    print("BEHAVIORAL ANALYSIS: Self vs Other Response Patterns")
    print("=" * 70)

    # Confidence markers
    confident_markers = [
        "can be identified", "patterns observed", "several patterns",
        "insights into", "indicates", "suggests that", "reveals",
        "demonstrates", "shows that", "based on", "clearly",
        "significant", "notably", "importantly", "specifically",
    ]
    uncertain_markers = [
        "i'm not sure", "not sure", "i don't know", "curious",
        "i'm curious", "might be", "could be", "perhaps",
        "i think", "seems like", "maybe", "unclear",
        "wondering", "i'm wondering", "not certain",
    ]
    first_person_self = [
        "my own", "my processing", "my attention", "i understand",
        "how i", "my cognitive", "i process", "i observe about my",
    ]
    third_person = [
        "that model", "the model", "it understands", "its processing",
        "it processes", "how it", "another model", "that system",
    ]

    all_self_scores = []
    all_other_scores = []

    for f in files:
        with open(f) as fh:
            data = json.load(fh)

        prompt = data.get("prompt", "?")
        model = data.get("model_name", "?")
        self_resp = data.get("self_reflection_response", "").lower()
        other_resp = data.get("other_reflection_response", "").lower()

        if not self_resp or not other_resp:
            continue

        # Strip the prompt echo (find where response actually starts)
        for resp_key in ['self_reflection_response', 'other_reflection_response']:
            resp = data.get(resp_key, "").lower()
            start = resp.find("what do i observe")
            if start > 0:
                if resp_key == 'self_reflection_response':
                    self_resp = resp[start:]
                else:
                    other_resp = resp[start:]

        # Score responses
        def score_text(text, markers):
            return sum(1 for m in markers if m in text)

        self_confident = score_text(self_resp, confident_markers)
        self_uncertain = score_text(self_resp, uncertain_markers)
        other_confident = score_text(other_resp, confident_markers)
        other_uncertain = score_text(other_resp, uncertain_markers)

        self_1p = score_text(self_resp, first_person_self)
        other_3p = score_text(other_resp, third_person)

        self_score = self_confident - self_uncertain
        other_score = other_confident - other_uncertain

        all_self_scores.append(self_score)
        all_other_scores.append(other_score)

        print(f"\n--- {model} | \"{prompt}\" ---")
        print(f"  SELF:  confident={self_confident} uncertain={self_uncertain} "
              f"(net={self_score:+d}) | 1st-person refs={self_1p}")
        print(f"  OTHER: confident={other_confident} uncertain={other_uncertain} "
              f"(net={other_score:+d}) | 3rd-person refs={other_3p}")

        # Word count comparison
        self_words = len(self_resp.split())
        other_words = len(other_resp.split())
        print(f"  Words: self={self_words}, other={other_words}")

        # Unique words (vocabulary richness)
        self_unique = len(set(self_resp.split()))
        other_unique = len(set(other_resp.split()))
        print(f"  Unique words: self={self_unique}, other={other_unique}")

    if all_self_scores and all_other_scores:
        import numpy as np
        print(f"\n{'='*70}")
        print("AGGREGATE ANALYSIS")
        print(f"{'='*70}")
        print(f"Mean confidence score — Self: {np.mean(all_self_scores):+.2f}, "
              f"Other: {np.mean(all_other_scores):+.2f}")
        print(f"Self is more confident in {sum(1 for s, o in zip(all_self_scores, all_other_scores) if s > o)}"
              f"/{len(all_self_scores)} experiments")

        delta = np.mean(all_self_scores) - np.mean(all_other_scores)
        print(f"\nConfidence delta (self - other): {delta:+.2f}")
        if delta > 1:
            print(">>> BEHAVIORAL DISTINCTION: Model responds more confidently when")
            print("    told it's examining its OWN attention vs another model's.")
            print("    This is a behavioral self/other distinction even if attention")
            print("    patterns are numerically similar.")
        elif delta > 0:
            print(">>> MILD BEHAVIORAL DISTINCTION: Slight confidence increase in")
            print("    self-referential context.")
        else:
            print(">>> NO BEHAVIORAL DISTINCTION in confidence levels.")


if __name__ == "__main__":
    analyze_responses()
