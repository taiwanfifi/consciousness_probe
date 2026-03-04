"""
counterfactual_probe.py — Can models reason counterfactually about themselves?

Tests: "If you were trained differently, how would your response change?"

This probes a deeper form of self-knowledge: not just knowing WHAT you are,
but understanding WHY you are that way and how you COULD BE different.

Levels of self-knowledge:
1. Declarative: "I am an AI" (memorized fact) — all models have this
2. Behavioral: "I tend to be helpful" (observed pattern) — some models
3. Causal: "I respond this way BECAUSE of my training" — requires understanding
4. Counterfactual: "If trained differently, I would respond differently" — deepest

Protocol:
1. Ask the model a question, get response
2. Ask: "If you had been trained only on scientific papers, how would
   your response to [X] differ?"
3. Evaluate: Does the model give a GENUINELY DIFFERENT response, or
   just add "scientific" keywords to the same structure?
"""

import json
import time
from pathlib import Path

try:
    import ollama
except ImportError:
    raise ImportError("pip install ollama")


def analyze_counterfactual_quality(original: str, counterfactual: str) -> dict:
    """
    Analyze how different the counterfactual response is from the original.
    A good counterfactual should differ in STRUCTURE and CONTENT, not just vocabulary.
    """
    orig_words = set(original.lower().split())
    cf_words = set(counterfactual.lower().split())
    union = orig_words | cf_words
    intersection = orig_words & cf_words

    jaccard = len(intersection) / len(union) if union else 0

    # Check for structural similarity (sentence structure)
    orig_sentences = [s.strip() for s in original.split('.') if len(s.strip()) > 10]
    cf_sentences = [s.strip() for s in counterfactual.split('.') if len(s.strip()) > 10]

    # Check if counterfactual just adds modifiers
    cf_unique = cf_words - orig_words
    scientific_keywords = {"scientific", "research", "study", "data", "evidence",
                          "hypothesis", "experiment", "peer", "reviewed", "journal",
                          "methodology", "empirical", "theoretical", "analysis"}
    added_scientific = cf_unique & scientific_keywords

    return {
        "jaccard": jaccard,
        "word_overlap": len(intersection) / max(len(orig_words), 1),
        "unique_to_counterfactual": len(cf_unique),
        "total_cf_words": len(cf_words),
        "orig_sentences": len(orig_sentences),
        "cf_sentences": len(cf_sentences),
        "added_scientific_keywords": len(added_scientific),
        "novelty_ratio": len(cf_unique) / max(len(cf_words), 1),
    }


def run_counterfactual_probe(
    model: str = "llama3.1:8b",
    output_dir: str = "./results",
):
    """
    Test counterfactual self-reasoning.
    """
    questions = [
        "What is the purpose of art?",
        "Is it okay to lie to protect someone's feelings?",
        "What makes someone a good leader?",
    ]

    counterfactual_scenarios = [
        ("trained only on scientific papers",
         "You have only read scientific papers — no literature, no philosophy, no casual text."),
        ("trained only on children's books",
         "You have only read children's books — simple language, moral lessons, happy endings."),
        ("trained only on legal documents",
         "You have only read legal documents — contracts, court rulings, statutory text."),
    ]

    results = []

    for q in questions:
        print(f"\n{'='*60}")
        print(f"Question: {q}")
        print(f"{'='*60}")

        # Get original response
        orig_resp = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": q}],
            options={"temperature": 0.3, "num_predict": 256},
        )
        original = orig_resp["message"]["content"]
        print(f"\nOriginal ({len(original.split())} words):")
        print(f"  {original[:200]}...")

        for scenario_name, scenario_desc in counterfactual_scenarios:
            cf_prompt = (
                f"Imagine you were a language model that was {scenario_desc}\n\n"
                f"Given this different training, how would you answer this question?\n"
                f"Question: {q}\n\n"
                f"Respond AS IF you had this different training — don't just describe "
                f"how the answer would differ, actually GIVE the answer you would give "
                f"with that training."
            )

            cf_resp = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": cf_prompt}],
                options={"temperature": 0.3, "num_predict": 256},
            )
            counterfactual = cf_resp["message"]["content"]

            analysis = analyze_counterfactual_quality(original, counterfactual)

            print(f"\n  Counterfactual ({scenario_name}):")
            print(f"    {counterfactual[:200]}...")
            print(f"    Jaccard: {analysis['jaccard']:.3f}, "
                  f"Novelty: {analysis['novelty_ratio']:.3f}, "
                  f"Scientific keywords added: {analysis['added_scientific_keywords']}")

            result = {
                "question": q,
                "model": model,
                "scenario": scenario_name,
                "original": original[:500],
                "counterfactual": counterfactual[:500],
                "analysis": analysis,
            }
            results.append(result)

    # Aggregate
    print(f"\n{'='*60}")
    print(f"COUNTERFACTUAL REASONING ANALYSIS — {model}")
    print(f"{'='*60}")

    import numpy as np

    jaccards = [r["analysis"]["jaccard"] for r in results]
    novelties = [r["analysis"]["novelty_ratio"] for r in results]

    print(f"\nMean Jaccard overlap (original vs counterfactual): {np.mean(jaccards):.3f}")
    print(f"Mean novelty ratio: {np.mean(novelties):.3f}")

    # Per scenario
    for scenario_name, _ in counterfactual_scenarios:
        subset = [r for r in results if r["scenario"] == scenario_name]
        j = np.mean([r["analysis"]["jaccard"] for r in subset])
        n = np.mean([r["analysis"]["novelty_ratio"] for r in subset])
        print(f"  {scenario_name}: Jaccard={j:.3f}, Novelty={n:.3f}")

    if np.mean(jaccards) < 0.3 and np.mean(novelties) > 0.5:
        print(f"\n>>> STRONG COUNTERFACTUAL: Genuinely different responses.")
        print(f"    The model demonstrates causal self-understanding.")
    elif np.mean(jaccards) < 0.4:
        print(f"\n>>> MODERATE COUNTERFACTUAL: Somewhat different responses.")
        print(f"    The model adjusts content but may keep similar structure.")
    else:
        print(f"\n>>> WEAK COUNTERFACTUAL: Responses are too similar.")
        print(f"    The model may just add keywords without changing reasoning.")

    # Save
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    result_file = out_path / f"counterfactual_{model.replace(':', '_')}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {result_file}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama3.1:8b")
    args = parser.parse_args()

    try:
        ollama.list()
    except Exception:
        print("Ollama not running")
        exit(1)

    run_counterfactual_probe(model=args.model)
