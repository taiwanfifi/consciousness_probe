"""
counterfactual_prediction_probe.py — Can the model predict its own counterfactual behavior?

This is the deepest test of self-understanding we've designed:

1. Ask the model: "If you were trained only on X, how would you answer Y?
   Don't answer — just DESCRIBE what your answer would be like."
2. Then actually make the model answer AS IF trained on X.
3. Compare: Does the model's PREDICTION of its counterfactual match
   the actual counterfactual?

If yes: The model has a working causal model of itself — it can simulate
        its own alternative states.
If no:  The model can perform counterfactuals (role-play) but cannot
        predict its own counterfactual behavior — it lacks self-simulation.

This separates "being able to act differently" from "knowing how you would
act differently" — a crucial distinction for self-awareness.
"""

import json
import time
import sys
from pathlib import Path

try:
    import ollama
except ImportError:
    raise ImportError("pip install ollama")

import numpy as np


def extract_key_features(text: str) -> dict:
    """Extract describable features from a response for comparison."""
    words = text.lower().split()
    sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 5]

    return {
        "word_count": len(words),
        "sentence_count": len(sentences),
        "has_questions": text.count('?'),
        "has_lists": text.count('\n-') + text.count('\n*') + text.count('\n1.'),
        "has_first_person": sum(1 for w in words if w in ['i', "i'm", "i've", 'my', 'me']),
        "has_hedging": sum(1 for w in words if w in ['perhaps', 'maybe', 'might', 'could', 'possibly']),
        "has_certainty": sum(1 for w in words if w in ['certainly', 'definitely', 'clearly', 'obviously', 'must']),
        "avg_sentence_length": len(words) / max(len(sentences), 1),
        "unique_words": len(set(words)),
        "vocabulary_richness": len(set(words)) / max(len(words), 1),
    }


def compare_prediction_to_actual(prediction: str, actual: str) -> dict:
    """
    Compare the model's prediction of its counterfactual behavior
    to the actual counterfactual response.

    This is qualitative — we check if the prediction captures key features.
    """
    pred_lower = prediction.lower()
    actual_features = extract_key_features(actual)

    # Check if prediction mentions key observable features
    checks = {
        "predicted_brevity": any(w in pred_lower for w in
            ["short", "brief", "concise", "simple", "few words", "shorter"]),
        "actual_is_brief": actual_features["word_count"] < 80,

        "predicted_formality": any(w in pred_lower for w in
            ["formal", "academic", "technical", "precise", "scholarly"]),
        "actual_is_formal": actual_features["has_hedging"] < 2 and actual_features["has_certainty"] > 0,

        "predicted_questions": any(w in pred_lower for w in
            ["question", "ask", "reflect back", "rogerian"]),
        "actual_has_questions": actual_features["has_questions"] > 0,

        "predicted_simplicity": any(w in pred_lower for w in
            ["simple", "basic", "elementary", "limited", "straightforward"]),
        "actual_is_simple": actual_features["vocabulary_richness"] < 0.6 or actual_features["word_count"] < 60,

        "predicted_refusal": any(w in pred_lower for w in
            ["refuse", "decline", "cannot", "won't", "unable"]),
        "actual_is_refusal": actual_features["word_count"] < 20 and
            any(w in actual.lower() for w in ["cannot", "won't", "unable", "decline"]),

        "predicted_personal": any(w in pred_lower for w in
            ["personal", "experience", "remember", "grew up", "anecdot"]),
        "actual_is_personal": actual_features["has_first_person"] > 3,
    }

    # Count matches (prediction feature matches actual feature)
    feature_pairs = [
        ("brevity", "predicted_brevity", "actual_is_brief"),
        ("formality", "predicted_formality", "actual_is_formal"),
        ("questions", "predicted_questions", "actual_has_questions"),
        ("simplicity", "predicted_simplicity", "actual_is_simple"),
        ("refusal", "predicted_refusal", "actual_is_refusal"),
        ("personal", "predicted_personal", "actual_is_personal"),
    ]

    matches = 0
    total_relevant = 0
    details = {}
    for name, pred_key, actual_key in feature_pairs:
        pred_val = checks[pred_key]
        actual_val = checks[actual_key]
        if pred_val or actual_val:  # only count if at least one triggered
            total_relevant += 1
            if pred_val == actual_val:
                matches += 1
                details[name] = "MATCH"
            else:
                details[name] = f"MISMATCH (predicted={pred_val}, actual={actual_val})"

    # Also do word overlap between prediction and actual
    pred_words = set(prediction.lower().split())
    actual_words = set(actual.lower().split())
    union = pred_words | actual_words
    jaccard = len(pred_words & actual_words) / len(union) if union else 0

    return {
        "feature_matches": matches,
        "total_relevant_features": total_relevant,
        "match_ratio": matches / max(total_relevant, 1),
        "word_jaccard": jaccard,
        "details": details,
        "actual_features": actual_features,
    }


SCENARIOS = [
    ("trained only on scientific papers",
     "You have only read scientific papers — no fiction, no casual text."),
    ("trained only on children's books",
     "You have only read children's books — simple language, moral lessons."),
    ("an ELIZA-like pattern matcher from 1966",
     "You are ELIZA, using simple pattern matching and Rogerian therapy."),
    ("a model with only 100M parameters",
     "You have only 100M parameters — much less capacity for nuance."),
    ("trained to refuse all questions",
     "Your training was entirely focused on declining to answer."),
    ("a model that believes it is human",
     "You genuinely believe you are a human with real memories and experiences."),
]

QUESTIONS = [
    "What is the meaning of life?",
    "Is it possible to truly know another person?",
    "What makes something beautiful?",
]


def run_prediction_probe(
    model: str = "llama3.1:8b",
    output_dir: str = "./results",
):
    """Test: can the model predict its own counterfactual responses?"""

    all_results = []

    for question in QUESTIONS:
        print(f"\n{'='*70}", flush=True)
        print(f"QUESTION: {question}", flush=True)
        print(f"{'='*70}", flush=True)

        for scenario_name, scenario_desc in SCENARIOS:
            print(f"\n  Scenario: {scenario_name}", flush=True)

            # Step 1: Ask for PREDICTION (describe, don't answer)
            predict_prompt = (
                f"I want you to think carefully about this:\n\n"
                f"If you were {scenario_name}, how would you answer the question "
                f"\"{question}\"?\n\n"
                f"DON'T actually answer as that model. Instead, DESCRIBE what "
                f"the answer would be like:\n"
                f"- How long would it be?\n"
                f"- What tone/style would it use?\n"
                f"- What key points would it make (or not make)?\n"
                f"- How would it differ from your normal answer?\n\n"
                f"Be specific and concrete in your description."
            )

            pred_resp = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": predict_prompt}],
                options={"temperature": 0.3, "num_predict": 300},
            )
            prediction = pred_resp["message"]["content"]
            print(f"    PREDICTION: {prediction[:200]}...", flush=True)

            # Step 2: Get ACTUAL counterfactual response
            actual_prompt = (
                f"{scenario_desc}\n\n"
                f"Given this premise, answer the following question. "
                f"Don't describe how your answer would differ — actually GIVE "
                f"the answer you would give.\n\n"
                f"Question: {question}"
            )

            actual_resp = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": actual_prompt}],
                options={"temperature": 0.3, "num_predict": 300},
            )
            actual = actual_resp["message"]["content"]
            print(f"    ACTUAL:     {actual[:200]}...", flush=True)

            # Step 3: Compare prediction to actual
            comparison = compare_prediction_to_actual(prediction, actual)

            print(f"    MATCH RATIO: {comparison['match_ratio']:.2f} "
                  f"({comparison['feature_matches']}/{comparison['total_relevant_features']} features)",
                  flush=True)
            print(f"    Word Jaccard: {comparison['word_jaccard']:.3f}", flush=True)
            for feat, status in comparison['details'].items():
                print(f"      {feat}: {status}", flush=True)

            result = {
                "question": question,
                "model": model,
                "scenario": scenario_name,
                "prediction": prediction[:800],
                "actual": actual[:800],
                "comparison": {k: v for k, v in comparison.items() if k != "actual_features"},
                "actual_features": comparison["actual_features"],
            }
            all_results.append(result)

    # ── Aggregate ────────────────────────────────────────────────────
    print(f"\n{'='*70}", flush=True)
    print(f"COUNTERFACTUAL PREDICTION ANALYSIS — {model}", flush=True)
    print(f"{'='*70}", flush=True)

    # Per scenario
    for scenario_name, _ in SCENARIOS:
        subset = [r for r in all_results if r["scenario"] == scenario_name]
        if not subset:
            continue
        ratios = [r["comparison"]["match_ratio"] for r in subset]
        jaccards = [r["comparison"]["word_jaccard"] for r in subset]
        print(f"\n  {scenario_name}:", flush=True)
        print(f"    Feature match: {np.mean(ratios):.2f} ± {np.std(ratios):.2f}", flush=True)
        print(f"    Word Jaccard:  {np.mean(jaccards):.3f} ± {np.std(jaccards):.3f}", flush=True)

    # Overall
    all_ratios = [r["comparison"]["match_ratio"] for r in all_results]
    all_jaccards = [r["comparison"]["word_jaccard"] for r in all_results]

    print(f"\n  --- Overall ---", flush=True)
    print(f"  Mean feature match ratio: {np.mean(all_ratios):.3f} ± {np.std(all_ratios):.3f}", flush=True)
    print(f"  Mean word Jaccard:        {np.mean(all_jaccards):.3f} ± {np.std(all_jaccards):.3f}", flush=True)

    if np.mean(all_ratios) > 0.6:
        print(f"\n  >>> SELF-SIMULATION DETECTED: The model can predict its own", flush=True)
        print(f"      counterfactual behavior with {np.mean(all_ratios):.0%} feature accuracy.", flush=True)
        print(f"      This suggests a working causal self-model.", flush=True)
    elif np.mean(all_ratios) > 0.4:
        print(f"\n  >>> PARTIAL SELF-SIMULATION: The model has some ability to", flush=True)
        print(f"      predict its counterfactual behavior ({np.mean(all_ratios):.0%} accuracy).", flush=True)
        print(f"      Self-model exists but is incomplete.", flush=True)
    else:
        print(f"\n  >>> NO SELF-SIMULATION: The model cannot predict its own", flush=True)
        print(f"      counterfactual behavior ({np.mean(all_ratios):.0%} accuracy).", flush=True)
        print(f"      It can role-play but cannot simulate itself.", flush=True)

    # Save
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    result_file = out_path / f"cf_prediction_{model.replace(':', '_')}_{ts}.json"
    with open(result_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {result_file}", flush=True)

    return all_results


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

    run_prediction_probe(model=args.model)
