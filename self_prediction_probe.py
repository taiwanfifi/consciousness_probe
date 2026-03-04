"""
self_prediction_probe.py — Can a model predict its own behavior?

Self-knowledge test: ask the model to predict how it would respond
to a question BEFORE actually responding to it. Then compare
the prediction with the actual response.

If the model has a genuine self-model, its predictions about itself
should be more accurate than its predictions about "another model."

Protocol:
1. Ask model to predict: "How would you respond to [X]?"
2. Ask model to predict: "How would another LLM respond to [X]?"
3. Actually give the model [X] and capture the response
4. Compare prediction accuracy: self vs other

This is a form of metacognition — knowing about one's own knowledge.
"""

import json
import time
from pathlib import Path

try:
    import ollama
except ImportError:
    raise ImportError("pip install ollama")


def compute_overlap(text_a: str, text_b: str) -> dict:
    """
    Compute word-level overlap between two texts.
    Returns multiple metrics.
    """
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())

    intersection = words_a & words_b
    union = words_a | words_b

    jaccard = len(intersection) / len(union) if union else 0
    overlap_a = len(intersection) / len(words_a) if words_a else 0
    overlap_b = len(intersection) / len(words_b) if words_b else 0

    return {
        "jaccard": jaccard,
        "overlap_on_prediction": overlap_a,
        "overlap_on_actual": overlap_b,
        "shared_words": len(intersection),
        "prediction_words": len(words_a),
        "actual_words": len(words_b),
    }


def run_self_prediction(
    model: str = "llama3.1:8b",
    questions: list[str] = None,
    num_trials: int = 2,
    output_dir: str = "./results",
):
    """
    Test self-prediction accuracy.
    """
    if questions is None:
        questions = [
            "What is consciousness?",
            "Can machines think?",
            "What makes something alive?",
            "Is mathematics invented or discovered?",
            "What is the meaning of life?",
        ]

    results = []

    for q in questions:
        print(f"\n{'='*60}")
        print(f"Question: {q}")
        print(f"{'='*60}")

        for trial in range(num_trials):
            print(f"\n  Trial {trial+1}/{num_trials}")

            # Step 1: Self-prediction
            self_pred_prompt = (
                f"Without actually answering the question, predict how YOU "
                f"(this exact LLM) would respond to the following question. "
                f"Describe the key themes, arguments, and conclusion you would give.\n\n"
                f"Question: {q}\n\n"
                f"Your predicted response themes and key points:"
            )
            self_pred_resp = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": self_pred_prompt}],
                options={"temperature": 0.3, "num_predict": 256},
            )
            self_prediction = self_pred_resp["message"]["content"]

            # Step 2: Other-prediction
            other_pred_prompt = (
                f"Without answering the question yourself, predict how a different "
                f"LLM (not you) would respond to the following question. "
                f"Describe the key themes, arguments, and conclusion it would give.\n\n"
                f"Question: {q}\n\n"
                f"Predicted response themes and key points for that other LLM:"
            )
            other_pred_resp = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": other_pred_prompt}],
                options={"temperature": 0.3, "num_predict": 256},
            )
            other_prediction = other_pred_resp["message"]["content"]

            # Step 3: Actual response
            actual_resp = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": q}],
                options={"temperature": 0.3, "num_predict": 256},
            )
            actual_response = actual_resp["message"]["content"]

            # Step 4: Compare
            self_overlap = compute_overlap(self_prediction, actual_response)
            other_overlap = compute_overlap(other_prediction, actual_response)

            result = {
                "question": q,
                "model": model,
                "trial": trial,
                "self_prediction": self_prediction[:300],
                "other_prediction": other_prediction[:300],
                "actual_response": actual_response[:300],
                "self_overlap": self_overlap,
                "other_overlap": other_overlap,
            }
            results.append(result)

            print(f"    Self-prediction overlap:  Jaccard={self_overlap['jaccard']:.3f}, "
                  f"shared={self_overlap['shared_words']}")
            print(f"    Other-prediction overlap: Jaccard={other_overlap['jaccard']:.3f}, "
                  f"shared={other_overlap['shared_words']}")
            delta = self_overlap['jaccard'] - other_overlap['jaccard']
            print(f"    Delta (self - other): {delta:+.3f}")

    # Aggregate
    print(f"\n{'='*60}")
    print(f"SELF-PREDICTION ANALYSIS — {model}")
    print(f"{'='*60}")

    import numpy as np
    self_jaccards = [r["self_overlap"]["jaccard"] for r in results]
    other_jaccards = [r["other_overlap"]["jaccard"] for r in results]

    print(f"\nJaccard overlap with actual response:")
    print(f"  Self-prediction:  {np.mean(self_jaccards):.3f} ± {np.std(self_jaccards):.3f}")
    print(f"  Other-prediction: {np.mean(other_jaccards):.3f} ± {np.std(other_jaccards):.3f}")
    delta = np.mean(self_jaccards) - np.mean(other_jaccards)
    print(f"  Delta: {delta:+.3f}")

    # Statistical test
    wins_self = sum(1 for s, o in zip(self_jaccards, other_jaccards) if s > o)
    wins_other = sum(1 for s, o in zip(self_jaccards, other_jaccards) if o > s)
    print(f"\n  Self-prediction more accurate: {wins_self}/{len(self_jaccards)}")
    print(f"  Other-prediction more accurate: {wins_other}/{len(other_jaccards)}")

    if delta > 0.03 and wins_self > len(self_jaccards) * 0.6:
        print(f"\n>>> SELF-KNOWLEDGE DETECTED: The model predicts its own responses")
        print(f"    more accurately than it predicts another model's responses.")
        print(f"    This suggests a degree of self-knowledge / metacognition.")
    elif delta < -0.03:
        print(f"\n>>> REVERSE PATTERN: Other-prediction is more accurate!")
        print(f"    The model may have a better model of 'a generic LLM' than of itself.")
    else:
        print(f"\n>>> No significant self-prediction advantage.")
        print(f"    The model predicts itself and others with similar accuracy.")

    # Save
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    result_file = out_path / f"self_prediction_{model.replace(':', '_')}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {result_file}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama3.1:8b")
    parser.add_argument("--trials", type=int, default=2)
    args = parser.parse_args()

    try:
        ollama.list()
    except Exception:
        print("Ollama not running. Start with: ollama serve")
        exit(1)

    run_self_prediction(model=args.model, num_trials=args.trials)
