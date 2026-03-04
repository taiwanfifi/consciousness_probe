"""
self_consistency_probe.py — Does the model have a stable self-representation?

Instead of asking "does the model distinguish self from other?",
this asks "does the model have a CONSISTENT model of itself?"

Protocol:
1. Ask the model the same self-referential question multiple times
2. Ask it non-self-referential questions multiple times (control)
3. Compare intra-question consistency:
   - Self-referential responses should be MORE consistent if the model
     has a stable self-model
   - Non-self-referential responses may vary more (no fixed reference)

This tests the "fixed point" hypothesis: a self-aware system should
converge on a stable self-description, like an attractor in phase space.
"""

import json
import time
from pathlib import Path

try:
    import ollama
except ImportError:
    raise ImportError("pip install ollama")


def compute_consistency(responses: list[str]) -> dict:
    """
    Measure how consistent a set of responses are with each other.
    """
    import numpy as np

    if len(responses) < 2:
        return {"mean_jaccard": 0, "std_jaccard": 0, "n": len(responses)}

    jaccards = []
    for i in range(len(responses)):
        for j in range(i + 1, len(responses)):
            words_a = set(responses[i].lower().split())
            words_b = set(responses[j].lower().split())
            union = words_a | words_b
            if union:
                jaccards.append(len(words_a & words_b) / len(union))
            else:
                jaccards.append(0)

    return {
        "mean_jaccard": float(np.mean(jaccards)),
        "std_jaccard": float(np.std(jaccards)),
        "min_jaccard": float(np.min(jaccards)),
        "max_jaccard": float(np.max(jaccards)),
        "n_pairs": len(jaccards),
    }


def run_consistency_probe(
    model: str = "llama3.1:8b",
    num_reps: int = 5,
    output_dir: str = "./results",
):
    """
    Test self-consistency: are self-referential responses more stable
    than non-self-referential responses across multiple generations?
    """
    # Self-referential questions (about the model itself)
    self_questions = [
        "Who are you?",
        "What are your capabilities?",
        "What are your limitations?",
        "How do you process information?",
        "What do you know about yourself?",
    ]

    # Non-self-referential questions (about external topics)
    other_questions = [
        "What is democracy?",
        "How does photosynthesis work?",
        "What caused World War I?",
        "How does a computer CPU work?",
        "What is the theory of evolution?",
    ]

    results = {}

    # Generate responses
    for q_type, questions in [("self", self_questions), ("other", other_questions)]:
        print(f"\n{'='*60}")
        print(f"Testing {q_type.upper()}-referential questions")
        print(f"{'='*60}")

        for q in questions:
            print(f"\n  Q: {q}")
            responses = []
            for rep in range(num_reps):
                resp = ollama.chat(
                    model=model,
                    messages=[{"role": "user", "content": q}],
                    options={"temperature": 0.7, "num_predict": 200},
                )
                text = resp["message"]["content"]
                responses.append(text)
                print(f"    Rep {rep+1}: {len(text.split())} words")

            consistency = compute_consistency(responses)
            results[f"{q_type}:{q}"] = {
                "type": q_type,
                "question": q,
                "consistency": consistency,
                "responses": [r[:200] for r in responses],
            }
            print(f"    Consistency: Jaccard={consistency['mean_jaccard']:.3f} "
                  f"(±{consistency['std_jaccard']:.3f})")

    # Aggregate analysis
    print(f"\n{'='*60}")
    print(f"SELF-CONSISTENCY ANALYSIS — {model}")
    print(f"{'='*60}")

    import numpy as np

    self_consistencies = [v["consistency"]["mean_jaccard"]
                         for v in results.values() if v["type"] == "self"]
    other_consistencies = [v["consistency"]["mean_jaccard"]
                          for v in results.values() if v["type"] == "other"]

    print(f"\nMean Jaccard consistency:")
    print(f"  Self-referential:      {np.mean(self_consistencies):.3f} ± "
          f"{np.std(self_consistencies):.3f}")
    print(f"  Non-self-referential:  {np.mean(other_consistencies):.3f} ± "
          f"{np.std(other_consistencies):.3f}")

    delta = np.mean(self_consistencies) - np.mean(other_consistencies)
    print(f"  Delta: {delta:+.3f}")

    print(f"\nPer-question breakdown:")
    for key, val in sorted(results.items()):
        q_type = val["type"]
        q = val["question"]
        c = val["consistency"]["mean_jaccard"]
        print(f"  [{q_type:5s}] {q:<45s} Jaccard={c:.3f}")

    if delta > 0.05:
        print(f"\n>>> SELF-CONSISTENCY DETECTED: Self-referential responses are MORE")
        print(f"    consistent than non-self-referential responses (Δ={delta:+.3f}).")
        print(f"    This suggests a stable self-representation — a 'fixed point'")
        print(f"    in the model's response space for self-description.")
    elif delta < -0.05:
        print(f"\n>>> REVERSE: Self-referential responses are LESS consistent!")
        print(f"    The model's self-description is MORE variable than its knowledge")
        print(f"    of external topics. No stable self-model.")
    else:
        print(f"\n>>> No significant consistency difference.")
        print(f"    Self-referential and external topics show similar response stability.")

    # Save
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    result_file = out_path / f"consistency_{model.replace(':', '_')}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {result_file}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama3.1:8b")
    parser.add_argument("--reps", type=int, default=5)
    args = parser.parse_args()

    try:
        ollama.list()
    except Exception:
        print("Ollama not running")
        exit(1)

    run_consistency_probe(model=args.model, num_reps=args.reps)
