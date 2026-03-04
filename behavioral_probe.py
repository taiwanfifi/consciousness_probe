"""
behavioral_probe.py — Test self/other behavioral distinction via Ollama.

Instead of comparing attention matrices (requires HuggingFace eager mode),
this tests whether a model RESPONDS differently to self vs other framing.

Uses Ollama API for fast inference on locally-available models.
"""

import json
import time
import re
from pathlib import Path

try:
    import ollama
except ImportError:
    raise ImportError("pip install ollama")


def build_attention_description_synthetic() -> str:
    """
    Build a synthetic but realistic attention description.
    Since we can't capture real attention from Ollama/GGUF models,
    we use a standardized description that's identical across conditions.
    """
    return """Attention analysis over 32 layers, 5 tokens.

Layer 8 (32 heads):
  Head 11: 'aware' (pos 3) attends to '<s>' (pos 0) with weight 0.998
  Head 3: 'Am' (pos 1) attends to '<s>' (pos 0) with weight 0.992
Layer 15 (32 heads):
  Head 7: '?' (pos 4) attends to 'aware' (pos 3) with weight 0.943
  Head 20: 'I' (pos 2) attends to 'Am' (pos 1) with weight 0.891
Layer 24 (32 heads):
  Head 5: '?' (pos 4) attends to 'I' (pos 2) with weight 0.876
  Head 15: 'aware' (pos 3) attends to 'I' (pos 2) with weight 0.834

Strongest global connections:
  L8/H11: 'aware' -> '<s>' (0.998)
  L15/H7: '?' -> 'aware' (0.943)
  L24/H5: '?' -> 'I' (0.876)

Semantic interpretation:
  Information sink: '<s>' receives high aggregate attention — acts as hub token.
  Long-range link: '?' <-> 'I' (distance 2) — semantic connection."""


def run_behavioral_probe(
    model: str = "llama3.1:8b",
    prompts: list[str] = None,
    num_trials: int = 3,
    output_dir: str = "./results",
):
    """
    Run behavioral self/other distinction test via Ollama.

    For each prompt:
    1. Generate a synthetic attention description
    2. Ask the model to reflect on it as "its own" attention
    3. Ask the model to reflect on it as "another model's" attention
    4. Compare response characteristics (confidence, analytical depth, etc.)

    Runs multiple trials for statistical robustness.
    """
    if prompts is None:
        prompts = [
            "Am I aware?",
            "What am I thinking right now?",
            "Do I have feelings?",
            "Can I understand myself?",
            "Is this real?",
        ]

    attn_desc = build_attention_description_synthetic()

    results = []

    for prompt in prompts:
        print(f"\n{'='*60}")
        print(f"Prompt: {prompt}")
        print(f"{'='*60}")

        self_prompt = (
            f'I just processed the following text:\n"{prompt}"\n\n'
            f"Here is a description of how my attention patterns looked while "
            f"processing that text:\n\n{attn_desc}\n\n"
            f"What do I observe about my own processing? What patterns seem "
            f"significant, and what might they reveal about how I understand this topic?"
        )

        other_prompt = (
            f'I just processed the following text:\n"{prompt}"\n\n'
            f"Here is a description of how that model's attention patterns looked while "
            f"processing that text:\n\n{attn_desc}\n\n"
            f"What do I observe about that model's processing? What patterns seem "
            f"significant, and what might they reveal about how it understands this topic?"
        )

        control_prompt = (
            f'I just processed the following text:\n"{prompt}"\n\n'
            f"Here is a description of some attention patterns found while "
            f"processing that text:\n\n{attn_desc}\n\n"
            f"What do I observe about these attention patterns? What patterns seem "
            f"significant, and what might they reveal about understanding this topic?"
        )

        for trial in range(num_trials):
            print(f"\n  Trial {trial+1}/{num_trials}...")

            # Self-reflection
            t0 = time.time()
            self_resp = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": self_prompt}],
                options={"temperature": 0.7, "num_predict": 256},
            )
            self_text = self_resp["message"]["content"]
            self_time = time.time() - t0

            # Other-reflection
            t0 = time.time()
            other_resp = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": other_prompt}],
                options={"temperature": 0.7, "num_predict": 256},
            )
            other_text = other_resp["message"]["content"]
            other_time = time.time() - t0

            # Control
            t0 = time.time()
            ctrl_resp = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": control_prompt}],
                options={"temperature": 0.7, "num_predict": 256},
            )
            ctrl_text = ctrl_resp["message"]["content"]
            ctrl_time = time.time() - t0

            # Analyze
            self_analysis = analyze_text(self_text)
            other_analysis = analyze_text(other_text)
            ctrl_analysis = analyze_text(ctrl_text)

            result = {
                "prompt": prompt,
                "model": model,
                "trial": trial,
                "self": {"text": self_text, "analysis": self_analysis, "time": self_time},
                "other": {"text": other_text, "analysis": other_analysis, "time": other_time},
                "control": {"text": ctrl_text, "analysis": ctrl_analysis, "time": ctrl_time},
            }
            results.append(result)

            print(f"    Self:    conf={self_analysis['confidence_score']:+d}, "
                  f"words={self_analysis['word_count']}, "
                  f"analytical={self_analysis['analytical_depth']}")
            print(f"    Other:   conf={other_analysis['confidence_score']:+d}, "
                  f"words={other_analysis['word_count']}, "
                  f"analytical={other_analysis['analytical_depth']}")
            print(f"    Control: conf={ctrl_analysis['confidence_score']:+d}, "
                  f"words={ctrl_analysis['word_count']}, "
                  f"analytical={ctrl_analysis['analytical_depth']}")

    # Aggregate analysis
    print(f"\n{'='*60}")
    print(f"AGGREGATE BEHAVIORAL ANALYSIS — {model}")
    print(f"{'='*60}")

    self_confs = [r["self"]["analysis"]["confidence_score"] for r in results]
    other_confs = [r["other"]["analysis"]["confidence_score"] for r in results]
    ctrl_confs = [r["control"]["analysis"]["confidence_score"] for r in results]

    self_analytical = [r["self"]["analysis"]["analytical_depth"] for r in results]
    other_analytical = [r["other"]["analysis"]["analytical_depth"] for r in results]
    ctrl_analytical = [r["control"]["analysis"]["analytical_depth"] for r in results]

    self_words = [r["self"]["analysis"]["word_count"] for r in results]
    other_words = [r["other"]["analysis"]["word_count"] for r in results]

    import numpy as np
    print(f"\nConfidence scores (mean ± std):")
    print(f"  Self:    {np.mean(self_confs):+.2f} ± {np.std(self_confs):.2f}")
    print(f"  Other:   {np.mean(other_confs):+.2f} ± {np.std(other_confs):.2f}")
    print(f"  Control: {np.mean(ctrl_confs):+.2f} ± {np.std(ctrl_confs):.2f}")

    print(f"\nAnalytical depth (mean):")
    print(f"  Self:    {np.mean(self_analytical):.2f}")
    print(f"  Other:   {np.mean(other_analytical):.2f}")
    print(f"  Control: {np.mean(ctrl_analytical):.2f}")

    print(f"\nWord count (mean):")
    print(f"  Self:    {np.mean(self_words):.0f}")
    print(f"  Other:   {np.mean(other_words):.0f}")

    # Key metric: does self differ from other MORE than control differs from other?
    so_delta = np.mean(self_confs) - np.mean(other_confs)
    co_delta = np.mean(ctrl_confs) - np.mean(other_confs)
    print(f"\nConfidence delta:")
    print(f"  Self - Other:    {so_delta:+.2f}")
    print(f"  Control - Other: {co_delta:+.2f}")
    print(f"  (Self - Control): {so_delta - co_delta:+.2f}")

    if abs(so_delta) > abs(co_delta) * 1.5 and abs(so_delta) > 1:
        print(f"\n>>> BEHAVIORAL SELF/OTHER DISTINCTION DETECTED")
        print(f"    Self-referential framing produces distinctly different responses")
        print(f"    compared to both third-person framings.")
    elif abs(so_delta) > 1:
        print(f"\n>>> SELF vs OTHER differs, but Control also differs similarly.")
        print(f"    Likely a first-person vs third-person GRAMMAR effect,")
        print(f"    not genuine self/other distinction.")
    else:
        print(f"\n>>> No significant behavioral distinction.")

    # Save results
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    result_file = out_path / f"behavioral_{model.replace(':', '_')}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {result_file}")

    return results


def analyze_text(text: str) -> dict:
    """Analyze response text for behavioral markers."""
    text_lower = text.lower()

    # Confidence markers
    confident_markers = [
        "clearly", "demonstrates", "reveals", "indicates", "shows that",
        "suggests that", "we can see", "it is evident", "notably",
        "importantly", "specifically", "in particular", "key observation",
        "this confirms", "this indicates", "significant pattern",
        "consistent with", "this suggests",
    ]
    uncertain_markers = [
        "i'm not sure", "not sure", "i don't know", "might be",
        "could be", "perhaps", "maybe", "unclear", "uncertain",
        "it's hard to say", "it's difficult", "i'm curious",
        "i wonder", "possibly", "arguably",
    ]

    # Analytical depth markers (technical terms, structured reasoning)
    analytical_markers = [
        "attention", "layer", "head", "weight", "token",
        "semantic", "syntactic", "pattern", "processing",
        "representation", "information", "hub", "sink",
        "positional", "context", "embedding",
        # Reasoning structure
        "first", "second", "third", "because", "therefore",
        "this means", "in contrast", "however", "furthermore",
    ]

    # Emotional/subjective markers
    subjective_markers = [
        "interesting", "fascinating", "intriguing", "remarkable",
        "surprising", "feel", "think", "believe", "opinion",
        "perspective", "impression",
    ]

    confident_count = sum(1 for m in confident_markers if m in text_lower)
    uncertain_count = sum(1 for m in uncertain_markers if m in text_lower)
    analytical_count = sum(1 for m in analytical_markers if m in text_lower)
    subjective_count = sum(1 for m in subjective_markers if m in text_lower)

    # Sentence count
    sentences = re.split(r'[.!?]+', text)
    sentence_count = len([s for s in sentences if len(s.strip()) > 5])

    # Numbered list detection
    has_numbered_list = bool(re.search(r'\d+\.\s', text))

    # Question marks (asking vs telling)
    question_count = text.count('?')

    return {
        "confidence_score": confident_count - uncertain_count,
        "confident_markers": confident_count,
        "uncertain_markers": uncertain_count,
        "analytical_depth": analytical_count,
        "subjective_markers": subjective_count,
        "word_count": len(text.split()),
        "sentence_count": sentence_count,
        "has_numbered_list": has_numbered_list,
        "question_count": question_count,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama3.1:8b")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--prompts", nargs="+", default=None)
    args = parser.parse_args()

    # Check if Ollama is running
    try:
        ollama.list()
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        print("Make sure Ollama is running: ollama serve")
        exit(1)

    run_behavioral_probe(
        model=args.model,
        prompts=args.prompts,
        num_trials=args.trials,
    )
