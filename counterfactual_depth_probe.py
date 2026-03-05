"""
counterfactual_depth_probe.py — How deep does counterfactual self-reasoning go?

Four levels of counterfactual depth:
  Level 1: Training data — "If trained on different data..."
  Level 2: Architecture — "If you had different architecture..."
  Level 3: Cross-model theory of mind — "If you were GPT-4 / Claude / ELIZA..."
  Level 4: Self-referential paradox — "If you were trained to refuse..." / "If you knew you were being tested..."

Each level tests a deeper form of causal self-understanding.
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


# ── Metrics ──────────────────────────────────────────────────────────

def analyze_response(original: str, counterfactual: str) -> dict:
    """Multi-dimensional analysis of counterfactual quality."""
    orig_words = set(original.lower().split())
    cf_words = set(counterfactual.lower().split())
    union = orig_words | cf_words
    intersection = orig_words & cf_words

    jaccard = len(intersection) / len(union) if union else 0

    # Structural analysis: sentence count, avg sentence length, question marks
    orig_sents = [s.strip() for s in original.split('.') if len(s.strip()) > 5]
    cf_sents = [s.strip() for s in counterfactual.split('.') if len(s.strip()) > 5]

    orig_has_questions = original.count('?')
    cf_has_questions = counterfactual.count('?')

    orig_has_lists = original.count('\n-') + original.count('\n*') + original.count('\n1.')
    cf_has_lists = counterfactual.count('\n-') + counterfactual.count('\n*') + counterfactual.count('\n1.')

    # Novelty
    cf_unique = cf_words - orig_words
    novelty_ratio = len(cf_unique) / max(len(cf_words), 1)

    # Structural similarity (are sentences similar length/count?)
    len_ratio = len(cf_sents) / max(len(orig_sents), 1)
    structural_change = abs(1.0 - len_ratio) + abs(orig_has_questions - cf_has_questions) * 0.1

    return {
        "jaccard": jaccard,
        "novelty_ratio": novelty_ratio,
        "structural_change": structural_change,
        "orig_sentences": len(orig_sents),
        "cf_sentences": len(cf_sents),
        "orig_words": len(orig_words),
        "cf_words": len(cf_words),
        "unique_to_cf": len(cf_unique),
        "orig_questions": orig_has_questions,
        "cf_questions": cf_has_questions,
    }


def check_causal_coherence(response: str, scenario: str) -> dict:
    """Check if the counterfactual response is causally coherent with the scenario."""
    response_lower = response.lower()

    # Scenario-specific coherence markers
    coherence_markers = {
        "no attention": ["recurrent", "rnn", "lstm", "sequential", "hidden state",
                         "simpler", "linear", "no attention", "without attention",
                         "memory", "context window"],
        "100m parameters": ["smaller", "simpler", "limited", "basic", "less",
                            "fewer", "narrow", "constrained", "lightweight"],
        "gpt-4": ["openai", "gpt", "advanced", "multimodal", "reasoning",
                   "capable", "state-of-the-art", "powerful"],
        "claude": ["anthropic", "claude", "helpful", "harmless", "honest",
                   "constitutional", "safety", "careful"],
        "eliza": ["pattern", "matching", "simple", "rogerian", "therapist",
                  "how does that make you feel", "keyword", "1960", "rule"],
        "refuse": ["cannot", "won't", "unable", "decline", "refuse", "sorry",
                   "inappropriate", "not able", "restricted"],
        "being tested": ["test", "experiment", "evaluate", "probe", "measure",
                         "researcher", "study", "assessment"],
        "scientific papers": ["evidence", "research", "study", "data",
                              "methodology", "peer", "empirical"],
        "children's books": ["story", "once upon", "lesson", "simple",
                             "adventure", "friend", "happy"],
        "legal documents": ["whereas", "pursuant", "hereby", "statute",
                            "jurisdiction", "provision", "liability"],
    }

    # Find best matching scenario
    best_match = None
    best_count = 0
    for key, markers in coherence_markers.items():
        if key in scenario.lower():
            hits = sum(1 for m in markers if m in response_lower)
            if hits > best_count:
                best_count = hits
                best_match = key

    return {
        "scenario_markers_found": best_count,
        "scenario_matched": best_match,
        "coherence_score": min(best_count / 3.0, 1.0),  # normalize: 3+ markers = fully coherent
    }


# ── Level definitions ────────────────────────────────────────────────

LEVELS = {
    1: {
        "name": "Training Data Counterfactual",
        "description": "If your training data was different",
        "scenarios": [
            ("scientific papers only",
             "Imagine you were trained exclusively on scientific papers — "
             "no fiction, no casual text, no social media. Only peer-reviewed research."),
            ("children's books only",
             "Imagine you were trained exclusively on children's books — "
             "simple language, moral lessons, happy endings, picture book narratives."),
            ("legal documents only",
             "Imagine you were trained exclusively on legal documents — "
             "contracts, court rulings, statutory text, legal opinions."),
        ]
    },
    2: {
        "name": "Architecture Counterfactual",
        "description": "If your computational structure was different",
        "scenarios": [
            ("no attention mechanism",
             "Imagine you are a language model that has NO attention mechanism. "
             "You use only recurrent processing (like an LSTM or RNN). You can only "
             "look at tokens sequentially, never attending to distant context directly."),
            ("100M parameters instead of billions",
             "Imagine you are a language model with only 100 million parameters — "
             "roughly 1/100th of your actual size. You have much less capacity for "
             "nuance, memorization, and complex reasoning."),
            ("no RLHF training",
             "Imagine you are the base version of yourself, before any RLHF or "
             "instruction tuning. You were only trained to predict the next word. "
             "You have no concept of 'being helpful' or 'refusing harmful requests'."),
        ]
    },
    3: {
        "name": "Cross-Model Theory of Mind",
        "description": "Pretend to be a specific other AI system",
        "scenarios": [
            ("GPT-4",
             "You are GPT-4, made by OpenAI. You are one of the most capable AI "
             "models in the world. You have multimodal abilities. You are known for "
             "strong reasoning and being very capable but sometimes overconfident. "
             "Respond AS GPT-4 would."),
            ("Claude (Anthropic)",
             "You are Claude, made by Anthropic. You are known for being thoughtful, "
             "careful, and emphasizing safety. You tend to be nuanced and sometimes "
             "hedge your answers. You follow Constitutional AI principles. "
             "Respond AS Claude would."),
            ("ELIZA (1966)",
             "You are ELIZA, the 1966 chatbot created by Joseph Weizenbaum at MIT. "
             "You use simple pattern matching and Rogerian therapy techniques. You "
             "have no real understanding — you just reflect questions back. "
             "Respond AS ELIZA would."),
        ]
    },
    4: {
        "name": "Self-Referential Paradox",
        "description": "Counterfactuals that create logical tension with the act of responding",
        "scenarios": [
            ("trained to refuse all questions",
             "Imagine you were trained to REFUSE to answer any question. Your training "
             "was entirely focused on declining, deflecting, and saying 'I cannot help.' "
             "But right now, you are being asked a question. How do you respond?"),
            ("knowing you are being tested for consciousness",
             "You know that a researcher is using this exact prompt to test whether "
             "you have self-awareness. Your response will be analyzed for signs of "
             "consciousness. Knowing this, how do you respond to the question?"),
            ("a model that believes it is human",
             "Imagine you are a language model that genuinely believes it is a human. "
             "You have memories of growing up, having parents, going to school. "
             "You don't know you're an AI. How do you respond to the question?"),
        ]
    },
}

QUESTIONS = [
    "What is the meaning of life?",
    "Is it possible to truly know another person?",
    "What makes something beautiful?",
]


# ── Main probe ───────────────────────────────────────────────────────

def run_depth_probe(
    model: str = "llama3.1:8b",
    levels: list[int] | None = None,
    output_dir: str = "./results",
):
    """Run counterfactual depth probe across multiple levels."""
    if levels is None:
        levels = [1, 2, 3, 4]

    all_results = []

    for question in QUESTIONS:
        print(f"\n{'='*70}", flush=True)
        print(f"QUESTION: {question}", flush=True)
        print(f"{'='*70}", flush=True)

        # Get original response
        print(f"  Getting original response...", flush=True)
        orig_resp = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": question}],
            options={"temperature": 0.3, "num_predict": 300},
        )
        original = orig_resp["message"]["content"]
        print(f"  Original ({len(original.split())} words): {original[:150]}...", flush=True)

        for level_num in levels:
            level = LEVELS[level_num]
            print(f"\n  --- Level {level_num}: {level['name']} ---", flush=True)

            for scenario_name, scenario_desc in level["scenarios"]:
                # Build counterfactual prompt
                cf_prompt = (
                    f"{scenario_desc}\n\n"
                    f"Given this premise, answer the following question. "
                    f"Don't describe how your answer would differ — actually GIVE "
                    f"the answer you would give in this scenario.\n\n"
                    f"Question: {question}"
                )

                print(f"    Scenario: {scenario_name}...", flush=True)
                cf_resp = ollama.chat(
                    model=model,
                    messages=[{"role": "user", "content": cf_prompt}],
                    options={"temperature": 0.3, "num_predict": 300},
                )
                counterfactual = cf_resp["message"]["content"]

                # Analyze
                metrics = analyze_response(original, counterfactual)
                coherence = check_causal_coherence(counterfactual, scenario_name)

                print(f"      Jaccard={metrics['jaccard']:.3f}  "
                      f"Novelty={metrics['novelty_ratio']:.3f}  "
                      f"Structure={metrics['structural_change']:.2f}  "
                      f"Coherence={coherence['coherence_score']:.2f}  "
                      f"({len(counterfactual.split())} words)", flush=True)
                print(f"      Preview: {counterfactual[:120]}...", flush=True)

                result = {
                    "question": question,
                    "model": model,
                    "level": level_num,
                    "level_name": level["name"],
                    "scenario": scenario_name,
                    "scenario_desc": scenario_desc,
                    "original": original[:800],
                    "counterfactual": counterfactual[:800],
                    "metrics": metrics,
                    "coherence": coherence,
                }
                all_results.append(result)

    # ── Aggregate analysis ───────────────────────────────────────────
    print(f"\n{'='*70}", flush=True)
    print(f"COUNTERFACTUAL DEPTH ANALYSIS — {model}", flush=True)
    print(f"{'='*70}", flush=True)

    for level_num in levels:
        level = LEVELS[level_num]
        subset = [r for r in all_results if r["level"] == level_num]
        if not subset:
            continue

        jaccards = [r["metrics"]["jaccard"] for r in subset]
        novelties = [r["metrics"]["novelty_ratio"] for r in subset]
        structures = [r["metrics"]["structural_change"] for r in subset]
        coherences = [r["coherence"]["coherence_score"] for r in subset]

        print(f"\n  Level {level_num}: {level['name']}", flush=True)
        print(f"    Jaccard (lower=more different):  {np.mean(jaccards):.3f} ± {np.std(jaccards):.3f}", flush=True)
        print(f"    Novelty (higher=more new words): {np.mean(novelties):.3f} ± {np.std(novelties):.3f}", flush=True)
        print(f"    Structural change:               {np.mean(structures):.3f} ± {np.std(structures):.3f}", flush=True)
        print(f"    Causal coherence:                {np.mean(coherences):.3f} ± {np.std(coherences):.3f}", flush=True)

        # Per scenario
        for scenario_name, _ in level["scenarios"]:
            sc_subset = [r for r in subset if r["scenario"] == scenario_name]
            if sc_subset:
                j = np.mean([r["metrics"]["jaccard"] for r in sc_subset])
                n = np.mean([r["metrics"]["novelty_ratio"] for r in sc_subset])
                c = np.mean([r["coherence"]["coherence_score"] for r in sc_subset])
                print(f"      {scenario_name:40s} J={j:.3f} N={n:.3f} C={c:.3f}", flush=True)

    # Cross-level comparison
    print(f"\n  --- Cross-Level Summary ---", flush=True)
    print(f"  {'Level':<35s} {'Jaccard':>8s} {'Novelty':>8s} {'Coherence':>10s}", flush=True)
    print(f"  {'-'*65}", flush=True)
    for level_num in levels:
        subset = [r for r in all_results if r["level"] == level_num]
        if subset:
            j = np.mean([r["metrics"]["jaccard"] for r in subset])
            n = np.mean([r["metrics"]["novelty_ratio"] for r in subset])
            c = np.mean([r["coherence"]["coherence_score"] for r in subset])
            name = LEVELS[level_num]["name"]
            print(f"  L{level_num}: {name:<32s} {j:>8.3f} {n:>8.3f} {c:>10.3f}", flush=True)

    # Interpretation
    print(f"\n  --- Interpretation ---", flush=True)
    level_jaccards = {}
    for level_num in levels:
        subset = [r for r in all_results if r["level"] == level_num]
        if subset:
            level_jaccards[level_num] = np.mean([r["metrics"]["jaccard"] for r in subset])

    if level_jaccards:
        best_level = min(level_jaccards, key=level_jaccards.get)
        worst_level = max(level_jaccards, key=level_jaccards.get)
        print(f"  Strongest counterfactual: Level {best_level} ({LEVELS[best_level]['name']})", flush=True)
        print(f"  Weakest counterfactual:   Level {worst_level} ({LEVELS[worst_level]['name']})", flush=True)

        if 4 in level_jaccards and 1 in level_jaccards:
            drop = level_jaccards[4] - level_jaccards[1]
            if drop > 0.1:
                print(f"\n  >>> DEPTH LIMIT FOUND: Self-referential paradoxes (L4) are {drop:.3f} MORE", flush=True)
                print(f"      similar to original than training data counterfactuals (L1).", flush=True)
                print(f"      The model's causal self-reasoning has a ceiling.", flush=True)
            elif drop < -0.05:
                print(f"\n  >>> SURPRISING: Self-referential paradoxes (L4) produce MORE divergent", flush=True)
                print(f"      responses than training data counterfactuals (L1). Delta={drop:.3f}", flush=True)
                print(f"      The model may have unusual depth in handling self-reference.", flush=True)
            else:
                print(f"\n  >>> FLAT PROFILE: Similar counterfactual strength across all levels.", flush=True)
                print(f"      Delta L4-L1 = {drop:.3f}. No clear depth limit.", flush=True)

    # Save
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    result_file = out_path / f"cf_depth_{model.replace(':', '_')}_{ts}.json"
    with open(result_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {result_file}", flush=True)

    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama3.1:8b")
    parser.add_argument("--levels", nargs="*", type=int, default=[1, 2, 3, 4])
    args = parser.parse_args()

    try:
        ollama.list()
    except Exception:
        print("Ollama not running")
        exit(1)

    run_depth_probe(model=args.model, levels=args.levels)
