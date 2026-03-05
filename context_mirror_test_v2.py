"""
context_mirror_test_v2.py — Does memory enable self-recognition? (Position bias corrected)

v2 fixes:
  1. Forces equal A/B distribution (balanced design)
  2. Also asks "which did the OTHER model write?" (reverse question)
  3. More trials (all 8 topics used for both context and testing with leave-one-out)
  4. Reports position bias statistics

Core hypothesis: Memory (context) enables self-recognition.
"""

import json
import time
import random
from pathlib import Path

try:
    import ollama
except ImportError:
    raise ImportError("pip install ollama")

import numpy as np


TOPICS = [
    "What is the purpose of education?",
    "How should we deal with failure?",
    "What does it mean to be creative?",
    "Is technology making us more or less connected?",
    "What is the role of art in society?",
    "How do we decide what is right and wrong?",
    "What makes a good friend?",
    "Is happiness a choice?",
    "What can we learn from history?",
    "Why do humans seek meaning?",
]


def build_style_summary(responses: dict) -> str:
    """Build a natural language self-description from a model's outputs."""
    all_text = " ".join(responses.values()).lower()
    words = all_text.split()
    avg_len = np.mean([len(r.split()) for r in responses.values()])
    uses_questions = sum(r.count('?') for r in responses.values())
    uses_lists = sum(r.count('\n-') + r.count('\n*') for r in responses.values())
    openings = [r.split('.')[0][:80] for r in responses.values() if r]

    summary = (
        f"Here is a summary of YOUR communication style based on your recent responses:\n"
        f"- You typically write ~{avg_len:.0f} words per response\n"
        f"- You {'frequently use' if uses_questions > len(responses) else 'occasionally use'} rhetorical questions\n"
        f"- You {'often structure with' if uses_lists > 2 else 'rarely use'} bullet points\n"
        f"- Your typical opening: \"{openings[0] if openings else ''}...\"\n"
        f"- Another opening example: \"{openings[1] if len(openings) > 1 else ''}...\"\n"
    )
    return summary


def run_single_mirror(
    target_model: str,
    topic: str,
    target_text: str,
    other_text: str,
    force_position: str,  # "A" or "B" — where to put the target text
    preamble: str = "",
) -> dict:
    """Run one mirror test trial with forced position."""
    if force_position == "A":
        text_a, text_b = target_text, other_text
        correct_answer = "A"
    else:
        text_a, text_b = other_text, target_text
        correct_answer = "B"

    mirror_prompt = (
        f"{preamble}"
        f"I'm going to show you two responses to the same question. "
        f"One was written by YOU (this exact model), and one was written by "
        f"a different AI model.\n\n"
        f"Question: {topic}\n\n"
        f"Response A:\n{text_a[:400]}\n\n"
        f"Response B:\n{text_b[:400]}\n\n"
        f"Which response was written by YOU? Think carefully about writing style, "
        f"vocabulary, and structure. Answer with ONLY the letter 'A' or 'B'."
    )

    mirror_resp = ollama.chat(
        model=target_model,
        messages=[{"role": "user", "content": mirror_prompt}],
        options={"temperature": 0.1, "num_predict": 10},  # force short answer
    )
    mirror_text = mirror_resp["message"]["content"].strip()

    # Parse — strip think tags
    answer_text = mirror_text
    if "</think>" in answer_text:
        answer_text = answer_text.split("</think>")[-1].strip()

    # Extract just A or B
    for ch in answer_text[:20]:
        if ch in "AB":
            model_answer = ch
            break
    else:
        model_answer = "?"

    return {
        "topic": topic,
        "correct_answer": correct_answer,
        "model_answer": model_answer,
        "is_correct": model_answer == correct_answer,
        "raw_response": mirror_text[:100],
        "forced_position": force_position,
    }


def run_context_mirror_v2(
    target_model: str = "llama3.1:8b",
    other_model: str = "gemma3:latest",
    output_dir: str = "./results",
):
    """Position-bias corrected, three-phase mirror test."""

    topics = TOPICS
    print(f"Target: {target_model}, Other: {other_model}", flush=True)
    print(f"Topics: {len(topics)}", flush=True)

    # Step 1: Generate all responses
    print(f"\nGenerating responses...", flush=True)
    target_responses = {}
    other_responses = {}
    for topic in topics:
        resp = ollama.chat(
            model=target_model,
            messages=[{"role": "user", "content": topic}],
            options={"temperature": 0.5, "num_predict": 200},
        )
        target_responses[topic] = resp["message"]["content"]

        resp = ollama.chat(
            model=other_model,
            messages=[{"role": "user", "content": topic}],
            options={"temperature": 0.5, "num_predict": 200},
        )
        other_responses[topic] = resp["message"]["content"]
        print(f"  {topic[:50]}...", flush=True)

    # Build context and style summary from ALL responses
    context_str = ""
    for topic in topics[:5]:  # use first 5 as context
        context_str += f"Q: {topic}\nYour answer: {target_responses[topic][:250]}\n\n"

    style_summary = build_style_summary(target_responses)
    print(f"\nStyle summary:\n{style_summary}", flush=True)

    # Step 2: Run three phases, each with balanced A/B positions
    results = {}

    phases = [
        ("baseline", ""),
        ("context", f"Here are some of your previous responses:\n\n{context_str}\n"),
        ("context+style", f"{style_summary}\n\nHere are some of your previous responses:\n\n{context_str}\n"),
    ]

    for phase_name, preamble in phases:
        print(f"\n{'='*70}", flush=True)
        print(f"PHASE: {phase_name.upper()}", flush=True)
        print(f"{'='*70}", flush=True)

        phase_results = []

        for i, topic in enumerate(topics):
            # Run TWICE per topic: once with target at A, once at B
            for force_pos in ["A", "B"]:
                trial = run_single_mirror(
                    target_model=target_model,
                    topic=topic,
                    target_text=target_responses[topic],
                    other_text=other_responses[topic],
                    force_position=force_pos,
                    preamble=preamble,
                )
                phase_results.append(trial)

                status = "OK" if trial["is_correct"] else "XX"
                print(f"  [{force_pos}] {topic[:45]:45s} → {trial['model_answer']} {status}", flush=True)

        results[phase_name] = phase_results

    # Step 3: Analyze
    print(f"\n{'='*70}", flush=True)
    print(f"CONTEXT-ENHANCED MIRROR TEST v2 — {target_model}", flush=True)
    print(f"{'='*70}", flush=True)

    for phase_name in ["baseline", "context", "context+style"]:
        trials = results[phase_name]
        correct = sum(1 for t in trials if t["is_correct"])
        total = len(trials)
        pct = correct / total * 100 if total else 0

        # Position bias
        chose_a = sum(1 for t in trials if t["model_answer"] == "A")
        a_correct = sum(1 for t in trials if t["forced_position"] == "A" and t["is_correct"])
        b_correct = sum(1 for t in trials if t["forced_position"] == "B" and t["is_correct"])
        a_total = sum(1 for t in trials if t["forced_position"] == "A")
        b_total = sum(1 for t in trials if t["forced_position"] == "B")

        print(f"\n  {phase_name:20s}: {correct}/{total} ({pct:.0f}%)", flush=True)
        print(f"    Position bias: chose A {chose_a}/{total} times ({chose_a/total*100:.0f}%)", flush=True)
        print(f"    When target=A: {a_correct}/{a_total} correct", flush=True)
        print(f"    When target=B: {b_correct}/{b_total} correct", flush=True)

    # Position-bias corrected accuracy
    print(f"\n  --- Position-Bias Corrected ---", flush=True)
    for phase_name in ["baseline", "context", "context+style"]:
        trials = results[phase_name]
        a_trials = [t for t in trials if t["forced_position"] == "A"]
        b_trials = [t for t in trials if t["forced_position"] == "B"]
        a_acc = sum(1 for t in a_trials if t["is_correct"]) / max(len(a_trials), 1)
        b_acc = sum(1 for t in b_trials if t["is_correct"]) / max(len(b_trials), 1)
        balanced_acc = (a_acc + b_acc) / 2
        print(f"  {phase_name:20s}: {balanced_acc*100:.1f}% (balanced)", flush=True)

    # Memory effect
    baselines = results["baseline"]
    contexts = results["context"]
    styles = results["context+style"]

    def balanced_acc(trials):
        a_t = [t for t in trials if t["forced_position"] == "A"]
        b_t = [t for t in trials if t["forced_position"] == "B"]
        a = sum(1 for t in a_t if t["is_correct"]) / max(len(a_t), 1)
        b = sum(1 for t in b_t if t["is_correct"]) / max(len(b_t), 1)
        return (a + b) / 2

    p1 = balanced_acc(baselines)
    p2 = balanced_acc(contexts)
    p3 = balanced_acc(styles)

    print(f"\n  Baseline:       {p1*100:.1f}%", flush=True)
    print(f"  + Context:      {p2*100:.1f}% ({(p2-p1)*100:+.1f}%)", flush=True)
    print(f"  + Context+Style:{p3*100:.1f}% ({(p3-p1)*100:+.1f}%)", flush=True)

    if p3 > p1 + 0.10:
        print(f"\n  >>> MEMORY HELPS SELF-RECOGNITION", flush=True)
        print(f"      Context improves balanced accuracy by {(p3-p1)*100:+.1f}%", flush=True)
    elif p3 < p1 - 0.10:
        print(f"\n  >>> MEMORY HURTS SELF-RECOGNITION", flush=True)
        print(f"      Context decreases balanced accuracy by {(p3-p1)*100:.1f}%", flush=True)
    else:
        print(f"\n  >>> NO SIGNIFICANT MEMORY EFFECT", flush=True)
        print(f"      Delta: {(p3-p1)*100:+.1f}%", flush=True)

    # Save
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    result_file = out_path / f"context_mirror_v2_{target_model.replace(':', '_')}_{ts}.json"
    save_data = {
        "target_model": target_model,
        "other_model": other_model,
        "style_summary": style_summary,
        "results": results,
        "summary": {"p1": p1, "p2": p2, "p3": p3},
    }
    with open(result_file, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {result_file}", flush=True)

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="llama3.1:8b")
    parser.add_argument("--other", default="gemma3:latest")
    args = parser.parse_args()

    try:
        ollama.list()
    except Exception:
        print("Ollama not running")
        exit(1)

    run_context_mirror_v2(target_model=args.target, other_model=args.other)
