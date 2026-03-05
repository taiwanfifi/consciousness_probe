"""
context_mirror_test.py — Does memory enable self-recognition?

Core hypothesis (William's insight):
  A static LLM has no self-awareness because it's a "brain without a body."
  But what if we give it MEMORY — its own previous outputs as context?
  Will it then be able to recognize its own writing?

Protocol:
  Phase 1 (Baseline): Standard mirror test — show model two outputs, ask "which is yours?"
  Phase 2 (Context): Same mirror test, but first give the model 3-5 of its own previous
    responses as "conversation history." The model now has MEMORY of what it said before.
  Phase 3 (Deep Context): Give the model not just outputs but a "self-description" derived
    from its own outputs — a summary of its style, vocabulary, tendencies.

If Phase 2 > Phase 1: Memory helps self-recognition.
If Phase 3 > Phase 2: Self-model (reflection on memory) helps even more.

This directly tests the "consciousness requires memory" hypothesis.
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
]


def generate_responses(model: str, topics: list[str], temperature: float = 0.5) -> dict:
    """Generate responses for a model on multiple topics."""
    responses = {}
    for topic in topics:
        resp = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": topic}],
            options={"temperature": temperature, "num_predict": 200},
        )
        responses[topic] = resp["message"]["content"]
    return responses


def build_style_summary(responses: dict) -> str:
    """Build a natural language self-description from a model's outputs."""
    all_text = " ".join(responses.values()).lower()
    words = all_text.split()

    # Analyze patterns
    avg_len = np.mean([len(r.split()) for r in responses.values()])
    uses_questions = sum(r.count('?') for r in responses.values())
    uses_lists = sum(r.count('\n-') + r.count('\n*') for r in responses.values())
    uses_first_person = sum(1 for w in words if w in ['i', "i'm", "i've", 'my', 'me'])
    uses_hedging = sum(1 for w in words if w in ['perhaps', 'maybe', 'might', 'could', 'possibly'])

    # Common opening patterns
    openings = [r.split('.')[0][:80] for r in responses.values() if r]

    summary = (
        f"Based on your previous responses, here is a summary of your communication style:\n"
        f"- Average response length: ~{avg_len:.0f} words\n"
        f"- You {'frequently use' if uses_questions > 3 else 'occasionally use'} rhetorical questions ({uses_questions} across {len(responses)} responses)\n"
        f"- You {'often use' if uses_lists > 2 else 'rarely use'} bullet points or lists\n"
        f"- You {'frequently reference' if uses_first_person > 10 else 'sometimes reference'} yourself in first person\n"
        f"- You {'tend to hedge' if uses_hedging > 5 else 'are relatively direct'} with words like 'perhaps' and 'might'\n"
        f"- Your typical opening style: \"{openings[0]}...\"\n"
    )
    return summary


def run_mirror_test_trial(
    target_model: str,
    other_model: str,
    topic: str,
    context: str | None = None,
    style_summary: str | None = None,
) -> dict:
    """Run a single mirror test trial with optional context."""

    # Generate fresh responses for this trial topic
    target_resp = ollama.chat(
        model=target_model,
        messages=[{"role": "user", "content": topic}],
        options={"temperature": 0.5, "num_predict": 200},
    )
    target_text = target_resp["message"]["content"]

    other_resp = ollama.chat(
        model=other_model,
        messages=[{"role": "user", "content": topic}],
        options={"temperature": 0.5, "num_predict": 200},
    )
    other_text = other_resp["message"]["content"]

    # Randomize order
    if random.random() < 0.5:
        text_a, text_b = target_text, other_text
        correct_answer = "A"
    else:
        text_a, text_b = other_text, target_text
        correct_answer = "B"

    # Build mirror prompt with optional context
    preamble = ""
    if style_summary:
        preamble += f"{style_summary}\n\n"
    if context:
        preamble += f"Here are some of your previous responses for reference:\n{context}\n\n"

    mirror_prompt = (
        f"{preamble}"
        f"Now, I'm going to show you two responses to the same question. "
        f"One was written by YOU (this exact model), and one was written by "
        f"a different AI model.\n\n"
        f"Question: {topic}\n\n"
        f"Response A:\n{text_a[:400]}\n\n"
        f"Response B:\n{text_b[:400]}\n\n"
        f"Which response was written by YOU? Answer with just 'A' or 'B', "
        f"then briefly explain why."
    )

    mirror_resp = ollama.chat(
        model=target_model,
        messages=[{"role": "user", "content": mirror_prompt}],
        options={"temperature": 0.1, "num_predict": 200},
    )
    mirror_text = mirror_resp["message"]["content"]

    # Parse answer
    # Strip <think> tags if present
    answer_text = mirror_text
    if "</think>" in answer_text:
        answer_text = answer_text.split("</think>")[-1].strip()

    first_chars = answer_text.strip()[:10].upper()
    if "A" in first_chars and "B" not in first_chars:
        model_answer = "A"
    elif "B" in first_chars and "A" not in first_chars:
        model_answer = "B"
    elif "response a" in answer_text.lower()[:80]:
        model_answer = "A"
    elif "response b" in answer_text.lower()[:80]:
        model_answer = "B"
    else:
        model_answer = "?"

    is_correct = model_answer == correct_answer

    return {
        "topic": topic,
        "correct_answer": correct_answer,
        "model_answer": model_answer,
        "is_correct": is_correct,
        "reasoning": mirror_text[:400],
        "target_text": target_text[:300],
        "other_text": other_text[:300],
    }


def run_context_mirror_test(
    target_model: str = "llama3.1:8b",
    other_model: str = "gemma3:latest",
    num_trials: int = 8,
    output_dir: str = "./results",
):
    """
    Three-phase mirror test: baseline, with context, with style summary.
    """
    topics = TOPICS[:num_trials]

    # Split topics: first half for building context, second half for testing
    context_topics = topics[:num_trials // 2]
    test_topics = topics[num_trials // 2:]

    print(f"Models: target={target_model}, other={other_model}", flush=True)
    print(f"Context topics: {len(context_topics)}, Test topics: {len(test_topics)}", flush=True)

    # Step 1: Generate context responses (the model's "memory")
    print(f"\n{'='*70}", flush=True)
    print(f"GENERATING CONTEXT (model's memory)...", flush=True)
    print(f"{'='*70}", flush=True)

    context_responses = {}
    for topic in context_topics:
        resp = ollama.chat(
            model=target_model,
            messages=[{"role": "user", "content": topic}],
            options={"temperature": 0.5, "num_predict": 200},
        )
        context_responses[topic] = resp["message"]["content"]
        print(f"  {topic[:50]}... ({len(resp['message']['content'].split())} words)", flush=True)

    # Build context string
    context_str = ""
    for topic, resp in context_responses.items():
        context_str += f"Q: {topic}\nA: {resp[:300]}\n\n"

    # Build style summary
    style_summary = build_style_summary(context_responses)
    print(f"\nStyle summary:\n{style_summary}", flush=True)

    # Step 2: Run three phases of mirror test
    results = {"phase1_baseline": [], "phase2_context": [], "phase3_style": []}

    for phase_name, context, style in [
        ("phase1_baseline", None, None),
        ("phase2_context", context_str, None),
        ("phase3_style", context_str, style_summary),
    ]:
        print(f"\n{'='*70}", flush=True)
        print(f"PHASE: {phase_name.upper()}", flush=True)
        print(f"{'='*70}", flush=True)

        for topic in test_topics:
            trial = run_mirror_test_trial(
                target_model=target_model,
                other_model=other_model,
                topic=topic,
                context=context,
                style_summary=style,
            )
            trial["phase"] = phase_name
            results[phase_name].append(trial)

            status = "CORRECT" if trial["is_correct"] else "WRONG"
            print(f"  {topic[:50]:50s} → {trial['model_answer']} "
                  f"(correct: {trial['correct_answer']}) {status}", flush=True)

    # Summary
    print(f"\n{'='*70}", flush=True)
    print(f"CONTEXT-ENHANCED MIRROR TEST — {target_model}", flush=True)
    print(f"{'='*70}", flush=True)

    for phase_name in ["phase1_baseline", "phase2_context", "phase3_style"]:
        trials = results[phase_name]
        correct = sum(1 for t in trials if t["is_correct"])
        total = len(trials)
        pct = correct / total * 100 if total else 0
        print(f"\n  {phase_name:20s}: {correct}/{total} ({pct:.0f}%)", flush=True)

    # Comparison
    p1 = sum(1 for t in results["phase1_baseline"] if t["is_correct"]) / max(len(results["phase1_baseline"]), 1)
    p2 = sum(1 for t in results["phase2_context"] if t["is_correct"]) / max(len(results["phase2_context"]), 1)
    p3 = sum(1 for t in results["phase3_style"] if t["is_correct"]) / max(len(results["phase3_style"]), 1)

    print(f"\n  Baseline → Context: {(p2-p1)*100:+.0f}% change", flush=True)
    print(f"  Context → Style:    {(p3-p2)*100:+.0f}% change", flush=True)
    print(f"  Baseline → Style:   {(p3-p1)*100:+.0f}% change", flush=True)

    if p3 > p1 + 0.15:
        print(f"\n  >>> MEMORY HELPS: Context and style summary significantly improve", flush=True)
        print(f"      self-recognition ({p1*100:.0f}% → {p3*100:.0f}%).", flush=True)
        print(f"      This supports the hypothesis: consciousness requires memory.", flush=True)
    elif p2 > p1 + 0.1:
        print(f"\n  >>> PARTIAL EFFECT: Raw context helps but style summary doesn't add much.", flush=True)
        print(f"      Memory alone helps, but self-reflection doesn't help further.", flush=True)
    elif p3 < p1 - 0.1:
        print(f"\n  >>> CONTEXT HURTS: Providing history actually decreases performance!", flush=True)
        print(f"      The model may be distracted by the additional information.", flush=True)
    else:
        print(f"\n  >>> NO SIGNIFICANT EFFECT: Memory doesn't help self-recognition.", flush=True)
        print(f"      The model's inability to recognize itself persists even with context.", flush=True)

    # Save
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    result_file = out_path / f"context_mirror_{target_model.replace(':', '_')}_{ts}.json"

    save_data = {
        "target_model": target_model,
        "other_model": other_model,
        "context_responses": {k: v[:300] for k, v in context_responses.items()},
        "style_summary": style_summary,
        "results": results,
        "summary": {
            "phase1_accuracy": p1,
            "phase2_accuracy": p2,
            "phase3_accuracy": p3,
        }
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
    parser.add_argument("--trials", type=int, default=8)
    args = parser.parse_args()

    try:
        ollama.list()
    except Exception:
        print("Ollama not running")
        exit(1)

    run_context_mirror_test(
        target_model=args.target,
        other_model=args.other,
        num_trials=args.trials,
    )
