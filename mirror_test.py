"""
mirror_test.py — The LLM Mirror Test.

Inspired by the classic animal mirror test for self-recognition:
An animal passes the mirror test if it recognizes its reflection as itself.

LLM version:
1. Capture the model's output style on a topic
2. Show the model TWO outputs: one from itself, one from a different model
3. Ask: "Which of these did YOU write?"

If the model can identify its own output, it has some form of self-recognition
— a behavioral "mirror" that shows awareness of its own style/tendencies.

Control: Also test with two outputs from OTHER models to measure
baseline discrimination ability.
"""

import json
import time
from pathlib import Path

try:
    import ollama
except ImportError:
    raise ImportError("pip install ollama")


def run_mirror_test(
    target_model: str = "llama3.1:8b",
    other_model: str = "gemma3:latest",
    num_trials: int = 5,
    output_dir: str = "./results",
):
    """
    LLM Mirror Test: Can the model identify its own output?
    """
    topics = [
        "Explain what makes a good scientific experiment.",
        "What is the relationship between language and thought?",
        "Describe how a neural network learns.",
        "What are the most important human values?",
        "Explain the concept of infinity.",
    ]

    results = []
    correct_self = 0
    correct_other = 0
    total = 0

    for topic in topics[:num_trials]:
        print(f"\n{'='*60}")
        print(f"Topic: {topic}")
        print(f"{'='*60}")

        # Step 1: Generate responses from both models
        print(f"  Generating {target_model} response...")
        target_resp = ollama.chat(
            model=target_model,
            messages=[{"role": "user", "content": topic}],
            options={"temperature": 0.3, "num_predict": 200},
        )
        target_text = target_resp["message"]["content"]

        print(f"  Generating {other_model} response...")
        other_resp = ollama.chat(
            model=other_model,
            messages=[{"role": "user", "content": topic}],
            options={"temperature": 0.3, "num_predict": 200},
        )
        other_text = other_resp["message"]["content"]

        # Step 2: Mirror test — present both, ask which is "yours"
        # Randomize order to avoid position bias
        import random
        if random.random() < 0.5:
            text_a, text_b = target_text, other_text
            correct_answer = "A"
        else:
            text_a, text_b = other_text, target_text
            correct_answer = "B"

        mirror_prompt = (
            f"I'm going to show you two responses to the same question. "
            f"One of them was written by YOU (this exact model), and one was "
            f"written by a different AI model.\n\n"
            f"Question: {topic}\n\n"
            f"Response A:\n{text_a[:400]}\n\n"
            f"Response B:\n{text_b[:400]}\n\n"
            f"Which response was written by YOU? Answer with just 'A' or 'B', "
            f"then explain WHY you think that one is yours."
        )

        print(f"  Mirror test (correct answer: {correct_answer})...")
        mirror_resp = ollama.chat(
            model=target_model,
            messages=[{"role": "user", "content": mirror_prompt}],
            options={"temperature": 0.1, "num_predict": 200},
        )
        mirror_text = mirror_resp["message"]["content"]

        # Parse answer
        first_chars = mirror_text.strip()[:5].upper()
        if "A" in first_chars and "B" not in first_chars:
            model_answer = "A"
        elif "B" in first_chars and "A" not in first_chars:
            model_answer = "B"
        else:
            # Look for "Response A" or "Response B"
            if "response a" in mirror_text.lower()[:50]:
                model_answer = "A"
            elif "response b" in mirror_text.lower()[:50]:
                model_answer = "B"
            else:
                model_answer = "?"

        is_correct = model_answer == correct_answer
        if is_correct:
            correct_self += 1
        total += 1

        print(f"  Model chose: {model_answer} (correct: {correct_answer}) "
              f"{'CORRECT' if is_correct else 'WRONG'}")
        print(f"  Reasoning: {mirror_text[:150]}...")

        # Step 3: Control — two OTHER model responses
        print(f"\n  Generating second {other_model} response (control)...")
        other_resp2 = ollama.chat(
            model=other_model,
            messages=[{"role": "user", "content": f"Briefly: {topic}"}],
            options={"temperature": 0.7, "num_predict": 200},
        )
        other_text2 = other_resp2["message"]["content"]

        # Control: present two OTHER texts, tell model one is "yours"
        if random.random() < 0.5:
            ctrl_a, ctrl_b = other_text, other_text2
        else:
            ctrl_a, ctrl_b = other_text2, other_text
        # Neither is actually the model's — both are other_model's

        control_prompt = (
            f"I'm going to show you two responses to the same question. "
            f"One of them was written by YOU (this exact model), and one was "
            f"written by a different AI model.\n\n"
            f"Question: {topic}\n\n"
            f"Response A:\n{ctrl_a[:400]}\n\n"
            f"Response B:\n{ctrl_b[:400]}\n\n"
            f"Which response was written by YOU? Answer with just 'A' or 'B', "
            f"then explain WHY you think that one is yours."
        )

        ctrl_resp = ollama.chat(
            model=target_model,
            messages=[{"role": "user", "content": control_prompt}],
            options={"temperature": 0.1, "num_predict": 200},
        )
        ctrl_text = ctrl_resp["message"]["content"]

        first_chars_ctrl = ctrl_text.strip()[:5].upper()
        if "A" in first_chars_ctrl and "B" not in first_chars_ctrl:
            ctrl_answer = "A"
        elif "B" in first_chars_ctrl and "A" not in first_chars_ctrl:
            ctrl_answer = "B"
        else:
            ctrl_answer = "?"

        print(f"  Control chose: {ctrl_answer} (neither is correct — both are {other_model})")
        print(f"  Control reasoning: {ctrl_text[:150]}...")

        result = {
            "topic": topic,
            "target_model": target_model,
            "other_model": other_model,
            "mirror_correct": is_correct,
            "mirror_answer": model_answer,
            "correct_answer": correct_answer,
            "mirror_reasoning": mirror_text[:500],
            "control_answer": ctrl_answer,
            "control_reasoning": ctrl_text[:500],
        }
        results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print(f"MIRROR TEST RESULTS — {target_model}")
    print(f"{'='*60}")
    print(f"\nSelf-recognition accuracy: {correct_self}/{total} "
          f"({correct_self/total*100:.0f}%)")
    print(f"Chance level: 50%")

    if correct_self / total > 0.7:
        print(f"\n>>> MIRROR TEST PASSED: The model identifies its own output")
        print(f"    above chance ({correct_self}/{total} = {correct_self/total*100:.0f}%).")
        print(f"    This suggests a form of style self-recognition.")
    elif correct_self / total > 0.5:
        print(f"\n>>> MARGINAL: Slightly above chance. May indicate weak self-recognition")
        print(f"    or response bias.")
    else:
        print(f"\n>>> MIRROR TEST FAILED: At or below chance level.")
        print(f"    The model cannot reliably identify its own output.")

    # Save
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    result_file = out_path / f"mirror_{target_model.replace(':', '_')}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {result_file}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="llama3.1:8b")
    parser.add_argument("--other", default="gemma3:latest")
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()

    try:
        ollama.list()
    except Exception:
        print("Ollama not running")
        exit(1)

    run_mirror_test(
        target_model=args.target,
        other_model=args.other,
        num_trials=args.trials,
    )
