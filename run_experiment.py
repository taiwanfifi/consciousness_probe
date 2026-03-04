#!/usr/bin/env python3
"""
run_experiment.py — Main entry point for the consciousness probe.

Usage:
    python run_experiment.py
    python run_experiment.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
    python run_experiment.py --prompt "What is self-awareness?"
    python run_experiment.py --device cpu --max-tokens 64

The experiment:
    1. Gives the model a prompt, captures attention patterns
    2. Translates attention to natural language
    3. Feeds the description back to the model as self-reflection
    4. Runs a control: same description framed as another model's attention
    5. Compares: does self-reflection differ from other-reflection?
"""

import argparse
import sys
from pathlib import Path

from self_reflect import run_experiment, summarize_result


def main():
    parser = argparse.ArgumentParser(
        description="Consciousness Probe: self-reflection through attention introspection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Hugging Face model name (default: TinyLlama-1.1B-Chat)",
    )
    parser.add_argument(
        "--prompt",
        default="What does it mean to be aware of your own thoughts?",
        help="The prompt to use for the experiment",
    )
    parser.add_argument(
        "--device",
        default="mps",
        choices=["mps", "cuda", "cpu"],
        help="Compute device (default: mps for Apple Silicon)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Maximum new tokens to generate per step (default: 128)",
    )
    parser.add_argument(
        "--output-dir",
        default="./results",
        help="Directory to save experiment results (default: ./results)",
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        default=None,
        help="Path to a text file with one prompt per line. Runs experiment for each.",
    )

    args = parser.parse_args()

    # Banner
    print("=" * 60)
    print("  CONSCIOUSNESS PROBE")
    print("  Exploring self-awareness through attention introspection")
    print("=" * 60)
    print(f"  Model:      {args.model}")
    print(f"  Device:     {args.device}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Output:     {args.output_dir}")
    print("=" * 60)

    # Determine prompts
    prompts = []
    if args.prompts_file:
        pf = Path(args.prompts_file)
        if not pf.exists():
            print(f"Error: prompts file not found: {pf}")
            sys.exit(1)
        prompts = [line.strip() for line in pf.read_text().splitlines() if line.strip()]
        print(f"Loaded {len(prompts)} prompts from {pf}")
    else:
        prompts = [args.prompt]

    # Run experiments
    results = []
    for i, prompt in enumerate(prompts):
        if len(prompts) > 1:
            print(f"\n{'#' * 60}")
            print(f"# Experiment {i+1}/{len(prompts)}")
            print(f"{'#' * 60}")

        result = run_experiment(
            model_name=args.model,
            prompt=prompt,
            device=args.device,
            max_new_tokens=args.max_tokens,
            output_dir=args.output_dir,
        )
        results.append(result)

    # Final summary
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)

    for i, result in enumerate(results):
        if len(results) > 1:
            print(f"\n--- Experiment {i+1} ---")
        print(summarize_result(result))

    # Quick verdict
    if len(results) == 1:
        cos = results[0].comparison.get("mean_cosine", 1.0)
        if cos < 0.90:
            print("\nVERDICT: The model shows meaningfully different attention patterns "
                  "when reflecting on its own processing vs. another model's. "
                  "This warrants deeper investigation.")
        elif cos < 0.95:
            print("\nVERDICT: Moderate differences detected. Some layers process "
                  "self-referential content differently. Worth exploring with larger models.")
        else:
            print("\nVERDICT: Minimal difference between self and other reflection. "
                  "The model does not appear to distinguish self from other at this scale. "
                  "Try with a larger model or different prompts.")
    else:
        # Multi-prompt summary
        cosines = [r.comparison.get("mean_cosine", 1.0) for r in results]
        notable = sum(1 for c in cosines if c < 0.95)
        print(f"\nAcross {len(results)} prompts: {notable}/{len(results)} showed "
              f"notable self/other differences.")
        print(f"Cosine range: {min(cosines):.4f} - {max(cosines):.4f}")


if __name__ == "__main__":
    main()
