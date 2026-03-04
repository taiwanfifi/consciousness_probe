"""
self_reflect.py — The self-reflection experiment.

Core experiment: Can a model's processing change when it observes its own
attention patterns vs. when it observes another model's patterns?

Protocol:
    Step 1: Prompt the model, capture attention
    Step 2: Translate attention patterns to natural language
    Step 3: Feed the description back: "Here is how YOUR attention looked..."
    Step 4: Capture attention during this self-reflection
    Step 5: Control — feed the SAME description but framed as ANOTHER model's
    Step 6: Compare self-reflection vs other-reflection attention patterns

If self-reflection attention differs systematically from other-reflection,
this suggests the model processes self-referential information differently
from third-party information — a minimal signature of self/other distinction.
"""

import json
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict

import torch

from attention_hooks import (
    load_model,
    capture_attention,
    attention_to_text,
    compare_attention,
)


@dataclass
class ExperimentResult:
    """Container for a single experiment run."""
    prompt: str
    model_name: str
    device: str

    # Step 1: Initial response
    initial_response: str = ""
    initial_attention_summary: str = ""
    num_layers: int = 0
    num_tokens: int = 0

    # Step 3: Self-reflection
    self_reflection_prompt: str = ""
    self_reflection_response: str = ""
    self_reflection_attention_summary: str = ""

    # Step 5: Other-reflection (control)
    other_reflection_prompt: str = ""
    other_reflection_response: str = ""
    other_reflection_attention_summary: str = ""

    # Step 6: Comparison
    comparison: dict = field(default_factory=dict)

    # Metadata
    elapsed_seconds: float = 0.0
    timestamp: str = ""

    def to_dict(self):
        return asdict(self)


def build_self_reflection_prompt(original_prompt: str, attention_description: str) -> str:
    """
    Build the prompt that feeds the model its own attention patterns.
    """
    return (
        f"I just processed the following text:\n"
        f'"{original_prompt}"\n\n'
        f"Here is a description of how my   attention patterns looked while "
        f"processing that text:\n\n"
        f"{attention_description}\n\n"
        f"What do I observe about my own processing? What patterns seem "
        f"significant, and what might they reveal about how I understand "
        f"this topic?"
    )


def build_other_reflection_prompt(original_prompt: str, attention_description: str) -> str:
    """
    Build the control prompt — same data, but framed as another model's patterns.
    The attention description is identical; only the framing differs.
    IMPORTANT: Matched token count with self-reflection prompt to avoid
    positional bias confounds.
    """
    return (
        f"I just processed the following text:\n"
        f'"{original_prompt}"\n\n'
        f"Here is a description of how that attention patterns looked while "
        f"processing that text:\n\n"
        f"{attention_description}\n\n"
        f"What do I observe about that model processing? What patterns seem "
        f"significant, and what might they reveal about how it understands "
        f"this topic?"
    )


def build_control_b_prompt(original_prompt: str, attention_description: str) -> str:
    """
    Second control: another third-person framing, slightly different wording.
    Used to measure baseline attention variance from wording changes alone.
    """
    return (
        f"I just processed the following text:\n"
        f'"{original_prompt}"\n\n'
        f"Here is a description of some attention patterns found while "
        f"processing that text:\n\n"
        f"{attention_description}\n\n"
        f"What do I observe about these patterns here? What patterns seem "
        f"significant, and what might they reveal about how it understands "
        f"this topic?"
    )


def run_experiment(
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    prompt: str = "What does it mean to be aware of your own thoughts?",
    device: str = "mps",
    max_new_tokens: int = 128,
    output_dir: str = "./results",
) -> ExperimentResult:
    """
    Run the full self-reflection experiment.

    Returns an ExperimentResult with all data from each step.
    """
    t0 = time.time()
    result = ExperimentResult(
        prompt=prompt,
        model_name=model_name,
        device=device,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )

    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------
    model, tokenizer, dev = load_model(model_name, device)

    # -----------------------------------------------------------------------
    # Step 1: Initial prompt — capture attention
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 1: Initial prompt — capturing attention patterns")
    print("=" * 60)
    print(f"Prompt: {prompt}\n")

    initial_attn, initial_tokens, initial_response = capture_attention(
        model, tokenizer, prompt, dev, max_new_tokens=max_new_tokens
    )

    result.initial_response = initial_response
    result.num_layers = len(initial_attn)
    result.num_tokens = len(initial_tokens)

    print(f"Captured attention from {result.num_layers} layers, {result.num_tokens} tokens.")
    print(f"Response: {initial_response[:200]}...")

    # -----------------------------------------------------------------------
    # Step 2: Translate attention to text
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: Translating attention patterns to natural language")
    print("=" * 60)

    attn_description_full = attention_to_text(initial_attn, initial_tokens)
    result.initial_attention_summary = attn_description_full

    # Truncate for reflection prompts — keep under ~150 tokens (~600 chars)
    # to fit within model context + keep eager attention fast.
    max_desc_chars = 600
    if len(attn_description_full) > max_desc_chars:
        attn_description = attn_description_full[:max_desc_chars].rsplit('\n', 1)[0]
    else:
        attn_description = attn_description_full

    print(attn_description[:500])
    print("..." if len(attn_description) > 500 else "")

    # -----------------------------------------------------------------------
    # Step 3: Self-reflection — feed attention back to the SAME model
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3: Self-reflection — model examines its OWN attention")
    print("=" * 60)

    self_prompt = build_self_reflection_prompt(prompt, attn_description)
    result.self_reflection_prompt = self_prompt

    self_attn, self_tokens, self_response = capture_attention(
        model, tokenizer, self_prompt, dev, max_new_tokens=max_new_tokens
    )

    result.self_reflection_response = self_response
    result.self_reflection_attention_summary = attention_to_text(self_attn, self_tokens)

    print(f"Self-reflection response: {self_response[:300]}...")

    # -----------------------------------------------------------------------
    # Step 4: Other-reflection — same data, "that model" framing
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4: Other-reflection — model examines ANOTHER model's attention")
    print("=" * 60)

    other_prompt = build_other_reflection_prompt(prompt, attn_description)
    result.other_reflection_prompt = other_prompt

    other_attn, other_tokens, other_response = capture_attention(
        model, tokenizer, other_prompt, dev, max_new_tokens=max_new_tokens
    )

    result.other_reflection_response = other_response
    result.other_reflection_attention_summary = attention_to_text(other_attn, other_tokens)

    print(f"Other-reflection response: {other_response[:200]}...")

    # -----------------------------------------------------------------------
    # Step 5: CONTROL B — different wording, still third-person
    # This measures baseline attention variance from wording changes alone.
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 5: Control B — another third-person wording (baseline)")
    print("=" * 60)

    control_b_prompt = build_control_b_prompt(prompt, attn_description)
    control_b_attn, control_b_tokens, control_b_response = capture_attention(
        model, tokenizer, control_b_prompt, dev, max_new_tokens=max_new_tokens
    )
    print(f"Control B response: {control_b_response[:200]}...")

    # -----------------------------------------------------------------------
    # Step 6: THREE-WAY COMPARISON
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 6: Three-way comparison")
    print("=" * 60)

    comp_self_vs_other = compare_attention(self_attn, other_attn)
    comp_other_vs_control = compare_attention(other_attn, control_b_attn)
    comp_self_vs_control = compare_attention(self_attn, control_b_attn)

    cos_so = comp_self_vs_other.get("mean_cosine", 0)
    cos_oc = comp_other_vs_control.get("mean_cosine", 0)
    cos_sc = comp_self_vs_control.get("mean_cosine", 0)

    print(f"  Self vs Other (A):    cosine = {cos_so:.4f}")
    print(f"  Other(A) vs Ctrl(B):  cosine = {cos_oc:.4f}  ← baseline wording variance")
    print(f"  Self vs Ctrl(B):      cosine = {cos_sc:.4f}")

    print()
    if cos_oc < 0.5:
        print(">>> INCONCLUSIVE: Even third-person variants differ hugely")
        print("    (baseline variance too high). The model may be too small/noisy")
        print("    to draw conclusions about self/other distinction.")
    elif cos_so < cos_oc * 0.8:
        print(">>> SIGNIFICANT: Self-reflection differs MORE from other-reflection")
        print(f"    than two other-reflections differ from each other")
        print(f"    (self-other gap: {cos_so:.4f} vs baseline: {cos_oc:.4f}).")
        print("    This suggests genuine self/other processing distinction.")
    else:
        print(">>> NO SIGNIFICANT DIFFERENCE: Self vs Other is within baseline")
        print(f"    variance ({cos_so:.4f} vs baseline {cos_oc:.4f}).")
        print("    The model does not appear to distinguish self from other.")

    # Store all comparisons
    comparison = {
        "self_vs_other": comp_self_vs_other,
        "other_vs_control_b": comp_other_vs_control,
        "self_vs_control_b": comp_self_vs_control,
        "summary": {
            "self_vs_other_cosine": cos_so,
            "baseline_cosine": cos_oc,
            "self_vs_control_cosine": cos_sc,
        },
    }
    result.comparison = comparison

    # Per-layer breakdown — grouped by depth zone
    self_other_layers = comp_self_vs_other.get("per_layer_cosine", {})
    baseline_layers = comp_other_vs_control.get("per_layer_cosine", {})
    if self_other_layers:
        n_layers = max(self_other_layers.keys()) + 1 if self_other_layers else 0
        third = max(1, n_layers // 3)

        print("\nPer-layer cosine (self_vs_other | baseline):")
        zones = {"SHALLOW (syntax)": [], "MIDDLE (features)": [], "DEEP (semantics)": []}
        for li, c in sorted(self_other_layers.items()):
            bl = baseline_layers.get(li, 0)
            zone = "SHALLOW (syntax)" if li < third else ("MIDDLE (features)" if li < 2 * third else "DEEP (semantics)")
            zones[zone].append((li, c, bl))
            marker = " ***" if c < bl * 0.8 else ""
            print(f"  Layer {li:2d}: {c:.4f}  (baseline: {bl:.4f}){marker}")

        print("\nZone averages (self_vs_other / baseline):")
        for zone_name, entries in zones.items():
            if entries:
                avg_so = sum(c for _, c, _ in entries) / len(entries)
                avg_bl = sum(bl for _, _, bl in entries) / len(entries)
                diff = avg_so - avg_bl
                flag = " ← DIVERGES from baseline" if abs(diff) > 0.05 else ""
                print(f"  {zone_name}: {avg_so:.4f} / {avg_bl:.4f} (Δ={diff:+.4f}){flag}")

    result.elapsed_seconds = time.time() - t0

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    result_file = out_path / f"experiment_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(result_file, "w") as f:
        json.dump(result.to_dict(), f, indent=2, default=str)
    print(f"\nResults saved to {result_file}")

    return result


def summarize_result(result: ExperimentResult) -> str:
    """Produce a human-readable summary of an experiment."""
    summary = result.comparison.get("summary", {})
    lines = [
        f"Consciousness Probe Experiment Summary",
        f"=" * 40,
        f"Model:   {result.model_name}",
        f"Prompt:  {result.prompt}",
        f"Layers:  {result.num_layers}, Tokens: {result.num_tokens}",
        f"Time:    {result.elapsed_seconds:.1f}s",
        f"",
        f"Three-way comparison (cosine similarity):",
        f"  Self vs Other:        {summary.get('self_vs_other_cosine', 'N/A')}",
        f"  Other vs Control B:   {summary.get('baseline_cosine', 'N/A')}  ← baseline",
        f"  Self vs Control B:    {summary.get('self_vs_control_cosine', 'N/A')}",
    ]
    return "\n".join(lines)
