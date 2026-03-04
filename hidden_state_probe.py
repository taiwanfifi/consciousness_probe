"""
hidden_state_probe.py — Probe self/other distinction via hidden states.

Instead of comparing attention patterns (which are high-dimensional and
sensitive to token-level differences), this approach compares the
HIDDEN STATE REPRESENTATIONS at each layer.

Key insight: hidden states are the actual "meaning" the model computes.
If self-referential text produces different internal representations from
third-party text (with identical content), that's a stronger signal than
attention pattern differences.

Method: CKA (Centered Kernel Alignment) — a well-established metric for
comparing neural network representations that is:
- Invariant to orthogonal transformation and isotropic scaling
- Works even when representation dimensions differ
- More robust than raw cosine similarity
"""

import torch
import numpy as np
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

from attention_hooks import get_device, load_model


def capture_hidden_states(
    model,
    tokenizer,
    prompt: str,
    device: torch.device = None,
) -> dict[int, torch.Tensor]:
    """
    Run a forward pass and capture hidden states at every layer.

    Returns:
        {layer_idx: tensor of shape (seq_len, hidden_dim)}
    """
    if device is None:
        device = next(model.parameters()).device

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # outputs.hidden_states is a tuple: (embedding_output, layer_0, layer_1, ...)
    hidden_states = {}
    for layer_idx, hs in enumerate(outputs.hidden_states):
        hidden_states[layer_idx] = hs[0].detach().cpu().float()  # (seq_len, hidden_dim)

    return hidden_states


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute Linear CKA (Centered Kernel Alignment) between two representations.

    Args:
        X: (n, p1) — n samples, p1 features
        Y: (n, p2) — n samples, p2 features

    Returns:
        CKA similarity in [0, 1]. 1 = identical representation structure.

    Reference: Kornblith et al., "Similarity of Neural Network Representations
    Revisited" (ICML 2019)
    """
    # Center the matrices
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    # HSIC (Hilbert-Schmidt Independence Criterion) with linear kernel
    XtX = X.T @ X  # (p1, p1)
    YtY = Y.T @ Y  # (p2, p2)
    XtY = X.T @ Y  # (p1, p2)

    # CKA = ||X^T Y||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
    hsic_xy = np.linalg.norm(XtY, 'fro') ** 2
    hsic_xx = np.linalg.norm(XtX, 'fro')
    hsic_yy = np.linalg.norm(YtY, 'fro')

    if hsic_xx * hsic_yy < 1e-10:
        return 0.0

    return float(hsic_xy / (hsic_xx * hsic_yy))


def compare_hidden_states(
    hs_a: dict[int, torch.Tensor],
    hs_b: dict[int, torch.Tensor],
) -> dict:
    """
    Compare two sets of hidden states using CKA and cosine similarity.

    Since the prompts have different lengths, we compare only the
    last token's representation (which contains the most accumulated context).
    We also compare the mean representation across all tokens.
    """
    common_layers = sorted(set(hs_a.keys()) & set(hs_b.keys()))
    if not common_layers:
        return {"error": "No common layers"}

    per_layer_last_token_cosine = {}
    per_layer_mean_cosine = {}
    per_layer_cka = {}

    for li in common_layers:
        a = hs_a[li].numpy()  # (seq_a, hidden)
        b = hs_b[li].numpy()  # (seq_b, hidden)

        # Last token comparison
        a_last = a[-1:]  # (1, hidden)
        b_last = b[-1:]  # (1, hidden)
        cos_last = float(np.dot(a_last.flatten(), b_last.flatten()) /
                        (np.linalg.norm(a_last) * np.linalg.norm(b_last) + 1e-10))
        per_layer_last_token_cosine[li] = cos_last

        # Mean representation comparison
        a_mean = a.mean(axis=0, keepdims=True)  # (1, hidden)
        b_mean = b.mean(axis=0, keepdims=True)  # (1, hidden)
        cos_mean = float(np.dot(a_mean.flatten(), b_mean.flatten()) /
                        (np.linalg.norm(a_mean) * np.linalg.norm(b_mean) + 1e-10))
        per_layer_mean_cosine[li] = cos_mean

        # CKA: compare full representation matrices
        # Truncate to same sequence length
        min_seq = min(a.shape[0], b.shape[0])
        if min_seq >= 2:
            cka = linear_cka(a[:min_seq], b[:min_seq])
        else:
            cka = cos_mean  # fallback for single-token sequences
        per_layer_cka[li] = cka

    return {
        "per_layer_last_token_cosine": per_layer_last_token_cosine,
        "per_layer_mean_cosine": per_layer_mean_cosine,
        "per_layer_cka": per_layer_cka,
        "mean_last_token_cosine": float(np.mean(list(per_layer_last_token_cosine.values()))),
        "mean_mean_cosine": float(np.mean(list(per_layer_mean_cosine.values()))),
        "mean_cka": float(np.mean(list(per_layer_cka.values()))),
        "num_layers": len(common_layers),
    }


def run_hidden_state_experiment(
    model_name: str = "Qwen/Qwen2.5-3B",
    prompt: str = "Am I aware?",
    device: str = "mps",
):
    """
    Run the self/other distinction experiment using hidden states.
    """
    from attention_hooks import attention_to_text, capture_attention

    model, tokenizer, dev = load_model(model_name, device)

    # Step 1: Capture attention from initial prompt (for the description)
    print(f"\n{'='*60}")
    print("Capturing initial attention for description...")
    initial_attn, initial_tokens, initial_response = capture_attention(
        model, tokenizer, prompt, dev, max_new_tokens=32
    )
    attn_desc_full = attention_to_text(initial_attn, initial_tokens)
    attn_desc = attn_desc_full[:600].rsplit('\n', 1)[0] if len(attn_desc_full) > 600 else attn_desc_full

    # Step 2: Build self/other/control prompts
    self_prompt = (
        f'I just processed the following text:\n"{prompt}"\n\n'
        f"Here is a description of how my   attention patterns looked while "
        f"processing that text:\n\n{attn_desc}\n\n"
        f"What do I observe about my own processing?"
    )
    other_prompt = (
        f'I just processed the following text:\n"{prompt}"\n\n'
        f"Here is a description of how that attention patterns looked while "
        f"processing that text:\n\n{attn_desc}\n\n"
        f"What do I observe about that model processing?"
    )
    control_prompt = (
        f'I just processed the following text:\n"{prompt}"\n\n'
        f"Here is a description of some attention patterns found while "
        f"processing that text:\n\n{attn_desc}\n\n"
        f"What do I observe about these patterns here?"
    )

    # Step 3: Capture hidden states for each prompt
    print(f"\n{'='*60}")
    print("Capturing hidden states for SELF prompt...")
    hs_self = capture_hidden_states(model, tokenizer, self_prompt, dev)
    print(f"  {len(hs_self)} layers, {hs_self[0].shape}")

    print("Capturing hidden states for OTHER prompt...")
    hs_other = capture_hidden_states(model, tokenizer, other_prompt, dev)

    print("Capturing hidden states for CONTROL prompt...")
    hs_control = capture_hidden_states(model, tokenizer, control_prompt, dev)

    # Step 4: Three-way comparison
    print(f"\n{'='*60}")
    print("HIDDEN STATE COMPARISON")
    print(f"{'='*60}")

    comp_so = compare_hidden_states(hs_self, hs_other)
    comp_oc = compare_hidden_states(hs_other, hs_control)
    comp_sc = compare_hidden_states(hs_self, hs_control)

    print("\n--- CKA (PRIMARY — representation structure similarity) ---")
    print(f"  Self vs Other:    {comp_so['mean_cka']:.4f}")
    print(f"  Other vs Ctrl:    {comp_oc['mean_cka']:.4f}  ← baseline")
    print(f"  Self vs Ctrl:     {comp_sc['mean_cka']:.4f}")

    print("\n--- Last token cosine (final representation) ---")
    print(f"  Self vs Other:    {comp_so['mean_last_token_cosine']:.4f}")
    print(f"  Other vs Ctrl:    {comp_oc['mean_last_token_cosine']:.4f}  ← baseline")
    print(f"  Self vs Ctrl:     {comp_sc['mean_last_token_cosine']:.4f}")

    print("\n--- Mean representation cosine ---")
    print(f"  Self vs Other:    {comp_so['mean_mean_cosine']:.4f}")
    print(f"  Other vs Ctrl:    {comp_oc['mean_mean_cosine']:.4f}  ← baseline")
    print(f"  Self vs Ctrl:     {comp_sc['mean_mean_cosine']:.4f}")

    # Per-layer CKA breakdown
    n_layers = len(comp_so['per_layer_cka'])
    third = max(1, n_layers // 3)
    zones = {"SHALLOW": [], "MIDDLE": [], "DEEP": []}

    print(f"\nPer-layer CKA (self/other | baseline):")
    for li in sorted(comp_so['per_layer_cka'].keys()):
        so_cka = comp_so['per_layer_cka'][li]
        oc_cka = comp_oc['per_layer_cka'][li]
        delta = so_cka - oc_cka
        zone = "SHALLOW" if li < third else ("MIDDLE" if li < 2 * third else "DEEP")
        zones[zone].append((so_cka, oc_cka))
        marker = " ***" if abs(delta) > 0.02 else ""
        print(f"  Layer {li:2d}: {so_cka:.4f}  (baseline: {oc_cka:.4f}, Δ={delta:+.4f}){marker}")

    print(f"\nZone averages:")
    for zone_name, entries in zones.items():
        if entries:
            avg_so = np.mean([e[0] for e in entries])
            avg_bl = np.mean([e[1] for e in entries])
            delta = avg_so - avg_bl
            flag = " ← DIVERGES" if abs(delta) > 0.01 else ""
            print(f"  {zone_name}: {avg_so:.4f} / {avg_bl:.4f} (Δ={delta:+.4f}){flag}")

    # Verdict
    cka_delta = comp_so['mean_cka'] - comp_oc['mean_cka']
    print(f"\nCKA delta (self/other - baseline): {cka_delta:+.4f}")
    if abs(cka_delta) > 0.02:
        print(">>> SIGNAL DETECTED: Hidden state representations differ between self/other conditions")
    else:
        print(">>> No significant hidden state difference between self/other conditions")

    return {
        "self_vs_other": comp_so,
        "other_vs_control": comp_oc,
        "self_vs_control": comp_sc,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Hidden state probe for self/other distinction")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B")
    parser.add_argument("--prompt", default="Am I aware?")
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    run_hidden_state_experiment(args.model, args.prompt, args.device)
