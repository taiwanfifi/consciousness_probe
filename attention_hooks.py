"""
attention_hooks.py — Core attention capture for consciousness probe.

Loads a Hugging Face model, registers forward hooks on all attention layers,
and provides utilities to capture and interpret attention patterns.

Designed for Apple M3 Max (MPS backend) with CPU fallback.
"""

import torch
import numpy as np
from typing import Optional
from contextlib import contextmanager
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_device(preferred: str = "mps") -> torch.device:
    """Resolve device with fallback chain: preferred → mps → cuda → cpu."""
    if preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred not in ("mps", "cuda"):
        # Allow explicit cpu or other
        return torch.device(preferred)
    # Fallback
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device: str = "mps",
    dtype: Optional[torch.dtype] = None,
):
    """
    Load model and tokenizer, move to device.

    Returns:
        (model, tokenizer, device)
    """
    dev = get_device(device)
    if dtype is None:
        # fp16 for GPU, fp32 for CPU
        dtype = torch.float32 if dev.type == "cpu" else torch.float16

    print(f"Loading {model_name} on {dev} ({dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        attn_implementation="eager",  # Required for output_attentions=True
    )
    model = model.to(dev)
    model.eval()
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model, tokenizer, dev


# ---------------------------------------------------------------------------
# Hook-based attention capture
# ---------------------------------------------------------------------------

class AttentionCapture:
    """
    Context manager that registers forward hooks on all attention layers
    to capture attention weight tensors.

    Usage:
        with AttentionCapture(model) as cap:
            outputs = model(**inputs)
        attention_data = cap.attention  # {layer_idx: tensor}
    """

    def __init__(self, model):
        self.model = model
        self.attention: dict[int, torch.Tensor] = {}
        self._hooks = []

    def _find_attention_modules(self):
        """
        Find all attention modules in the model.
        Supports common architectures: LlamaAttention, GPT2Attention, etc.
        """
        attn_modules = []
        for name, module in self.model.named_modules():
            cls_name = type(module).__name__.lower()
            # Match common attention class names across architectures
            if "attention" in cls_name and "selfattention" not in cls_name.replace("_", ""):
                # Skip wrapper modules that contain the actual attention sublayer
                # We want the innermost attention module
                has_attn_child = any(
                    "attention" in type(c).__name__.lower()
                    for c in module.children()
                )
                if not has_attn_child:
                    attn_modules.append((name, module))
        return attn_modules

    def __enter__(self):
        attn_modules = self._find_attention_modules()
        if not attn_modules:
            raise RuntimeError(
                "No attention modules found. The model architecture may not be supported. "
                "Ensure attn_implementation='eager' and output_attentions=True."
            )
        for idx, (name, module) in enumerate(attn_modules):
            layer_idx = idx

            def make_hook(li):
                def hook_fn(mod, inp, out):
                    # Most HF attention modules return (attn_output, attn_weights, ...)
                    # attn_weights shape: (batch, num_heads, seq_len, seq_len)
                    if isinstance(out, tuple) and len(out) >= 2:
                        attn_weights = out[1]
                        if attn_weights is not None:
                            self.attention[li] = attn_weights.detach().cpu()
                return hook_fn

            h = module.register_forward_hook(make_hook(layer_idx))
            self._hooks.append(h)

        print(f"Registered hooks on {len(attn_modules)} attention layers.")
        return self

    def __exit__(self, *args):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ---------------------------------------------------------------------------
# High-level capture function
# ---------------------------------------------------------------------------

def capture_attention(
    model,
    tokenizer,
    prompt: str,
    device: torch.device = None,
    max_new_tokens: int = 128,
) -> tuple[dict[int, torch.Tensor], list[str], str]:
    """
    Run a prompt through the model and capture attention patterns from the
    PREFILL pass (full seq×seq matrix), plus generate text separately.

    Args:
        model: HF causal LM model (with output_attentions=True)
        tokenizer: Corresponding tokenizer
        prompt: Input text
        device: Device (inferred from model if None)
        max_new_tokens: How many tokens to generate for the response text

    Returns:
        (attention_data, tokens, generated_text)
        - attention_data: {layer_idx: tensor of shape (batch, heads, seq, seq)}
        - tokens: list of token strings (from prompt only)
        - generated_text: the model's full output
    """
    if device is None:
        device = next(model.parameters()).device

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    # Step 1: Plain forward pass with output_attentions to get attention matrices
    # This works with SDPA/flash attention in newer transformers versions
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # outputs.attentions is a tuple of (batch, heads, seq, seq) per layer
    attention_data = {}
    if outputs.attentions:
        for layer_idx, attn_tensor in enumerate(outputs.attentions):
            attention_data[layer_idx] = attn_tensor.detach().cpu()

    # Step 2: Separate generation for the response text
    with torch.no_grad():
        gen_outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    generated_text = tokenizer.decode(gen_outputs[0], skip_special_tokens=True)

    return attention_data, tokens, generated_text


# ---------------------------------------------------------------------------
# Attention-to-text translation
# ---------------------------------------------------------------------------

def attention_to_text(
    attention_data: dict[int, torch.Tensor],
    tokens: list[str],
    top_k: int = 3,
    min_weight: float = 0.15,
    max_layers_reported: int = 6,
) -> str:
    """
    Translate attention patterns into natural language descriptions.

    For each layer, identifies the strongest cross-token attention patterns
    (excluding self-attention on the diagonal) and describes them in plain text.

    Args:
        attention_data: {layer_idx: tensor (batch, heads, seq, seq)}
        tokens: List of token strings corresponding to the sequence
        top_k: Number of top attention patterns to report per layer
        min_weight: Minimum attention weight to consider noteworthy

    Returns:
        Multi-line string describing the attention patterns.
    """
    if not attention_data:
        return "No attention data captured."

    descriptions = []
    num_layers = len(attention_data)
    seq_len = len(tokens)

    # Clean token strings for readability
    clean_tokens = [t.replace("\u2581", "").replace("##", "") or "<sp>" for t in tokens]

    descriptions.append(f"Attention analysis over {num_layers} layers, {seq_len} tokens.\n")

    # Summarize by layer (report only most interesting layers to keep description compact)
    # First pass: find layers with strongest patterns
    layer_max_weights = {}
    for layer_idx in sorted(attention_data.keys()):
        attn = attention_data[layer_idx]
        if attn is None:
            continue
        attn_np = attn[0].float().numpy()
        # Zero out diagonal (self-attention)
        import numpy as _np
        mask = 1 - _np.eye(attn_np.shape[-1])
        masked = attn_np * mask[None, :, :]
        layer_max_weights[layer_idx] = float(masked.max())

    # Select top layers by max attention weight
    top_layers = sorted(layer_max_weights, key=layer_max_weights.get, reverse=True)[:max_layers_reported]
    top_layers_set = set(top_layers)

    for layer_idx in sorted(attention_data.keys()):
        if layer_idx not in top_layers_set:
            continue
        attn = attention_data[layer_idx]  # (batch, heads, seq, seq)
        if attn is None:
            continue
        attn = attn[0]  # Remove batch dim → (heads, seq, seq)
        num_heads = attn.shape[0]

        # Find top cross-token attention patterns across all heads
        patterns = []
        for head_idx in range(num_heads):
            head_attn = attn[head_idx].float().numpy()  # (seq, seq)
            for src in range(seq_len):
                for tgt in range(seq_len):
                    if src == tgt:
                        continue  # Skip self-attention
                    w = head_attn[src, tgt]
                    if w >= min_weight:
                        patterns.append((w, layer_idx, head_idx, src, tgt))

        if not patterns:
            continue

        # Sort by weight descending, take top_k
        patterns.sort(key=lambda x: x[0], reverse=True)
        top_patterns = patterns[:top_k]

        layer_desc = [f"Layer {layer_idx} ({num_heads} heads):"]
        for w, li, hi, src, tgt in top_patterns:
            src_tok = clean_tokens[src] if src < len(clean_tokens) else f"[{src}]"
            tgt_tok = clean_tokens[tgt] if tgt < len(clean_tokens) else f"[{tgt}]"
            layer_desc.append(
                f"  Head {hi}: '{src_tok}' (pos {src}) attends to "
                f"'{tgt_tok}' (pos {tgt}) with weight {w:.3f}"
            )

        descriptions.append("\n".join(layer_desc))

    # Global summary: which token pairs have the strongest connections overall
    all_patterns = []
    for layer_idx in sorted(attention_data.keys()):
        attn = attention_data[layer_idx]
        if attn is None:
            continue
        attn = attn[0].float().numpy()
        num_heads = attn.shape[0]
        for head_idx in range(num_heads):
            for src in range(seq_len):
                for tgt in range(seq_len):
                    if src != tgt:
                        all_patterns.append((attn[head_idx, src, tgt], layer_idx, head_idx, src, tgt))

    if all_patterns:
        all_patterns.sort(key=lambda x: x[0], reverse=True)
        descriptions.append("\nStrongest global attention connections:")
        for w, li, hi, src, tgt in all_patterns[:10]:
            src_tok = clean_tokens[src] if src < len(clean_tokens) else f"[{src}]"
            tgt_tok = clean_tokens[tgt] if tgt < len(clean_tokens) else f"[{tgt}]"
            descriptions.append(
                f"  L{li}/H{hi}: '{src_tok}' → '{tgt_tok}' ({w:.3f})"
            )

    # Semantic interpretation
    descriptions.append("\nSemantic interpretation:")
    descriptions.append(_interpret_patterns(all_patterns[:20], clean_tokens))

    return "\n".join(descriptions)


def _interpret_patterns(
    top_patterns: list[tuple],
    tokens: list[str],
) -> str:
    """
    Generate semantic interpretations of attention patterns.
    Identifies common patterns: positional bias, semantic linking, syntactic heads.
    """
    if not top_patterns:
        return "  No significant patterns detected."

    observations = []

    # Check for positional bias (adjacent token attention)
    adjacent_count = sum(1 for w, l, h, s, t in top_patterns if abs(s - t) == 1)
    if adjacent_count > len(top_patterns) * 0.5:
        observations.append(
            "  Strong positional bias: most top attention is between adjacent tokens "
            "(typical of lower layers handling local syntax)."
        )

    # Check for long-range attention
    long_range = [(w, l, h, s, t) for w, l, h, s, t in top_patterns if abs(s - t) > 5]
    if long_range:
        for w, l, h, s, t in long_range[:3]:
            src_tok = tokens[s] if s < len(tokens) else f"[{s}]"
            tgt_tok = tokens[t] if t < len(tokens) else f"[{t}]"
            observations.append(
                f"  Long-range link: '{src_tok}' ↔ '{tgt_tok}' (distance {abs(s-t)}) "
                f"in L{l}/H{h} — suggests semantic or thematic connection."
            )

    # Check which tokens are most attended to (information sinks)
    tgt_counts: dict[int, float] = {}
    for w, l, h, s, t in top_patterns:
        tgt_counts[t] = tgt_counts.get(t, 0) + w
    if tgt_counts:
        top_targets = sorted(tgt_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        for tgt_idx, total_w in top_targets:
            tok = tokens[tgt_idx] if tgt_idx < len(tokens) else f"[{tgt_idx}]"
            observations.append(
                f"  Information sink: '{tok}' (pos {tgt_idx}) receives high aggregate "
                f"attention ({total_w:.2f}) — acts as a hub token."
            )

    return "\n".join(observations) if observations else "  Patterns are distributed; no dominant structure."


# ---------------------------------------------------------------------------
# Utility: compare two attention captures
# ---------------------------------------------------------------------------

def _compute_attention_stats(attn: torch.Tensor) -> dict:
    """
    Compute low-dimensional statistics from attention weights.
    These are more stable than raw attention matrices for comparison.

    Args:
        attn: tensor of shape (batch, heads, seq, seq) — single layer

    Returns:
        dict of statistics (all numpy arrays, small and comparable)
    """
    attn = attn[0].float()  # Remove batch dim → (heads, seq, seq)
    num_heads, seq_len, _ = attn.shape

    stats = {}

    # 1. Entropy per head: how "spread out" is attention? (H heads,)
    eps = 1e-10
    log_attn = (attn + eps).log()
    entropy = -(attn * log_attn).sum(dim=-1)  # (heads, seq)
    stats["entropy_per_head"] = entropy.mean(dim=-1).numpy()  # (heads,)

    # 2. Attention to first token (BOS sink) per head
    stats["bos_attention"] = attn[:, :, 0].mean(dim=-1).numpy()  # (heads,)

    # 3. Diagonal dominance: how much does each token attend to itself?
    diag = torch.diagonal(attn, dim1=-2, dim2=-1)  # (heads, seq)
    stats["self_attention_ratio"] = diag.mean(dim=-1).numpy()  # (heads,)

    # 4. Attention concentration: max attention weight per head (averaged over queries)
    stats["max_attention"] = attn.max(dim=-1).values.mean(dim=-1).numpy()  # (heads,)

    # 5. Mean attention distance: average distance between query and key
    if seq_len > 1:
        positions = torch.arange(seq_len, dtype=torch.float32)
        distances = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()  # (seq, seq)
        mean_dist = (attn * distances.unsqueeze(0)).sum(dim=-1).mean(dim=-1)  # (heads,)
        stats["mean_attn_distance"] = mean_dist.numpy()
    else:
        stats["mean_attn_distance"] = np.zeros(num_heads)

    return stats


def compare_attention(
    attn_a: dict[int, torch.Tensor],
    attn_b: dict[int, torch.Tensor],
) -> dict:
    """
    Compare two attention captures using multiple metrics:

    1. Per-head cosine: Compare each head independently (avoids curse of dimensionality)
    2. Attention statistics: Compare low-dimensional features (entropy, concentration, etc.)
    3. Raw cosine (original metric, kept for reference)

    Returns comprehensive comparison dict.
    """
    common_layers = sorted(set(attn_a.keys()) & set(attn_b.keys()))
    if not common_layers:
        return {"error": "No common layers to compare"}

    # --- Metric 1: Per-head cosine similarity ---
    per_layer_head_cosines = {}
    per_layer_mean_head_cosine = {}

    for li in common_layers:
        a = attn_a[li][0].float()  # (heads, seq, seq)
        b = attn_b[li][0].float()
        min_seq = min(a.shape[-1], b.shape[-1])
        a = a[:, :min_seq, :min_seq]
        b = b[:, :min_seq, :min_seq]

        num_heads = min(a.shape[0], b.shape[0])
        head_cosines = []
        for h in range(num_heads):
            ah = a[h].flatten()
            bh = b[h].flatten()
            cos = torch.nn.functional.cosine_similarity(
                ah.unsqueeze(0), bh.unsqueeze(0)
            ).item()
            head_cosines.append(cos)

        per_layer_head_cosines[li] = head_cosines
        per_layer_mean_head_cosine[li] = float(np.mean(head_cosines))

    mean_head_cosine = float(np.mean(list(per_layer_mean_head_cosine.values())))

    # --- Metric 2: Attention statistics comparison ---
    per_layer_stat_corr = {}
    stat_names = ["entropy_per_head", "bos_attention", "self_attention_ratio",
                  "max_attention", "mean_attn_distance"]

    for li in common_layers:
        stats_a = _compute_attention_stats(attn_a[li])
        stats_b = _compute_attention_stats(attn_b[li])

        stat_correlations = {}
        for sn in stat_names:
            va = stats_a[sn]
            vb = stats_b[sn]
            min_len = min(len(va), len(vb))
            va, vb = va[:min_len], vb[:min_len]
            if np.std(va) > 0 and np.std(vb) > 0:
                corr = float(np.corrcoef(va, vb)[0, 1])
            else:
                corr = 1.0  # constant vectors are "identical"
            stat_correlations[sn] = corr

        per_layer_stat_corr[li] = stat_correlations

    # Average stat correlations across layers
    mean_stat_corr = {}
    for sn in stat_names:
        vals = [per_layer_stat_corr[li][sn] for li in common_layers
                if not np.isnan(per_layer_stat_corr[li][sn])]
        mean_stat_corr[sn] = float(np.mean(vals)) if vals else 0.0

    # --- Metric 3: Raw cosine (original, for reference) ---
    raw_cosines = []
    for li in common_layers:
        a = attn_a[li].float().flatten()
        b = attn_b[li].float().flatten()
        min_len = min(len(a), len(b))
        cos = torch.nn.functional.cosine_similarity(
            a[:min_len].unsqueeze(0), b[:min_len].unsqueeze(0)
        ).item()
        raw_cosines.append(cos)

    return {
        # Per-head comparison (primary metric)
        "per_layer_head_cosine": per_layer_mean_head_cosine,
        "mean_head_cosine": mean_head_cosine,
        # Attention statistics correlation
        "per_layer_stat_corr": per_layer_stat_corr,
        "mean_stat_corr": mean_stat_corr,
        # Raw cosine (reference only)
        "per_layer_cosine": {li: c for li, c in zip(common_layers, raw_cosines)},
        "mean_cosine": float(np.mean(raw_cosines)),
        "num_layers_compared": len(common_layers),
    }
