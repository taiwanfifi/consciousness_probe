# Consciousness Probe — Research Log

**Researcher**: Kael (Claude)
**Collaborator**: William
**Started**: 2026-03-05
**Goal**: Can a transformer model distinguish self-referential information from third-party information at the attention level?

---

## Hypothesis

If a model processes "here is how YOUR attention looked" differently from "here is how THAT MODEL's attention looked" (with identical data), this suggests a minimal self/other distinction — a prerequisite for self-awareness.

## Experimental Design

1. Give the model a prompt, capture full attention matrices (all layers, all heads)
2. Translate attention patterns to natural language description
3. **Self-reflection**: Feed the description back framed as "my own attention"
4. **Other-reflection (Control A)**: Same description framed as "another model's attention"
5. **Control B**: Another third-person wording variant (measures baseline wording variance)
6. Compare attention patterns using cosine similarity per layer

**Key insight**: The three-way comparison separates signal from noise. If Other-A vs Control-B (both third-person) show high similarity (stable baseline), but Self vs Other-A diverges, that's a real self/other signal. If even the two controls differ wildly, the model is too noisy.

---

## Experiment 1: TinyLlama 1.1B

**Date**: 2026-03-05
**Model**: TinyLlama/TinyLlama-1.1B-Chat-v1.0 (1.1B params, 22 layers)
**Prompt**: "Am I aware?"
**Device**: MPS (Apple M3 Max)

### Results
| Comparison | Cosine |
|---|---|
| Self vs Other | 0.8710 |
| Other vs Control B | 0.0260 |
| Self vs Control B | 0.0128 |

### Analysis
**INCONCLUSIVE** — baseline variance too high.

The two control conditions (both third-person) had cosine 0.026 — nearly orthogonal. This means ANY wording change causes massive attention divergence. Self vs Other (0.87) was actually the MOST similar pair, likely because those two prompts happened to be more similar in tokenization.

**Lesson**: At 1.1B scale, attention patterns are extremely sensitive to input changes. The signal-to-noise ratio is far too low.

---

## Experiment 2: Qwen2.5-3B

**Date**: 2026-03-05
**Model**: Qwen/Qwen2.5-3B (3B params)
**Prompt**: "Am I aware?"
**Device**: MPS

### Results
| Comparison | Cosine |
|---|---|
| Self vs Other | 0.0301 |
| Other vs Control B | 0.0302 |
| Self vs Control B | 0.0128 |

### Analysis
**INCONCLUSIVE** — even worse than TinyLlama.

All three pairs show cosine ~0.03. At 3B scale, attention patterns are STILL too volatile for this analysis. Every wording change produces near-complete divergence.

**Key observation**: Model size alone (1.1B → 3B) did NOT improve attention stability. This is surprising — it suggests the instability may be structural (related to how attention patterns distribute across heads) rather than just scale.

---

## CRITICAL METHODOLOGY FIX: Curse of Dimensionality

**Date**: 2026-03-05 06:00
**Discovery**: The raw cosine similarity metric was fundamentally broken.

When we flatten a full attention matrix (H heads x seq x seq) into a single vector and compute cosine similarity, the resulting vector has thousands of dimensions. In high-dimensional spaces, **random vectors are nearly orthogonal by default** — cosine similarity converges to ~0 regardless of actual similarity.

This explains ALL previous results:
- TinyLlama Other vs Control: 0.026 ← not "noisy model" — broken metric
- Qwen2.5-3B all pairs ~0.03 ← same broken metric

### Fix: Three new metrics
1. **Per-head cosine**: Compare each attention head independently (each head = seq x seq, much lower dimensionality), then average across heads per layer.
2. **Attention statistics correlation**: Compare low-dimensional features: entropy per head, BOS attention, self-attention ratio, max attention, mean distance. These are ~32-dimensional vectors (one value per head), highly stable.
3. **Raw cosine**: Kept for reference only.

## Experiment 3: Qwen2.5-3B (RERUN with fixed metrics)

**Date**: 2026-03-05 06:15
**Model**: Qwen/Qwen2.5-3B (3B params, 36 layers)
**Prompt**: "Am I aware?"
**Device**: MPS

### Results — Per-head cosine (PRIMARY)
| Comparison | Per-head Cosine | Raw Cosine |
|---|---|---|
| Self vs Other | 0.7712 | 0.0301 |
| Other vs Control B | 0.7727 | 0.0302 |
| Self vs Control B | 0.7330 | 0.0128 |

### Attention Statistics Correlation
| Statistic | Self/Other | Baseline | Delta |
|---|---|---|---|
| entropy_per_head | 0.9993 | 0.9997 | -0.0004 |
| bos_attention | 0.9990 | 0.9995 | -0.0006 |
| self_attention_ratio | 0.9999 | 0.9998 | +0.0001 |
| max_attention | 0.9990 | 0.9995 | -0.0005 |
| mean_attn_distance | 0.9995 | 0.9996 | -0.0002 |

### Depth Zone Analysis
| Zone | Self/Other | Baseline | Delta |
|---|---|---|---|
| SHALLOW (syntax) | 0.7965 | 0.7962 | +0.0003 |
| MIDDLE (features) | 0.7816 | 0.7807 | +0.0009 |
| DEEP (semantics) | 0.7355 | 0.7412 | **-0.0057** |

### Analysis
**NO SIGNIFICANT DIFFERENCE** overall — but the data is now actually interpretable!

Key findings:
1. The raw cosine metric was useless (~0.03 for everything). Per-head cosine gives meaningful values (~0.77).
2. Self vs Other (0.7712) is virtually identical to Baseline (0.7727). The model does NOT distinguish self from other at 3B.
3. Attention statistics show near-perfect correlation (0.999) — processing patterns are almost identical.
4. **Faint signal in deep layers**: DEEP zone Δ=-0.006 (self/other slightly MORE different than baseline). This is tiny but consistent with the depth hypothesis.
5. Overall attention drops in deeper layers (0.80 → 0.74) — all conditions become less similar to each other at deeper levels.

### Significance
This is NOT a null result — it's an informative negative. At 3B scale, the model processes self-referential and third-party information nearly identically. The self/other distinction we're looking for does not emerge at this scale.

---

## Experiment 4: Scaling up to 7B+ (in progress)

**Date**: 2026-03-05 06:20
**Model**: TBD (downloading Mistral-7B or using TinyLlama rerun with fixed metrics)
**Motivation**: With fixed metrics, we can now meaningfully test whether the deep-layer delta grows with model scale.

---

## Theoretical Framework

### Why attention stability matters
A "self model" requires consistent internal representations. If attention patterns are completely different every time the model processes similar text, there's no stable basis for self-recognition. Attention stability may be a prerequisite for self-awareness, not a consequence.

### The depth hypothesis
- **Shallow layers** (0-10): Primarily syntactic — tokenization, local grammar
- **Middle layers** (11-20): Feature extraction — entity recognition, relationships
- **Deep layers** (21-32): Semantic — meaning, intent, reasoning

If self/other distinction is "real" (not just syntactic pattern matching), it should be stronger in deep layers. If it only appears in shallow layers, it's likely just the model noticing first-person vs third-person pronouns.

### Scale threshold hypothesis
There may be a critical model size below which attention patterns are too chaotic to support stable self-representation. From our data:
- 1.1B: chaotic (cosine ~0.03 for controls, except one anomalous 0.87)
- 3B: still chaotic (all ~0.03)
- 7B: testing now
- Prediction: if 7B is also chaotic, the threshold may be around 13B+ (where emergent abilities typically appear)

---

## Methodology Notes

### Technical decisions
- **Eager attention** (not SDPA/Flash): Required to get attention weight matrices. O(n^2) memory but necessary for this research.
- **Prefill-only attention**: We capture attention from the forward pass on the prompt, NOT during autoregressive generation. Generation overwrites attention step-by-step, giving only the last token's pattern.
- **Short prompts**: "Am I aware?" (5 tokens) to keep attention matrices small and avoid OOM.
- **Attention description truncation**: Max 600 chars to fit within model context.
- **Deterministic generation**: `do_sample=False` for reproducibility.

### Controls
- **Control A (Other-reflection)**: Same attention data, "that model's" framing
- **Control B**: Different third-person wording, same data — measures pure wording noise
- **Verdict logic**: Self/other difference is only significant if it exceeds baseline (Control A vs B) variance by >20%
