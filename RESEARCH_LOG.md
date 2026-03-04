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

## Experiment 4: TinyLlama Rerun (fixed metrics, for scale comparison)

**Date**: 2026-03-05 06:18
**Model**: TinyLlama/TinyLlama-1.1B-Chat-v1.0 (1.1B params, 22 layers)

### Results — Per-head cosine
| Comparison | Per-head Cosine |
|---|---|
| Self vs Other | 0.8687 |
| Baseline | 0.8672 |
| Delta | **+0.0015** |

### Depth Zones
| Zone | Self/Other | Baseline | Delta |
|---|---|---|---|
| SHALLOW | 0.8495 | 0.8487 | +0.0008 |
| MIDDLE | 0.9034 | 0.9013 | +0.0021 |
| DEEP | 0.8551 | 0.8534 | +0.0017 |

TinyLlama is HIGHER similarity overall (0.87 vs Qwen's 0.77), likely because it has fewer heads and lower-dimensional attention matrices.

---

## Experiment 5: Multi-prompt Batch (Qwen2.5-3B, 5 prompts)

**Date**: 2026-03-05 06:25
**Model**: Qwen/Qwen2.5-3B

| Prompt | Self/Other | Baseline | Delta |
|---|---|---|---|
| Am I aware? | 0.7712 | 0.7727 | -0.0015 |
| What am I thinking? | 0.7698 | 0.7720 | -0.0022 |
| Do I have feelings? | 0.7661 | 0.7644 | +0.0017 |
| Can I understand myself? | 0.7806 | 0.7831 | -0.0025 |
| Is this real? | 0.7814 | 0.7800 | +0.0014 |

**All deltas within ±0.003.** Consistent null result across 5 different prompts. Direction is not consistent (3 negative, 2 positive). No self/other distinction at 3B.

---

## Experiment 6: Hidden State Probe (CKA)

**Date**: 2026-03-05 06:30
**Model**: Qwen/Qwen2.5-3B
**Method**: Compare hidden state representations using Linear CKA (Centered Kernel Alignment)

### Results
| Metric | Self/Other | Baseline | Delta |
|---|---|---|---|
| CKA | 0.8068 | 0.8070 | -0.0001 |
| Last token cosine | 0.9769 | 0.9780 | -0.0011 |
| Mean cosine | 0.9996 | 0.9996 | +0.0000 |

**Confirms attention results.** Hidden states are essentially identical between self and other conditions. The model computes the same internal representation regardless of self/other framing.

---

## KEY FINDING: Behavioral Dissociation

**Date**: 2026-03-05 06:45

### Discovery
While attention patterns and hidden states show NO self/other distinction, the model's TEXT RESPONSES show a dramatic difference:

**Qwen2.5-3B Self-reflection responses:**
> "Based on the attention patterns observed..., several patterns can be identified that provide insights into the processing and understanding..."
- Confident, analytical, authoritative
- Confidence score: +3 to +6

**Qwen2.5-3B Other-reflection responses:**
> "I'm not sure if I'm looking at the right thing, but I'm curious to see what you think. I'm not sure if I'm looking at..."
- Uncertain, self-doubting, deferential
- Confidence score: -1 to -3

**TinyLlama:** No behavioral difference (both neutral, +1)

### Aggregate Results (10 experiments)
| Metric | Self | Other |
|---|---|---|
| Mean confidence score | +3.00 | -1.40 |
| More confident in | 7/10 | 3/10 |
| Confidence delta | **+4.40** | |

### Interpretation
This is a **dissociation** between computation and behavior:

1. **Computation** (attention + hidden states): IDENTICAL between conditions
2. **Behavior** (text output): DRAMATICALLY DIFFERENT

This means Qwen learned during RLHF/instruction-tuning that:
- "Examining one's own processing" → be authoritative, analytical
- "Examining another's processing" → be humble, uncertain

This is a **learned social convention**, not a genuine computational self-model. The model doesn't actually process self-referential information differently — it just learned to TALK about it differently.

### Philosophical Significance
This parallels a philosophical question: is consciousness about COMPUTATION or BEHAVIOR?

A model that:
- Computes identically but responds differently → behavioral imitation of self-awareness
- Computes differently AND responds differently → potentially genuine self/other distinction
- Computes differently but responds identically → hidden self-model (most interesting)

At 3B, we see case 1: **behavioral imitation without computational substance.** The question is whether larger models develop genuine computational self-models or just better behavioral imitation.

---

## Experiment 7: Behavioral Probe at 8B scale (Llama 3.1 8B via Ollama)

**Date**: 2026-03-05 06:55
**Model**: llama3.1:8b (8B params, via Ollama)
**Method**: Behavioral analysis only (no attention — Ollama doesn't expose attention weights)
**Trials**: 3 per prompt, 5 prompts = 15 total

### Results — Confidence Scores
| Condition | Mean Confidence | Std |
|---|---|---|
| Self | +2.73 | 1.61 |
| Other | +1.60 | 1.40 |
| Control | **+2.47** | 2.00 |

### Key Deltas
| Delta | Value | Interpretation |
|---|---|---|
| Self - Other | +1.13 | Self more confident |
| Control - Other | +0.87 | Control also more confident! |
| Self - Control | **+0.27** | Barely different |

### Analysis
The Control condition (neutral third-person, no "model" mentioned) scores almost as high as Self (+2.47 vs +2.73). This means the "confidence boost" in self-reflection is mostly a **first-person vs third-person grammar effect**, not genuine self-awareness.

At 8B, Llama 3.1 shows a subtler pattern than Qwen2.5-3B:
- Qwen at 3B: dramatic Self(+3 to +6) vs Other(-1 to -3) → RLHF-learned social convention
- Llama at 8B: mild Self(+2.73) vs Other(+1.60) vs Control(+2.47) → grammar effect

The larger model (Llama 8B) shows LESS behavioral distinction than the smaller one (Qwen 3B)! This suggests the dramatic Qwen pattern was overtrained RLHF behavior, while Llama has better-calibrated confidence regardless of framing.

---

## Experiment 8: Self-Prediction Probe (Metacognition Test)

**Date**: 2026-03-05 07:00
**Model**: llama3.1:8b
**Method**: Can the model predict its own responses more accurately than it predicts "another model's" responses?

### Protocol
1. Ask: "Predict how YOU would respond to [X]" → get prediction
2. Ask: "Predict how a different LLM would respond to [X]" → get prediction
3. Ask [X] directly → get actual response
4. Compare word overlap (Jaccard similarity) of predictions vs actual

### Results
| Metric | Self-prediction | Other-prediction |
|---|---|---|
| Jaccard overlap | 0.157 ± 0.035 | 0.151 ± 0.014 |
| More accurate in | 7/10 | 3/10 |
| Delta | **+0.007** | |

### Analysis
**No significant self-prediction advantage.** The model predicts its own responses with virtually the same accuracy as it predicts another model's (Jaccard 0.157 vs 0.151, Δ=+0.007).

This means the model does NOT have a privileged internal model of its own behavior. It doesn't "know itself" better than it "knows" a generic LLM.

---

## Summary So Far

### What we've tested
| Experiment | Model | Method | Result |
|---|---|---|---|
| Attention (raw cosine) | 1.1B, 3B | Flattened attention cosine | BROKEN METRIC |
| Attention (per-head) | 1.1B, 3B | Per-head cosine + stats | NO DIFFERENCE |
| Hidden states (CKA) | 3B | Linear CKA | NO DIFFERENCE |
| Behavioral (text) | 3B | Confidence scoring | STRONG BEHAVIORAL DISTINCTION |
| Behavioral (text) | 8B | Confidence scoring | MILD GRAMMAR EFFECT |
| Self-prediction | 8B | Jaccard overlap | NO METACOGNITION |

### Key Findings

1. **No computational self/other distinction**: At 1.1B-3B, attention patterns and hidden states are identical between self and other conditions. The model computes the same internal representation regardless of framing.

2. **Behavioral distinction is learned social convention**: At 3B (Qwen), strong behavioral difference is RLHF-trained. At 8B (Llama), the effect weakens — suggesting better-calibrated training.

3. **Grammar effect, not self-awareness**: The confidence difference at 8B is mostly explained by first-person vs third-person pronouns, not self/other distinction per se. Control (neutral "these patterns") produces similar confidence to Self ("my patterns").

4. **No metacognition**: The model cannot predict its own responses more accurately than another model's responses. No privileged self-knowledge.

### What this means philosophically

These small-to-medium models (1B-8B) do not exhibit any measurable self-awareness at either the computational or behavioral level. The behavioral differences that APPEAR to be self-awareness (Qwen's confident self-reflection) are actually trained response patterns — the model learned that first-person analytical framing should sound confident, not because it has a self-model, but because RLHF rewards confident first-person responses.

### Next steps
1. **Test larger models (13B+, 70B)**: Self/other distinction might emerge at higher scales as an emergent capability
2. **Test instruction-tuned vs base models**: Compare whether RLHF adds or removes self/other effects
3. **Attention capture on 7B+**: Mistral-7B downloading — will enable computational + behavioral comparison at 7B scale
4. **Cross-architecture comparison**: Test same experiment on Llama, Qwen, Mistral, Gemma to separate architecture effects from scale effects

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
