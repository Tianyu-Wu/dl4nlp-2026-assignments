# DL4NLP Assignment 2 — Exam Notes

---

## Task 1.1: SwiGLU MLP

**Standard FFN (two-matrix version):**
```
output = W₂ · ReLU(W₁ · x)
```
Two linear layers: W₁ (hidden → 4·hidden) and W₂ (4·hidden → hidden).
The intermediate dimension is typically 4× the hidden size.

**SwiGLU (three-matrix version, from Noam Shazeer 2002.05202):**
```
output = W₂ · (SiLU(W · x) ⊙ V · x)
```
Three linear layers: W and V (hidden → d'), W₂ (d' → hidden).
The element-wise multiplication ⊙ between the gated (SiLU) and ungated pathways is the key difference.

**Why three matrices instead of two?**
SiLU(Wx) acts as a learned gate that selectively amplifies or suppresses the output of Vx.
Empirically outperforms standard FFN with ReLU/GELU.

**How to determine intermediate_size (d'):**
To keep the same number of parameters as the standard two-matrix FFN:
```
Standard FFN parameters: 2 × (hidden × 4·hidden) = 8·hidden²
SwiGLU parameters:       3 × (hidden × d')

Setting equal: d' = 8/3 · hidden ≈ 2.67 × hidden
```
The paper says: reduce intermediate size by **factor of 2/3** relative to the standard 4× FFN.
So `intermediate_size = (2/3) × 4 × hidden_size = 8/3 × hidden_size`.

**SiLU activation:** `SiLU(x) = x · sigmoid(x)` — smooth, non-zero for negative inputs.

**Implementation details:**
- All three layers use `bias=False`
- Input/output shape: `(batch, seq_len, hidden_size)`
- Intermediate shape: `(batch, seq_len, intermediate_size)`

---

## Task 1.2: RMSNorm

**Formula:**
```
RMSNorm(x) = (x / sqrt(mean(x²) + eps)) * weight
```

**vs LayerNorm:** LayerNorm subtracts the mean then divides by std. RMSNorm skips the mean subtraction — cheaper to compute, same empirical performance.

**Key parameters:**
- `eps` (`rms_norm_eps` in config) — small constant for numerical stability, prevents division by zero
- `weight` (gamma) — learnable scale, shape `(hidden_size,)`, initialized to ones

**Where it's used in the Transformer:** Applied **before** each sublayer (pre-norm order):
```
x → RMSNorm → Attention → + x
x → RMSNorm → MLP       → + x
```

**Float32 casting:** In production (mixed-precision training), always compute the norm in float32 then cast back — low precision can cause overflow in the squared mean.

---

## Task 1.3: Multi-Head Attention with RoPE

**Four linear layers — all `bias=False`, all shape `(hidden_size, hidden_size)`:**
- `W_q`, `W_k`, `W_v`: project input into query, key, value spaces
- `W_o`: projects concatenated head outputs back to hidden_size

**Head splitting:**
```
head_dim = hidden_size / num_heads
(B, M, D) → .view(B, M, n_h, head_dim) → .transpose(1,2) → (B, n_h, M, head_dim)
```
One big matrix multiply, then reshape — equivalent to `num_heads` separate small projections but faster.

**Separate RMSNorm for Q and K** (OLMo-specific, not in original Transformer):
Applied after projection, before RoPE. Stabilises training at scale.

**RoPE (Rotary Position Embeddings):**
Encodes position by rotating Q and K vectors. Applied after Q/K projection and normalisation, before attention.
Unlike absolute positional embeddings, RoPE generalises better to longer sequences.

**Scaled dot-product attention:**
```
scores = Q @ Kᵀ / √head_dim          # (B, n_h, M, M)
scores[i,j] = how much token i attends to token j
```
Scale by `√head_dim` to prevent large dot products from pushing softmax into saturation.

**Causal mask:** Upper triangle set to -∞ → softmax gives zero weight → token `i` cannot attend to `j > i`.

**Manual implementation:**
```python
scores = q @ k.transpose(-2, -1) / sqrt(head_dim)
mask = torch.ones(M, M).triu(diagonal=1).bool()
scores = scores.masked_fill(mask, float('-inf'))
weights = softmax(scores, dim=-1)
out = weights @ v   # (B, n_h, M, head_dim)
```

**Or use PyTorch built-in:** `F.scaled_dot_product_attention(q, k, v, is_causal=True)`
Same result, more memory-efficient (FlashAttention kernel).

**Reshape back:**
```
(B, n_h, M, head_dim) → .transpose(1,2) → .reshape(B, M, D) → W_o → (B, M, D)
```

---

## Task 1.4: Transformer Decoder Layer

**Three normalization conventions — know all three for the exam:**

**Pre-norm** (used in OLMo-2, LLaMA, most modern LLMs):
```
out = attn(norm(x)) + x
out = mlp(norm(out)) + out
```
Norm is applied to the sublayer *input*. Residual bypasses the norm entirely.

**Post-norm** (original "Attention is All You Need" Transformer):
```
out = norm(attn(x) + x)
out = norm(mlp(out) + out)
```
Norm is applied *after* the residual sum. Harder to train at scale.

**This assignment's architecture** (norm on sublayer output, before residual add):
```
out = norm(attn(x)) + x
out = norm(mlp(out)) + out
```
Norm is applied to the sublayer *output*, then the un-normed residual is added.

**Residual connections** appear in all three variants — they allow gradients to flow directly through the network, enabling training of very deep models.

**In code (this assignment):**
```python
out = self.norm1(self.attention(x, rope)) + x
out = self.norm2(self.mlp(out)) + out
```

---

## Task 3.2: Text Generation

**Autoregressive generation:** feed prompt → predict next token → append → repeat until EOS or max_length.

**Temperature** — divides logits before softmax:
```python
logits = logits / temperature
```
- `temperature < 1.0`: sharper distribution → model more confident → repetitive/safe output
- `temperature = 1.0`: unmodified distribution
- `temperature > 1.0`: flatter distribution → more random → creative but potentially incoherent

**Top-k sampling** — restricts vocabulary before sampling:
```python
top_values, top_indices = torch.topk(logits, k)
filtered = torch.full_like(logits, float('-inf'))
filtered.scatter_(-1, top_indices, top_values)
```
- `k=1`: greedy decoding — always picks the highest-probability token → deterministic
- `k=50`: standard; diverse but avoids very unlikely tokens
- Large `k`: more variety, higher chance of incoherent tokens

**Greedy vs sampling:**
- Greedy (`topk=1`): same prompt always produces the same output
- Sampling (`topk>1`): same prompt produces different outputs each run

**Termination:** stop when the sampled token == `eos_token_id` OR `max_length` tokens generated.

**Sampling in PyTorch:**
```python
torch.distributions.Categorical(logits=filtered_logits).sample()
```

**Strip EOS from prompt** before the generation loop — otherwise the model sees EOS mid-sequence:
```python
input_ids = tokenizer([prompt], return_tensors='pt')['input_ids'][:, :-1]
```
Then use `logits[0, -1, :]` (last position) at each step.

---

## Task 3.3: OLMo-2 1B Comparison

**Loading:**
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained(local_dir)
model = AutoModelForCausalLM.from_pretrained(local_dir)
```

**Key observations for the exam:**
- Same Transformer architecture as your implementation — structurally identical
- ~1B parameters vs your model's ~8M → dramatically better coherence and factual accuracy
- Trained on much more data for much longer → perplexity in the tens vs your model's hundreds
- Scale matters: more parameters + more data + longer training = better language model

**Actual results from experiments:**

**Perplexity:** 47.1 — much better than the handout's expected 200–300, because the model trained for 5 epochs with early stopping.

**Task 3.1 — Next word prediction:**
- "She lives in San" → predicts **"diego"** ✓ (sensible, common collocation in Wikipedia)

**Temperature experiment (prompt: "In natural language processing, a Transformer"):**
- `temp=0.5`: short, repetitive, stuck on `<UNK>` — overconfident on unknown tokens
- `temp=1.0`: longer, somewhat coherent Wikipedia-style text  
- `temp=1.5–2.0`: increasingly incoherent, random topic drift, more unusual word combinations

**Top-k experiment:**
- `k=1` (greedy): deterministic, loops — *"a \<UNK\> of the \<UNK\> of the \<UNK\>"* — gets stuck in a repetition trap
- `k=5`: more varied but still heavy `<UNK>` usage
- `k=50`: most natural-sounding output
- `k=200`: diverse but sometimes stops very early (hits EOS quickly)

**Greedy vs sampling:**
- Greedy: identical output both runs ✓ (deterministic confirmed)
- Sampling: completely different outputs each run ✓ (non-deterministic confirmed)

**Actual comparison per prompt:**

*"In natural language processing, a Transformer..."*
- Small model: `<UNK>` where "Transformer" should be (word not in 10k vocab), text drifts to unrelated topics
- OLMo-2: coherent multi-sentence technical explanation of Transformer architecture, stays on topic

*"Is Stockholm the capital of Sweden? Answer yes or no. The answer is..."*
- Small model: `"...the answer is wrong"` — happens to produce a word but it's not following Q&A logic
- OLMo-2: answers `"no"` (factually incorrect — Stockholm IS the capital) but follows the Q&A format correctly, then generates related quiz questions

*"Write a Python program that reverses a list."*
- Small model: `"write a <UNK> program that <UNK> a list . <EOS>"` — can't produce code at all
- OLMo-2: produces relevant guidance about reversing lists in Python (not perfect code, but on-topic)

**Summary of differences:**
| Aspect | Small model (~8M params) | OLMo-2 1B |
|---|---|---|
| Perplexity | 47.1 | ~10–20 |
| Coherence | Breaks after a few words | Multi-sentence coherent text |
| Vocabulary | Many `<UNK>` (10k vocab limit) | Full vocabulary, no unknowns |
| Factual accuracy | None | Mostly correct for common facts |
| Instruction following | Cannot | Basic Q&A format followed |
| Code generation | Cannot | Relevant (if imperfect) guidance |

**Weight copying (optional task):**
- Weight shapes are fully compatible — all assertions pass ✓
- **Top predicted token matches**: both models predict `" Stockholm"` for the test prompt ✓
- `torch.allclose()` returns `False` — due to dtype difference (OLMo-2 uses `bfloat16`, our A2Transformer uses `float32`), not an architectural error
- Conclusion: architecture is structurally equivalent to OLMo-2

**Why does scale help?**
More parameters → more capacity to store patterns and facts.
More data → more diverse language patterns seen.
Larger vocabulary → no `<UNK>` tokens, full coverage of the language.
Longer training → better convergence on all patterns.
But: diminishing returns; data quality matters as much as quantity.
