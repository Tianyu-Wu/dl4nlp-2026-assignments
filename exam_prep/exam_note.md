# DL4NLP Exam Notes — All Assignments

---

# Assignment 1: RNN Language Model

---

## Task 1.2: Building the vocabulary

**Goal:** Map token strings → integer IDs, with a fixed vocabulary size.

**Steps:**
1. Tokenize the training corpus with NLTK word tokenizer + lowercase
2. Count token frequencies with `Counter`
3. Keep the top `max_voc_size - 4` most frequent tokens
4. Prepend 4 special tokens: `<BOS>`, `<EOS>`, `<UNK>`, `<PAD>`
5. Build `word_to_id` and `id_to_word` dicts

**Why each special token:**
- `<PAD>` — pads shorter sequences to the same length in a batch; loss masked with -100
- `<UNK>` — replaces out-of-vocabulary words at inference time
- `<BOS>` — marks the beginning of a sequence (optional but conventional)
- `<EOS>` — marks the end of a sequence; generation stops here

**Why limit vocabulary size?**
The embedding matrix has shape `(vocab_size, embedding_size)` and the unembedding layer has shape `(hidden_size, vocab_size)` — both scale linearly with vocab. Rare words add parameters without contributing useful signal.

---

## Task 3.1: Setting up the network

**Architecture — three components:**
```python
self.embedding   = nn.Embedding(vocab_size, embedding_size)   # (V, 128)
self.rnn         = nn.LSTM(embedding_size, hidden_size, batch_first=True)  # (128 → 512)
self.unembedding = nn.Linear(hidden_size, vocab_size)         # (512 → V)
```

**Forward pass:**
```
input_ids → embedding → (B, T, 128) → LSTM → (B, T, 512) → Linear → (B, T, V) logits
```

**Why LSTM over basic RNN?**
LSTM has a cell state and gates (input, forget, output) that allow it to remember information over longer sequences. Basic RNNs suffer from vanishing gradients on long sequences.

**`batch_first=True`:** input/output shape is `(batch, seq_len, features)` instead of `(seq_len, batch, features)`.

---

## Task 3.2: Computing the loss

**Next-token prediction:** at each position `t`, the model predicts token `t+1`. This requires shifting:
```python
shifted_logits = logits[:, :-1, :]   # drop last prediction
shifted_labels = labels[:, 1:]       # drop first label (BOS)
```
Then compute cross-entropy between shifted logits and shifted labels.

**Why shift?**
- `logits[t]` is the prediction made *after seeing* token `t`
- The correct answer is the *next* token: `labels[t+1]`
- Without shifting, you'd be training the model to predict the token it just saw

**-100 masking:** padding tokens in `labels` are replaced with -100 before computing loss. `CrossEntropyLoss(ignore_index=-100)` skips these positions.

---

## Task 4.1: Implementing the trainer

**Training loop structure:**
```
for each epoch:
    for each batch:
        tokenize batch → input_ids
        labels = input_ids.clone(); labels[pad] = -100
        outputs = model(input_ids, labels)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    evaluate on validation set
    early stopping check
```

**Early stopping:** track `best_val_loss`; if validation loss doesn't improve for `patience=2` epochs, stop training and keep the best checkpoint.

**Why AdamW?** Adam with weight decay. Weight decay regularises the model by penalising large weights, which reduces overfitting.

**Key implementation detail:** padding tokens must be masked to -100 in `labels` (not `input_ids`) so the model still sees PAD tokens as context but doesn't get penalised for predicting them.

---

## Task 5.2: Computing perplexity

**Formula:**
```
perplexity = exp(mean cross-entropy loss over the dataset)
```

**Implementation:**
```python
avg_loss = total_loss / num_batches
perplexity = torch.exp(torch.tensor(avg_loss))
```

**What perplexity measures:** how surprised the model is on average. A perplexity of K means the model is as uncertain as choosing uniformly among K options at each step.

**Baselines:**
- Random model: perplexity ≈ vocab_size (10,000 for this assignment)
- Trained model: expected 200–300 (handout); achieved ~47 after 5 epochs with early stopping
- Large pretrained model (OLMo-2 1B): ~10–20

**Lower is always better.**

---

## Task 5.3: Inspecting learned word embeddings

**What embeddings capture:** words used in similar contexts end up with similar vectors. This encodes both semantic similarity (king ≈ queen) and syntactic patterns (run ≈ walk).

**Nearest neighbours using cosine similarity:**
```python
sim_func = nn.CosineSimilarity(dim=1)
cosine_scores = sim_func(test_embedding, all_embeddings)
top_k = cosine_scores.topk(k+1)  # +1 because the word itself is always #1
```

**Why cosine similarity?**
Measures the angle between vectors — direction matters, not magnitude. Two words can have very different frequency-based magnitudes but still be semantically similar.

**Why skip the first result?**
The nearest neighbour of any word is always itself (cosine similarity = 1.0), so we skip index 0.

**PCA visualisation:** reduces 128-dimensional embeddings to 2D using TruncatedSVD (PCA variant). Allows visual inspection of clusters — semantically related words should appear near each other.

---

# Assignment 2: Transformer from Scratch

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

**Key observations for the exam:**
- Same Transformer architecture as your implementation — structurally identical
- ~1B parameters vs your model's ~8M → dramatically better coherence and factual accuracy
- Trained on much more data for much longer → perplexity in the tens vs your model's hundreds
- Scale matters: more parameters + more data + longer training = better language model

**Actual results from experiments:**

**Perplexity:** 47.1 — much better than the handout's expected 200–300, because the model trained for 5 epochs with early stopping.

**Temperature experiment (prompt: "In natural language processing, a Transformer"):**
- `temp=0.5`: short, repetitive, stuck on `<UNK>` — overconfident on unknown tokens
- `temp=1.0`: longer, somewhat coherent Wikipedia-style text
- `temp=1.5–2.0`: increasingly incoherent, random topic drift

**Top-k experiment:**
- `k=1` (greedy): deterministic, loops — *"a \<UNK\> of the \<UNK\> of the \<UNK\>"*
- `k=50`: most natural-sounding output
- `k=200`: diverse but sometimes stops very early

**Summary of differences:**
| Aspect | Small model (~8M params) | OLMo-2 1B |
|---|---|---|
| Perplexity | 47.1 | ~10–20 |
| Coherence | Breaks after a few words | Multi-sentence coherent text |
| Vocabulary | Many `<UNK>` (10k vocab limit) | Full vocabulary, no unknowns |
| Factual accuracy | None | Mostly correct for common facts |
| Instruction following | Cannot | Basic Q&A format followed |

**Why does scale help?**
More parameters → more capacity to store patterns and facts.
More data → more diverse language patterns seen.
Larger vocabulary → no `<UNK>` tokens, full coverage of the language.
Longer training → better convergence on all patterns.

---

# Assignment 3: Supervised Fine-Tuning + LoRA

---

## Task 1.2: Formatting data for instruction tuning

**What is instruction tuning (SFT)?**
Supervised Fine-Tuning (SFT) trains a base language model to follow instructions by showing it many examples of (instruction → response) pairs and training it to produce the correct response given the instruction.

**ChatML format** (used by SmolLM2 and others):
```
<|im_start|>system
{system prompt}<|im_end|>
<|im_start|>user
{user message}<|im_end|>
<|im_start|>assistant
{assistant response}<|im_end|>
```

**Why use a standard template?**
The base model was pre-trained on data already formatted in ChatML, so it recognises the special tokens (`<|im_start|>`, `<|im_end|>`) as structural markers. Using a consistent format at fine-tuning time aligns with what the model already learned.

**Prompt vs response split:**
- `prompt` = everything up to and including `<|im_start|>assistant\n` — what the model *receives*
- `response` = assistant content + `<|im_end|>` — what the model *generates*

**System prompt is optional:** Not all examples have a system message. The function must handle both cases (2-message and 3-message examples).

---

## Task 1.3: Tokenizing with label masking

**Why separate `input_ids` and `labels`?**
The model sees the full sequence (prompt + response) as input. But we only want it to *learn* from the response tokens — not the prompt. The loss is only computed on positions where `labels != -100`.

```
input_ids: [10, 20, 30, 40, 50, 60, 70]
             |___prompt___|  |__response__|

labels:    [-100, -100, -100, 40, 50, 60, 70]
```

**Why -100?** PyTorch's `CrossEntropyLoss` ignores positions with label `-100` by default. This is a hard-coded convention in PyTorch.

**Why tokenize separately?**
Tokenizing prompt and response separately lets you know where the prompt ends (its length), so you can create the correct -100 mask:
```python
prompt_ids   = tokenizer(prompt,   add_special_tokens=False).input_ids
response_ids = tokenizer(response, add_special_tokens=False).input_ids

input_ids      = prompt_ids + response_ids
labels         = [-100] * len(prompt_ids) + response_ids
attention_mask = [1] * len(input_ids)
```

**`add_special_tokens=False`:** Prevents the tokenizer from automatically adding BOS/EOS tokens, since those are already handled by the ChatML special tokens in the text.

**Attention mask at tokenization time:** All `1`s because every token is real — no padding yet. Padding happens later in the `data_collator`.

---

## Task 2.2: Evaluating the pre-trained model (baseline)

**Baseline results (SmolLM2-135M, no instruction tuning):**
- `eval_loss`: 2.48
- `eval_rougeL`: 0.563

**Why is ROUGE-L so high without instruction tuning?**

The Trainer evaluates using **teacher forcing**: it feeds the full `input_ids` (prompt + response) to the model at once. At each response position `i`, the model sees all previous *gold* tokens as context — including earlier response tokens.

This is a form of **information leakage**: the model partially sees the answer while being scored on it.

**Key exam point:** Teacher-forcing ROUGE-L overstates instruction-following ability. Real evaluation requires autoregressive generation (no gold context tokens), as done in Task 4.4.

**ROUGE-L reminder:** Measures the longest common subsequence (LCS) between predicted and reference text, normalised by reference length.

---

## Task 3.1 & 3.3: Full SFT training and parameter count

**Results after 1 epoch of SFT (SmolLM2-135M, 5000 examples):**

| | Pre-trained (no SFT) | After SFT |
|---|---|---|
| eval_loss | 2.48 | 1.09 |
| eval_rougeL | 0.563 | 0.668 |
| Trainable params | 134.5M | 134.5M |

**`num_trainable_parameters`:**
```python
sum(p.numel() for p in model.parameters() if p.requires_grad)
```

---

## Task 4.2 & 4.3: LoRA (Low-Rank Adaptation)

**What is LoRA?**
Instead of updating the full weight matrix `W` during fine-tuning, LoRA learns two small matrices `A` and `B` such that the weight update is `ΔW = B·A`. The forward pass becomes:
```
output = W·x + (alpha/r) · B·A·x
```
- `W` is **frozen** — no gradient, no optimizer state
- `A` shape `(r, in_features)` — initialized with random Gaussian
- `B` shape `(out_features, r)` — initialized to **zero** so ΔW=0 at start
- `alpha/r` is a scaling factor

**Why initialize B to zero?**
So that at the start of training, the LoRA branch contributes nothing and the model behaves exactly like the base model. Training starts from a stable point.

**Parameter efficiency:**
For SmolLM2-135M with r=8, targeting q_proj and v_proj across 30 layers:
- q_proj per layer: (8×576) + (576×8) = 9,216
- v_proj per layer: (8×576) + (192×8) = 6,144
- Total: 30 × 15,360 = **460,800** trainable params vs 134.5M for full SFT → **0.34%**

**Comparison: Full SFT vs LoRA (SmolLM2-135M, 1 epoch, 5000 examples):**

| | Pretrained | Full SFT | LoRA (r=8) |
|---|---|---|---|
| eval_loss | 2.48 | 1.09 | 1.38 |
| ROUGE-L | 0.563 | 0.668 | 0.643 |
| Trainable params | 134.5M | 134.5M | 460K |

**Key exam points:**
- LoRA is faster to train (fewer gradients and optimizer states)
- LoRA uses much less GPU memory — critical for large models
- Quality is slightly lower than full SFT but surprisingly close
- `r` controls the tradeoff: lower r = fewer params but rougher approximation
- `alpha/r` scaling: alpha is typically set to 2× r (e.g. r=8, alpha=16)

**Freezing strategy:** Freeze all model parameters first, then replace target layers with `LoRALayer` instances — the new `lora_a` and `lora_b` `nn.Parameter`s default to `requires_grad=True`.

---

## Task 4.4: Qualitative inspection

**Actual outputs (SmolLM2-135M, 3 examples):**

| Model | Behaviour |
|---|---|
| **Pretrained** | Ignores the instruction entirely — continues with unrelated text patterns from training data |
| **SFT** | First sentence is on-topic and often correct, then loops into related Q&A pairs |
| **LoRA** | First sentence is often the most concise and direct, also loops |

**Why do all models loop?**
They don't generate `<|im_end|>` reliably to stop. Fix: pass `eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>")` to `model.generate()`.

**Exam answer — "Do models show instruction-following behavior?"**
- Pretrained: **No** — ignores the prompt, generates unrelated continuation
- SFT: **Yes** — first sentence is relevant and correct; looping is a generation artifact
- LoRA: **Yes** — first sentence is often the sharpest despite training only 0.34% of parameters

**Key point:** Repetition loops are a failure mode of greedy decoding on small models — they get trapped in high-probability cycles because the model doesn't learn when to stop without a proper EOS signal.

---

# Assignment 4: Retrieval-Augmented Generation (RAG)

---

## What is RAG?

**Retrieval-Augmented Generation** combines a retrieval system with a language model:
1. Given a query, retrieve relevant documents from a knowledge base
2. Provide the retrieved documents as context to the LLM
3. The LLM generates an answer conditioned on both the query and the context

**Why RAG?** LLMs have a fixed knowledge cutoff and can hallucinate. RAG grounds the answer in retrieved evidence, improving factual accuracy without retraining the model.

---

## Task 3.1: Embedding model

**What are dense embeddings?**
A sentence-transformer maps text → a fixed-size dense vector (e.g. 768 dimensions for `all-mpnet-base-v2`). Semantically similar texts map to nearby vectors in embedding space.

**Why normalize embeddings?**
Setting `normalize_embeddings=True` ensures all vectors have unit length. Then cosine similarity = dot product, which is faster to compute and avoids magnitude bias.

**`all-mpnet-base-v2`:** Based on MPNet-base (Microsoft), hidden size = 768. A strong general-purpose sentence encoder.

```python
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    encode_kwargs={"normalize_embeddings": True},
)
```

---

## Task 3.3: Vector store (Chroma)

**What is a vector store?**
A database that stores embedding vectors and supports fast approximate nearest-neighbour (ANN) search. Given a query embedding, it retrieves the k most similar document chunks.

**Cosine similarity vs L2 (Euclidean):**
- **Cosine**: measures the angle between vectors — direction matters, not magnitude. Preferred for text because two documents about the same topic should be close regardless of length.
- **L2**: measures absolute distance — sensitive to vector magnitude. Less suitable for text embeddings.

**HNSW (Hierarchical Navigable Small World):** The ANN index algorithm used by Chroma. Builds a multi-layer graph where each node connects to nearby nodes — allows sub-linear search time (O(log n)) instead of brute-force O(n).

**Setup:**
```python
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_metadata={"hnsw:space": "cosine"}
)
```

**Chunking affects retrieval quality:**
- Too large: chunks contain irrelevant text, similarity is diluted
- Too small: chunks lose context, answer may be split across chunks
- Overlap (`chunk_overlap=50`) ensures context is not lost at boundaries

---

## Task 4.1: RAG pipeline (LCEL)

**LangChain Expression Language (LCEL):** Composes components with `|` operator, similar to Unix pipes.

**Chain structure:**
```python
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt_template
    | model
    | StrOutputParser()
)
```

- `retriever`: fetches k most similar chunks for the query
- `format_docs`: joins Document objects into a single string
- `RunnablePassthrough()`: passes the question through unchanged
- `prompt_template`: injects context + question into a prompt string
- `model`: generates the answer
- `StrOutputParser()`: extracts the string from the model output

**Key design choice:** Prompt the model to answer "yes or no only" — constrains output for classification tasks and makes parsing reliable.

---

## Task 5.1: High-level evaluation

**Metrics:**
- **Accuracy**: fraction of questions answered correctly (including invalid responses as wrong)
- **F1 (binary)**: harmonic mean of precision and recall for the "yes" class — better than accuracy when class distribution is skewed

**Invalid responses:** Model outputs that are neither "yes" nor "no" (e.g. long explanations). Treat as a separate class or filter them out before computing F1.

**Parse function:**
```python
def parse_answer(text):
    text = text.strip().lower()
    if text.startswith("yes"): return "yes"
    elif text.startswith("no"): return "no"
    else: return "invalid"
```

**Expected pattern:** RAG should outperform the baseline LLM — especially on questions where the answer depends on specific medical facts not in the model's training data.

---

## Task 5.2: Detailed inspection

**Retrieval accuracy:** Fraction of questions where the top-1 retrieved chunk comes from the gold document. Measures whether the retrieval step is actually finding the right evidence.

**What can go wrong in RAG:**
1. **Retrieval failure**: wrong document retrieved → model has no useful context → may hallucinate or default to prior knowledge
2. **Context too long/irrelevant**: retrieved chunk is from the right document but the wrong section
3. **Model ignores context**: LLM answers from its own prior knowledge even when context contradicts it

**Qualitative inspection questions to answer:**
- When retrieval succeeds and model is correct: RAG working as intended
- When retrieval fails but model is still correct: model's prior knowledge is sufficient
- When retrieval succeeds but model is wrong: model failed to use the context
- When retrieval fails and model is wrong: worst case — both components failed

---
