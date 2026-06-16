# Oral Exam Q&A — Role-play: Instructor Marco

Four tasks selected. For each task: code walkthrough → shapes → design choices → alternatives → theory extensions.

---

# Task 1 — a1-task1.2: Building the Vocabulary (`build_tokenizer`)

**Relevant code:** `a1_1/A1_skeleton.py` — `build_tokenizer()` and `A1Tokenizer.__call__()`

---

**Q1. Walk me through your `build_tokenizer` function. What does it do step by step?**

> It reads all lines from the training file, tokenizes each line using the provided tokenize function (NLTK word tokenizer + lowercase), and counts how many times each token appears using a `Counter`. If `max_voc_size` is set, it keeps only the top `max_voc_size - 4` most frequent tokens. It then builds the vocabulary list by prepending the four special tokens (BOS, EOS, UNK, PAD) to the most-common tokens. Finally it creates `word_to_id` and `id_to_word` dictionaries and returns an `A1Tokenizer` object.

---

**Q2. Why do you subtract 4 when computing `n = max_voc_size - 4`?**

> Because the final vocabulary must include the four special tokens (BOS, EOS, UNK, PAD). If I want a vocabulary of size 10,000, I can only include 9,996 real word tokens — the remaining 4 slots are reserved for the special tokens.

---

**Q3. Why do you place the special tokens at the beginning of the vocabulary list?**

> It is a convention, but it has a practical benefit: the special tokens end up at indices 0–3 regardless of the training data. This makes it easy to hard-code or look up their IDs. It also avoids any chance of a word from the corpus accidentally overwriting a special token index.

---

**Q4. What is the purpose of each special token — PAD, UNK, BOS, EOS?**

> - **BOS** (beginning of sequence): prepended to every tokenized text. Tells the model "a new sequence is starting here."
> - **EOS** (end of sequence): appended to every tokenized text. Signals the model that the sequence has ended; generation stops when this token is produced.
> - **UNK** (unknown): replaces any word at inference time that was not seen in training (out-of-vocabulary word). Without this, the model would crash on unseen words.
> - **PAD** (padding): added to the right of shorter sequences to make all sequences in a batch the same length. The loss function is masked to ignore these positions.

---

**Q5. Walk me through the `__call__` method. What are the inputs and outputs?**

> **Input:** a list of N text strings, plus keyword arguments `truncation`, `padding`, `return_tensors`.
>
> **Steps:**
> 1. For each text: tokenize → map each token to an integer ID (use `unk_token_id` if not in vocabulary).
> 2. If `truncation=True`: clip the sequence to `model_max_length - 2` tokens, leaving room for BOS and EOS.
> 3. Prepend BOS and append EOS to each sequence.
> 4. Build an attention mask of all 1s (same length as the sequence).
> 5. Track the maximum length across all sequences.
> 6. If `padding=True`: extend shorter sequences with `pad_token_id` on the right; extend their attention masks with 0s.
> 7. If `return_tensors='pt'`: convert both lists to PyTorch tensors.
>
> **Output:** a `BatchEncoding` object with fields `input_ids` and `attention_mask`.

---

**Q6. What is the shape of `input_ids` when you call `tokenizer(texts, padding=True, truncation=True, return_tensors='pt')`? Assume the batch has N texts and the longest (after truncation) is L tokens.**

> `input_ids.shape = (N, L)` — a 2D tensor where N is the batch size and L is the length of the longest sequence in the batch (including BOS and EOS). Shorter sequences are padded on the right to length L.

---

**Q7. Why do you subtract 2 for truncation (`self.model_max_length - 2`)?**

> Because BOS and EOS are added *after* truncation. If `model_max_length = 512`, I truncate the raw tokens to 510, then add BOS and EOS to get exactly 512. If I didn't subtract 2, the final sequence would be 514 tokens — longer than the model can handle.

---

**Q8. What is the attention mask and why do we return it?**

> The attention mask is a tensor of the same shape as `input_ids`, where real tokens get a value of 1 and padding tokens get 0. The model uses this mask to ignore padding positions during attention — it prevents padding tokens from influencing the computation of real tokens. Without it, padding would pollute the attention scores and degrade performance.

---

**Q9. What does `counter.most_common(n)` do, and why is it a good vocabulary selection strategy?**

> `counter.most_common(n)` returns the `n` most frequent (token, count) pairs in descending order of frequency. This is a good strategy because frequent words carry the most information in the corpus — rare words contribute very little to the model's predictions and waste capacity in the embedding and unembedding matrices. Rare words that appear only once or twice are likely to be typos, proper nouns, or domain-specific terms that the model cannot generalise from anyway.

---

**Q10. What happens when the tokenizer encounters a word at inference time that is not in the vocabulary?**

> It maps it to `self.unk_token_id` using `self.word_to_id.get(t, self.unk_token_id)`. The `.get()` with a default value is the key — it avoids a `KeyError` and silently replaces the unknown token. At the model output side, the model can never predict an `<UNK>` token as the most likely next word in practice (unless it was very common in training), which means generation quality degrades for text with many OOV terms.

---

**Q11. What is the difference between this word-level tokenizer and BPE? What are the trade-offs?**

> This tokenizer splits on word boundaries (NLTK) and then maps whole words to IDs. BPE (Byte Pair Encoding) starts from characters or bytes and iteratively merges the most frequent pair of symbols into a new sub-word token. Trade-offs:
> - **Vocabulary coverage**: BPE can handle any word by decomposing it into known sub-words — no `<UNK>` needed. Word-level tokenizers produce `<UNK>` for unseen words.
> - **Vocabulary size**: BPE vocabularies are typically 30k–100k sub-words; word-level vocabularies are limited because the embedding table would be too large.
> - **Morphology**: BPE implicitly shares representations across "run", "running", "runner" via common sub-words. Word-level treats them as completely separate.
> - **Implementation simplicity**: word-level is much simpler to implement, which is why it is used in Assignment 1.

---

**Q12. Why do you use `lowercase_tokenizer` by default? What is the trade-off?**

> Lowercasing reduces vocabulary size — "The" and "the" map to the same token, so the model sees twice as many examples of each word form. The trade-off is that case information is lost: "US" (the country) and "us" (the pronoun) become indistinguishable, and named entities like "Apple" lose their capitalisation signal.

---

# Task 2 — a2-task1.1: SwiGLU MLP Layer (`A2MLP`)

**Relevant code:** `a1_2/A2_skeleton.py` — `A2MLP`

```python
self.linear_w  = nn.Linear(hidden_size, intermediate_size, bias=False)
self.linear_v  = nn.Linear(hidden_size, intermediate_size, bias=False)
self.linear_w2 = nn.Linear(intermediate_size, hidden_size, bias=False)
self.activation = nn.SiLU()

def forward(self, hidden_states):
    w_out = self.linear_w(hidden_states)
    v_out = self.linear_v(hidden_states)
    return self.linear_w2(self.activation(w_out) * v_out)
```

---

**Q1. Walk me through the MLP layer. What is the input, what happens step by step, and what is the output?**

> **Input:** `hidden_states` of shape `(B, M, hidden_size)` — a batch of B sequences, each of length M tokens, each token represented by a `hidden_size`-dimensional vector.
>
> **Steps:**
> 1. `w_out = linear_w(hidden_states)` — project to intermediate size: `(B, M, intermediate_size)`
> 2. `v_out = linear_v(hidden_states)` — a second projection to intermediate size: `(B, M, intermediate_size)`
> 3. `activation(w_out)` — apply SiLU to `w_out`: `(B, M, intermediate_size)` — this is the **gate**
> 4. `activation(w_out) * v_out` — element-wise multiply gate and value: `(B, M, intermediate_size)`
> 5. `linear_w2(...)` — project back to hidden size: `(B, M, hidden_size)`
>
> **Output:** `(B, M, hidden_size)` — same shape as input.

---

**Q2. What are the shapes of `linear_w`, `linear_v`, and `linear_w2`'s weight matrices?**

> - `linear_w.weight`: `(intermediate_size, hidden_size)`
> - `linear_v.weight`: `(intermediate_size, hidden_size)`
> - `linear_w2.weight`: `(hidden_size, intermediate_size)`
>
> For OLMo-2 1B as a reference: hidden_size=2048, intermediate_size=8192, num_heads=16.
> For SmolLM2-135M: hidden_size=576, intermediate_size=1536.

---

**Q3. This is called SwiGLU. What does that mean, and how does it differ from a standard MLP?**

> **Standard 2-layer FFN:**
> `output = W2 · ReLU(W1 · x)` — one input projection, one activation, one output projection.
>
> **SwiGLU** (Swish-Gated Linear Unit, from Shazeer 2020):
> `output = W2 · (SiLU(W · x) ⊙ V · x)`
>
> The key difference is the **gating mechanism**: a second projection `V` computes a "value" stream, while `SiLU(W·x)` acts as a learned gate that selectively amplifies or suppresses each dimension of the value stream via element-wise multiplication. This adds expressivity without adding depth.

---

**Q4. Why does SwiGLU use three linear layers instead of two?**

> Because the gated architecture requires two input projections: `W` for the gate and `V` for the value. A third matrix `W2` projects the result back. This is an inherent property of gated units — you need separate learned transformations for the gate path and the value path.

---

**Q5. Why is `bias=False` in all three linear layers?**

> Because the Transformer uses pre-norm (RMSNorm is applied before each sublayer). The normalisation already re-centres the activations, so an additive bias provides no additional expressive power. Removing biases reduces parameters slightly and avoids redundancy.

---

**Q6. What is the SiLU activation, and why is it used instead of ReLU?**

> `SiLU(x) = x · σ(x)` where `σ` is the sigmoid function. It is sometimes called the Swish activation.
>
> Compared to ReLU:
> - SiLU is **smooth** (infinitely differentiable) — no sharp corner at 0. This can improve gradient flow.
> - SiLU is **non-monotonic** — it has a small dip for negative x before rising, which gives it more expressivity.
> - SiLU is **non-zero for negative inputs** — unlike ReLU, which completely zeros out negative activations (the "dying ReLU" problem).
>
> Empirically, SiLU with gating (SwiGLU) consistently outperforms ReLU FFNs at the same parameter count.

---

**Q7. The code asserts `config.hidden_act == 'silu'`. Why?**

> Because `A2MLP` is hardcoded for SiLU — the assertion guards against misconfiguration. If someone accidentally loaded a config with `hidden_act='relu'` and used this MLP class, the model would silently compute the wrong thing. The assertion makes the error explicit immediately.

---

**Q8. How does the MLP layer fit into the overall Transformer decoder layer?**

> The decoder layer is:
> ```
> x  →  Attention(RMSNorm(x)) + x   →  MLP(RMSNorm(...)) + ...
> ```
> The MLP is the second sublayer in each Transformer block. It processes each token position **independently** (unlike attention, which mixes information across positions). Its role is to store and retrieve factual knowledge — research suggests that the MLP layers act as a key-value memory.

---

**Q9. Why is `intermediate_size` typically larger than `hidden_size`? What is the typical ratio?**

> The MLP needs to project into a higher-dimensional space to compute a richer representation before projecting back. For standard FFNs, the ratio is 4× (`intermediate_size = 4 × hidden_size`). For SwiGLU, to keep the same total parameter count as a 2-matrix FFN with 4× expansion, the ratio is reduced to `8/3 ≈ 2.67×`:
> ```
> Standard:  2 × hidden × 4·hidden = 8·hidden²
> SwiGLU:    3 × hidden × d'       → d' = 8/3 × hidden
> ```
> In SmolLM2-135M: 576 × (8/3) ≈ 1536 → exactly the intermediate_size in the config.

---

# Task 3 — a3-task4.2: LoRA Layer (`LoRALayer`)

**Relevant code:** `a2_1/WASP_NLP_A3_skeleton.ipynb` — `LoRALayer`

```python
class LoRALayer(nn.Module):
    def __init__(self, W, r, alpha):
        super().__init__()
        self.W = W
        self.W.requires_grad_(False)
        self.lora_b = nn.Parameter(torch.zeros(W.out_features, r))
        self.lora_a = nn.Parameter(torch.randn(r, W.in_features))
        self.scaling_factor = alpha / r

    def forward(self, x):
        return self.W(x) + self.scaling_factor * (x @ self.lora_a.T) @ self.lora_b.T
```

---

**Q1. Walk me through the LoRA layer. What does it do?**

> LoRA adds a low-rank trainable branch alongside a frozen pretrained weight matrix `W`. The forward pass is:
>
> `output = W(x) + (alpha/r) · B · A · x`
>
> - `W(x)` is the frozen original layer's output — this is computed as before, no gradients flow through `W`.
> - `A · x` projects `x` from `in_features` down to `r` dimensions (the "low-rank bottleneck").
> - `B · (A · x)` projects back up to `out_features`.
> - The result is scaled by `alpha/r` and added to the original output.
>
> Only `A` and `B` are trained; `W` is frozen.

---

**Q2. What are the shapes of `lora_a` and `lora_b`?**

> - `lora_a`: `(r, in_features)` — for q_proj in SmolLM2-135M: `(8, 576)`
> - `lora_b`: `(out_features, r)` — for q_proj: `(576, 8)`
>
> The matrix product `lora_b @ lora_a` has shape `(out_features, in_features)` — exactly the same shape as `W.weight`. This is the low-rank decomposition: instead of updating a `576×576` = 331,776-parameter matrix, we update `2 × 8 × 576` = 9,216 parameters.

---

**Q3. What are the shapes at each step of the forward pass? Assume batch size B, sequence length M, and hidden size D=576, r=8.**

> ```
> x              : (B, M, 576)
> x @ lora_a.T   : (B, M, 576) @ (576, 8)  →  (B, M, 8)
> (...) @ lora_b.T: (B, M, 8)  @ (8, 576)  →  (B, M, 576)
> W(x)           : (B, M, 576)    [from frozen linear layer]
> output         : (B, M, 576)    [sum of both paths]
> ```

---

**Q4. Why is `lora_b` initialized to zeros?**

> So that at the very start of training, the LoRA branch contributes exactly zero: `B · A · x = 0 · A · x = 0`. The model therefore behaves identically to the frozen pretrained model at step 0. This means training starts from a stable, well-understood baseline — the pretrained model's behavior — rather than from random noise added on top.

---

**Q5. Why is `lora_a` initialized with random values (`torch.randn`)?**

> Because if both `A` and `B` started at zero, the gradient with respect to `A` would be zero at step 0 (the gradient of `B·A·x` with respect to `A` contains `B`, which is zero). The model would never update `A`. By initializing `A` randomly, gradients can flow through from step 1.
>
> The pair zero-B + random-A is specifically designed so that the initial output is zero while still allowing gradients to propagate and break symmetry.

---

**Q6. What happens on the very first forward pass?**

> The output is identical to what the frozen pretrained model would produce, because:
> `(alpha/r) · (x @ lora_a.T) @ lora_b.T = (alpha/r) · (x @ lora_a.T) @ zeros.T = 0`
>
> So `output = W(x) + 0 = W(x)`. The LoRA branch is silent at initialization.

---

**Q7. Why is `W` frozen (`requires_grad_(False)`)?**

> To save GPU memory and computation during training. For each parameter with `requires_grad=True`, PyTorch stores the gradient tensor (same size as the parameter) and the optimizer keeps additional states (Adam needs two: first and second moment). For a 576×576 weight matrix, that's ~1M floats just for gradient + optimizer state. Freezing `W` eliminates all of this. Since LoRA only trains `A` and `B`, the memory overhead is proportional to `r` not to `in_features × out_features`.

---

**Q8. What is the purpose of the scaling factor `alpha / r`?**

> It keeps the effective magnitude of the LoRA update approximately constant regardless of the rank `r`. Without scaling, a larger `r` would produce a larger update (more parameters → larger product `B·A·x`). The `alpha/r` factor cancels this out. `alpha` is then a single hyperparameter that controls the overall update magnitude independently of the rank. In practice `alpha = 2r` is a common choice (e.g., `r=8, alpha=16`).

---

**Q9. Which layers do you apply LoRA to, and why those specifically?**

> In `extract_lora_targets`, I apply LoRA to `q_proj` and `v_proj` (query and value projection matrices) in every attention block. This is the choice from the original LoRA paper (Hu et al. 2021). The reasoning is that Q and V projections have the most influence on *what information is retrieved* during attention — Q determines what the current token is looking for, and V determines what information is extracted from each position. K (key) and the output projection `W_o` are left frozen.

---

**Q10. How many trainable parameters does LoRA have compared to full SFT? What do the actual numbers show?**

> - **Full SFT (SmolLM2-135M):** 134,515,008 trainable parameters — all of them.
> - **LoRA (r=8, targeting q_proj + v_proj, 30 layers):** 460,800 trainable parameters — **0.34%** of the full model.
>
> Despite training only 0.34% of the parameters, LoRA achieves ROUGE-L 0.643 vs full SFT's 0.668 — a small gap for a dramatic reduction in memory and compute.

---

**Q11. If you wanted to improve LoRA performance, what would you change?**

> 1. **Increase rank `r`**: Higher rank allows a better approximation of the full weight update, at the cost of more trainable parameters.
> 2. **Target more layers**: Add LoRA to `k_proj`, `o_proj`, or even MLP layers.
> 3. **Train for more epochs** or on more data: LoRA is inherently underfitting relative to full SFT with only 0.34% of the parameters.
> 4. **Adjust `alpha`**: Try alpha=r (unit scaling) or sweep over values.

---

**Q12. What is the mathematical motivation for the low-rank decomposition?**

> The hypothesis is that the weight updates `ΔW` learned during fine-tuning have low intrinsic rank — meaning the update can be well-approximated by a product of two small matrices `B·A` where rank `r << min(in_features, out_features)`. This is plausible because fine-tuning for a specific task (e.g., instruction following) only needs to adjust a small number of "directions" in the weight space. The full pretrained model already encodes most of the needed knowledge.

---

# Task 4 — a4-task3.1: Embedding Model (`HuggingFaceEmbeddings`)

**Relevant code:** `a2_2/a2_2.ipynb` — Task 3.1

```python
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    encode_kwargs={"normalize_embeddings": True},
)
test_embedding = embeddings.embed_query("Does aspirin reduce fever?")
# Output: Embedding shape: (768,)
```

---

**Q1. What does this code do? Walk me through it.**

> This sets up a **sentence embedding model** using the `all-mpnet-base-v2` model from the `sentence-transformers` library, wrapped by LangChain's `HuggingFaceEmbeddings` class. The model takes a text string as input and produces a single dense vector of 768 dimensions that represents the semantic meaning of the text.
>
> The `embed_query` call encodes the sentence "Does aspirin reduce fever?" and returns a list of 768 floats. When converted to a numpy array, the shape is `(768,)`.
>
> This embedding model is the retrieval backbone of the RAG pipeline — documents are also embedded and stored in a vector database, and at query time the query embedding is compared to document embeddings to find relevant context.

---

**Q2. What is the output shape and what does each dimension represent?**

> The output is a 1D vector of shape `(768,)`. The number 768 is the hidden size of the MPNet-base model (`all-mpnet-base-v2`). Each of the 768 dimensions is a learned feature from the sentence encoder — they do not have interpretable meanings individually, but the overall direction of the vector captures the semantic content of the sentence. The model was trained so that semantically similar sentences map to nearby vectors.

---

**Q3. Why do you set `normalize_embeddings=True`? What effect does it have?**

> Normalization scales every embedding vector to unit length (L2 norm = 1). The practical effect is that:
> - **Cosine similarity becomes equivalent to a dot product**: `cos(u,v) = u·v / (|u||v|)`. If both are unit vectors, `|u| = |v| = 1`, so `cos(u,v) = u·v`. Dot products are faster to compute than full cosine similarity.
> - **Magnitude bias is removed**: Without normalization, longer documents would produce larger-magnitude vectors and dominate similarity scores even if they are less relevant.

---

**Q4. Why do you use a sentence-transformer model for retrieval instead of the same Qwen language model used for generation?**

> Sentence-transformer models are trained specifically for **semantic similarity** using contrastive learning (e.g., with cosine-similarity loss on sentence pairs). They encode a full sentence into a fixed-size vector where similar meanings are close together.
>
> Generative models like Qwen are trained to predict the next token — their hidden states do not directly represent sentence-level semantic similarity. They are much larger (billions of parameters) and designed for generation, not for fast batch encoding of thousands of documents.
>
> In a RAG pipeline, you need to embed thousands of documents offline and then run similarity search at query time. A dedicated, efficient sentence encoder is much better suited for this.

---

**Q5. What is `all-mpnet-base-v2` and why was it chosen?**

> It is a sentence-transformer model based on **MPNet-base** (Microsoft 2020), fine-tuned by the `sentence-transformers` team on over 1 billion sentence pairs. It produces 768-dimensional embeddings. At the time of its release it was one of the strongest general-purpose sentence encoders on the MTEB (Massive Text Embedding Benchmark). It is a solid default choice for retrieval tasks across a wide range of domains, including the medical domain of this assignment.

---

**Q6. How does the embedding model connect to the rest of the RAG pipeline?**

> The pipeline has two phases:
>
> **Offline (indexing):**
> 1. Each document abstract is split into chunks (`RecursiveCharacterTextSplitter`, chunk_size=500).
> 2. Each chunk is embedded by this model → a 768-dimensional vector.
> 3. Chunks + vectors are stored in Chroma (a vector database using HNSW for fast ANN search).
>
> **Online (query time):**
> 1. The user's question is embedded by the same model → one 768-dimensional query vector.
> 2. Chroma finds the k most similar chunk vectors (by cosine similarity).
> 3. The retrieved chunks are injected as context into the prompt sent to the Qwen LLM.
> 4. The LLM generates a yes/no answer conditioned on the retrieved context.

---

**Q7. What is cosine similarity and why is it used for retrieval?**

> Cosine similarity measures the cosine of the angle between two vectors:
> `cos(u, v) = (u · v) / (|u| · |v|)`
>
> It ranges from -1 (opposite directions) to +1 (same direction). For text embeddings it is preferred because:
> - It is **scale-invariant**: two documents about the same topic should be similar regardless of their length (longer documents would have larger L2 norms but the same direction).
> - With normalised embeddings (unit vectors), it reduces to a dot product, which is fast to compute and hardware-optimised.
>
> The vector database is configured with `hnsw:space = "cosine"` so Chroma uses cosine distance (1 − cosine similarity) as its search metric.

---

**Q8. What is chunking and why is it needed before indexing?**

> Chunking splits long documents into smaller pieces before embedding. It is needed because:
> 1. **Context window limit**: most embedding models have a maximum input length (e.g., 512 tokens for many BERT-based models). Long abstracts would be truncated.
> 2. **Retrieval precision**: if you embed an entire abstract as one vector, a query about one specific claim in the abstract may not match well — the vector averages over many topics. Smaller chunks allow more targeted retrieval.
> 3. **LLM context limit**: the retrieved context must fit in the LLM's prompt. Smaller chunks allow retrieving more of them.
>
> `chunk_overlap=50` means consecutive chunks share 50 characters, preventing information from being lost at the boundary between two chunks.

---

**Q9. In your experiment, what retrieval accuracy did you achieve, and what does that tell you?**

> Retrieval accuracy with `k=1` was **0.984** — for 98.4% of questions, the top-1 retrieved chunk came from the correct gold document. This means the embedding model is very effective at finding the right evidence for these medical questions. The fact that RAG only marginally outperformed the baseline (accuracy 0.649 vs 0.626) despite near-perfect retrieval suggests that the bottleneck is the **LLM's ability to extract the correct answer from the retrieved context**, not the retrieval step itself.

---

**Q10. What is the difference between dense retrieval (what you implemented) and sparse retrieval like BM25?**

> - **Sparse retrieval (BM25)**: matches query terms to document terms using term frequency and inverse document frequency. Fast, no training required, but fails when query and document use different vocabulary (synonyms, paraphrases).
> - **Dense retrieval**: both query and documents are encoded into dense vectors; similarity is computed in the embedding space. Captures semantic similarity beyond exact keyword match — a query about "cardiac arrest" can retrieve documents about "heart failure." Requires a trained encoder model.
>
> Dense retrieval generally outperforms BM25 for paraphrase-heavy or domain-specific tasks like medical QA, at the cost of requiring an embedding model and a vector database.

---
