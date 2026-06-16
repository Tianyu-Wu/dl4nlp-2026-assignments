# Oral Exam Q&A — Other Tasks (Instructor-Selected)

Only 🎓-marked tasks can be tested. This file covers every 🎓 task that is NOT among your four selected ones.

## Complete 🎓 task map

| Assignment | 🎓 Tasks | Your selection? |
|---|---|---|
| A1 | 1.2, 3.1, 3.2, 4.1, 5.2, 5.3 | 1.2 ✓ |
| A2 | 1.1, 1.3, 1.4, 3.2, 3.3 | 1.1 ✓ |
| A3 | 1.2, 2.2, 3.1, 4.2, 4.3, 4.4 | 4.2 ✓ |
| A4 | 3.1, 3.3, 4.1, 5.1, 5.2 | 3.1 ✓ |

---

# Assignment 1 — RNN Language Model

---

## A1 Task 3.1: Setting up the RNN model (`A1RNNModel`)

**Code:**
```python
self.embedding   = nn.Embedding(config.vocab_size, config.embedding_size)   # (V, 128)
self.rnn         = nn.LSTM(config.embedding_size, config.hidden_size, batch_first=True)
self.unembedding = nn.Linear(config.hidden_size, config.vocab_size)
self.loss_func   = CrossEntropyLoss(ignore_index=-100)

def forward(self, input_ids, labels=None):
    embedded   = self.embedding(input_ids)     # (B, T) → (B, T, 128)
    rnn_out, _ = self.rnn(embedded)            # (B, T, 128) → (B, T, 512)
    logits     = self.unembedding(rnn_out)     # (B, T, 512) → (B, T, V)
    ...
```

---

**Q1. Walk me through the architecture and forward pass. What are the shapes at each step?**

> Assume B=batch size, T=sequence length, V=vocab size (10,000), E=embedding size (128), H=hidden size (512).
>
> 1. `input_ids`: `(B, T)` — integer token IDs
> 2. `embedding(input_ids)` → `(B, T, 128)` — each token ID replaced by its learned 128-dim vector
> 3. `rnn(embedded)` → `rnn_out: (B, T, 512)` — LSTM hidden state at every position
> 4. `unembedding(rnn_out)` → `logits: (B, T, V)` — score for each vocabulary word at each position
>
> The model predicts the next token at every position, so logits at position `t` are used to predict token `t+1` (after shifting in the loss).

---

**Q2. What are the three components and why do we need each one?**

> - **Embedding**: maps discrete integer token IDs to continuous vectors. Neural networks require real-valued inputs — integers cannot be fed directly.
> - **LSTM**: processes the sequence left-to-right, maintaining a hidden state that accumulates context. This makes it a language model — each position "remembers" what came before.
> - **Unembedding** (linear): projects from hidden space back to vocabulary scores. This is where the "which word comes next" prediction happens.

---

**Q3. What is `batch_first=True` and why is it needed?**

> By default, PyTorch's LSTM expects `(T, B, E)` — time-first ordering. Setting `batch_first=True` changes this to `(B, T, E)`, which matches the convention used everywhere else. Without it, LSTM would treat the batch dimension as time steps and vice versa, producing wrong results.

---

**Q4. Why do we discard the second LSTM return value `_`? What does it contain?**

> `nn.LSTM` returns `(output, (h_n, c_n))` where `h_n` is the final hidden state and `c_n` is the final cell state. We only need `output` (hidden states at every time step) to compute logits at all positions in one pass. The final states would only matter in step-by-step generation, not during training.

---

**Q5. What is the difference between an LSTM and a basic RNN? Why use LSTM?**

> A basic RNN: `h_t = tanh(W·x_t + U·h_{t-1})`. Gradients vanish over long sequences because they are multiplied by the weight matrix at every step — the model cannot learn long-range dependencies.
>
> An LSTM adds a cell state `c_t` and three gates (input, forget, output) that control how information flows. The cell state is updated additively, providing a gradient "highway" that preserves information over many steps and avoids vanishing gradients.

---

## A1 Task 3.2: Computing the loss (label shifting)

**Code:**
```python
shifted_logits = logits[:, :-1, :].contiguous()     # (B, T-1, V)
shifted_labels = labels[:, 1:].contiguous()          # (B, T-1)
loss = self.loss_func(
    shifted_logits.view(-1, shifted_logits.shape[-1]),  # (B*(T-1), V)
    shifted_labels.view(-1)                             # (B*(T-1),)
)
```

---

**Q1. Why do you shift the logits and labels? What would go wrong without the shift?**

> `logits[:, t, :]` is the prediction made *after seeing* token `t` — it should predict token `t+1`. So:
> - We drop the last logit (`[:, :-1]`) — there is no next token after the final position
> - We drop the first label (`[:, 1:]`) — the first label (BOS) has no logit to predict it from nothing
>
> Without the shift, position `t` would be trained to predict token `t` — the one it just saw. That is trivially easy (look at your own input) and teaches the model nothing about language.

---

**Q2. What are the shapes before and after the shift, and after `.view()`?**

> - `logits`: `(B, T, V)` → `shifted_logits = logits[:, :-1, :]` → `(B, T-1, V)` → `.view(-1, V)` → `(B·(T-1), V)`
> - `labels`: `(B, T)` → `shifted_labels = labels[:, 1:]` → `(B, T-1)` → `.view(-1)` → `(B·(T-1),)`
>
> `CrossEntropyLoss` expects 2D predictions and 1D targets, so we flatten batch and time into one dimension.

---

**Q3. Why is `ignore_index=-100` used, and where do the -100 values come from?**

> In the training loop, padding tokens in `labels` are set to `-100` before the forward pass. `CrossEntropyLoss(ignore_index=-100)` skips those positions entirely — they do not contribute to the gradient. This is necessary because padding is artificial; the model should not be penalised for predicting any particular word after a padding token.

---

**Q4. What is cross-entropy loss and why is it the right loss for language modelling?**

> At each position: `L = -log P(correct_token)` where probability comes from softmax over logits. Minimising cross-entropy maximises the log-likelihood of the training text — which is exactly the definition of a language model. The mean cross-entropy over all positions also defines perplexity: `PPL = exp(mean CE loss)`.

---

## A1 Task 4.1: Implementing the Trainer

**Code (core loop):**
```python
optimizer  = AdamW(model.parameters(), lr=args.learning_rate)
train_loader = DataLoader(train_dataset, batch_size=..., shuffle=True)

for epoch in range(args.num_train_epochs):
    for batch in train_loader:
        input_ids = tokenizer(batch['text'], padding=True, truncation=True, return_tensors='pt')['input_ids']
        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

**Q1. Walk me through one training step in order.**

> 1. **Tokenize** the batch of text strings → `input_ids: (B, T)`
> 2. **Create labels**: clone `input_ids`, replace pad token positions with `-100`
> 3. **Forward pass**: model computes logits and loss (shift + CE internally)
> 4. **Zero gradients**: clear accumulated gradients from the previous step
> 5. **Backward**: compute gradients of loss w.r.t. all parameters
> 6. **Update**: AdamW applies the gradient update to each parameter

---

**Q2. Why call `optimizer.zero_grad()` before `loss.backward()`? What if you forget?**

> PyTorch accumulates gradients by default — `.backward()` adds to existing gradients rather than replacing them. If you skip `zero_grad()`, each update step adds gradients from the current batch on top of all previous batches, causing the model to take wildly incorrect steps and diverge.

---

**Q3. Why `shuffle=True` for training but not for validation?**

> Shuffling prevents the model from memorising example order (a form of overfitting) and acts as additional regularisation. Validation does not need shuffling — we want deterministic, comparable results across runs. The metric is an average over the whole set, so ordering does not change its value, but reproducibility is good practice.

---

**Q4. What is AdamW and why is it preferred over plain SGD?**

> Adam maintains per-parameter adaptive learning rates using estimates of the first and second gradient moments. This makes it much faster and more stable than SGD, especially for parameters with very different gradient magnitudes (common in NLP).
>
> The "W" means weight decay is applied directly to the parameters (not the gradient moments), which is the correct way to regularise Adam — plain Adam applies weight decay to the gradients, which is mathematically inconsistent and performs worse in practice.

---

**Q5. What is early stopping and how is it implemented here?**

> Track `best_val_loss = inf` and `epochs_without_improvement = 0`. After each epoch: if validation loss improved, save the model and reset the counter; otherwise increment. If the counter reaches `patience=2`, stop training. This prevents overfitting — when the model starts memorising the training set, validation loss rises and training stops, keeping only the best-generalising checkpoint.

---

## A1 Task 5.2: Computing Perplexity

**Code:**
```python
avg_loss = total_loss / num_batches
perplexity = torch.exp(torch.tensor(avg_loss))
```

---

**Q1. What is perplexity and how is it computed from the loss?**

> `PPL = exp(mean cross-entropy loss)`. If the average CE loss over the dataset is `L`, then `PPL = e^L`. It measures how "surprised" the model is on average: a perplexity of K means the model is as uncertain as choosing uniformly among K options at each step. Lower is better.

---

**Q2. What perplexity would a random model have? What did your model achieve?**

> A random model assigns probability `1/V = 1/10,000` to every word, giving CE = `log(10,000) ≈ 9.21` and `PPL = 10,000`. After training, the model achieved PPL ≈ 47 — dramatically better than random, and better than the assignment's expected 200–300.

---

**Q3. Why report perplexity instead of just the cross-entropy loss?**

> Perplexity is more interpretable. A drop in loss from 3.9 to 3.5 is not intuitive, but PPL 49 → 33 immediately suggests "the model is now choosing from a third fewer options on average." Both are equivalent for model comparison since `exp` is monotonic, but PPL communicates magnitude more naturally.

---

## A1 Task 5.3: Inspecting the learned word embeddings

**Code:**
```python
sim_func = nn.CosineSimilarity(dim=1)
cosine_scores = sim_func(test_emb, emb.weight)   # compare one vec to all vecs
near_nbr = cosine_scores.topk(n_neighbors+1)
topk_indices = near_nbr.indices[1:]              # skip index 0 (the word itself)
```

---

**Q1. Why cosine similarity instead of Euclidean distance?**

> Frequent words tend to have larger embedding magnitudes because they appear in more training contexts and receive more gradient updates. Euclidean distance would bias the search toward high-magnitude (high-frequency) words regardless of semantic content. Cosine similarity measures the angle between vectors — direction only, not magnitude — which captures semantic relatedness independent of word frequency.

---

**Q2. Why skip `indices[0]`?**

> The closest vector to any word embedding under cosine similarity is always that word's own embedding (similarity = 1.0 exactly). It is trivially the top result. Skipping index 0 gives the actual nearest neighbours from the rest of the vocabulary.

---

**Q3. What kinds of relationships do the learned embeddings capture?**

> The distributional hypothesis: words used in similar contexts end up with similar vectors. The embeddings capture semantic similarity (king ≈ queen, car ≈ truck), syntactic patterns (running ≈ walking ≈ jumping — all progressive verbs), and domain-specific co-occurrence patterns from the training corpus.

---

**Q4. What does the PCA visualisation tell you?**

> PCA (here: TruncatedSVD) projects 128-dimensional embeddings to 2D while preserving as much variance as possible. Clustering in the 2D plot indicates that the learned embeddings have genuinely organised semantically related words into the same region of embedding space. It provides visual confirmation that the model learned something non-trivial.

---

---

# Assignment 2 — Transformer from Scratch

---

## A2 Task 1.3: Multi-head attention with RoPE (`A2Attention`)

**Code:**
```python
# Four projections, all (D, D), all bias=False
self.W_q, self.W_k, self.W_v, self.W_o = ...
self.q_norm = A2RMSNorm(config)
self.k_norm = A2RMSNorm(config)

def forward(self, hidden_states, rope_rotations):
    q, k, v = W_q(x), W_k(x), W_v(x)
    q, k = q_norm(q), k_norm(k)
    # reshape: (B, M, D) → (B, n_h, M, head_dim)
    q, k, v = [t.view(B, M, n_h, hd).transpose(1,2) for t in [q,k,v]]
    q, k = apply_rotary_pos_emb(q, k, rope_rotations)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    out = out.transpose(1,2).reshape(B, M, D)
    return W_o(out)
```

---

**Q1. Walk me through the attention forward pass with shapes. Assume B=2, M=10, D=576, n_h=9.**

> - `head_dim = 576 / 9 = 64`
> - `W_q/k/v(hidden_states)` → `(2, 10, 576)` each
> - After RMSNorm on q, k: `(2, 10, 576)`
> - `.view(2, 10, 9, 64).transpose(1,2)` → `(2, 9, 10, 64)` — split into 9 heads of 64 dims
> - After RoPE: q, k still `(2, 9, 10, 64)`
> - `scaled_dot_product_attention` → `(2, 9, 10, 64)`
> - `.transpose(1,2).reshape(2, 10, 576)` — merge heads back
> - `W_o(...)` → `(2, 10, 576)`

---

**Q2. Why split into multiple heads? What does each head independently learn?**

> Multi-head attention lets the model attend to different representation subspaces simultaneously. One head might focus on syntactic agreement (subject-verb), another on coreference (pronoun → noun), another on local context (adjacent words). A single head would have to average all these signals into one representation, losing information. Using H heads with dimension `D/H` each gives the same parameter count as one head of dimension D but much richer representations.

---

**Q3. What is causal masking (`is_causal=True`) and why is it essential for a language model?**

> Causal masking prevents position `i` from attending to positions `j > i` — it can only see its own past. This is enforced by setting attention scores for future positions to `-∞` before softmax, making their attention weights exactly zero.
>
> Without it, the model would see future tokens during training (a form of cheating) and learn trivial shortcuts that completely fail at inference time when future tokens do not exist. A causal language model by definition generates one token at a time conditioned only on the past.

---

**Q4. What is the scaled dot-product attention formula?**

> ```
> scores  = Q @ Kᵀ / √head_dim    # (B, n_h, M, M)
> scores  = mask(scores)           # set upper triangle to -inf
> weights = softmax(scores, dim=-1)
> output  = weights @ V            # (B, n_h, M, head_dim)
> ```
> The `1/√head_dim` scaling prevents dot products from becoming very large when `head_dim` is large, which would push softmax into near-one-hot territory and cause vanishing gradients.

---

**Q5. What is RoPE and why is it applied to Q and K but not V?**

> RoPE (Rotary Position Embeddings) encodes position by rotating Q and K vectors. The rotation at position `p` is a fixed rotation matrix `R_p`. The key property: `Q_i · K_j` (dot product of rotated vectors) encodes the relative position `i − j` — so attention scores automatically incorporate position without adding separate positional embeddings to the input.
>
> V is not rotated because it holds the *content* to be retrieved — what information to pass along. Position information needs to enter through the attention scores (Q·K), not through the values.

---

**Q6. Why is there a separate RMSNorm on Q and K? Is this standard?**

> No — this is OLMo-2 specific (also seen in some other recent LLMs). Q and K projections can produce vectors of very different magnitudes, which destabilises attention scores at scale. Normalising Q and K before RoPE keeps them on a consistent scale and improves training stability. The standard Transformer from "Attention is All You Need" does not do this.

---

## A2 Task 1.4: Transformer Decoder Layer (`A2DecoderLayer`)

**Code:**
```python
def forward(self, hidden_states, rope_rotations):
    out     = self.attention(hidden_states, rope_rotations)
    out     = self.norm1(out) + hidden_states    # norm on sublayer OUTPUT, then residual
    out_mlp = self.mlp(out)
    out     = self.norm2(out_mlp) + out          # norm on sublayer OUTPUT, then residual
    return out
```

---

**Q1. Describe the three normalisation conventions (post-norm, pre-norm, this assignment's variant) and how they differ.**

> **Post-norm** (original "Attention is All You Need"):
> `out = norm(sublayer(x) + x)` — norm applied *after* the residual sum.
>
> **Pre-norm** (LLaMA, Mistral, most modern LLMs):
> `out = sublayer(norm(x)) + x` — norm applied to the sublayer *input*; the raw residual bypasses the norm.
>
> **This assignment** (OLMo-2 style):
> `out = norm(sublayer(x)) + x` — norm applied to the sublayer *output*, then the un-normed residual is added.
>
> All three produce stable training. Pre-norm and this variant are generally preferred over post-norm because they prevent gradient explosions in very deep networks.

---

**Q2. What is the purpose of the residual connection?**

> Adding the input `x` directly to the sublayer output creates a gradient "shortcut" that allows gradients to flow through many layers without passing through any nonlinearity. Without residual connections, gradients would vanish in deep networks (they get multiplied by the Jacobian of each layer at every step). With residuals, the gradient can always take the identity path, enabling training of Transformers with dozens or hundreds of layers.

---

**Q3. If the model has 30 layers, what is the shape of the tensor that flows through all 30?**

> Always `(B, M, hidden_size)`. Each decoder layer maps `(B, M, D) → (B, M, D)`. The residual connection enforces this: the attention and MLP outputs must match the input shape for the addition to be valid. Only the input embedding and the final unembedding layer change dimensionality.

---

## A2 Task 3.2: Text generation (`generate_text`)

**Code:**
```python
generated = tokenizer([prompt])['input_ids'][:, :-1]  # remove EOS
for _ in range(max_length):
    logits = model(generated).logits[0, -1, :] / temperature
    topk_logits, topk_indices = torch.topk(logits, topk)
    next_id = topk_indices[Categorical(logits=topk_logits).sample()]
    generated = torch.cat([generated, next_id.reshape(1,-1)], dim=1)
    if next_id == tokenizer.eos_token_id:
        break
```

---

**Q1. Walk me through the generation loop.**

> 1. Tokenize the prompt, strip the EOS token (don't want model to see EOS mid-sequence)
> 2. Forward pass → `logits[0, -1, :]`: the next-token prediction at the last position, shape `(V,)`
> 3. Divide by temperature to scale the distribution
> 4. Take top-k logits; sample from the resulting Categorical distribution
> 5. Append sampled token to `generated`
> 6. Stop if EOS sampled or `max_length` reached

---

**Q2. What does temperature do? What happens at the extremes?**

> `logits / temperature` then softmax. Temperature scales the "peakiness" of the distribution:
> - **T < 1** (e.g., 0.5): sharper → model more confident → repetitive, conservative text
> - **T = 1**: unchanged distribution
> - **T > 1** (e.g., 2.0): flatter → more random → creative but potentially incoherent
> - **T → 0**: equivalent to greedy decoding (always pick the argmax token)
> - **T → ∞**: uniform random sampling over the vocabulary

---

**Q3. What is top-k sampling and why use it?**

> Restrict the vocabulary to the k most likely tokens, set all others to `-∞`, then sample from those k. Prevents the model from sampling very low-probability tokens that produce clearly wrong or incoherent words, while still allowing variety among the most plausible options. `k=1` is greedy (deterministic); `k=50` is a common default for diverse but sensible generation.

---

**Q4. What does `logits[0, -1, :]` mean? Why index -1?**

> - `[0]`: first (only) item in the batch
> - `[-1]`: the last token position — where the model predicts what comes *next*
> - `[:]`: all vocabulary logits at that position
>
> This is the autoregressive pattern: at each step, we only care about the prediction at the rightmost position because that's where the model's estimate of the next token lives.

---

**Q5. Why remove EOS from the prompt before generation?**

> The tokenizer appends EOS automatically. If we kept it, the model sees `[... tokens, EOS]` as the prompt and may immediately produce another EOS (since EOS → EOS has high probability — "the sequence ended, so it ends again"). Removing EOS ensures the model treats the prompt as an open-ended prefix to continue.

---

## A2 Task 3.3: Comparing to a pre-trained Transformer (`copy_olmo_weights`)

**Code:**
```python
def copy_olmo_weights(olmo_model):
    mapping = {
        'model.embed_tokens.weight': 'token_embedding.weight',
        'model.norm.weight': 'top_level_norm.gamma',
        'lm_head.weight': 'unembedding.weight',
        # + per-layer mappings for attn + MLP weights
    }
    for olmo_key, our_key in mapping.items():
        assert olmo_sd[olmo_key].shape == our_sd[our_key].shape
        our_sd[our_key] = olmo_sd[olmo_key]
    model.load_state_dict(our_sd)
```

---

**Q1. What does this function do and why is it useful?**

> It copies weights from a publicly available pretrained OLMo-2 1B model into your `A2Transformer` implementation. This verifies that your architecture is structurally identical to OLMo-2 (the shape assertions would fail if anything were mismatched), and lets you run a 1B-parameter model through your own code, comparing its language quality to your small trained model.

---

**Q2. How does the weight mapping work? What does the shape assertion check?**

> OLMo-2 uses its own naming conventions (e.g., `model.layers.0.self_attn.q_proj.weight`) while your implementation uses different names (e.g., `transformer_decoder_layers.0.attention.W_q.weight`). The mapping dictionary translates between the two. The shape assertion `olmo_sd[olmo_key].shape == our_sd[our_key].shape` ensures that the corresponding layers are dimensionally identical — if shapes matched, the architecture is verified to be compatible.

---

**Q3. How does OLMo-2 1B compare to your trained small model? What explains the differences?**

> | Aspect | Small model (~8M params, your model) | OLMo-2 1B |
> |---|---|---|
> | Perplexity | ~47 | ~10–20 |
> | Coherence | Breaks after a few words | Multi-sentence coherent text |
> | Vocabulary | Many `<UNK>` (10k vocab) | Full coverage, no unknowns |
> | Factual accuracy | None | Mostly correct for common facts |
>
> Scale explains the differences: 1B parameters can store far more language patterns and world knowledge. More training data provides more diverse examples. A larger vocabulary eliminates OOV tokens. Longer training improves convergence.

---

---

# Assignment 3 — Fine-tuning with LoRA

---

## A3 Task 1.2: Formatting data for instruction tuning (`format_input_output`)

**Code:**
```python
def format_input_output(example):
    start, end = "<|im_start|>", "<|im_end|>"
    system_content = user_content = assistant_content = ""
    for message in example['messages']:
        if   message['role'] == 'system':    system_content    += f"{start}system\n{message['content']}{end}\n"
        elif message['role'] == 'user':      user_content      += f"{start}user\n{message['content']}{end}\n"
        elif message['role'] == 'assistant': assistant_content += message['content'] + f"{end}\n"
    return {"prompt":   system_content + user_content + f"{start}assistant\n",
            "response": assistant_content}
```

---

**Q1. What format did you choose and why?**

> I used **ChatML format** (`<|im_start|>role\ncontent<|im_end|>`). SmolLM2-135M was pretrained on data already formatted in ChatML — these special tokens exist in its vocabulary and appeared billions of times. Fine-tuning with the same template means the model immediately understands the structural markers. Any other format (e.g., Llama's `[INST]...[/INST]`) would be tokenised into sub-word pieces that SmolLM2 doesn't recognise as structural delimiters.

---

**Q2. What is the `prompt` / `response` split? Why does it matter?**

> - **Prompt**: system + user message + `<|im_start|>assistant\n` — the input the model conditions on
> - **Response**: assistant content + `<|im_end|>` — what the model must generate
>
> The split is used in tokenization to create the label mask: prompt positions get `-100` (excluded from loss), response positions get their real token IDs (included in loss). Without this split, the model would also be trained to "predict" the prompt, which is wasteful and can hurt convergence since the prompt is given context, not generated output.

---

**Q3. Why is the system prompt optional? How does your code handle its absence?**

> The dataset doesn't always include a system message. The code initialises `system_content = ""` and only appends to it when a `system` role message is found. When there's no system message, the prompt starts directly with `<|im_start|>user\n...` with no preceding system block — this is valid ChatML.

---

**Q4. What would happen if you used ChatML format with a model not pretrained on it?**

> The tokenizer would split `<|im_start|>` into its character sub-words (e.g., `<`, `|`, `im`, `_`, `start`, `|`, `>`). The model would see a sequence of unusual tokens with no learned meaning for the structural role. Fine-tuning might eventually teach the structure, but it would be much slower and the model might never associate these character fragments with turn-taking.

---

## A3 Task 2.2: Why is baseline ROUGE-L so high?

**Result:** Pretrained SmolLM2-135M (no instruction tuning) scores ROUGE-L = 0.564 on the test set.

---

**Q1. Why is the baseline ROUGE-L so high if the model has never been trained to follow instructions?**

> The HuggingFace Trainer evaluates using **teacher forcing**: the full `input_ids = [prompt_tokens] + [response_tokens]` is fed to the model in one forward pass. At each response position, the model sees all prior gold tokens as context — including earlier response tokens.
>
> This is much easier than generating a response from scratch. The model doesn't need to know what to say — it just needs to predict the next token given the actual previous tokens, which a strong language model can do well. The baseline ROUGE-L of 0.564 is inflated by this information leakage.

---

**Q2. How would you evaluate genuine instruction-following?**

> Use **autoregressive generation**: give the model only the prompt tokens and generate the response token by token with no access to gold response tokens. This is what Task 4.4's `generate_response` function does. The pretrained model's actual generated outputs (ignoring the instruction, repeating patterns from pretraining) reveal it cannot follow instructions at all — confirming that teacher-forcing ROUGE-L is not a reliable measure of instruction-following.

---

## A3 Task 3.1: Full SFT — results and comparison

---

**Q1. How do the results change after one epoch of full SFT?**

> | | Pretrained | After 1 epoch SFT |
> |---|---|---|
> | eval_loss | 2.48 | 1.09 |
> | ROUGE-L | 0.564 | 0.668 |
> | Trainable params | 134.5M | 134.5M |
>
> The loss dropped significantly and ROUGE-L improved. More importantly, in autoregressive generation (Task 4.4), the SFT model actually answers the question correctly — the pretrained model does not.

---

**Q2. Why does the SFT model still sometimes generate repetitive loops?**

> Repetition loops are a failure mode of greedy or low-diversity decoding on small models. Once the model generates a high-probability sequence, that same sequence becomes the highest-probability continuation in its own right (the model has seen many "assistant: X" followed by "assistant: X" patterns). The underlying cause is the model not reliably generating `<|im_end|>` as an EOS signal. Fix: pass `eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")` to `model.generate()`.

---

## A3 Task 4.3: Fine-tuning with LoRA — setup and results

**Code:**
```python
lora_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
for p in lora_model.parameters():
    p.requires_grad = False                            # freeze everything
lora_layers = extract_lora_targets(lora_model)         # get q_proj, v_proj
lora_layers = {n: LoRALayer(l, r=8, alpha=16) for n, l in lora_layers.items()}
lora_model  = replace_layers(lora_model, lora_layers)  # plug in LoRA layers
```

---

**Q1. Walk me through exactly how LoRA is set up before training.**

> 1. Load the pretrained model
> 2. Freeze every parameter (`requires_grad = False` for all 134.5M)
> 3. Extract `q_proj` and `v_proj` layers from every attention block
> 4. Wrap each in `LoRALayer(W, r=8, alpha=16)` — this creates new `lora_a` (8×576) and `lora_b` (576×8) nn.Parameters, which default to `requires_grad=True`
> 5. Replace the original layers back into the model with `replace_layers`
>
> Result: 134.5M frozen params + 460K trainable LoRA params.

---

**Q2. Compare full SFT vs LoRA results. What is the trade-off?**

> | | Pretrained | Full SFT | LoRA (r=8) |
> |---|---|---|---|
> | eval_loss | 2.48 | 1.09 | 1.38 |
> | ROUGE-L | 0.564 | 0.668 | 0.643 |
> | Trainable params | 134.5M | 134.5M | 460K (0.34%) |
>
> LoRA achieves surprisingly close performance to full SFT while training only 0.34% of the parameters. The trade-off: slightly lower quality in exchange for drastically lower GPU memory usage (no gradient/optimizer state for the frozen 134.5M parameters) and potentially faster iteration.

---

## A3 Task 4.4: Qualitative inspection

---

**Q1. What did the three models produce and what does it show?**

> | Model | Behaviour |
> |---|---|
> | **Pretrained** | Ignores the instruction; continues with patterns from pretraining (e.g., repeating similar questions, topic drift) |
> | **SFT** | Correctly follows the instruction format; relevant first sentence; may loop |
> | **LoRA** | Also follows instructions; often the most concise and direct first response despite 0.34% of the parameters |
>
> This demonstrates that SFT teaches instruction-following and that LoRA achieves comparable quality at a tiny fraction of the training cost.

---

**Q2. Why does the pretrained model loop/repeat itself?**

> Without instruction tuning, the model has no concept of "answer and stop." Given the ChatML prompt, the most probable continuation in its training distribution is a similar question-answer pair from raw text. Each generated token then reinforces the next high-probability token from the same pattern, creating a deterministic loop. SFT teaches the model to produce `<|im_end|>` after the response, breaking the loop.

---

---

# Assignment 4 — Retrieval-Augmented Generation

---

## A4 Task 3.3: Vector store (Chroma)

**Code:**
```python
vector_store = Chroma.from_documents(
    documents=texts,
    embedding=embeddings,
    collection_name="pqal_collection",
    collection_metadata={"hnsw:space": "cosine"}
)
results = vector_store.similarity_search_with_score("What is programmed cell death?", k=3)
```

---

**Q1. What is a vector store and what does it do?**

> A vector store is a database that stores embedding vectors alongside their source documents and supports fast approximate nearest-neighbour (ANN) search. `Chroma.from_documents` embeds all 3,510 chunks using the embedding model and stores the vectors. At query time, `similarity_search` embeds the query and returns the k chunks with the highest cosine similarity to the query vector.

---

**Q2. What is HNSW and why is it used instead of brute-force search?**

> HNSW (Hierarchical Navigable Small World) is an ANN index algorithm. It builds a layered graph where each chunk connects to its approximate nearest neighbours. Search navigates this graph in O(log n) time instead of comparing to all n vectors (O(n)). For 3,510 chunks this does not matter much, but for millions of documents the difference is enormous. `hnsw:space: "cosine"` configures the graph to use cosine distance.

---

**Q3. The scores returned by `similarity_search_with_score` are 0.37, 0.62, 0.64. What do these mean?**

> These are **cosine distances** (= 1 − cosine similarity), not similarities. Lower = more similar. The best match (`SIM=0.37`) has cosine similarity `1 − 0.37 = 0.63`, and the others are less similar. The naming `similarity_search_with_score` is misleading — the returned score is actually a distance. Always prefer the result at index 0 (lowest distance = highest similarity).

---

**Q4. Why cosine distance instead of L2 (Euclidean) distance?**

> For text embeddings, direction matters more than magnitude. Documents about the same topic should be similar regardless of their length (longer documents → larger L2 norm). Cosine distance is scale-invariant — it only measures the angle between vectors. Combined with normalised embeddings (unit vectors), cosine distance = `1 − dot_product`, which is extremely fast to compute.

---

## A4 Task 4.1: RAG pipeline (LCEL)

**Code:**
```python
retriever = vector_store.as_retriever()
template = "Answer the question based only on the following context:\n{context}\n\nQuestion: {question}\n\nAnswer yes or no:"
prompt = ChatPromptTemplate.from_template(template)

retrieval_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt | model | StrOutputParser()
)
```

---

**Q1. Walk me through what happens when `retrieval_chain.invoke(question)` is called.**

> 1. The input `question` string is fanned out to two runnables in parallel:
>    - `retriever`: embeds the question and fetches the top-k most similar chunks from Chroma
>    - `RunnablePassthrough()`: passes the question string through unchanged
> 2. Both outputs are assembled into `{"context": chunks, "question": question}`
> 3. `ChatPromptTemplate` fills in the template → a formatted prompt string
> 4. The LLM (Qwen) generates a response
> 5. `StrOutputParser()` extracts the plain string
>
> Final output: a string, ideally "Yes" or "No."

---

**Q2. What is `RunnablePassthrough()` and why is it needed?**

> It is a LangChain identity runnable — passes its input through unchanged. It is needed because the `RunnableParallel` block must fan the single `question` string out to two places: the retriever (to search) and the prompt template (to fill in `{question}`). Without `RunnablePassthrough()`, the question would only reach the retriever and `{question}` in the prompt would be unfilled.

---

**Q3. Why prompt with "answer yes or no"?**

> PubMedQA has binary yes/no labels. Constraining the output to "yes or no" makes it parseable with a simple `text.startswith("yes")` check, avoids long explanations, and keeps the invalid-response rate low. Without this constraint, the model generates multi-sentence answers that require more sophisticated parsing and have higher invalid rates.

---

## A4 Task 5.1: High-level evaluation

**Results:**
```
Baseline  — Accuracy: 0.626, F1: 0.767, Invalid Rate: 0.065
RAG       — Accuracy: 0.649, F1: 0.770, Invalid Rate: 0.000
```

---

**Q1. What metrics did you use and why those?**

> - **Accuracy**: simple fraction correct. Intuitive but sensitive to class imbalance.
> - **F1 (binary, pos_label="yes")**: harmonic mean of precision and recall for the "yes" class. Robust to class imbalance — if most labels are "yes", a model that always says "yes" gets high accuracy but low recall for "no." F1 catches this.
> - **Invalid rate**: fraction of outputs that are neither "yes" nor "no." Measures how reliably the model follows the output constraint.

---

**Q2. RAG improved accuracy by only 2.3 points despite 98.4% retrieval accuracy. Why is the improvement so small?**

> The bottleneck is the LLM, not retrieval. Even with the correct document retrieved, Qwen-0.5B doesn't always correctly extract and apply the answer from the context — it sometimes overrides the context with its prior beliefs. Additional causes: k=1 retrieval (only one chunk per question, which may miss relevant information), and the model's limited reading-comprehension ability at 0.5B parameters.

---

**Q3. The RAG invalid rate is 0.000 but the baseline is 0.065. Why?**

> The retrieved context gives the model concrete text to react to, which anchors it more firmly on the "yes or no" constraint. Without retrieval, the model sometimes generates long explanations instead of a binary answer (especially for harder questions where it is uncertain). Having specific evidence to respond to makes it easier to produce a short, constrained answer.

---

## A4 Task 5.2: Detailed inspection

**Results:** Retrieval accuracy (k=1) = 0.984

---

**Q1. What does retrieval accuracy of 98.4% tell you?**

> For 98.4% of questions, the top-1 retrieved chunk came from the correct gold document. The embedding model (`all-mpnet-base-v2`) is extremely effective at matching medical questions to their source abstracts in this dataset. The semantic similarity between a question and its corresponding abstract is reliably higher than to any other document.

---

**Q2. Describe the four possible combinations of retrieval success/failure and model correctness. Which is most interesting?**

> | Retrieval | Model | Interpretation |
> |---|---|---|
> | ✓ correct doc | ✓ correct answer | RAG working as intended |
> | ✗ wrong doc | ✓ correct answer | Model's prior knowledge sufficient (RAG not needed) |
> | ✓ correct doc | ✗ wrong answer | **Most interesting** — model fails to use the context |
> | ✗ wrong doc | ✗ wrong answer | Both components failed — worst case |
>
> The third case (correct retrieval but wrong answer) is most instructive: it means the bottleneck is the LLM's reading comprehension, not the retrieval. This is consistent with the small RAG vs. baseline improvement.

---

**Q3. What would you improve to get higher accuracy?**

> 1. **Retrieve more chunks** (k=3 or 5): more evidence, less chance of missing the key fact
> 2. **Larger LLM**: better reading comprehension from context
> 3. **Re-ranking**: use a cross-encoder to re-rank the top retrieved chunks by relevance before passing to LLM
> 4. **Chain-of-thought**: ask the model to reason before giving the binary answer
> 5. **Better chunking**: semantic splitting at paragraph boundaries rather than fixed character count

---
