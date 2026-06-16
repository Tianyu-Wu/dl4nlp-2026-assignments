# DL4NLP Lecture Notes — Course Content Summary

Based on slides from https://liu-nlp.ai/dl4nlp/units/

---

## Unit 1: Tokenisation and Embeddings

---

### 1.1 Introduction to Tokenisation

**What is tokenisation?**
Breaking running text into smaller segments (words, subwords, characters). It is the first step in mapping text to a numerical representation.

**Pipeline overview:**
```
str → tokenise → list[str] → encode → Tensor[int] → embed → Tensor[float]
"one ring to"  →  ["one","ring","to"]  →  [28993,31365,15169]  →  [[-0.82,0.37], ...]
```

**Whitespace tokenisation:**
```python
def tokenize(text): return text.split()
vocab = set(tokenize(text))
stoid = {s: i for i, s in enumerate(vocab)}
```
Problem: doesn't handle punctuation, contractions, or non-English scripts.

**Regex tokenisation:**
```
re.findall(r"[A-Za-z]\.|\w+(?:-\w+)*|'\w+|[^\w\s]+", text)
```
Handles: single letters with periods (J.R.R.), hyphenated words, genitives ('s), punctuation.

**Text normalisation:** Converting text to a standard form. Techniques: case folding, harmonisation (color→colour), lemmatisation (runs→run), removing punctuation. Less commonly used today with subword models.

**Heaps' Law:** Vocabulary grows unboundedly with more text. New text always contains unknown words (OOV problem).

**Handling OOV words:**
1. Build vocabulary with frequency threshold
2. Add `[UNK]` special token
3. Replace OOV words with `[UNK]` at test time

**Problem with word tokenisation:**
- Concept of "word" is not universal (Chinese has no spaces; Inuktitut forms whole sentences as single words)
- Vocabulary size explodes; OOV inevitable

**Three tokenisation options:**
1. **Words:** Universal concept missing; OOV problem
2. **Characters:** Too small a unit for learning
3. **Subwords:** Best of both — words composed of morphemes ← the modern choice

---

### 1.2 Byte Pair Encoding (BPE)

**BPE Algorithm** (Sennrich et al., 2016):
1. Encode text as a sequence of bytes; initialise vocabulary with all single bytes (256 tokens for UTF-8)
2. Find the most frequent consecutive token pair
3. Merge that pair into a new single token; add it to vocabulary
4. Repeat until vocabulary reaches the target size

**Unicode and UTF-8:**
- Unicode: standard supporting all writing systems; v16 supports 154,998 characters from 168 scripts
- UTF-8: variable-width encoding, 1–4 bytes per character (ASCII = 1 byte; many Asian scripts = 3 bytes)
- The first 128 codepoints of Unicode = ASCII (backwards compatibility)

**BPE example trace:**
Starting bytes → merge `e ` → `[e ]` → merge `t+h` → `[th]` → merge `d+ ` → `[d ]` → merge `e+r` → `[er]` → ...
Tokens gradually grow from bytes to morpheme-like units to whole words.

**Properties of BPE tokens:**
- Span variable lengths of source text (characters → words → multi-word)
- Not guaranteed linguistic meaning, but often resemble morphemes → "poor man's morphology"
- Solves OOV: every text can be tokenised (worst case: byte-by-byte)

**Tokenisation in major LLMs:**

| Model | Method | Vocab size |
|-------|--------|-----------|
| BERT | WordPiece | 30K |
| GPT-2 | BPE | 50K |
| GPT-3.5 | BPE | 100K |
| GPT-4o | BPE | 200K |
| Llama 3 | BPE | 128K |

---

### 1.3 Tokenisation Fairness

**Fairness in AI:** Whether model behaviours systematically advantage/disadvantage certain users. Bias enters at: data collection, model architecture, optimisation, deployment. Fairness is contextual — no single technical fix.

**Tokenisation premium** (Petrov et al., 2023): The ratio of token count for a language to token count for English on the same content.

| Language | GPT-4 premium |
|----------|---------------|
| Spanish | 1.2× |
| Swedish | 1.58× |
| German | 1.5× |
| Hindi | 4.8× |
| Shan | 15.05× |

**Consequences of high tokenisation premium:**
- **Higher latency:** Users wait longer for same content
- **Higher cost:** LLM APIs charge per token; users of penalised languages pay more (GPT-4 pricing is per-token)
- **Lower quality:** Fixed context windows → less effective content per window

**Root cause:** BPE uses byte sequences from UTF-8. High Unicode codepoints (non-Latin scripts) require more UTF-8 bytes, giving fewer tokens per character. English = 1 byte/char, Shan = 4 bytes/char.

**Compression rate:** `CR(b) = |b| / |tokenize(b)|` — ratio of original bytes to number of tokens. Higher compression = better efficiency for that language.

**Proposed fixes:**
- **Block-structured encoding (SCRIPT, Land & Arnett 2025):** Maps each codepoint to a block token + index token, decoupling encoding from Unicode codepoint range
- **Parity-aware BPE (Foroutan et al., 2025):** Instead of always merging the globally most frequent pair, find the next merge that most improves the language with the currently *worst* compression rate. Apply that merge to the full corpus.

**Key takeaway:** Tokenisation encodes decisions about which languages get fast, cheap, expressive technology. Fairness metrics can be optimised during training, not just evaluated after. The problem is not "solved" — each fix introduces new trade-offs.

---

### 1.4 Introduction to Embeddings

**Embedding layer:** Maps each token (integer) to a fixed-size vector of floats. Initially random; tuned via backpropagation during training.

**PyTorch:**
```python
emb = torch.nn.Embedding(num_embeddings=3, embedding_dim=2)
emb(torch.tensor([0, 1, 2]))  # returns (3, 2) tensor
```

**Embedding as linear layer:** An embedding lookup is equivalent to multiplying a one-hot vector by the weight matrix W of shape `(V × d)`. This means embeddings are differentiable.

**Bag-of-words classifier:**
```
tokens → Embed each → mean pooling → Linear → softmax → class
```
Simple but loses word order.

**Transfer learning with embeddings:**
- Pre-train embeddings on a general task (e.g., language modelling on raw text)
- Re-use to initialise embedding layer for a downstream task
- Option 1: Fine-tune (continue updating) — adapts to the task
- Option 2: Freeze weights — preserves general representations

**Why language modelling for pre-training?**
- Uses raw text → data is extremely abundant
- Forces the model to learn generally useful representations of language

---

### 1.5 Word Embeddings

**Problem with one-hot vectors:** Size = vocabulary (thousands of dimensions); no notion of similarity between words.

**Word embeddings:** Dense vectors, typically 50–300 dimensions; support similarity; learnable from data.

**Distributional hypothesis:** "You shall know a word by the company it keeps" (Firth, 1957). Words with similar distributions (appear in similar contexts) have similar meanings.

**Co-occurrence matrix:** Count how often words appear together in a context window.
Entry `M[i,j]` = number of times word `i` co-occurs with word `j`.

**Cosine similarity:**
$$\cos(a, b) = \frac{a \cdot b}{\|a\| \|b\|}$$

**Two approaches to learning word embeddings:**
1. **Count-based (matrix factorisation):** Minimise the reconstruction error between the co-occurrence matrix and a low-rank approximation. Example: GloVe.
2. **Prediction-based (neural):** Train a network to predict context words from a target word (or vice versa). Embeddings emerge as side-effect. Example: Word2Vec (skip-gram, CBOW).

**Evaluation methods:**
- **Visualisation:** PCA/t-SNE/UMAP to 2D — inspect clusters
- **Similarity benchmarks:** "odd one out" (breakfast lunch dinner **surgery**)
- **Analogy benchmarks:** `woman - man + brother = ?` → should be `sister`

---

### 1.6 Contextualised Word Embeddings

**Problem with static embeddings:** Each word type gets a *single* vector regardless of context. Cannot handle polysemy:
- "The children **play** in the park." (verb)
- "The **play** premiered yesterday." (noun)

**Contextualised embeddings:** Each token gets a representation that depends on its full context in the sentence.

**ELMo (Peters et al., 2018) — Embeddings from Language Models:**
- Pre-trained as a **bidirectional language model** (both forward and backward)
- Architecture: character-level CNN for input → two bidirectional LSTM layers with residual connections → softmax output
- Final ELMo representation: **task-specific weighted sum** of all intermediate representations
  - `ELMo_token = γ · (s₀·h₀ + s₁·h₁ + s₂·h₂)` where weights s are learned per task
- The base ELMo model is frozen after pre-training; task-specific weights are learned for each downstream task
- Can optionally fine-tune the whole ELMo model

**ELMo character CNN details:**
- Filter widths 1–7, channels 32–1024
- Followed by max pooling then **highway layers**
- Highway layer: `y = t ⊙ f(Wx) + (1-t) ⊙ x` where `t = sigmoid(W_t x)` is a learned gate
  - Allows the network to selectively pass or transform the input at each layer
  - Enables training of very deep character-level models

**ELMo improvements over non-contextual baselines:**
| Task | Baseline | +ELMo | Relative ↑ |
|------|----------|-------|------------|
| SQuAD (QA) | 81.1 | 85.8 | +5.7% |
| Coreference | 67.2 | 70.4 | +4.7% |
| SST-5 (sentiment) | 54.7 | 57.1 | +4.4% |
| SNLI (entailment) | 88.0 | 88.7 | +0.8% |

**Historical context:** ELMo (2018) is the first widely-used contextualised embedding approach. It was soon superseded by BERT (2019), which uses a Transformer instead of LSTM.

---

## Unit 2: LLM Architectures

### 2.1 Attention

**Motivation — contextual embeddings:**
Static word embeddings assign one fixed vector per word type. But word meaning depends on context:
- "Dogs may **bark** at strangers." vs "Trees shed **bark** in winter."
- "Children **play** games." vs "The **play** flopped."

Contextual embeddings assign different vectors to the same word depending on its context. Attention is the mechanism that achieves this inside the Transformer.

**Attention as contextual embedding:**
Each token's new representation `h_i^{l+1}` is computed as a **weighted sum** of all token vectors at the previous layer:

$$h_i^{\ell+1} = \sum_j \alpha_{ij} \, h_j^{\ell}$$

The weights `α_ij` express how much token `i` should "attend to" token `j`. They are proportional to the **vector similarity** between `h_i` and `h_j`, and are learned during training.

**Scaled dot-product attention:**
Similarity is computed as a scaled dot product:

$$h_i \cdot h_j = \frac{h_i^\top h_j}{\sqrt{d}}$$

Then softmaxed to get the weights:

$$\alpha_i = \text{softmax}\!\left(\frac{h_i^\top [h_1;\ldots;h_n]}{\sqrt{d}}\right)$$

The `1/√d` scaling prevents large dot products from pushing the softmax into saturation (near-zero gradients).

**Queries, keys, and values:**
In general, attention separates three roles with learned linear projections:
- **Query** `q`: what the current position is looking for
- **Keys** `K`: what each position "advertises" about itself
- **Values** `V`: the actual content to be retrieved

Output: `α = softmax(q, K)` then `result = α · V`

Formally (Vaswani et al., 2017):
$$\alpha_i = \text{softmax}(q, K), \quad q \in \mathbb{R}^{d_Q},\; K \in \mathbb{R}^{n \times d_K},\; V \in \mathbb{R}^{n \times d_V},\; d_Q = d_K$$

**Attention in the Transformer:**
Learned input projections `W_Q, W_K, W_V` map the input to queries, keys, values. An output projection `W_O` maps the result back.
```
input → Linear_Q → queries
input → Linear_K → keys
input → Linear_V → values
scaled dot-product attention → output → Linear_O → output
```

**PyTorch implementation:**
```python
# q, k, v shape: [num_words, d]
scores = q @ k.transpose(-1, -2) / hidden_dim**0.5   # [num_words, num_words]
alphas = F.softmax(scores, dim=-1)                    # [num_words, num_words]
result = alphas @ v                                   # [num_words, d]
```

**Multi-head attention:**
Run `h` attention heads in parallel, each operating on a lower-dimensional projection:
```
input (1024) → split into 2 heads of 512 each
  Head 1: Linear_Q1, Linear_K1, Linear_V1 → scaled dot-product attention → 512
  Head 2: Linear_Q2, Linear_K2, Linear_V2 → scaled dot-product attention → 512
concat(Head 1, Head 2) → 1024 → Linear_O → 1024 output
```
Different heads can attend to different aspects of the input simultaneously.

---

### 2.2–2.3 The Transformer Architecture

**History:** Vaswani et al. (2017), "Attention is All You Need" — original design for machine translation (encoder-decoder).

**Three types of Transformers:**

| Type | Models | Output | Examples |
|------|--------|--------|---------|
| Encoder-only | f(X) = representation | Vectors | BERT, RoBERTa |
| Decoder-only | P(Y) autoregressive | Text | GPT, Llama |
| Encoder-decoder | P(Y\|X) | Text | T5, original Transformer |

**Transformer encoder block — the repeating unit:**
1. **Multi-Head Attention (MHA):** Each token attends to all other tokens
2. **Add & Norm:** Residual connection + Layer normalisation
3. **Feed-Forward Network (FFN):** Two linear layers with ReLU; applied per token
4. **Add & Norm:** Residual connection + Layer normalisation

**Matrix form of self-attention:**
- Project to queries, keys, values: `Q = W_Q X`, `K = W_K X`, `V = W_V X`
- Compute scores: `Scores = QKᵀ / √d`  (shape: seq_len × seq_len)
- Attention output: `A = softmax(Scores) V`

**Multi-Head Attention (MHA):**
- Run `h` parallel attention heads, each with its own `W_Q^h, W_K^h, W_V^h`
- Each head produces output of size `d/h` (head_dim)
- Concatenate all heads → project with `W_O` back to model dimension `d`
- Benefit: different heads can attend to different aspects / positions simultaneously

**FFN in the Transformer:**
```
FFN(x) = W₂ · ReLU(W₁ · x)
```
- W₁: d → 4d (expansion)
- W₂: 4d → d (compression)
- Applied independently at each position

**Residual connections** (He et al., 2016): `output = sublayer(x) + x`. Allow gradients to flow directly through the network, enabling training of very deep models.

**Layer Normalisation** (Ba et al., 2016): Normalise each position's representation across features. Stabilises training.

**Normalisation placement:**
- **Post-norm** (original Vaswani 2017): `x' = LayerNorm(x + sublayer(x))` — harder to train at scale
- **Pre-norm** (modern LLMs — Llama, GPT-3): `x' = x + sublayer(LayerNorm(x))` — more stable

**GPT-style decoder-only Transformer:**
- Same as encoder but with **causal (triangular) mask** on attention
- Each position can only attend to itself and *earlier* positions
- Upper triangle of attention score matrix = -∞ → softmax → zero weight
- Enables autoregressive text generation

**Encoder-decoder (original Transformer for MT):**
- Encoder: processes source sentence (fully bidirectional)
- Decoder: same as decoder-only BUT adds a **cross-attention layer** between self-attention and FFN
  - Cross-attention queries come from decoder, keys+values from encoder output

---

### 2.4 Representing Positions in Transformers

**Problem:** Self-attention is permutation-invariant — if you shuffle the input tokens, you get the same output in shuffled order. Position must be explicitly encoded.

**Sinusoidal encoding (Vaswani et al., 2017):**
$$PE_{2i}(p) = \sin\left(p \cdot 10000^{-2i/d}\right), \quad PE_{2i+1}(p) = \cos\left(p \cdot 10000^{-2i/d}\right)$$
Added to token embeddings at the bottom layer only. Deterministic, no learned parameters.

**Learned position embeddings (BERT, GPT-2):**
- One embedding vector per absolute position (e.g., position 0–511)
- Learned end-to-end with the model
- Simple and effective; does **not** extrapolate to lengths beyond training max

**Fixed vs learned trade-off:**
- Fixed sinusoidal: nothing to learn; may theoretically extrapolate
- Learned: easy to implement, works well in practice, fails on longer sequences

**RoPE — Rotary Position Embedding** (Su et al., 2021):
- Applied to Q and K vectors inside attention (not at the bottom layer)
- Multiplies Q and K by rotation matrices; rotation angle depends on position
- 2D case: `RoPE(x, m) = [[cos(mθ), -sin(mθ)], [sin(mθ), cos(mθ)]] · [x₁, x₂]`
- d-dimensional: block-diagonal rotation matrix with angle pairs (mθ₁, …, mθ_{d/2})
- Key property: `qₘᵀ kₙ` depends only on the relative position `m-n`
  $$q_m^T k_n = x_m^T W_Q^T R^d_{\Theta, m-n} W_K x_n$$
- **As of 2025, RoPE is the dominant method:** used by Llama, Qwen, OLMo, gpt-oss, DeepSeek, …

**ALiBi (Press et al., 2022):**
- Add a linear bias to attention scores based on token distance: `score_{ij} -= |i-j| · slope`
- No position embeddings needed at all
- Enables training on short sequences and extrapolation to longer ones

---

### 2.5 Generating Text from a Language Model

**Autoregressive generation:** Generate tokens one at a time, conditioning on all previous tokens.

**Greedy decoding:**
```
for i in range(max_len):
    xi ← argmax_x P(x | x₁,...,x_{i-1})
    append xi
```
Fast and simple. Does not find the globally best sequence. Prone to **repetition** and bland output.

**Beam search:**
- Maintain k candidates (the "beam") at each step
- Expand each candidate by all vocabulary words; keep top-k by cumulative log-probability
- More complete search than greedy; still prone to bland/repetitive output (Holtzman et al., 2020)

**Sampling:**
```
xi ~ P(x | x₁,...,x_{i-1})
```
Produces diverse output; may pick low-probability tokens.

**Top-k sampling:**
- Restrict vocabulary to the k most probable words at each step; then sample
- `k=1` = greedy; larger k = more diversity

**Top-p (nucleus) sampling** (Holtzman et al., 2020):
- Select the smallest set of tokens whose cumulative probability ≥ p; then sample
- Adapts to the shape of the distribution (narrow when confident, wider when uncertain)

**Temperature (T):**
```
logits_scaled = logits / T
probabilities = softmax(logits_scaled)
```
- T < 1: sharper distribution → repetitive/conservative
- T = 1: unmodified
- T > 1: flatter → more random/creative

**Repetition problem:** Greedy and beam search can get trapped in high-probability cycles, producing "the the the the" or similar degenerate outputs. Theoretically not well understood.

---

### 2.6 Transformer Representation Models (BERT)

**Document vs word representations:**
- **Document representation:** Single vector per document → used for retrieval, classification
- **Word/token representation:** Per-token vectors → used for NER, parsing, QA (span extraction)

**Transfer learning paradigm:**
1. Pre-train on large unlabelled corpus (general representations)
2. Fine-tune on labelled task-specific data (specialise)

**BERT (Devlin et al., 2019) — Bidirectional Encoder Representations from Transformers:**
- Architecture: encoder-only Transformer (12 layers = BERT-base, 24 layers = BERT-large)
- Key: **bidirectional attention** — each token attends to all other tokens in both directions
- Special tokens: `[CLS]` at start (used for classification), `[SEP]` between sentences

**BERT pre-training tasks:**

1. **Masked Language Modelling (MLM):**
   - Randomly mask 15% of input tokens with `[MASK]`
   - Train to predict the original token at each masked position
   - Allows bidirectional context — model must use both left and right context

2. **Next Sentence Prediction (NSP):**
   - Given two sentences A and B, predict whether B actually follows A in the corpus
   - Meant to learn inter-sentence relationships
   - Note: later work (RoBERTa) found NSP to be unhelpful or even harmful

**Fine-tuning BERT:**
| Task type | How to use BERT |
|-----------|----------------|
| Sentence classification | `[CLS]` token → linear classifier |
| Token classification (NER, POS) | Each token representation → linear classifier |
| Question answering | Predict start and end token positions |

**Key BERT variants:**
- **RoBERTa:** Removed NSP, trained longer with larger batches on more data → better
- **DistilBERT:** Knowledge distillation → 40% smaller, 60% faster, 97% of BERT performance
- **BioBERT, SciBERT:** Domain-specific pre-training on biomedical / scientific text
- **SBERT (Sentence-BERT):** Fine-tuned with siamese network for sentence similarity

**Modern use case:** As of 2025, BERT-style models are primarily used for **dense retrieval** (document similarity, RAG embedding) rather than task-specific fine-tuning.

**Benchmarks:**
- **MTEB** (Muennighoff et al., 2023): 8 task types, English only
- **MMTEB** (Enevoldsen et al., 2025): 131 tasks, 1038 languages — massively multilingual

---

## Unit 3: Pretraining and Fine-Tuning

---

### 3.1 Introduction to LLM Development

**Major LLM families (2025):**

| Model | Org | Parameters | Context | License |
|-------|-----|-----------|---------|---------|
| GPT-4o | OpenAI | undisclosed | 128K | Proprietary |
| Gemini 1.5 | Google | undisclosed | 1M | Proprietary |
| Llama 3.1 | Meta | 405B | 128K | Open-source |
| DeepSeek-R1 | DeepSeek | 671B | 128K | Open-source |

**Mixture of Experts (MoE):**
- Architecture that dynamically selects sub-models ("experts") per input
- A **router** (small learned network) decides which experts to activate for each token
- Only a fraction of total parameters are active for any single token → efficient inference despite large total parameter count
- Used in: DeepSeek-V3, Mixtral, GPT-4 (unconfirmed)

**Full LLM development pipeline:**

| Stage | Data | Data scale | Algorithm | Resources |
|-------|------|-----------|-----------|-----------|
| Unsupervised pre-training | Raw internet text | Trillions of tokens | Next-token prediction (LM) | 1000s GPUs, months |
| Instruction fine-tuning (SFT) | Ideal dialogues | 10K–100K | Next-token prediction (on responses) | 1–100 GPUs, days |
| Reward modelling | Preference-annotated dialogues | 100K–1M | Binary classification | 1–100 GPUs, days |
| Reinforcement learning (RLHF/DPO) | Generated dialogues | 10K–100K | RL for reward maximisation | 1–100 GPUs, days |

**Evaluation benchmarks:**
| Benchmark | Focus | Size |
|-----------|-------|------|
| MMLU | General knowledge | 14,000 questions |
| MATH | Mathematical reasoning | 12,500 problems |
| BIG-Bench | Reasoning, extrapolation | 204 tasks |
| SWE-bench | Coding, software engineering | 2,300 GitHub issues |

---

### 3.2 Training LLMs

**Gradient descent:**
1. Start with random θ
2. Compute gradient `∇L(θ)` on a mini-batch
3. Update: `θ := θ - α ∇L(θ)` (α = learning rate)
4. Repeat

**SGD (Stochastic Gradient Descent):** Estimate gradient on a random mini-batch per step. Introduces noise but is much faster than full-batch gradient descent.

**Adam optimiser (Adaptive Moment Estimation):**
- Maintains exponential moving averages of: gradients (1st moment = direction), squared gradients (2nd moment = magnitude)
- Effectively adapts learning rate per parameter
- Large steps along high-gradient dimensions, small steps across — navigates curved loss landscapes
- Most popular optimiser for LLM training

**Gradient clipping:** Rescale gradient vector if its total norm exceeds a threshold.
$$g_\text{clipped} = g \cdot \min\left(1, \frac{\text{max\_norm}}{\|g\|}\right)$$
Prevents gradient explosion during early training when parameters are far from optimal.

**Learning rate scheduling — three phases:**
1. **Linear warm-up:** LR increases from ~0 to peak over first ~1000–4000 steps. Avoids large updates with noisy early gradients.
2. **Cosine decay:** LR decreases following a cosine curve back toward 0.
3. **Plateau:** May hold at a small non-zero value.

**Gradient accumulation:** Simulate large batches on limited GPU memory.
```python
optimizer.zero_grad()
for microstep in range(n_microsteps):
    x, y = next(data_loader)
    loss = cross_entropy(model(x), y, reduction="sum")
    loss.backward()
optimizer.step()
```
Key: use `reduction="sum"` and divide by total elements before the step to get the true mean loss gradient.

**Weight decay:** L₂ regularisation added to loss: `L_reg(θ) = L(θ) + λ‖θ‖²`.
- Penalises large weights → reduces overfitting
- Usually **not** applied to biases and 1D tensors (layer norm parameters)
- AdamW = Adam with decoupled weight decay

---

### 3.3 Data for LLM Pretraining

**Scale requirement:** Modern LLMs need trillions of tokens. Primary source: the public internet (Common Crawl).

**Common Crawl:** Public dataset of web crawls. Each snapshot contains billions of pages and several TiB of data, updated ~monthly.

**The FineWeb pipeline** (Penedo et al., 2024) — steps to clean internet data:

1. **URL filtering:** Blocklist (adult content, malware, phishing)
2. **Text extraction:** Use Trafilatura on WARC files — much better than default WET files (less boilerplate, menus)
3. **Language filtering:** fastText classifier; keep English with score ≥ 0.65
4. **Gopher heuristic filtering:** Filter for length, symbol-to-word ratio, presence of common words, etc.
5. **C4 filtering:** Additional quality heuristics from the C4 dataset
6. **MinHash deduplication:** Fuzzy near-duplicate removal across crawl snapshots
7. **PII removal:** Remove personally identifiable information

Each step provides measurable performance improvement on downstream benchmarks.

**Deduplication:** Web contains many mirrors, aggregators, and templated pages. Duplicates → overfitting and memorisation of training data. Methods:
- **Exact:** Suffix arrays (find identical sequences)
- **Fuzzy:** MinHash (hash-based similarity) — faster for massive scale

**FineWeb-Edu:**
- Subset of FineWeb filtered for educational content quality
- 1.3T tokens
- Classifier: linear regression on Snowflake-arctic-embed-m embeddings, trained on 460K pages scored (0–5) by Llama-3-70B-Instruct
- Threshold: score ≥ 3 (best trade-off on reasoning/knowledge benchmarks)
- Outperforms all other public web datasets on MMLU, ARC, OpenBookQA

**Key public datasets:**
| Dataset | Source | Size |
|---------|--------|------|
| C4 | Common Crawl | 750B tokens |
| MassiveText | Google | 2.35T |
| RefinedWeb | Common Crawl | 5T |
| Dolma | Common Crawl | 3T |
| FineWeb | Common Crawl | 15T |

---

### 3.4 Scaling Laws

**What are scaling laws?**
Empirical relationships between model performance and key scale factors: model size N (parameters), dataset size D (tokens), compute C (FLOPs). Performance follows a **power law** — improves smoothly but at diminishing rates.

**Compute cost:**
$$C \approx 6 P T$$
where P = parameters, T = training tokens. Standard unit: FLOPs (floating point operations).

**Kaplan et al. (2020) — OpenAI scaling laws:**
- Performance improves smoothly with N, D, and C over 6+ orders of magnitude
- Performance depends **strongly on scale**, weakly on model shape (depth vs width)
- Large models are more **sample efficient** than small ones — reach the same loss with fewer examples
- Optimal batch size: ~1-2M tokens at convergence

**Chinchilla (Hoffmann et al., 2022) — compute-optimal training:**
- Key question: Given a fixed FLOPs budget C, how to allocate between N (model size) and T (tokens)?
- Kaplan said: scale N more aggressively
- Chinchilla said: scale N and T **equally** (exponents ≈ 0.49 and 0.51)
- Parametric loss model: `L(N, T) = E + A/N^α + B/T^β`
- Finding: Gopher (280B params, 300B tokens) was significantly under-trained
- Chinchilla (70B params, 1.4T tokens) **outperforms Gopher** with 4× fewer parameters

**IsoFLOP curves:** For a fixed compute budget, train models of varying sizes (keep N × T ≈ constant). Plot loss vs N — find the valley = optimal N for that budget.

**Practical implications today:**
- LLMs like Llama 3 are "over-trained" relative to Chinchilla optimal — trained on far more tokens than the formula suggests
- Over-training is intentional: a smaller, more-trained model has lower **inference cost** per query
- Training cost is amortised over many inference calls

**Old paradigm vs new paradigm:**
- Old: Train a few large models, evaluate, select the best
- New: Train many small models at various (N, T) combinations → fit the scaling law → predict optimal N for a large compute budget → train one large model

---

### 3.5 Efficient Fine-Tuning (PEFT)

**Why PEFT?** Full fine-tuning of large models is expensive in: time, storage (need one copy per task), memory (full gradients + Adam optimizer states for all parameters). PEFT adds a tiny number of trainable parameters.

**Prompt tuning / Prefix tuning:**
- Prepend learnable "virtual tokens" (soft prompts) to the input sequence
- Only these added embeddings are trained; the model is completely frozen
- Prefix tuning extends this to all layers, adding learnable prefix vectors to each layer's key/value

**Adapters (Houlsby et al., 2019):**
- Insert small bottleneck modules inside each Transformer layer after attention and FFN
- Module structure: Linear (down) → activation → Linear (up) → residual add
- Only adapter weights are trained; the rest of the model is frozen
- Available via the AdapterHub library

**LoRA (Hu et al., 2022) — Low-Rank Adaptation:**
- Decompose weight update: `ΔW = B · A` where `B ∈ ℝ^{d×r}`, `A ∈ ℝ^{r×k}`, rank `r ≪ min(d,k)`
- Forward pass: `output = W·x + (α/r)·B·A·x`
- A initialised with random Gaussian; **B initialised to zero** → ΔW = 0 at start (stable initialisation)
- Only A and B are trained; W is frozen
- Typically applied to Q and V projection matrices; sometimes K, O, and FFN too
- `α/r` scaling factor (usually α = 2r for simplicity)

**QLoRA (Dettmers et al., 2023):**
- Combine LoRA with 4-bit quantisation of the base model
- Enables fine-tuning 65B parameter models on a single consumer GPU

**Comparison:**
- Full SFT: best performance, most expensive
- LoRA: slightly below full SFT but surprisingly close; < 1% of parameters
- Prompt tuning: least parameters, works better at very large model scales

---

### 3.6 Training LLMs to Follow Instructions (SFT)

**Problem:** A base LM trained on raw text generates continuations in whatever style it learned — it doesn't follow instructions as a conversational assistant.

**Supervised Fine-Tuning (SFT) for instruction following:**
- Collect a dataset of (instruction, ideal response) pairs
- Fine-tune the model on these examples using language modelling loss
- Crucially: **only compute loss on response tokens** (mask instruction with -100)

**Flan (Chung et al., 2022):**
- Fine-tuned on 1836 diverse tasks; evaluated on 102 held-out tasks
- Key finding: training on more diverse task types improves generalisation to unseen tasks
- Improvements from: more tasks, paraphrasing instructions, step-by-step (chain-of-thought) reasoning examples

**Problems with SFT alone:**
- Hard to capture all desirable qualities in instruction-response pairs
- Model imitates the format of training data without truly understanding intent
- No signal about *quality* of responses beyond whether they match training examples
- Can be misled into producing harmful outputs if training data contains such examples

**Solution:** Use human preference feedback to train a reward model, then use RL to maximise that reward → RLHF (Unit 4.1)

---

### 3.7 Emergent Abilities of LLMs (TDDE09 Supplemental)

*Source: NLP-2026-35 (Kuhlmann, also used as TDDE09 Unit 3.5)*

**GPT model growth:**

| Model | Dimensions | Layers | Parameters | Training data |
|-------|------------|--------|------------|---------------|
| GPT-1 | 768 | 12 | 0.117B | 4 GB |
| GPT-2 | 1,600 | 48 | 1.542B | 40 GB |
| GPT-3 | 12,288 | 96 | 175B | 570 GB |
| GPT-4 | ? | 120 | ~1,800B | ? |

**GPT-1: Effective pretraining (Radford et al., 2018)**
- Key insight: next-word prediction is an effective pre-training strategy for a broad range of NLP tasks
- Fine-tuned with task-specific layers on MNLI, QNLI, RTE, SNLI — exceeded previous state of the art
- Standard pretraining → fine-tuning paradigm; all parameters updated during fine-tuning

**GPT-2: Emergent zero-shot learning (Radford et al., 2019)**
- **Zero-shot learning:** the ability of a model to solve tasks out-of-the-box, with no examples and no gradient updates
- Demonstrated via sequence modelling: QA, coreference resolution (Winograd schema) emerged naturally
- Winograd example: "The trophy doesn't fit into the brown suitcase because **it** is too large." → model correctly assigns higher probability to "trophy" as referent

**GPT-3: Emergent in-context learning (Brown et al., 2020)**
- **In-context learning:** the ability to learn tasks from a few examples in the prompt, with no gradient updates
- Word unscrambling, translation — shown to work by placing examples before the query token-by-token
- Quantitative benchmark: in-context learning improves as model scale increases (unlike small models)

**Chain-of-thought prompting (Wei et al., 2022)**
- Standard prompt: Q → A directly; accuracy suffers on multi-step reasoning
- Chain-of-thought: Q → reasoning steps → A; model shows intermediate work, dramatically improves accuracy
- Example: cafeteria apples — standard gives 11 (wrong); CoT gives 9 (correct) by explicitly computing 23−20=3, 3+6=9

**Zero-shot chain-of-thought (Kojima et al., 2022)**
- Adding "Let's think step by step." to a standard prompt activates CoT-like reasoning
- No task-specific few-shot examples needed; the instruction suffices to trigger step-by-step reasoning

**Prompt engineering**
- LLM-designed prompts (using LLM to optimise the prompt itself) can outperform human-designed prompts
- Best LLM prompt: "Let's work this out in a step by step way to be sure we have the right answer." → 82.0%
- Best human prompt: "Let's think step by step." → 78.7%  
- **Margin: 3.3 accuracy points** (not percent; not ×3.3)

---

## Unit 4: Alignment and Current Research

---

### 4.1 Reinforcement Learning with Human Feedback (RLHF)

**Motivation:** Maximising token-level log-likelihood (SFT) is a primitive objective that doesn't capture high-level text quality. RL allows optimising any differentiable or learned reward function.

**Full RLHF pipeline (InstructGPT, Ouyang et al., 2022):**

**Step 1 — Supervised Fine-Tuning (SFT):**
- Fine-tune the base LM on ~10K high-quality instruction-response pairs
- Creates a well-behaved starting point for RL

**Step 2 — Reward Model Training:**
- For each prompt, generate several responses using the SFT model
- Human annotators rank responses (which is better/worse?)
- Train a reward model `r_θ(x, y)` to assign higher scores to preferred responses
- Uses the **Bradley-Terry preference model:**
  $$P(y_1 \succ y_0) = \sigma\left(r_\theta(x, y_1) - r_\theta(x, y_0)\right)$$
- Loss: `L = -log P(yw ≻ yl)` for each preference pair (yw=winner, yl=loser)

**Step 3 — RL with PPO:**
- Optimise the SFT model to maximise expected reward while staying close to SFT
- **PPO objective (InstructGPT):**
  $$\text{obj}(\phi) = \mathbb{E}_{(x,y) \sim D}\left[r_\theta(x,y) - \beta\log\frac{\pi^\text{RL}_\phi(y|x)}{\pi^\text{SFT}(y|x)}\right] + \gamma \mathbb{E}_{x \sim D_\text{pretrain}}\left[\log\pi^\text{RL}_\phi(x)\right]$$
  - Term 1: maximise reward `r_θ`
  - KL term `β·KL(πRL ‖ πSFT)`: penalise large deviations from SFT (prevents reward hacking)
  - Term 3: pretraining loss (prevents catastrophic forgetting of language modelling ability)
  - β controls the KL penalty strength

**PPO involves 4 models:** πRL (being trained), πSFT (frozen reference), r(x,y) (reward model), value network (token-level utility). Complex to implement and train.

**Effect of RLHF:** Model generates responses that humans prefer — more helpful, more honest, less harmful.

---

### 4.2 Direct Preference Optimisation (DPO)

**Motivation:** PPO is complex (4 models), unstable to train, and sensitive to hyperparameters. DPO achieves the same goal without RL.

**Key mathematical insight** (Rafailov et al., 2023):
For PPO-style RL with a KL penalty, the optimal policy satisfies:
$$r(x, y) = \beta\log\frac{\pi^*(y|x)}{\pi^\text{SFT}(y|x)} + \beta\log Z(x)$$
This means the reward can be expressed in terms of the ratio between the optimal policy and the SFT reference policy — no explicit reward model needed.

**Derivation of DPO objective:**
1. Express reward in terms of optimal policy (above)
2. Plug into Bradley-Terry model: `P(yw ≻ yl) = σ(r(x,yw) - r(x,yl))`
3. Cancel the partition function Z(x) (it appears in both terms)
4. Result: **DPO loss:**
$$\mathcal{L}_\text{DPO} = -\log\sigma\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_\text{SFT}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_\text{SFT}(y_l|x)}\right)$$

**Interpreting DPO:**
- Increases `P(yw|x)` relative to the SFT reference (reward the winner)
- Decreases `P(yl|x)` relative to the SFT reference (penalise the loser)
- β: higher β → stay closer to SFT model
- `-log σ` = BCEWithLogitsLoss (binary cross-entropy)

**Advantages over RLHF/PPO:**
- Only 2 models: πDPO (trainable) + πSFT (frozen reference)
- No explicit reward model needed
- No value network
- No sampling/rollouts during training
- Much simpler to implement

**DPO training data:** Preference pairs `(x, yw, yl)`. Example: UltraFeedback dataset.

**DPO in practice (as of 2025):**
- Competitive with PPO-based approaches
- Used in: Qwen, Phi, OLMo, Mixtral
- Common pipeline: SFT → DPO → (optional additional RL)

---

### 4.3 Retrieval-Augmented Generation (RAG)

*No lecture slides for this topic — see Assignment 4 notes in exam_note.md for full coverage.*

**Core concept:** Combine retrieval (find relevant passages from a knowledge base) with LLM generation (produce an answer conditioned on retrieved passages + query).

**Why RAG?**
- LLMs have a fixed training knowledge cutoff and can hallucinate facts
- RAG grounds responses in retrieved evidence without retraining
- Allows updating the knowledge base without touching the model

**RAG pipeline:**
1. Index a corpus: chunk documents, embed each chunk, store in vector database
2. At query time: embed the query, retrieve top-k similar chunks
3. Provide query + retrieved chunks as context to the LLM
4. LLM generates an answer grounded in the retrieved context

---

### 4.4 LLMs for Fact Completion — PrISM

**Paper:** "Fact Recall, Heuristics or Pure Guesswork? Precise Interpretations of Language Models for Fact Completion" (Saynova et al., LiU/Chalmers)

**Task: Fact completion**
Given a subject + relation, predict the object:
> "Astrid Lindgren is originally from ___" → Sweden

The question: when an LLM gets this right, *why* does it get it right?

**PrISM taxonomy — 4 prediction scenarios:**

| Scenario | Description | Example |
|----------|-------------|---------|
| **Exact fact recall** | Model truly recalls the specific fact | "Kye Ji-Su, a citizen of South Korea" (with synthetic name → no name bias) |
| **Heuristics recall** | Model exploits statistical patterns instead of the fact | "Nokia N9 was produced by Nokia" (lexical overlap); "It was produced by Apple" (prompt bias) |
| **Guesswork** | Model is uncertain; generates inconsistent answers across similar prompts | "Eksi Ekso originated in Russia" / "Eksi Ekso was started in France" (inconsistent) |
| **Generic language modelling** | Not a factual recall scenario; model produces generic continuations | "Kun-Woo Paik is also a regular guest artist at the ___" |

**Key insight:** Accuracy alone does not indicate consistent fact recall. A model can get the right answer for the wrong reason (heuristics or coincidence).

**Heuristics (biases) that inflate apparent accuracy:**
- **Name bias:** "Kye Ji-Su" → South Korea (because Kye Ji-Su is a common Korean name)
- **Prompt bias:** "It was produced by Apple" → Apple (the word "Apple" already in prompt)
- **Lexical overlap:** "Nokia N9 was produced by ___" → Nokia (subject name in prompt)

**Methodology:** Use synthetic/novel subjects to eliminate name bias; test with low-confidence prompts to detect guesswork; use causal tracing and information flow analysis.

**Conflict with prior mechanistic interpretability work:**
- **Causal Tracing** (Meng et al., 2022): attributes fact storage to middle-layer MLP modules
- **Information Flow** (Geva et al., 2023): attributes it to attention heads retrieving from subject representations
- PrISM result: CT and IFA indicate *different* recall mechanisms depending on which scenario the prediction actually belongs to → prior interpretability results may be confounded by the mix of scenarios

**Summary of PrISM contributions:**
- Provides a principled taxonomy of *why* models produce fact completions
- Shows accuracy is not a reliable proxy for genuine fact recall
- Exposes how heuristics inflate benchmark numbers
- Causal Tracing and Information Flow point to different mechanisms — neither alone is sufficient

---

### 4.5 Adversarial Tokenization

**Paper:** "Adversarial Tokenization" (Geh, Shao, Van den Broeck, ACL 2025) — University of California, Los Angeles

**Background: the canonical tokenization assumption**
Most LLMs use exactly one tokenization per string (the canonical tokenization output by BPE). For example, Llama 3 always tokenizes "penguin" as `[p, enguin]`. However, `[peng, uin]` is equally valid. In fact, any string can be tokenized in an **exponentially large** number of ways (e.g., "tokenization" has 784 valid tokenizations in Llama 2's vocabulary).

**Key observation 1 — Semantic signal in non-canonical tokenizations:**
LLMs retain semantic understanding of non-canonical tokenizations. Even if a string is tokenized differently from the canonical form, the model still "understands" the meaning — **but only up to a point.**

Measurement: Use Levenshtein (edit) distance from the canonical tokenization. As distance increases, the model's ability to correctly answer questions about the string *decreases*, but remains meaningful at short distances.

**Key observation 2 — Alignment is fragile to non-canonical tokenizations:**
Post-training alignment (RLHF, DPO) is performed only on **canonical tokenizations**. This creates a gap:
- Non-canonical tokenizations that are semantically understood by the model are NOT covered by safety training
- A malicious string can be adversarially re-tokenized to retain meaning but bypass alignment

**Why does this gap exist?**
| Training phase | Data characteristics |
|----------------|---------------------|
| **Pre-training** | Huge datasets; includes typos, multilingual text, weird spacing → model learns semantics from many non-canonical forms |
| **Post-training (RLHF/DPO)** | Smaller, curated datasets; mostly monolingual; fewer typos → safety signal trained only on canonical tokenizations |

Result: Semantic understanding generalises to non-canonical tokenizations, but safety restrictions do not.

**Adversarial tokenization — the optimisation problem:**
Find the tokenization `v` of malicious string `x` that maximises the probability of generating a harmful target response `y*`:

$$v^* = \arg\max_v \pi_\theta(y^* \mid v)$$

This problem is **NP-hard** (reducible from 3-SAT).

**Solution: greedy local search**
- Define neighbourhood `N(v)` = all tokenizations at Levenshtein distance 2 from current `v`
- Distance-2 neighbourhood chosen for: (1) bounded number of edges, (2) connectivity of the search space
- Greedily move to the neighbour with highest likelihood of the target response
- Can be seamlessly combined with existing jailbreaking pipelines

**Three case studies:**

| Attack type | Description | Improvement |
|-------------|-------------|-------------|
| **Jailbreaking** | Elicit harmful responses from aligned LLMs | State-of-the-art when combined with existing methods |
| **Safety model evasion** | Bypass guard classifiers (Llama Guard, ShieldLlama) | +5–10% bypass rate |
| **Prompt injection** | Man-in-the-middle: attacker alters user input to provoke malicious response | +65% over canonical baseline |

**Limitations:**
- Only works on **open-source models** — requires access to logits and ability to input raw token sequences
- Closed-source APIs (GPT-4, Claude) that do not expose logits or force canonical tokenization are not vulnerable

**Main takeaways:**
1. Non-canonical tokenizations retain semantic understanding
2. Non-canonical tokenizations break alignment
3. Adversarial tokenization is easy to implement and effective
4. Computing the optimal tokenization exactly is NP-hard → use greedy search
5. **Open question:** Should post-training pipelines incorporate non-canonical tokenizations to make safety training more robust?

---

### 4.6 Environmental Cost of Chatbot Technology

**The problem:** Chatbots feel "ethereal" but have concrete environmental impact: electricity, water, carbon emissions.

**Energy consumption of ChatGPT** (Verma and Tan, 2024) — generating one 100-word email:

| Usage | Energy | Equivalent |
|-------|--------|-----------|
| One person, once | 0.14 kWh | 14 LED bulbs for 1 hour |
| One person, weekly for 1 year | 7.5 kWh | Power a home for 10 hours |
| 10% of Sweden's workforce, weekly | 3,900,000 kWh | Power 650 homes for 1 year |

**Water consumption** (Verma and Tan, 2024) — same 100-word email:
| Usage | Water |
|-------|-------|
| One person, once | 519 ml (1 bottle) |
| One person, weekly for 1 year | 27 L |
| 10% of Sweden's workforce, weekly | 14,600,000 L |

**BLOOM training cost** (Luccioni et al., 2022):
| Metric | Value |
|--------|-------|
| Parameters | 176B |
| Training time | 118 days, 5 hrs, 41 min |
| Total GPU time | 1,082,990 GPU-hours |
| Energy used | 433,196 kWh |
| Carbon emissions (with PUE) | 30 tonnes CO₂ |

Carbon emissions (training only): GPT-3 > OPT > BLOOM (same scale, different infrastructure/energy mix).

**LLM lifecycle:** Materials extraction → manufacturing → model training → deployment → disposal.
Most data is available for training and deployment; full lifecycle assessment is difficult.

**Data centre key performance indicators:**
- **PUE (Power Usage Effectiveness):** Total data centre energy / energy delivered to compute. Ideal = 1.0.
- **WUE (Water Usage Effectiveness):** Water consumed (m³) / energy to compute (MWh).

**European data centre comparison:**
| Country | PUE | WUE |
|---------|-----|-----|
| Sweden | 1.14 | 0.04 |
| Finland | 1.16 | 0.01 |
| Germany | 1.40 | 1.43 |

Sweden is highly efficient due to cold climate (natural air cooling) and hydroelectric power.

**Cooling methods:**
- **Air cooling:** Fans + air conditioning move heat out. Simpler but less efficient at high densities.
- **Liquid cooling:** Liquid absorbs heat from servers directly. More efficient, used for high-density GPU clusters.

**Interventions to reduce impact:**
1. **Technical:** Quantisation, distillation, model compression → smaller models with similar performance
2. **Behavioural:** Use task-specific tools; don't use AI where simpler tools suffice
3. **Organisational:** Choose compute providers powered by renewable energy
4. **Policy:** Mandatory transparency reporting on resource consumption and environmental impact

**Concern:** Google (2024) announced it is no longer maintaining operational carbon neutrality, attributed largely to AI infrastructure growth.

**Societal concerns — "Stochastic Parrots" (Bender et al., 2021)**

LLM-based chatbots differ from traditional chatbots in **capability breadth** — traditional chatbots had narrow, predefined affordances; modern LLMs handle open-ended tasks across domains.

**Bender's "stochastic parrot" claim:**
- LLMs are stochastic parrots: they randomly recombine linguistic forms without reference to meaning
- They predict next tokens based on statistical patterns in training data — not genuine understanding
- Producing fluent, coherent text does not imply semantics, understanding, or intent

**Anthropomorphism problem:**
- Users project understanding onto coherent outputs — this is a perception, not a model property
- The model "seems to understand" because we interpret linguistic form as meaning
- Bender argues this is a category error regardless of model scale or RLHF fine-tuning

**Chatbots vs. search:**
- Traditional search exposes many sources; users evaluate and synthesise
- Chatbots present one confident answer — users are tricked into believing there is "the answer"
- Cuts users off from thinking independently and from understanding information provenance

**Information provenance:** tracking the origin and history of information (who said it, based on what, when)
- LLMs destroy provenance: outputs are statistically blended from billions of sources with no attribution
- Users cannot verify claims; errors and biases propagate invisibly

---

## Quick Reference: Key Formulas

| Formula | Topic |
|---------|-------|
| `CR(b) = |b| / |tokenize(b)|` | Tokenisation compression rate |
| `eij = qi·kj / √d` | Scaled dot-product attention energy |
| `αij = softmax(eij)` | Attention weights |
| `attention_i = Σ_j αij vj` | Attention output |
| `PE_{2i}(p) = sin(p · 10000^{-2i/d})` | Sinusoidal position encoding |
| `RoPE(x,m) = R_m · x` (rotation matrix) | Rotary position embedding |
| `C ≈ 6PT` | Compute cost (FLOPs) |
| `WFT = WPT + (α/r)·B·A` | LoRA weight update |
| `P(y1≻y0) = σ(rθ(x,y1) - rθ(x,y0))` | Bradley-Terry preference model |
| `LDPO = -log σ(β log π(yw)/πSFT(yw) - β log π(yl)/πSFT(yl))` | DPO objective |

---

## Quick Reference: Key Papers

| Paper | Contribution |
|-------|-------------|
| Sennrich et al. (2016) | BPE tokenisation for NMT |
| Petrov et al. (2023) | Tokenisation premiums across languages |
| Land & Arnett (2025) | SCRIPT block-structured encoding |
| Foroutan et al. (2025) | Parity-aware BPE |
| Peters et al. (2018) | ELMo contextualised embeddings |
| Vaswani et al. (2017) | Transformer ("Attention is All You Need") |
| Devlin et al. (2019) | BERT |
| He et al. (2016) | Residual connections |
| Ba et al. (2016) | Layer normalisation |
| Su et al. (2021) | RoPE |
| Press et al. (2022) | ALiBi positional bias |
| Holtzman et al. (2020) | Nucleus sampling; neural text degeneration |
| Muennighoff et al. (2023) | MTEB benchmark |
| Enevoldsen et al. (2025) | MMTEB benchmark (multilingual) |
| Kaplan et al. (2020) | OpenAI scaling laws |
| Hoffmann et al. (2022) | Chinchilla / compute-optimal training |
| Penedo et al. (2024) | FineWeb dataset |
| Hu et al. (2022) | LoRA |
| Dettmers et al. (2023) | QLoRA |
| Houlsby et al. (2019) | Adapters |
| Chung et al. (2022) | Flan instruction tuning |
| Ouyang et al. (2022) | InstructGPT / RLHF |
| Rafailov et al. (2023) | DPO |
| Luccioni et al. (2022) | BLOOM environmental cost analysis |
| Verma & Tan (2024) | ChatGPT energy/water consumption estimates |
