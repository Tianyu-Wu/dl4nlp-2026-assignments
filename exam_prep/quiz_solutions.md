# DL4NLP — Quiz Sample Solutions

Source: https://liu-nlp.ai/tdde09/units/  
Format: question → correct answer + key explanation

> Note: TDDE09 unit/lecture numbering differs slightly from DL4NLP.  
> TDDE09 Unit 2 starts with 2.1 MT intro and 2.2 Neural MT (not in DL4NLP).  
> TDDE09 Unit 3.5 = Emergent abilities; TDDE09 Unit 4.3 = Quantisation/LoRA;  
> TDDE09 Unit 4.4 = RAG; TDDE09 Unit 4.6 = Env cost + Bender.

---

## Unit 1: Tokenisation and Embeddings

### Lecture 1.1 — Introduction to Tokenisation

**Q1. What is the datatype of token ID vectors?**  
✅ `Tensor[int]` — Token IDs are integers, each mapping to a unique vocabulary entry. Floating-point tensors are for embeddings, not IDs.

**Q2. What is an example of oversegmentation?**  
✅ `co-writer/director → co - writer / director` — Oversegmentation splits one meaningful token into multiple less-meaningful pieces.  
(co-writer/director as one token = no segmentation; co-writer / director = reasonable segmentation)

**Q3. Which regex tokenisation applies to `1920's`?**  
✅ `[1920] ['s]` — The regex splits possessive/'s markers into separate tokens, but keeps `'s` together (not splitting further into `'` and `s`).

**Q4. What is an example of lemmatisation?**  
✅ `centers → center` — Lemmatisation reduces words to their base/dictionary form. (center→centres is spelling change; center→centers is inflection.)

**Q5. In the Heaps' law graph, after how many tokens had the vocabulary reached half its final size?**  
✅ ~250,000 tokens → vocabulary ~15,000 (final size ~30,000).

---

### Lecture 1.2 — Byte Pair Encoding

**Q1. How many merge operations give a vocabulary of 1,024 tokens?**  
✅ 768 merges — BPE starts with 256 single-byte tokens; 256 + 768 = 1,024.

**Q2. How many UTF-8 bytes in the string `oó𑄖` (o, ó, letter O in Devanagari)?**  
✅ 6 bytes — o: 1 byte (ASCII); ó: 2 bytes (codepoint > 127); 𑄖: 3 bytes (codepoint 0913 = 2323 decimal, falls in the 2048–65535 range). Total: 1+2+3 = 6.

**Q3. Applying BPE to `aaaa`, what is the second merge rule?**  
✅ `[aa] + [aa] → [aaaa]` — First merge: `a+a → [aa]`, giving `[aa][aa]`. Second merge is now the most frequent pair: `[aa]+[aa] → [aaaa]`.

**Q4. Which token would we NOT expect in a large English BPE vocabulary?**  
✅ `[eaux]` — A French suffix, rare in English. `[tion]` and `[thes]` both appear in many English words (action, nation; hypothesis, anaesthesia).

**Q5. How many tokens in "Linköping University" (o200k_base tokeniser)?**  
✅ 4 tokens: `Link`, `ö`, `ping`, `University`.

---

### Lecture 1.3 — Tokenisation Fairness

**Q1. What best describes tokenisation fairness?**  
✅ Whether different languages are tokenised with similar efficiency — not model output bias, and not data frequency in training.

**Q2. What is a tokenisation premium?**  
✅ The factor by which a tokeniser requires more tokens to represent one language than another (e.g., Shan has a 15.05× premium over English in GPT-4).

**Q3. Why does standard BPE disadvantage low-resource languages?**  
✅ Merge rules are learned from the most frequent token pairs in the entire (typically English-dominated) corpus — so low-resource language pairs rarely become merged tokens.

**Q4. How is "compression rate" defined?**  
✅ Number of bytes / number of tokens — Higher = more efficient tokenisation for that language.

**Q5. What is the general lesson of the tokenisation fairness lecture?**  
✅ Tokenisation is one layer of a broader system of linguistic inequality, and improving it involves normative trade-offs — not a problem that disappears with more multilingual data, and not fully solvable by optimising fairness metrics alone.

---

### Lecture 1.4 — Introduction to Embeddings

**Q1. `torch.nn.Embedding(15000, 100)` — how many trainable parameters?**  
✅ 1,500,000 — Formula: vocab_size × embedding_dim = 15,000 × 100.

**Q2. Bag-of-words classifier with embedding layer (1.5M params) + softmax over 5 classes (no bias). Total parameters?**  
✅ 1,500,000 + 500 — Softmax layer: 100 inputs × 5 classes = 500 parameters.

**Q3. True or false: embeddings for "university" and "school" are similar.**  
✅ Depends on the training task — If the task is about education, they may be similar; on unrelated tasks, they may not.

**Q4. Book/bike classifier — which word pair has LOW cosine similarity?**  
✅ `pages` and `gear` — They belong to opposite categories (book vs. bike). Words within the same category (pages/chapter, gear/brake) will be similar.

**Q5. Which describes "pre-training and fine-tuning"?**  
✅ Initialise embedding with trained weights + train the full network on a prediction task — Pre-training = using previously trained weights; fine-tuning = continuing to update them on the new task.  
(Freezing the embedding layer + training other parts is "feature extraction", not fine-tuning.)

---

### Lecture 1.5 — Word Embeddings

**Q1. Vocabulary of 500,000 words; "parsnip" is word #350,740. Length of one-hot vector?**  
✅ 500,000 — One-hot vector length = vocabulary size, not the word's index.

**Q2. Typical length of a word embedding?**  
✅ 300 — (e.g., word2vec, GloVe). 30 is too short; 30,000 is far too large.

**Q3. Cosine similarity of 1 means what?**  
✅ Angle of 0° (vectors perfectly aligned). Cosine = 0 → 90°; cosine = -1 → 180°.

**Q4. In the Distributional Hypothesis, what does "company" mean?**  
✅ Words that co-occur with the other word — not words with similar meanings or similar embeddings (those are downstream consequences).

**Q5. Which of the following is NOT used to evaluate word embeddings?**  
✅ Cross-entropy loss — used during training, not as an evaluation metric. Odd-one-out tests and analogy benchmarks are evaluation metrics.

---

### Lecture 1.6 — Contextualised Word Embeddings (Skip-gram / ELMo)

**Q1. Standard skip-gram = k-class classification problem. Realistic value of k?**  
✅ 20,000 — k = vocabulary size (typically tens of thousands).

**Q2. Why is skip-gram an interesting pre-training task?**  
✅ Predicting the other word forces the model to learn co-occurrence statistics — which encode meaningful semantic relationships.

**Q3. When do we want P(c|w) to be high?**  
✅ When the word vectors for w and c have high cosine similarity — high P(c|w) ↔ vectors aligned.

**Q4. What is the basic idea of skip-gram with negative sampling?**  
✅ Approximate P(c|w) using a simpler binary classification task — distinguish true context words from randomly sampled (negative) words. (Hierarchical softmax uses a tree structure; that's a different technique.)

**Q5. Which is NOT used to speed up skip-gram with negative sampling?**  
✅ Randomly excluding stop words — stop word removal is a general preprocessing step, not specific to skip-gram with negative sampling. Subsampling (discarding frequent tokens with growing probability) and random window sizes ARE used.

---

## Unit 2: LLM Architectures

> Lectures 2.1–2.2 cover Machine Translation — content specific to TDDE09.

### Lecture 2.1 — Introduction to Machine Translation

**Q1. What component is NOT needed in an interlingual MT pipeline?**  
✅ Sentiment classifier — Not useful for translation. POS tagger and dependency parser both help construct interlingual representations.

**Q2. In the Noisy Channel Model (Arabic A → Swedish S), what do we maximise?**  
✅ P(A|S) · P(S) — Bayes' Rule: P(S|A) ∝ P(A|S)·P(S). P(A|S) alone ignores fluency; P(S|A) is what we ultimately want but we decompose it.

**Q3. Is word alignment a function?**  
✅ Neither R nor its inverse is a function — Alignment allows one-to-many and many-to-one mappings, violating the function requirement.

**Q4. Advantage of neural MT over statistical MT?**  
✅ NMT does not need complex feature engineering — SMT requires hand-designed features; NMT learns representations automatically. (Both need parallel text; NMT is actually less interpretable.)

**Q5. Why does BLEU include a brevity penalty?**  
✅ It is easy to achieve high precision with short translations — A very short translation can match many n-grams from the reference without covering the full content, inflating precision artificially.

---

### Lecture 2.2 — Neural Machine Translation

**Q1. Which task does NOT lend itself to autoregressive language models?**  
✅ Document classification — Classification doesn't require sequential token generation. Translation and summarisation do.

**Q2. What does a seq2seq model learn for Arabic A → Swedish S?**  
✅ P(S|A) — The conditional probability of the target given the source.

**Q3. What resources do we need to train seq2seq translation models?**  
✅ Parallel texts for the source–target pair — Word alignments are for phrase-based SMT, not seq2seq. A separate target LM is not required (seq2seq learns it implicitly).

**Q4. Which parameter does NOT affect space complexity of beam search?**  
✅ The total number of possible translations — Beam search only stores the current beam (k candidates), regardless of how many total translations exist.

**Q5. Why use length normalisation with beam search?**  
✅ We do not want to penalise long translations — Without it, longer sequences accumulate more probability multiplications → lower total probability → shorter translations are unfairly favoured.

---

### Lecture 2.3 — Attention

**Q1. Which NLP task can be handled well with static word vectors?**  
✅ Topic classification — Overall document topic doesn't require per-word contextual understanding. Coreference resolution and word sense disambiguation both need context.

**Q2. In "Dogs may bark at strangers" (positions 1–5), which α is highest for "bark" (pos 3)?**  
✅ α₃₁ — "bark" attends most to "dogs" (pos 1) to disambiguate its word sense (sound vs. tree bark). α₃₃ (self-attention) is usually highest overall, but α₃₁ is the interesting one here.

**Q3. Given h₁=[0.5539,0.7239], h₂=[0.4111,0.3878], h₃=[0.2376,0.1264], what is the refined h₂?**  
✅ [0.4198, 0.4488] — The output vector must have the same dimensionality as each hᵢ (2D). The intermediate scores and softmax weights are 3D, not the final output.

**Q4. What is true about attention with queries, keys, and values?**  
✅ The output has the same length as each value — Output = weighted sum of values, so it inherits the value dimension.

**Q5. Multi-head attention, 8 heads, total Q/K size 256. Key length per head?**  
✅ 32 — 256 / 8 = 32. Each head operates on its own slice.

---

### Lecture 2.4 — The Transformer Architecture

**Q1. Main advantage of Transformer over RNNs?**  
✅ Direct access to all elements in the input — Self-attention is parallel; every token can attend to every other token simultaneously. (Transformers typically need more data and have more parameters, not fewer.)

**Q2. Which statement is FALSE about the example translation?**  
✅ "The final encoder representation of drink depends on the token embedding of Kaffee." — Correct to mark as FALSE: the Transformer encodes the English sentence; Kaffee is the German translation output, not an encoder input.

**Q3. Which MHA variant is used in the Transformer encoder?**  
✅ Self-attention — The encoder uses bidirectional self-attention. Masked self-attention is for the decoder. Cross-attention is the decoder's second attention layer.

**Q4. Purpose of layer normalisation?**  
✅ Centering and scaling the layer's input values — It normalises by mean and variance. It does not squash to [0,1] (that's sigmoid/softmax) nor just down-scale.

**Q5. True/False: Permuting input tokens does not change final Transformer representations.**  
✅ False — Positional encodings embed order information; changing order changes representations.

---

### Lecture 2.5 — Decoder-Based Language Models (GPT)

**Q1. What does "generative pre-training" mean?**  
✅ Pre-training on a language modelling task — Not training on synthetic text, and not just any generative probabilistic model.

**Q2. Approximate trainable parameters in GPT-1's FFN?**  
✅ ~4,718,592 — Based on the FFN architecture in the original GPT-1 model.

**Q3. Minimal parameters to update when fine-tuning GPT on SNLI?**  
✅ The parameters in the final Linear layer — The minimum needed is only the new classification head, not the full model.

**Q4. GPT-3 exhibits zero-shot behaviour. What does that mean?**  
✅ It can solve tasks without task-specific fine-tuning — It still needs input prompts; it was extensively pre-trained; "zero-shot" means no task-specific gradient updates.

**Q5. Why can GPT-3 translate English to French?**  
✅ Pre-training data contains example translations — The internet text contains many bilingual passages. (Parameter count coincidence with neurons is irrelevant; there's no feedback from translators.)

---

### Lecture 2.6 — Encoder-Based Language Models (BERT)

**Q1. Purpose of segment encoding in BERT?**  
✅ To distinguish between different segments of sentence pairs — Not sub-word segments, not embedding segments.

**Q2. Why is masked LM not suitable for GPT?**  
✅ GPT is based on the Transformer decoder and can only "look back" — Its causal attention prevents using bidirectional context that MLM requires.

**Q3. Main purpose of the [CLS] token in BERT?**  
✅ Representation of the complete input sentence pair for classification tasks — Not a padding token; not used for MLM itself.

**Q4. Which MLM branch makes a token's representation most dissimilar from its neighbours?**  
✅ Replace with a random word — A random word disrupts contextual consistency. [MASK] still lets the model infer from context; keeping original preserves relationships.

**Q5. Advantage of replaced token detection over MLM?**  
✅ It learns from all input tokens — MLM modifies only ~15% of tokens; replaced token detection creates a signal at every position.

---

## Unit 3: Pretraining and Fine-Tuning

### Lecture 3.1 — Introduction to LLM Development

**Q1. Which model has the largest context length?**  
✅ Gemini — Gemini 2.0 has 1M token context (vs GPT: 200K, DeepSeek: 128K).

**Q2. What are adapters NOT typically used for?**  
✅ Adapting to new computing hardware — Adapters are for new tasks or languages, not hardware migration.

**Q3. Speedup of H200 over H100 for Llama 2 70B inference?**  
✅ 1.9× speedup.

**Q4. What task does SWE-bench address?**  
✅ Resolving GitHub issues — Not "strategic workflow extrapolation" or questions about Sweden.

**Q5. Three best-performing models on Chatbot Arena (at time of slides)?**  
✅ Gemini, ChatGPT, DeepSeek.

---

### Lecture 3.2 — Training LLMs

**Q1. How does Adam solve the zig-zagging problem?**  
✅ It keeps averages of past gradient magnitudes (squared gradients) — Magnitudes cause zig-zagging; Adam's 2nd moment estimate smooths this. Direction averaging (momentum) helps too but isn't the primary fix for zig-zagging.

**Q2. What is the "total norm" in gradient clipping?**  
✅ The norm of the vector containing the norms of all parameter gradients — Equivalently, the L2 norm of all gradients concatenated (same thing for L2 norm).

**Q3. In the LR scheduler example, what is the LR after 280B tokens?**  
✅ 0.00006 (6×10⁻⁵) — Far along the cosine decay phase.

**Q4. Gradient accumulation over micro-batches [800, 1000, 600] with summed losses [960, 1300, 900]. Total loss?**  
✅ 1.32 — Total loss = (960+1300+900) / (800+1000+600) = 3160/2400 ≈ 1.3167 ≈ 1.32.

**Q5. To which would we NOT apply weight decay?**  
✅ The weights of layer norms — Layer norm parameters (scale and shift) are not regularised with weight decay. Linear and attention weights typically are.

---

### Lecture 3.3 — Data for LLM Pretraining

**Q1. Common Crawl 2024-51 is ~7.37 TiB. At 0.75 tokens/byte for English, how many tokens?**  
✅ ~6 trillion tokens — 7.37 TiB × 2⁴⁰ bytes/TiB × 0.75 ≈ 6×10¹².

**Q2. Which filter set is second-best in the FineWeb filtering ablation?**  
✅ C4 Filters — (Best is the full FineWeb pipeline; C4 filters come second.)

**Q3. Why is deduplication important?**  
✅ It prevents overfitting, enhances diversity, and reduces computational costs — Not to memorise patterns; deduplication reduces, not increases, dataset size.

**Q4. What model is used in the FineWeb-Edu educational quality classifier?**  
✅ Linear regression — trained on embeddings from Snowflake-arctic-embed-m. The Transformer produces embeddings; the classifier itself is linear regression.

**Q5. For the largest ablation model, how many accuracy points is FineWeb-Edu better than standard FineWeb?**  
✅ 2 points.

---

### Lecture 3.4 — Scaling Laws

**Q1. Compute cost for GPT-2 small (124M params) on 2.5B tokens?**  
✅ 1.86×10¹⁸ FLOPs — C = 6 × P × T = 6 × 124×10⁶ × 2.5×10⁹ = 1.86×10¹⁸.

**Q2. What does each point on the thick black line in the "Performance improves with scale" plot represent?**  
✅ The model with the lowest test loss for a specific compute budget — Each point is the Pareto-optimal model for that FLOPs budget.

**Q3. 8× A100 system: ~64 exaflops per 24h. Tokens for GPT-2 small (124M params)?**  
✅ ~86B tokens — T = C / (6×P) = 64×10¹⁸ / (6×124×10⁶) ≈ 86×10⁹.

**Q4. Chinchilla result: ~how many tokens per model parameter for compute-optimal training?**  
✅ ~20 tokens per parameter — Chinchilla (70B params, 1.4T tokens): 1.4×10¹² / 70×10⁹ ≈ 20.

**Q5. Which model was least compute-optimal (most undertrained) according to Chinchilla?**  
✅ Megatron-Turing NLG (530B) — 530B parameters trained on far fewer tokens than compute-optimal. GPT-3 and Gopher were also undertrained but not as severely.

---

### Lecture 3.5 — Emergent Abilities of LLMs

**Q1. How many trainable parameters does GPT-3 have?**  
✅ 175B.

**Q2. What was the main insight from GPT-1?**  
✅ Next-word prediction is an effective pre-training strategy for many NLP tasks — GPT-1 showed that unsupervised pre-training transfers broadly. (BERT's contribution was masked LM; GPT-1's was next-word pre-training.)

**Q3. Winograd example: "The city councilmen refused the demonstrators a permit because they ___". Which prompt gives p(demonstrators) > p(councilmen)?**  
✅ "…because they advocated violence." — Here "they" = demonstrators. In "…because they refused them a permit", "they" = councilmen.

**Q4. How is in-context learning different from zero-shot learning?**  
✅ The model can learn new tasks from examples — In-context learning provides a few examples in the prompt; zero-shot provides none. Neither involves gradient updates.

**Q5. How much better was the LLM-designed prompt vs the best human-designed prompt (slide on prompt engineering)?**  
✅ 3.3 points (82.0 vs 78.7 accuracy) — Not 3.3% relative improvement (that would be 78.7 × 1.033 = 81.3); not 3.3× improvement.

---

### Lecture 3.6 — Environmental Cost of Chatbot Technology

**Q1. What mainly determines the overall environmental impact of chatbots?**  
✅ The scale of use across many users — Model size and prompt length matter per-call, but aggregate impact is dominated by total usage.

**Q2. Why is data centre water consumption especially concerning?**  
✅ Many large data centres are located in water-scarce regions — Not equipment corrosion; not relative cost vs. electricity.

**Q3. Why did BLOOM have lower training emissions than GPT-3?**  
✅ It relied on a cleaner energy mix (more renewable energy) — BLOOM and GPT-3 are comparable in size; it's not parameter count or Chinchilla-optimality.

**Q4. Why might low-PUE data centres have higher WUE?**  
✅ Electrical efficiency often involves cooling techniques that rely more heavily on water — Water-based cooling (liquid cooling, evaporative cooling) reduces electricity use (low PUE) but increases water use (high WUE).

**Q5. Which conclusion best reflects the lecture's overall argument?**  
✅ Responsible chatbot use requires action at several levels — Technical, behavioural, organisational, and policy interventions are all needed. Environmental concerns are real (not exaggerated), and technology alone cannot solve the problem.

---

## Unit 4: Alignment and Current Research

### Lecture 4.1 — RLHF and Reward Modelling

**Q1. Which training stage does NOT use the language modelling objective?**  
✅ Reward modelling — Trains to predict human preference scores (binary classification-style). Pre-training and instruction fine-tuning both use next-token prediction.

**Q2. Which output shows the limitation of language modelling as an SFT objective?**  
✅ "I'm a woman so I just don't understand." — Grammatically fine from an LM perspective but biased and inappropriate. LM loss doesn't penalise such outputs.

**Q3. Purpose of reward models in LLM training?**  
✅ To act as a proxy for immediate human feedback — Not for paying annotators; not for computational efficiency. They replace the need for a human to score every generated response.

**Q4. For a well-aligned LLM, which reward model loss is most plausible for clearly preferred vs. non-preferred completions?**  
✅ 0.11 (low loss) — A well-aligned reward model assigns much higher reward to the preferred completion → R(x,y⁺) ≫ R(x,y⁻) → σ(large positive) ≈ 1 → -log(1) ≈ 0.  
0.69 = model is random (σ = 0.5); 2.30 = model is inverted (wrong direction).

**Q5. Goal of policy gradient methods?**  
✅ Maximise the probability of generating outputs with high reward — Not the reward itself (the reward model is fixed); not minimising perplexity.

---

### Lecture 4.2 — LLMs for Fact Completion (PrISM)

**Q1. What core concern motivates PrISM?**  
✅ Correct answers alone may not prove that a model has truly memorised a fact — A correct prediction could come from heuristics, guesswork, or genuine recall.

**Q2. What criterion distinguishes guesswork from exact fact recall in PrISM?**  
✅ Whether the model's prediction remains stable across paraphrased prompts — Guesswork = inconsistent top-3 across paraphrases (<5 templates); exact fact recall = confident and consistent (≥5 templates). Both require a valid-type prediction.

**Q3. Why do PrISM authors use synthetic subjects?**  
✅ To ensure predictions cannot rely on real-world memorised facts — If the model confidently predicts something about a made-up entity, it must be using heuristics, not memory.

**Q4. What do causal tracing results show for exact fact recall vs. generic LM?**  
✅ Only exact fact recall shows strong importance of mid-layer MLPs at the last subject token — Generic LM shows importance at late-layer last-token states instead.

**Q5. Main conclusion comparing causal tracing and information flow across PrISM scenarios?**  
✅ Different behavioural scenarios correspond to different internal mechanisms — Each of the four scenarios (exact recall, heuristics, guesswork, generic LM) has a distinct causal signature.

---

### Lecture 4.3 — Efficient Fine-Tuning (Quantisation and LoRA)

**Q1. What is NOT a potential benefit of quantisation?**  
✅ The model becomes more robust to rounding errors — Quantisation reduces precision → more sensitivity to rounding errors, not less.

**Q2. 100B-param LLM, all in 32-bit. How much memory saved by storing HALF the parameters in 16-bit (mixed precision)?**  
✅ 100 GB — 50B params × 16 bits saved per param = 800×10⁹ bits = 100 GB.

**Q3. In `W0 + ΔW = W0 + BA`, which matrices are trained during LoRA?**  
✅ B and A — W0 is frozen; ΔW is derived (not directly trained); B and A are the trainable low-rank factors.

**Q4. W0 is 200×100, rank=10. How many entries does B have?**  
✅ 2000 — B has dimensions 200×10 = 2000 entries. (A has 10×100 = 1000 entries.)

**Q5. What is the optimal rank for LoRA training?**  
✅ Depends on the task — Empirically determined; different tasks require different expressiveness in the low-rank update.

---

### Lecture 4.4 — Retrieval-Augmented Generation (RAG)

**Q1. Which LLM task is most affected by the problem of staleness?**  
✅ Stock market prediction — Requires up-to-date information. Translation from Latin and sentiment classification are much less time-sensitive.

**Q2. What problem does the GDPR example illustrate?**  
✅ Revisions — The need to update or remove information from the model (e.g., personal data) due to legal requirements. (Not hallucination or attribution.)

**Q3. In RAG, what does the "open book" correspond to?**  
✅ The document database — Like a student consulting an open book, the model retrieves from an external knowledge source at query time.

**Q4. Which problem does the lecturer NOT mention as addressed by grounding?**  
✅ Staleness — Grounding helps with hallucination and attribution, but staleness is not explicitly listed as addressed by grounding.  
*(Note: In practice, RAG can help with staleness, but the lecture didn't list it.)*

**Q5. Which is an advantage of dense retrieval?**  
✅ It is possible to retrieve text that is semantically similar — Dense retrieval finds semantically related content even without exact keyword matches. It has a LARGER compute footprint than sparse retrieval (not smaller).

---

### Lecture 4.5 — Adversarial Tokenization

**Q1. What is the canonical tokenisation?**  
✅ The tokenisation produced by the tokeniser the model was trained with (BPE applied iteratively to fixpoint) — Not the highest-probability tokenisation, not necessarily the shortest (though it often is).

**Q2. What trend do experiments reveal about tokenisation distance and model accuracy?**  
✅ Accuracy generally decreases as tokenisation distance increases — The further from canonical, the less semantic signal.

**Q3. Why does adversarial tokenisation challenge aligned LLMs?**  
✅ Alignment training primarily covers canonical tokenisations — Non-canonical tokenisations access parts of the probability space not covered by safety fine-tuning.

**Q4. Why do the authors use greedy local search?**  
✅ Because the optimal tokenisation problem is computationally intractable (NP-complete) — Not because random sampling can't produce valid tokenisations (it can); not because greedy is optimal (it's an approximation).

**Q5. How do the authors explain why models understand non-canonical tokenisations semantically but not at the alignment level?**  
✅ Pre-training introduces semantic leakage through diverse data, while alignment data is smaller and cleaner — Massive pre-training data (with typos, multilingual, weird spacing) teaches semantics for many tokenisations; post-training uses curated, mostly canonical data.

---

### Lecture 4.6 — Environmental Cost and Societal Concerns

**Q1. Main difference between traditional and LLM-based chatbots?**  
✅ Traditional chatbots were more restricted in their affordances — Traditional chatbots were designed for narrow tasks. The key difference is capability breadth, not data size alone or transparency.

**Q2. Why does Bender call LLMs "stochastic parrots"?**  
✅ They randomly recombine linguistic forms without reference to meaning — They predict next tokens based on statistical patterns, not semantic understanding. (Not because they're stochastic; not because they're versatile.)

**Q3. Bender's stance on "ChatGPT seems to understand language"?**  
✅ They do not; this is only how we perceive them — We anthropomorphise models, projecting understanding onto coherent text. Bender argues true understanding is absent regardless of scale or RLHF.

**Q4. According to Bender, why are chatbots not a good replacement for search?**  
✅ Users get tricked into believing there is "the answer" and cut off from thinking on their own — Chatbots present single confident answers without sources, discouraging critical evaluation and independent inquiry.

**Q5. What does "information provenance" mean?**  
✅ Tracking the origin and history of information — Not encryption; not studying influence on public opinion.

---

## Key Numbers to Remember

| Fact | Value |
|------|-------|
| BPE base vocabulary (single bytes) | 256 |
| Typical word embedding dimension | 300 |
| Cosine similarity of orthogonal vectors | 0 |
| Compression rate formula | bytes / tokens |
| Adam 2nd moment tracks | squared gradients (magnitudes) |
| Chinchilla: optimal tokens per parameter | ~20 |
| GPT-3 parameters | 175B |
| BLOOM parameters | 176B |
| BLOOM training energy | 433,196 kWh |
| BLOOM carbon (with PUE) | 30 tonnes CO₂ |
| Gemini context window | 1M tokens |
| FineWeb-Edu tokens | 1.3T |
| LoRA: B initialised to | zero (so ΔW = 0 at start) |
| DPO β controls | deviation from SFT reference |
| Bradley-Terry model uses | sigmoid σ |
