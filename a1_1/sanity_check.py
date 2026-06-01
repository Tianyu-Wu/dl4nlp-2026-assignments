"""
Sanity checks for each task in Assignment 1.
Run from the a1_1/ directory:  python sanity_check.py
Or call individual functions after importing.
"""

import torch

TRAIN_FILE = 'train.txt'
VAL_FILE   = 'val.txt'

# ── helpers ──────────────────────────────────────────────────────────────────

def ok(msg):  print(f'  [OK]  {msg}')
def fail(msg): print(f'  [FAIL] {msg}')

def check(cond, msg):
    if cond: ok(msg)
    else:    fail(msg)


# ── Task 1.1 ─────────────────────────────────────────────────────────────────

def check_task1_1():
    print('\n=== Task 1.1: lowercase_tokenizer ===')
    from A1_skeleton import lowercase_tokenizer
    result = lowercase_tokenizer("Let's test!!")
    # NLTK word_tokenize splits contractions and punctuation
    check(isinstance(result, list),           'returns a list')
    check(all(isinstance(t, str) for t in result), 'all tokens are strings')
    check(all(t == t.lower() for t in result),    'all tokens are lowercase')
    check("let" in result and "!" in result,      '"let" and "!" present')
    print(f'  Tokens: {result}')


# ── Task 1.2 ─────────────────────────────────────────────────────────────────

def check_task1_2(max_voc_size=10_000):
    print(f'\n=== Task 1.2: build_tokenizer (max_voc_size={max_voc_size}) ===')
    from A1_skeleton import build_tokenizer
    tokenizer = build_tokenizer(TRAIN_FILE, max_voc_size=max_voc_size)

    check(len(tokenizer) <= max_voc_size,
          f'vocab size {len(tokenizer)} ≤ {max_voc_size}')

    for special in ['<PAD>', '<UNK>', '<BOS>', '<EOS>']:
        check(special in tokenizer.word_to_id, f'{special} is in vocab')

    for common in ['the', 'of', 'and']:
        check(common in tokenizer.word_to_id, f'common word "{common}" in vocab')

    # round-trip
    sample = 'the'
    idx = tokenizer.word_to_id.get(sample)
    if idx is not None:
        check(tokenizer.id_to_word[idx] == sample, f'round-trip: "{sample}" → {idx} → "{tokenizer.id_to_word[idx]}"')

    check(tokenizer.pad_token_id == tokenizer.word_to_id['<PAD>'],
          'pad_token_id matches <PAD> entry')

    # no max_voc_size (should not crash)
    t2 = build_tokenizer(TRAIN_FILE, max_voc_size=None)
    check(len(t2) > max_voc_size, 'unlimited vocab is larger than 10k vocab')


# ── Task 1.3 ─────────────────────────────────────────────────────────────────

def check_task1_3(max_voc_size=10_000):
    print('\n=== Task 1.3: A1Tokenizer.__call__ ===')
    from A1_skeleton import build_tokenizer
    tokenizer = build_tokenizer(TRAIN_FILE, max_voc_size=max_voc_size, model_max_length=10)

    texts = ['This is a test.', 'Another test.']
    out = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)

    check('input_ids'    in out, 'output has input_ids')
    check('attention_mask' in out, 'output has attention_mask')

    ids  = out['input_ids']
    mask = out['attention_mask']

    check(ids.shape == mask.shape, f'input_ids and mask same shape: {ids.shape}')
    check(ids.shape[0] == 2,       'batch size = 2')
    check(ids.shape[1] <= 10,      f'length ≤ model_max_length (got {ids.shape[1]})')

    bos = tokenizer.bos_token_id
    pad = tokenizer.pad_token_id
    check(ids[0, 0].item() == bos, f'first token is BOS ({bos})')
    check(ids[1, 0].item() == bos, f'second seq also starts with BOS')

    # shorter sentence should have padding
    check((ids == pad).any().item(), 'padding tokens present for shorter sequence')
    check((mask == 0).any().item(),  'attention_mask has 0s for padding')

    # mask values are only 0 or 1
    check(mask.max().item() == 1 and mask.min().item() == 0, 'mask values are 0 or 1')

    # no padding/truncation — returns lists, not tensors
    out2 = tokenizer(texts, padding=False, truncation=False)
    check(out2['input_ids'][0][0] == bos, 'BOS present without padding/truncation')

    print(f'  input_ids:\n{ids}')
    print(f'  attention_mask:\n{mask}')


# ── Task 2.1 ─────────────────────────────────────────────────────────────────

def check_task2_1():
    print('\n=== Task 2.1: load_text_dataset ===')
    from A1_skeleton import load_text_dataset
    dataset = load_text_dataset(TRAIN_FILE, VAL_FILE)
    train = dataset['train']
    val   = dataset['val']

    check(100_000 < len(train) < 200_000,
          f'train size ~147k (got {len(train):,})')
    check(10_000 < len(val) < 30_000,
          f'val size ~18k (got {len(val):,})')
    # HuggingFace dataset rows are dicts with a 'text' key
    first = train[0]
    check('text' in first, "rows have a 'text' field")
    check(first['text'].strip() != '', 'no empty texts')
    print(f'  Train: {len(train):,}  Val: {len(val):,}')
    print(f'  First item: {first["text"][:80]}...')


# ── Task 3.1 / 3.2 ───────────────────────────────────────────────────────────

def check_task3(vocab_size=10_000, embedding_size=64, hidden_size=128, seq_len=15):
    print('\n=== Task 3: A1RNNModel ===')
    from A1_skeleton import A1RNNModel, A1RNNModelConfig

    config = A1RNNModelConfig(vocab_size=vocab_size,
                              embedding_size=embedding_size,
                              hidden_size=hidden_size)
    model = A1RNNModel(config)
    model.eval()

    # forward without labels
    x = torch.randint(0, vocab_size, (1, seq_len))
    out = model(input_ids=x)
    check(out.logits.shape == (1, seq_len, vocab_size),
          f'logits shape (1, {seq_len}, {vocab_size}): {out.logits.shape}')
    check(out.loss is None, 'loss is None when labels not provided')

    # forward with labels (loss should be a scalar)
    labels = x.clone()
    labels[labels == 0] = -100
    out2 = model(input_ids=x, labels=labels)
    check(out2.loss is not None,        'loss is computed when labels provided')
    check(out2.loss.shape == torch.Size([]), f'loss is a scalar: {out2.loss.shape}')
    check(out2.loss.item() > 0,         f'loss > 0 (got {out2.loss.item():.4f})')
    print(f'  logits shape: {out2.logits.shape}')
    print(f'  loss: {out2.loss.item():.4f}')


# ── Task 5.2: perplexity ──────────────────────────────────────────────────────

def check_task5_2(max_voc_size=10_000):
    print('\n=== Task 5.2: compute_perplexity ===')
    from A1_skeleton import (build_tokenizer, load_text_dataset,
                             A1RNNModel, A1RNNModelConfig, compute_perplexity)
    tokenizer = build_tokenizer(TRAIN_FILE, max_voc_size=max_voc_size, model_max_length=100)
    val = load_text_dataset(VAL_FILE)[:200]   # small subset for speed

    config = A1RNNModelConfig(vocab_size=len(tokenizer), embedding_size=64, hidden_size=128)
    model  = A1RNNModel(config)
    model.eval()

    ppl = compute_perplexity(model, tokenizer, val)
    check(isinstance(ppl, float), f'perplexity is a float: {ppl:.1f}')
    check(ppl > 1, 'perplexity > 1')
    print(f'  Perplexity (untrained model, expect very high): {ppl:.1f}')


# ── run all ───────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    tasks = {
        '1.1': check_task1_1,
        '1.2': check_task1_2,
        '1.3': check_task1_3,
        '2.1': check_task2_1,
        '3':   check_task3,
        '5.2': check_task5_2,
    }
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg in tasks:
                tasks[arg]()
            else:
                print(f'Unknown task: {arg}. Available: {list(tasks)}')
    else:
        for fn in tasks.values():
            try:
                fn()
            except Exception as e:
                print(f'  [ERROR] {e}')
