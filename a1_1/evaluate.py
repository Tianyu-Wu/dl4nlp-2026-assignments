import logging
from pathlib import Path
from A1_skeleton import (A1Tokenizer, A1RNNModel, load_text_dataset,
                         predict_next_word, compute_perplexity,
                         nearest_neighbors, plot_embeddings_pca)

TRAIN_FILE   = 'train.txt'
VAL_FILE     = 'val.txt'
OUTPUT_DIR   = Path('outputs/')
MODEL_DIR    = OUTPUT_DIR / 'model'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(OUTPUT_DIR / 'evaluate.log'),
    ]
)
log = logging.getLogger(__name__)

# --- Load tokenizer and model ---
log.info('Loading tokenizer...')
tokenizer = A1Tokenizer.from_file(MODEL_DIR / 'a1_tokenizer')

log.info('Loading model from %s...', MODEL_DIR)
model = A1RNNModel.from_pretrained(MODEL_DIR)
model.eval()

# --- Load validation set ---
_, val_dataset = load_text_dataset(TRAIN_FILE, VAL_FILE)

# --- Task 5.1: Predict next word ---
TEST_TEXT = 'She lives in San'
predicted = predict_next_word(model, tokenizer, TEST_TEXT)
log.info("Next word predictions for '%s': %s", TEST_TEXT, predicted)

# --- Task 5.2: Perplexity ---
log.info('Computing perplexity...')
ppl = compute_perplexity(model, tokenizer, val_dataset)
log.info('Perplexity on validation set: %.2f', ppl)

# --- Task 5.3: Nearest neighbors ---
TEST_WORDS = ['sweden', 'mary', 'good']
for word in TEST_WORDS:
    if word not in tokenizer.word_to_id:
        log.warning("'%s' not in vocabulary, skipping.", word)
        continue
    neighbors = nearest_neighbors(model.embedding, tokenizer.word_to_id, tokenizer.id_to_word, word)
    log.info("Nearest neighbors of '%s': %s", word, [(n, f'{s:.4f}') for n, s in neighbors])

# --- Task 5.3: PCA embedding plot ---
PCA_WORDS = [w for w in ['sweden', 'denmark', 'europe', 'africa', 'london', 'stockholm',
                          'large', 'small', 'great', 'black', '3', '7', '10',
                          'seven', 'three', 'ten', '1984', '2005', '2010']
             if w in tokenizer.word_to_id]
log.info('Plotting PCA embeddings for: %s', PCA_WORDS)
plot_embeddings_pca(model.embedding, tokenizer.word_to_id, PCA_WORDS, OUTPUT_DIR / 'embeddings_pca.png')
log.info('Plot saved to %s', OUTPUT_DIR / 'embeddings_pca.png')
