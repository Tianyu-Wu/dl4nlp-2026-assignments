from transformers import TrainingArguments
from A1_skeleton import *
from pathlib import Path
import logging

# --- Config ---
TRAIN_FILE       = 'train.txt'
VAL_FILE         = 'val.txt'
OUTPUT_DIR       = Path('outputs/')
MAX_VOC_SIZE     = 10000
MODEL_MAX_LENGTH = 100
EMBED_SIZE       = 128
HIDDEN_SIZE      = 512
NUM_EPOCHS       = 5
TRAIN_BATCH_SIZE = 64
EVAL_BATCH_SIZE  = 128
LEARNING_RATE    = 1e-3

OUTPUT_DIR.mkdir(exist_ok=True)
model_output_dir = OUTPUT_DIR / 'model'
model_output_dir.mkdir(exist_ok=True)

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('outputs/train.log'),
    ]
)
log = logging.getLogger(__name__)

# --- Tokenizer ---
tokenizer_path = model_output_dir / 'a1_tokenizer'
if tokenizer_path.exists():
    log.info('Loading tokenizer from %s', tokenizer_path)
    tokenizer = A1Tokenizer.from_file(tokenizer_path)
else:
    log.info('Building tokenizer from %s (max_voc_size=%d)', TRAIN_FILE, MAX_VOC_SIZE)
    tokenizer = build_tokenizer(TRAIN_FILE, max_voc_size=MAX_VOC_SIZE, model_max_length=MODEL_MAX_LENGTH)
    tokenizer.save(tokenizer_path)
log.info('Vocabulary size: %d', len(tokenizer))

# --- Dataset ---
log.info('Loading dataset...')
train_dataset, val_dataset = load_text_dataset(TRAIN_FILE, VAL_FILE)
log.info('Train: %d  Val: %d', len(train_dataset), len(val_dataset))

# --- Model ---
model_config = A1RNNModelConfig(vocab_size=len(tokenizer), embedding_size=EMBED_SIZE, hidden_size=HIDDEN_SIZE)
model = A1RNNModel(model_config)
log.info('Model parameters: %s', f'{sum(p.numel() for p in model.parameters()):,}')

# --- Training ---
# TODO: create a TrainingArguments object with appropriate hyperparameters
#       (output_dir, num_train_epochs, per_device_train_batch_size,
#        per_device_eval_batch_size, learning_rate, eval_strategy, optim)
training_args = TrainingArguments(
    output_dir=model_output_dir,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    eval_strategy="epoch",
    optim="adamw_torch"
)

# TODO: create an A1Trainer and call .train()
trainer = A1Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer)
trainer.train()

# --- Evaluation ---
log.info('Running evaluation...')

TEST_TEXT = 'She lives in San'
predicted_word = predict_next_word(model, tokenizer, TEST_TEXT)
log.info("Next word predictions for '%s': %s", TEST_TEXT, predicted_word)

perplexity = compute_perplexity(model, tokenizer, val_dataset)
log.info('Perplexity on validation set: %.2f', perplexity)

# TODO: inspect embeddings — find nearest neighbors for a few interesting words
TEST_WORDS = ["sweden", "mary", "good"]
for word in TEST_WORDS:
    if word not in tokenizer.word_to_id:
        log.warning("'%s' not in vocabulary, skipping.", word)
        continue
    neighbors = nearest_neighbors(model.embedding, tokenizer.word_to_id, tokenizer.id_to_word, word)
    log.info("Nearest neighbors of '%s': %s", word, [(n, f'{s:.4f}') for n, s in neighbors])

PCA_TEST_WORDS = [w for w in ['sweden', 'denmark', 'europe', 'africa', 'london', 'stockholm',
                               'large', 'small', 'great', 'black', '3', '7', '10',
                               'seven', 'three', 'ten', '1984', '2005', '2010']
                  if w in tokenizer.word_to_id]
log.info('Plotting PCA embeddings for: %s', PCA_TEST_WORDS)
plot_embeddings_pca(model.embedding, tokenizer.word_to_id, PCA_TEST_WORDS, OUTPUT_DIR / 'embeddings_pca.png')