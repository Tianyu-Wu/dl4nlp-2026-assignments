from transformers import TrainingArguments
from A2_skeleton import *
from pathlib import Path
import sys
sys.path.append('../a1_1')
from A1_skeleton import A1Tokenizer, build_tokenizer, load_text_dataset
import logging

# --- Config ---
TRAIN_FILE       = '../a1_1/train.txt'
VAL_FILE         = '../a1_1/val.txt'
OUTPUT_DIR       = Path('outputs/')
MAX_VOC_SIZE     = 10_000
MODEL_MAX_LENGTH = 512
NUM_EPOCHS       = 5
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE  = 64
LEARNING_RATE    = 1e-3

# Model hyperparameters
NUM_LAYERS        = 4
HIDDEN_SIZE       = 256
NUM_HEADS         = 4
INTERMEDIATE_SIZE = 682    # ≈ 8/3 * HIDDEN_SIZE
ROPE_THETA        = 10_000
RMS_NORM_EPS      = 1e-6

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
        logging.FileHandler(OUTPUT_DIR / 'train.log'),
    ]
)
log = logging.getLogger(__name__)

# --- Tokenizer ---
tokenizer_path = Path('../a1_1/outputs/model/a1_tokenizer')
if tokenizer_path.exists():
    log.info('Loading tokenizer from %s', tokenizer_path)
    tokenizer = A1Tokenizer.from_file(tokenizer_path)
else:
    log.info('Building tokenizer...')
    tokenizer = build_tokenizer(TRAIN_FILE, max_voc_size=MAX_VOC_SIZE, model_max_length=MODEL_MAX_LENGTH)
    tokenizer.save(model_output_dir / 'a1_tokenizer')
log.info('Vocabulary size: %d', len(tokenizer))

# --- Dataset ---
log.info('Loading dataset...')
train_dataset, val_dataset = load_text_dataset(TRAIN_FILE, VAL_FILE)
log.info('Train: %d  Val: %d', len(train_dataset), len(val_dataset))

# --- Model ---
# TODO: create an A2ModelConfig with the hyperparameters above
config = A2ModelConfig(
    vocab_size = len(tokenizer), 
    hidden_size=HIDDEN_SIZE, 
    intermediate_size=INTERMEDIATE_SIZE, 
    num_attention_heads=NUM_HEADS, 
    num_hidden_layers=NUM_LAYERS, 
    rope_theta=ROPE_THETA, 
    max_position_embeddings=MODEL_MAX_LENGTH,
    rms_norm_eps=RMS_NORM_EPS
)

# TODO: create the A2Transformer model
model = A2Transformer(config)
log.info('Parameters: %s', f'{sum(p.numel() for p in model.parameters()):,}')

# --- Training ---
# TODO: create TrainingArguments (same as A1)
training_args = TrainingArguments(
    output_dir=model_output_dir,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    eval_strategy="epoch",
    optim="adamw_torch"
)

# TODO: reuse A1Trainer from A1_skeleton — it works with any model that returns loss
sys.path.append('../a1_1')
from A1_skeleton import A1Trainer
trainer = A1Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer
)
trainer.train()

