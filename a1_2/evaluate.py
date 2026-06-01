import sys
import logging
from pathlib import Path
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.append('../a1_1')
from A1_skeleton import A1Tokenizer, load_text_dataset, compute_perplexity
from A2_skeleton import A2Transformer, generate_text

VAL_FILE         = '../a1_1/val.txt'
TRAIN_FILE       = '../a1_1/train.txt'
MODEL_DIR        = Path('outputs/model')
TOKENIZER_PATH   = Path('../a1_1/outputs/model/a1_tokenizer')
OLMO_MODEL_PATH  = 'allenai/OLMo-2-0425-1B'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path('outputs/') / 'evaluate.log'),
    ]
)
log = logging.getLogger(__name__)

# --- Load tokenizer and model ---
log.info('Loading tokenizer...')
tokenizer = A1Tokenizer.from_file(TOKENIZER_PATH)

log.info('Loading model from %s...', MODEL_DIR)
model = A2Transformer.from_pretrained(MODEL_DIR)
model.eval()

# --- Load validation set ---
_, val_dataset = load_text_dataset(TRAIN_FILE, VAL_FILE)

# --- Task 5.2: Perplexity ---
log.info('Computing perplexity...')
ppl = compute_perplexity(model, tokenizer, val_dataset)
log.info('Perplexity: %.1f', ppl)

# --- Task 3.1: Predict next word ---
log.info('\n=== Task 3.1: Next word prediction ===')
NEXT_WORD_PROMPT = 'She lives in San'
generated = generate_text(model, tokenizer, NEXT_WORD_PROMPT, max_length=1, temperature=1.0, topk=1)
log.info("Prompt: '%s' → next: %s", NEXT_WORD_PROMPT, generated)

# --- Task 3.2: Text generation experiments ---

PROMPTS = [
    'In natural language processing, a Transformer',
    'Is Stockholm the capital of Sweden? Answer yes or no. The answer is',
    'Write a Python program that reverses a list.',
]

# Experiment 1: Effect of temperature (fixed prompt, fixed topk)
log.info('\n=== Experiment 1: Effect of temperature ===')
prompt = PROMPTS[0]
for temperature in [0.5, 1.0, 1.5, 2.0]:
    generated = generate_text(model, tokenizer, prompt, max_length=50, temperature=temperature, topk=50)
    log.info('[temp=%.1f] %s', temperature, generated)

# Experiment 2: Effect of top-k (fixed prompt, fixed temperature)
log.info('\n=== Experiment 2: Effect of top-k ===')
for topk in [1, 5, 50, 200]:
    generated = generate_text(model, tokenizer, prompt, max_length=50, temperature=1.0, topk=topk)
    log.info('[topk=%d] %s', topk, generated)

# Experiment 3: All prompts with default parameters
log.info('\n=== Experiment 3: Different prompts (temp=1.0, topk=50) ===')
for prompt in PROMPTS:
    generated = generate_text(model, tokenizer, prompt, max_length=100, temperature=1.0, topk=50)
    log.info("Prompt: '%s'\nGenerated: %s\n", prompt, generated)

# Experiment 4: Greedy vs sampling — same prompt, run twice to show determinism
log.info('\n=== Experiment 4: Greedy (topk=1) is deterministic ===')
for _ in range(2):
    generated = generate_text(model, tokenizer, PROMPTS[0], max_length=50, temperature=1.0, topk=1)
    log.info('[greedy] %s', generated)

log.info('\n=== Experiment 4: Sampling (topk=50) is non-deterministic ===')
for _ in range(2):
    generated = generate_text(model, tokenizer, PROMPTS[0], max_length=50, temperature=1.0, topk=50)
    log.info('[sample] %s', generated)


# --- Task 3.3: Compare to pre-trained OLMo-2 ---
log.info('\n=== Task 3.3: OLMo-2 1B generation ===')
olmo_tokenizer = AutoTokenizer.from_pretrained(OLMO_MODEL_PATH)
olmo_model = AutoModelForCausalLM.from_pretrained(OLMO_MODEL_PATH)
olmo_model.eval()

for prompt in PROMPTS:
    inputs = olmo_tokenizer(prompt, return_tensors='pt')
    output_ids = olmo_model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=1.0,
        top_k=50,
    )
    generated = olmo_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    log.info("Prompt: '%s'\nOLMo-2: %s\n", prompt, generated)

# --- Task 3.3 (optional): Copy OLMo-2 weights into A2Transformer ---
from A2_skeleton import copy_olmo_weights

log.info('\n=== Task 3.3 (optional): Weight copying ===')
log.info('OLMo-2 config attributes: %s', {k: v for k, v in vars(olmo_model.config).items() if 'rope' in k.lower() or 'theta' in k.lower() or 'rotary' in k.lower()})
try:
    a2_olmo = copy_olmo_weights(olmo_model)
    a2_olmo.eval()
    log.info('Weight copy successful — shapes are compatible.')

    # Verify outputs match OLMo-2 on a short prompt
    test_prompt = 'The capital of Sweden is'
    inputs = olmo_tokenizer(test_prompt, return_tensors='pt')
    with torch.no_grad():
        olmo_logits = olmo_model(**inputs).logits[0, -1, :]
        a2_input_ids = inputs['input_ids']
        a2_logits = a2_olmo(input_ids=a2_input_ids).logits[0, -1, :]
    top_olmo = olmo_tokenizer.decode(olmo_logits.argmax())
    top_a2   = olmo_tokenizer.decode(a2_logits.argmax())
    log.info('OLMo-2 top next token: %s', top_olmo)
    log.info('A2 (copied) top next token: %s', top_a2)
    log.info('Top tokens match: %s', top_olmo == top_a2)
    log.info('OLMo-2 logits dtype: %s', olmo_logits.dtype)
    log.info('A2 logits dtype: %s', a2_logits.dtype)
    max_diff = (olmo_logits.float() - a2_logits.float()).abs().max().item()
    log.info('Max logit difference: %.4f', max_diff)
    log.info('Outputs match (atol=1e-3): %s', torch.allclose(olmo_logits.float(), a2_logits.float(), atol=1e-3))
    log.info('Outputs match (atol=0.1): %s', torch.allclose(olmo_logits.float(), a2_logits.float(), atol=0.1))
except Exception as e:
    log.warning('Weight copy failed: %s', e)