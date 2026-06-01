
import torch, nltk, pickle
from torch import nn
from collections import Counter
from datasets import load_dataset
from transformers import BatchEncoding, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput

from torch.utils.data import DataLoader
from torch.utils.data import Subset
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import numpy as np
import sys, time, os

###
### Part 1. Tokenization.
###
def lowercase_tokenizer(text):
    return [t.lower() for t in nltk.word_tokenize(text)]

def build_tokenizer(train_file, tokenize_fun=lowercase_tokenizer, max_voc_size=None, model_max_length=None,
                    pad_token='<PAD>', unk_token='<UNK>', bos_token='<BOS>', eos_token='<EOS>'):
    """ Build a tokenizer from the given file.

        Args:
             train_file:        The name of the file containing the training texts.
             tokenize_fun:      The function that maps a text to a list of string tokens.
             max_voc_size:      The maximally allowed size of the vocabulary.
             model_max_length:  Truncate texts longer than this length.
             pad_token:         The dummy string corresponding to padding.
             unk_token:         The dummy string corresponding to out-of-vocabulary tokens.
             bos_token:         The dummy string corresponding to the beginning of the text.
             eos_token:         The dummy string corresponding to the end the text.
    """

    # TODO: build the vocabulary, possibly truncating it to max_voc_size if that is specified.
    # Then return a tokenizer object (implemented below).
    counter = Counter()
    with open(train_file, 'r') as f:
        texts = f.readlines()
        for text in texts:
            tokens = tokenize_fun(text)
            counter.update(tokens)
    n = max_voc_size - 4 if max_voc_size is not None else None
    vocab = [bos_token, eos_token, unk_token, pad_token] + [t for t, _ in counter.most_common(n)]
    word_to_id = {w: i for i, w in enumerate(vocab)}
    id_to_word = {i: w for i, w in enumerate(vocab)}

    return A1Tokenizer(word_to_id, id_to_word, model_max_length, tokenize_fun, pad_token, unk_token, bos_token, eos_token)


class A1Tokenizer:
    """A minimal implementation of a tokenizer similar to tokenizers in the HuggingFace library."""

    def __init__(self, world_to_id, id_to_word, model_max_length, tokenize_fun=lowercase_tokenizer, 
                 pad_token='<PAD>', unk_token='<UNK>', bos_token='<BOS>', eos_token='<EOS>'):
        # TODO: store all values you need in order to implement __call__ below.
        self.pad_token_id = world_to_id[pad_token]     # Compulsory attribute.
        self.bos_token_id = world_to_id[bos_token]       # Compulsory attribute.
        self.eos_token_id = world_to_id[eos_token]       # Compulsory attribute.
        self.unk_token_id = world_to_id[unk_token]       # Compulsory attribute.
        self.model_max_length = model_max_length # Needed for truncation.
        self.tokenize_fun = tokenize_fun # Needed for tokenization.
        self.word_to_id = world_to_id
        self.id_to_word = id_to_word

    def __call__(self, texts, truncation=False, padding=False, return_tensors=None):
        """Tokenize the given texts and return a BatchEncoding containing the integer-encoded tokens.
           
           Args:
             texts:           The texts to tokenize.
             truncation:      Whether the texts should be truncated to model_max_length.
             padding:         Whether the tokenized texts should be padded on the right side.
             return_tensors:  If None, then return lists; if 'pt', then return PyTorch tensors.

           Returns:
             A BatchEncoding where the field `input_ids` stores the integer-encoded texts.
        """
        if return_tensors and return_tensors != 'pt':
            raise ValueError('Should be pt')
        
        # TODO: Your work here is to split the texts into words and map them to integer values.
        # 
        # - If `truncation` is set to True, the length of the encoded sequences should be 
        #   at most self.model_max_length.
        # - If `padding` is set to True, then all the integer-encoded sequences should be of the
        #   same length. That is: the shorter sequences should be "padded" by adding dummy padding
        #   tokens on the right side.
        # - If `return_tensors` is undefined, then the returned `input_ids` should be a list of lists.
        #   Otherwise, if `return_tensors` is 'pt', then `input_ids` should be a PyTorch 2D tensor.
        input_ids_list = []
        attention_mask_list = []
        max_length = 0
        for text in texts:
            tokens = self.tokenize_fun(text)
            input_ids = [self.word_to_id.get(t, self.unk_token_id) for t in tokens]
            if truncation and len(input_ids) > self.model_max_length - 2: # Account for BOS and EOS tokens.
                input_ids = input_ids[:self.model_max_length - 2]
            input_ids = [self.bos_token_id] + input_ids + [self.eos_token_id]
            attention_mask = [1] * len(input_ids)
            max_length = max(max_length, len(input_ids))
            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
        if padding:
            for input_ids, attention_mask in zip(input_ids_list, attention_mask_list):
                padding_length = max_length - len(input_ids)
                input_ids.extend([self.pad_token_id] * padding_length)
                attention_mask.extend([0] * padding_length)
                
        if return_tensors == 'pt':
            input_ids_list = torch.tensor(input_ids_list, dtype=torch.long)
            attention_mask_list = torch.tensor(attention_mask_list, dtype=torch.long)   

        # TODO: Return a BatchEncoding where input_ids stores the result of the integer encoding.
        # Optionally, if you want to be 100% HuggingFace-compatible, you should also include an 
        # attention mask of the same shape as input_ids. In this mask, padding tokens correspond
        # to the the value 0 and real tokens to the value 1.
        return BatchEncoding({'input_ids': input_ids_list, 'attention_mask': attention_mask_list})

    def __len__(self):
        """Return the size of the vocabulary."""
        return len(self.word_to_id)
    
    def save(self, filename):
        """Save the tokenizer to the given file."""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def from_file(filename):
        """Load a tokenizer from the given file."""
        with open(filename, 'rb') as f:
            return pickle.load(f)
   

###
### Part 2. Loading the text files and creating batches
###

def load_text_dataset(train_file, val_file, return_subset=False):
    """Load the text dataset from the given file and return a list of documents."""
    dataset = load_dataset('text', data_files={'train': train_file, 'val': val_file})
    dataset = dataset.filter(lambda x: x['text'].strip() != '')
    if return_subset:
        for sec in ['train', 'val']:
            dataset[sec] = Subset(dataset[sec], range(1000))
    return dataset['train'], dataset['val']

    
###
### Part 3. Defining the model.
###

class A1RNNModelConfig(PretrainedConfig):
    """Configuration object that stores hyperparameters that define the RNN-based language model."""
    def __init__(self, vocab_size=10000, embedding_size=128, hidden_size=512, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size

class A1RNNModel(PreTrainedModel):
    """The neural network model that implements a RNN-based language model."""
    config_class = A1RNNModelConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_size)
        self.rnn = nn.LSTM(config.embedding_size, config.hidden_size, batch_first=True)
        self.unembedding = nn.Linear(config.hidden_size, config.vocab_size)

        # Note: -100 is the value HuggingFace conventionally uses to refer to tokens
        # where we do not want to compute the loss.
        self.loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100)


    def forward(self, input_ids, labels=None):
        """The forward pass of the RNN-based language model.
        
           Args:
             - input_ids:  The input tensor (2D), consisting of a batch of integer-encoded texts.
             - labels:     The reference tensor (2D), consisting of a batch of integer-encoded texts.
           Returns:
             A CausalLMOutput containing
               - logits:   The output tensor (3D), consisting of logits for all token positions for all vocabulary items.
               - loss:     The loss computed on this batch.               
        """
        embedded = self.embedding(input_ids)
        rnn_out, _ = self.rnn(embedded)
        logits = self.unembedding(rnn_out)

        loss = None
        if labels is not None:
            shifted_logits = logits[:, :-1, :].contiguous()
            shifted_labels = labels[:, 1:].contiguous()
            loss = self.loss_func(shifted_logits.view(-1, shifted_logits.shape[-1]), shifted_labels.view(-1))

        return CausalLMOutput(logits=logits, loss=loss)


###
### Part 4. Training the language model.
###

## Hint: the following TrainingArguments hyperparameters may be relevant for your implementation:
#
# - optim:            What optimizer to use. You can assume that this is set to 'adamw_torch',
#                     meaning that we use the PyTorch AdamW optimizer.
# - eval_strategy:    You can assume that this is set to 'epoch', meaning that the model should
#                     be evaluated on the validation set after each epoch
# - use_cpu:          Force the trainer to use the CPU; otherwise, CUDA or MPS should be used.
#                     (In your code, you can just use the provided method select_device.)
# - learning_rate:    The optimizer's learning rate.
# - num_train_epochs: The number of epochs to use in the training loop.
# - per_device_train_batch_size: 
#                     The batch size to use while training.
# - per_device_eval_batch_size:
#                     The batch size to use while evaluating.
# - output_dir:       The directory where the trained model will be saved.

class A1Trainer:
    """A minimal implementation similar to a Trainer from the HuggingFace library."""

    def __init__(self, model, args, train_dataset, eval_dataset, tokenizer):
        """Set up the trainer.
           
           Args:
             model:          The model to train.
             args:           The training parameters stored in a TrainingArguments object.
             train_dataset:  The dataset containing the training documents.
             eval_dataset:   The dataset containing the validation documents.
             eval_dataset:   The dataset containing the validation documents.
             tokenizer:      The tokenizer.
        """
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer

        assert(args.optim == 'adamw_torch')
        assert(args.eval_strategy == 'epoch')

    def select_device(self):
        """Return the device to use for training, depending on the training arguments and the available backends."""
        if self.args.use_cpu:
            return torch.device('cpu')
        if torch.cuda.is_available():
            return torch.device('cuda')
        if torch.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
            
    def train(self, patience=2):
        """Train the model."""
        args = self.args

        device = self.select_device()
        print('Device:', device)
        self.model.to(device)
        
        loss_func = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

        # TODO: Relevant arguments: at least args.learning_rate, but you can optionally also consider
        # other Adam-related hyperparameters here.
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.learning_rate)

        # TODO: Relevant arguments: args.per_device_train_batch_size, args.per_device_eval_batch_size
        train_loader = DataLoader(self.train_dataset, batch_size=args.per_device_train_batch_size, shuffle=True)
        val_loader = DataLoader(self.eval_dataset, batch_size=args.per_device_eval_batch_size, shuffle=False)
        
        # TODO: Your work here is to implement the training loop.
        #       
        # for each training epoch (use args.num_train_epochs here):
        #   for each batch B in the training set:
        #
        #       PREPROCESSING AND FORWARD PASS:
        #       input_ids = apply your tokenizer to B
        #       labels = input_ids with padding replaced by -100
	    #       put input_ids and labels onto the GPU (or whatever device you use)
        #       apply the model to input_ids and labels
        #       get the loss from the model output
        #
        #       BACKWARD PASS AND MODEL UPDATE:
        #       optimizer.zero_grad()
        #       loss.backward()
        #       optimizer.step()

        best_val_loss = float('inf')
        epochs_without_improvement = 0
        for epoch in range(int(args.num_train_epochs)):
            self.model.train()
            for batch in train_loader:
                input_ids = self.tokenizer(batch['text'], padding=True, truncation=True, return_tensors='pt')['input_ids']
                labels = input_ids.clone()
                labels[labels == self.tokenizer.pad_token_id] = -100
                input_ids = input_ids.to(device)
                labels = labels.to(device)

                outputs = self.model(input_ids=input_ids, labels=labels)
                loss = outputs.loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # EVALUATION:
            if args.eval_strategy == 'epoch':
                # After each epoch, evaluate the model on the validation set and print the validation loss.
                self.model.eval()
                total_loss = 0
                num_batches = 0
                with torch.no_grad():
                    for batch in val_loader:
                        input_ids = self.tokenizer(batch['text'], padding=True, truncation=True, return_tensors='pt')['input_ids']
                        labels = input_ids.clone()
                        labels[labels == self.tokenizer.pad_token_id] = -100
                        input_ids = input_ids.to(device)
                        labels = labels.to(device)

                        outputs = self.model(input_ids=input_ids, labels=labels)
                        loss = outputs.loss
                        total_loss += loss.item()
                        num_batches += 1
                avg_loss = total_loss / num_batches
                print(f'Epoch {epoch + 1}/{args.num_train_epochs}, Validation Loss: {avg_loss:.4f}')
            
            # EARLY STOPPING:
            if avg_loss < best_val_loss:
                best_val_loss = avg_loss
                epochs_without_improvement = 0
                print(f'Saving to {args.output_dir}.')
                self.model.save_pretrained(args.output_dir)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f'Early stopping after {epoch + 1} epochs.')
                    break


###
### Part 5. Evaluation and analysis.
###

def predict_next_word(model, tokenizer, text, k=5):
    """Given a text, predict the top k most likely next words according to the model."""
    model.eval()
    with torch.no_grad():
        input_ids = tokenizer([text], return_tensors='pt')['input_ids']
        input_ids = input_ids.to(model.device)
        outputs = model(input_ids=input_ids)
        logits = outputs.logits
        last_token_logits = logits[0, -2, :] # Get the logits for the last token (before EOS).
        top_k_indices = torch.topk(last_token_logits, k).indices
        top_k_words = [tokenizer.id_to_word[idx.item()] for idx in top_k_indices]
    return top_k_words


def compute_perplexity(model, tokenizer, dataset):
    """Compute the perplexity of the model on the given dataset."""
    model.eval()
    total_loss = 0
    num_batches = 0
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=32):
            input_ids = tokenizer(batch['text'], padding=True, truncation=True, return_tensors='pt')['input_ids']
            labels = input_ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100
            input_ids = input_ids.to(model.device)
            labels = labels.to(model.device)
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            num_batches += 1
    avg_loss = total_loss / num_batches
    perplexity = torch.exp(torch.tensor(avg_loss))
    return perplexity.item()


def nearest_neighbors(emb, voc, inv_voc, word, n_neighbors=5):
    # Look up the embedding for the test word.
    test_emb = emb.weight[voc[word]]
    
    # We'll use a cosine similarity function to find the most similar words.
    sim_func = nn.CosineSimilarity(dim=1)
    cosine_scores = sim_func(test_emb, emb.weight)
    
    # Find the positions of the highest cosine values.
    near_nbr = cosine_scores.topk(n_neighbors+1)
    topk_cos = near_nbr.values[1:]
    topk_indices = near_nbr.indices[1:]
    # NB: the first word in the top-k list is the query word itself!
    # That's why we skip the first position in the code above.
    
    # Finally, map word indices back to strings, and put the result in a list.
    return [ (inv_voc[ix.item()], cos.item()) for ix, cos in zip(topk_indices, topk_cos) ]


def plot_embeddings_pca(emb, voc, words, filename='embeddings_pca.png'):
    vectors = np.vstack([emb.weight[voc[w]].cpu().detach().numpy() for w in words])
    vectors -= vectors.mean(axis=0)
    twodim = TruncatedSVD(n_components=2).fit_transform(vectors)
    plt.figure(figsize=(5,5))
    plt.scatter(twodim[:,0], twodim[:,1], edgecolors='k', c='r')
    for word, (x,y) in zip(words, twodim):
        plt.text(x+0.02, y, word)
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    

    