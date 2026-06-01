
import torch
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutput
from nltk.tokenize.treebank import TreebankWordDetokenizer



class A2ModelConfig(PretrainedConfig):
    """Configuration object that stores hyperparameters that define the Transformer language model."""
    def __init__(self, vocab_size=None, hidden_size=None, intermediate_size=None, num_attention_heads=None, 
                 num_hidden_layers=None,
                 rope_theta=None, hidden_act='silu', max_position_embeddings=None, rms_norm_eps=None, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.num_attention_heads = num_attention_heads
        self.rope_theta = rope_theta
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers



class A2MLP(nn.Module):
    """The MLP layer of the Transformer. Uses the SwiGLU architecture."""
    def __init__(self, config):
        super().__init__()
        assert(config.hidden_act == 'silu')
        # TODO: initalize components here
        self.linear_w = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.linear_v = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.linear_w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.activation = nn.SiLU()

    def forward(self, hidden_states):
        # https://arxiv.org/pdf/2002.05202
        w_out = self.linear_w(hidden_states)
        v_out = self.linear_v(hidden_states)
        return self.linear_w2(self.activation(w_out) * v_out)

# This is optional, since you can use PyTorch's RMSNorm.
class A2RMSNorm(nn.Module):
    """RMS layer normalization."""
    def __init__(self, config):
        super().__init__()
        # TODO: Use config.rms_norm_eps
        self.eps = config.rms_norm_eps
        # TODO: initalize weights here
        self.gamma = nn.Parameter(torch.ones(config.hidden_size))

    def forward(self, hidden_states):
        # equation according to https://docs.pytorch.org/docs/2.12/generated/torch.nn.RMSNorm.html
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        x_square = hidden_states.pow(2)
        x_mean_square = x_square.mean(dim=-1, keepdim=True)
        x_rms = torch.sqrt(x_mean_square + self.eps)
        result = (hidden_states / x_rms) * self.gamma
        return result.to(input_dtype)


class A2Attention(nn.Module):
    """The multi-head attention layer of the Transformer. Uses standard scaled dot-product attention with causal masking."""
    
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        # TODO: set up W_q, W_k, W_v, W_o here
        self.W_q = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.W_k = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.W_v = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.W_o = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        # TODO: set up normalizers here
        self.q_norm = A2RMSNorm(config)
        self.k_norm = A2RMSNorm(config)

    def forward(self, hidden_states, rope_rotations):
        # step 1
        q = self.W_q(hidden_states)
        k = self.W_k(hidden_states)
        v = self.W_v(hidden_states)
        q, k = self.q_norm(q), self.k_norm(k)  # RMSNorm before applying RoPE
        # reshape
        B, M, D = q.shape # batch size, sequence length, hidden size
        q = q.view(B, M, self.num_attention_heads, self.head_dim).transpose(1, 2) # (B, num_heads, M, head_dim)
        k = k.view(B, M, self.num_attention_heads, self.head_dim).transpose(1, 2) # (B, num_heads, M, head_dim)
        v = v.view(B, M, self.num_attention_heads, self.head_dim).transpose(1, 2) # (B, num_heads, M, head_dim)
        # apply RoPE
        q, k = apply_rotary_pos_emb(q, k, rope_rotations)
        # compute attention
        attn_out = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        # reshape back
        attn_out = attn_out.transpose(1, 2).reshape(B, M, D) # (B, M, hidden_size)
        out = self.W_o(attn_out)
        return out
    
    def my_scaled_dot_product_attention(self, q, k, v):
        # q, k, v: (B, num_heads, M, head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5) # (B, num_heads, M, M)
        attn_mask = torch.triu(torch.ones_like(attn_scores), diagonal=1).bool()
        attn_scores = attn_scores.masked_fill(attn_mask, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=-1) # (B, num_heads, M, M)
        attn_out = torch.matmul(attn_weights, v) # (B, num_heads, M, head_dim)
        return attn_out


class A2DecoderLayer(nn.Module):
    """A complete Transformer decoder layer."""
    def __init__(self, config):
        super().__init__()
        # TODO: set up attention, MLP, and normalizers here.
        self.attention = A2Attention(config)
        self.norm1 = A2RMSNorm(config)
        self.mlp = A2MLP(config)
        self.norm2 = A2RMSNorm(config)

    def forward(self, hidden_states, rope_rotations):
        out = self.attention(hidden_states, rope_rotations)
        out = self.norm1(out) + hidden_states # residual connection
        out_mlp = self.mlp(out)
        out = self.norm2(out_mlp) + out # residual connection
        return out


class A2Transformer(PreTrainedModel):
    """A language model based on the Transformer architecture."""
    
    config_class = A2ModelConfig

    def __init__(self, config):
        super().__init__(config)

        self.rotary_emb = A2RotaryEmbedding(config)
        # TODO: Set up the other components here.
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.top_level_norm = A2RMSNorm(config)
        self.unembedding = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # TODO: put all transformer decoder layers in a ModuleList.
        self.transformer_decoder_layers = nn.ModuleList([A2DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        
        # Note: -100 is the value HuggingFace conventionally uses to refer to tokens
        # where we do not want to compute the loss.
        self.loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100)

        # This line should be called after you have set up all components.
        self.post_init()


    def forward(self, input_ids, labels=None):
        rope_rotations = self.rotary_emb(input_ids) # pass this to all the transformer decoder layers

        # TODO: Call embedding, transformer decoder layers, last normalizer, and unembedding.
        hidden_states = self.token_embedding(input_ids)
        for layer in self.transformer_decoder_layers:
            hidden_states = layer(hidden_states, rope_rotations)
        hidden_states = self.top_level_norm(hidden_states)
        logits = self.unembedding(hidden_states)
        # TODO: Compute the loss as in Assignment 1 if labels is not None.
        loss = None
        if labels is not None:
            shifted_logits = logits[:, :-1, :].contiguous()
            shifted_labels = labels[:, 1:].contiguous()
            loss = self.loss_func(shifted_logits.view(-1, shifted_logits.shape[-1]), shifted_labels.view(-1))

        # return {"loss": loss, "logits": logits}
        return CausalLMOutput(logits=logits, loss=loss)


#### RoPE implementation (copied and simplified from HuggingFace). ####

def apply_rotary_pos_emb(q, k, rope_rotations, unsqueeze_dim=1):
    """Applies precomputed RoPE rotations to the query and key representations."""
    assert(q.shape == k.shape)
    assert(len(q.shape) == 4)
    cos, sin = rope_rotations
    assert(q.shape[2] == cos.shape[1])
    assert(q.shape[3] == cos.shape[2])    
    q_type, k_type = q.dtype, k.dtype
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(q_type), k_embed.to(k_type)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

class A2RotaryEmbedding(nn.Module):
    """RoPE position representation for use in Transformer attention."""

    def __init__(self, config, device=None):
        super().__init__()
        rope_theta = config.rope_theta
        head_dim = config.hidden_size // config.num_attention_heads
        partial_rotary_factor = 1.0
        dim = int(head_dim * partial_rotary_factor)
        self.register_buffer('inv_freq', 1.0 / (rope_theta ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)))

    @torch.no_grad()
    def forward(self, x):
        position_ids = torch.arange(0, x.shape[1], device=x.device).unsqueeze(0)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
            return cos, sin
        

def predict_next_word(model, tokenizer, text):
    """Given a text, predict the most likely next words according to the model."""
    model.eval()
    with torch.no_grad():
        input_ids = tokenizer([text], return_tensors='pt')['input_ids']
        input_ids = input_ids.to(model.device)
        outputs = model(input_ids=input_ids)
        logits = outputs.logits
        last_token_logits = logits[0, -2, :] # Get the logits for the last token (before EOS).
        index = torch.argmax(last_token_logits).item()
        predicted_word = tokenizer.id_to_word[index]
    return predicted_word


def generate_text(model, tokenizer, prompt, max_length=100, temperature=1.0, topk=50, detokenizer=TreebankWordDetokenizer()):
    """Given a prompt, generate text autoregressively from the model."""
    model.eval()
    input_ids = tokenizer([prompt], return_tensors='pt')['input_ids']
    input_ids = input_ids.to(model.device)
    generated = input_ids[:, :-1] # remove EOS token
    generated_text = prompt
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids=generated)
            logits = outputs.logits
            last_token_logits = logits[0, -1, :] / temperature
            topk_logits, topk_indices = torch.topk(last_token_logits, topk)
            next_token_id = torch.distributions.Categorical(logits=topk_logits).sample()
            next_token_id = topk_indices[next_token_id]
            generated = torch.cat([generated, next_token_id.reshape(1, -1)], dim=1)
            if next_token_id == tokenizer.eos_token_id:
                break
        tokens = [tokenizer.id_to_word[token_id.item()] for token_id in generated[0][1:]]
        generated_text = detokenizer.detokenize(tokens)
    return generated_text


def copy_olmo_weights(olmo_model):
    """Copy weights from a pre-trained OLMo-2 model into an A2Transformer.

    Verifies structural compatibility between OLMo-2 and our implementation
    by checking that all weight shapes match before copying.
    """
    olmo_cfg = olmo_model.config
    config = A2ModelConfig(
        vocab_size            = olmo_cfg.vocab_size,
        hidden_size           = olmo_cfg.hidden_size,
        intermediate_size     = olmo_cfg.intermediate_size,
        num_attention_heads   = olmo_cfg.num_attention_heads,
        num_hidden_layers     = olmo_cfg.num_hidden_layers,
        rope_theta            = (olmo_cfg.rope_parameters.get('rope_theta', 10000)
                                 if hasattr(olmo_cfg, 'rope_parameters')
                                 else getattr(olmo_cfg, 'rope_theta', 10000)),
        max_position_embeddings = olmo_cfg.max_position_embeddings,
        rms_norm_eps          = olmo_cfg.rms_norm_eps,
    )
    model = A2Transformer(config)

    olmo_sd = olmo_model.state_dict()
    our_sd  = model.state_dict()

    mapping = {'model.embed_tokens.weight': 'token_embedding.weight',
               'model.norm.weight':         'top_level_norm.gamma',
               'lm_head.weight':            'unembedding.weight'}

    for i in range(olmo_cfg.num_hidden_layers):
        p = f'model.layers.{i}'
        o = f'transformer_decoder_layers.{i}'
        mapping.update({
            f'{p}.self_attn.q_proj.weight':           f'{o}.attention.W_q.weight',
            f'{p}.self_attn.k_proj.weight':           f'{o}.attention.W_k.weight',
            f'{p}.self_attn.v_proj.weight':           f'{o}.attention.W_v.weight',
            f'{p}.self_attn.o_proj.weight':           f'{o}.attention.W_o.weight',
            f'{p}.self_attn.q_norm.weight':           f'{o}.attention.q_norm.gamma',
            f'{p}.self_attn.k_norm.weight':           f'{o}.attention.k_norm.gamma',
            f'{p}.post_attention_layernorm.weight':   f'{o}.norm1.gamma',
            f'{p}.mlp.gate_proj.weight':              f'{o}.mlp.linear_w.weight',
            f'{p}.mlp.up_proj.weight':                f'{o}.mlp.linear_v.weight',
            f'{p}.mlp.down_proj.weight':              f'{o}.mlp.linear_w2.weight',
            f'{p}.post_feedforward_layernorm.weight': f'{o}.norm2.gamma',
        })

    for olmo_key, our_key in mapping.items():
        assert olmo_sd[olmo_key].shape == our_sd[our_key].shape, \
            f'Shape mismatch: {olmo_key} {olmo_sd[olmo_key].shape} vs {our_key} {our_sd[our_key].shape}'
        our_sd[our_key] = olmo_sd[olmo_key]

    model.load_state_dict(our_sd)
    print(f'Successfully copied {len(mapping)} weight tensors from OLMo-2.')
    return model