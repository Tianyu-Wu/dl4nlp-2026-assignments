"""
Sanity checks for Assignment 2.
Run from a1_2/: python sanity_check.py [1.1] [1.2] [1.3] [1.4] [1.5]
"""
import sys
import torch
from A2_skeleton import A2ModelConfig, A2RotaryEmbedding

# ── helpers ──────────────────────────────────────────────────────────────────

def ok(msg):   print(f'  [OK]   {msg}')
def fail(msg): print(f'  [FAIL] {msg}')

def check(cond, msg):
    if cond: ok(msg)
    else:    fail(msg)

def small_config():
    """Minimal config for fast sanity checks."""
    return A2ModelConfig(
        vocab_size       = 1000,
        hidden_size      = 64,
        intermediate_size= 171,   # ≈ 8/3 * 64
        num_attention_heads = 4,
        num_hidden_layers   = 2,
        rope_theta          = 10000,
        max_position_embeddings = 128,
        rms_norm_eps        = 1e-6,
    )

def make_rope(config, batch=2, seq=10):
    """Return rope_rotations for a dummy batch."""
    rope = A2RotaryEmbedding(config)
    dummy = torch.zeros(batch, seq, dtype=torch.long)
    return rope(dummy)


# ── Task 1.1: MLP ─────────────────────────────────────────────────────────────

def check_task1_1():
    print('\n=== Task 1.1: A2MLP ===')
    from A2_skeleton import A2MLP
    config = small_config()
    mlp = A2MLP(config)
    mlp.eval()

    B, M, D = 2, 10, config.hidden_size
    x   = torch.randn(B, M, D)
    out = mlp(x)

    check(out.shape == (B, M, D), f'output shape {out.shape} == ({B}, {M}, {D})')
    check(not torch.isnan(out).any(), 'no NaNs in output')


# ── Task 1.2: RMSNorm ─────────────────────────────────────────────────────────

def check_task1_2():
    print('\n=== Task 1.2: A2RMSNorm ===')
    from A2_skeleton import A2RMSNorm
    config = small_config()
    norm = A2RMSNorm(config)
    norm.eval()

    B, M, D = 2, 10, config.hidden_size
    x   = torch.randn(B, M, D)
    out = norm(x)

    check(out.shape == (B, M, D), f'output shape {out.shape} == ({B}, {M}, {D})')
    check(not torch.isnan(out).any(), 'no NaNs in output')

    # RMSNorm should reduce the RMS of each vector close to 1 (before the learned scale)
    # At init the weight is ones, so output RMS ≈ 1
    rms = out.pow(2).mean(dim=-1).sqrt()
    check(rms.mean().item() < 2.0, f'output RMS roughly normalised (mean={rms.mean():.3f})')

    # Compare against PyTorch's built-in nn.RMSNorm
    ref_norm = torch.nn.RMSNorm(D, eps=config.rms_norm_eps)
    with torch.no_grad():
        ref_norm.weight.copy_(norm.gamma)   # same learned weights
    ref_out = ref_norm(x)
    check(torch.allclose(out, ref_out, atol=1e-5),
          f'matches nn.RMSNorm (max diff={( out - ref_out).abs().max().item():.2e})')


# ── Task 1.3: Attention ───────────────────────────────────────────────────────

def check_task1_3():
    print('\n=== Task 1.3: A2Attention ===')
    from A2_skeleton import A2Attention
    config = small_config()
    attn = A2Attention(config)
    attn.eval()

    B, M, D = 2, 10, config.hidden_size
    x              = torch.randn(B, M, D)
    rope_rotations = make_rope(config, B, M)
    out            = attn(x, rope_rotations)

    check(out.shape == (B, M, D), f'output shape {out.shape} == ({B}, {M}, {D})')
    check(not torch.isnan(out).any(), 'no NaNs in output')


# ── Task 1.4: Decoder Layer ───────────────────────────────────────────────────

def check_task1_4():
    print('\n=== Task 1.4: A2DecoderLayer ===')
    from A2_skeleton import A2DecoderLayer
    config = small_config()
    layer = A2DecoderLayer(config)
    layer.eval()

    B, M, D = 2, 10, config.hidden_size
    x              = torch.randn(B, M, D)
    rope_rotations = make_rope(config, B, M)
    out            = layer(x, rope_rotations)

    check(out.shape == (B, M, D), f'output shape {out.shape} == ({B}, {M}, {D})')
    check(not torch.isnan(out).any(), 'no NaNs in output')

    # Residual connection: output should not be identical to input
    check(not torch.allclose(out, x), 'output differs from input (residual is not a no-op)')


# ── Task 1.5: Full Transformer ────────────────────────────────────────────────

def check_task1_5():
    print('\n=== Task 1.5: A2Transformer ===')
    from A2_skeleton import A2Transformer
    config = small_config()
    model  = A2Transformer(config)
    model.eval()

    B, M = 2, 10
    input_ids = torch.randint(0, config.vocab_size, (B, M))

    # forward without labels
    out = model(input_ids=input_ids)
    check(out.logits.shape == (B, M, config.vocab_size),
          f'logits shape {out.logits.shape} == ({B}, {M}, {config.vocab_size})')
    check(out.loss is None, 'loss is None when labels not provided')
    check(not torch.isnan(out.logits).any(), 'no NaNs in logits')

    # forward with labels
    labels = input_ids.clone()
    labels[labels == 0] = -100
    out2 = model(input_ids=input_ids, labels=labels)
    check(out2.loss is not None,              'loss computed when labels provided')
    check(out2.loss.shape == torch.Size([]),  f'loss is scalar: {out2.loss.shape}')
    check(out2.loss.item() > 0,               f'loss > 0 (got {out2.loss.item():.4f})')
    print(f'  logits shape : {out2.logits.shape}')
    print(f'  loss         : {out2.loss.item():.4f}')


# ── run ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    tasks = {
        '1.1': check_task1_1,
        '1.2': check_task1_2,
        '1.3': check_task1_3,
        '1.4': check_task1_4,
        '1.5': check_task1_5,
    }
    targets = sys.argv[1:] if len(sys.argv) > 1 else list(tasks)
    for t in targets:
        if t in tasks:
            try:
                tasks[t]()
            except Exception as e:
                print(f'  [ERROR] {e}')
        else:
            print(f'Unknown task: {t}. Available: {list(tasks)}')
