import unittest
import torch
import torch.nn.functional as F

from mainrun.llama_model import (
    LlamaConfig,
    LlamaTransfomerDecoderBlock,
    LlamaModel,
    LlamaMLP,
    SwiGLU,
    RotaryPositionalEmbeddingFrequencies,
    apply_rotary_positional_embeddings,
)
    
class TestSwiGLU(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_swiglu_forward_shape(self):
        """Test output shape matches (B, T, d_model) after projection."""
        batch_size, seq_len, d_model, hidden_dim = 2, 8, 64, 172
        swiglu = SwiGLU(d_model=d_model, hidden_dim=hidden_dim)
        x = torch.randn(batch_size, seq_len, d_model)
        out = swiglu(x)

        print("\n[SwiGLU Forward Shape]")
        print(f"  Input shape      (B, T, d_model)   : {tuple(x.shape)}")
        print(f"  Intermediate dim (hidden_dim)      : {hidden_dim}")
        print(f"  Output shape     (B, T, d_model)   : {tuple(out.shape)}")

        self.assertEqual(out.shape, (batch_size, seq_len, d_model))

    def test_swiglu_zero_input(self):
        """With zero input and no biases, output should be strictly zero."""
        d_model, hidden_dim = 32, 64
        swiglu = SwiGLU(d_model=d_model, hidden_dim=hidden_dim)
        x = torch.zeros(2, 4, d_model)
        out = swiglu(x)

        print("\n[SwiGLU Zero Input]")
        print(f"  Input zero norm  : {x.norm().item():.4f}")
        print(f"  Output zero norm : {out.norm().item():.4f}")

        self.assertTrue(torch.allclose(out, torch.zeros_like(out), atol=1e-7))


class TestRotaryPositionalEmbeddingFrequencies(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_rope_projection_and_rotation_flow(self):
        """
        Verify the linear projection, head reshaping, and RoPE rotation workflow
        matching LLaMA attention preparation.
        """
        dim = 64
        n_heads = 4
        n_kv_heads = 4  
        head_dim = dim // n_heads
        max_seq_len = 32
        seq_len = 8
        batch_size = 2
        device = torch.device("cpu")

        # Projections
        wq = torch.nn.Linear(dim, n_heads * head_dim, bias=False, device=device)
        wk = torch.nn.Linear(dim, n_kv_heads * head_dim, bias=False, device=device)

        # Simulated normalized input x_norm of shape (B, T, C)
        x_norm = torch.randn(batch_size, seq_len, dim, device=device)

        # Project
        xq = wq(x_norm)
        xk = wk(x_norm)
        self.assertEqual(xq.shape, (batch_size, seq_len, n_heads * head_dim))
        self.assertEqual(xk.shape, (batch_size, seq_len, n_kv_heads * head_dim))

        # Reshape and transpose to canonical attention layout (B, n_heads, T, head_dim)
        xq = xq.view(xq.shape[0], xq.shape[1], n_heads, head_dim).transpose(1, 2)
        xk = xk.view(xk.shape[0], xk.shape[1], n_kv_heads, head_dim).transpose(1, 2)
        self.assertEqual(xq.shape, (batch_size, n_heads, seq_len, head_dim))
        self.assertEqual(xk.shape, (batch_size, n_kv_heads, seq_len, head_dim))

        # Rotary embeddings cache
        rope = RotaryPositionalEmbeddingFrequencies(head_dim=head_dim, max_seq_len=max_seq_len).to(device)
        cos, sin = rope(seq_len)
        self.assertEqual(cos.shape, (1, 1, seq_len, head_dim))
        self.assertEqual(sin.shape, (1, 1, seq_len, head_dim))

        # Rotate queries and keys
        xq_rotate, xk_rotate = apply_rotary_positional_embeddings(xq, xk, cos, sin)
        self.assertEqual(xq_rotate.shape, xq.shape)
        self.assertEqual(xk_rotate.shape, xk.shape)

        print("\n[RoPE Projection and Rotation Workflow]")
        print(f"  1. Input normalized x_norm : {tuple(x_norm.shape)} (B, T, C)")
        print(f"  2. Q projection (wq)       : {tuple(wq(x_norm).shape)} (B, T, n_heads * head_dim)")
        print(f"  3. Q transposed attention  : {tuple(xq.shape)} (B, n_heads, T, head_dim)")
        print(f"  4. RoPE cos cache          : {tuple(cos.shape)} (1, 1, T, head_dim)")
        print(f"  5. RoPE sin cache          : {tuple(sin.shape)} (1, 1, T, head_dim)")
        print(f"  6. Q rotated (xq_rotate)   : {tuple(xq_rotate.shape)} (B, n_heads, T, head_dim)")

    def test_rope_attention_transposed_layout(self):
        """
        Verify RoPE rotation on (B, n_heads, T, head_dim) format
        as used inside CausalSelfAttention.
        """
        batch_size, n_heads, seq_len, head_dim = 2, 4, 8, 16
        xq = torch.randn(batch_size, n_heads, seq_len, head_dim)
        xk = torch.randn(batch_size, n_heads, seq_len, head_dim)

        rope = RotaryPositionalEmbeddingFrequencies(head_dim=head_dim, max_seq_len=16)
        cos, sin = rope(seq_len)

        xq_rot, xk_rot = apply_rotary_positional_embeddings(xq, xk, cos, sin)

        print("\n[RoPE Attention Transposed Layout]")
        print(f"  xq layout before RoPE: {tuple(xq.shape)} (B, n_heads, T, head_dim)")
        print(f"  xq layout after RoPE : {tuple(xq_rot.shape)}")

        self.assertEqual(xq_rot.shape, (batch_size, n_heads, seq_len, head_dim))
        self.assertEqual(xk_rot.shape, (batch_size, n_heads, seq_len, head_dim))

    def test_rope_preserves_vector_norm(self):
        """Rotations are orthogonal transformations, so vector L2 norm must be preserved."""
        batch_size, n_heads, seq_len, head_dim = 2, 4, 8, 16
        xq = torch.randn(batch_size, n_heads, seq_len, head_dim)
        xk = torch.randn(batch_size, n_heads, seq_len, head_dim)

        rope = RotaryPositionalEmbeddingFrequencies(head_dim=head_dim, max_seq_len=32)
        cos, sin = rope(seq_len)

        xq_rot, xk_rot = apply_rotary_positional_embeddings(xq, xk, cos, sin)

        norm_before = torch.norm(xq, dim=-1)
        norm_after = torch.norm(xq_rot, dim=-1)

        print("\n[RoPE Preserves Vector Norm]")
        print(f"  Sample L2 norm before rotation (B=0, head=0, :4 tokens): {[round(v, 4) for v in norm_before[0, 0, :4].tolist()]}")
        print(f"  Sample L2 norm after rotation  (B=0, head=0, :4 tokens): {[round(v, 4) for v in norm_after[0, 0, :4].tolist()]}")
        print(f"  Max norm discrepancy: {(norm_before - norm_after).abs().max().item():.2e}")

        self.assertTrue(torch.allclose(norm_before, norm_after, atol=1e-5))

    def test_rope_position_zero_identity(self):
        """At position t=0, rotation angle is 0 (cos=1, sin=0), so output equals input."""
        batch_size, n_heads, seq_len, head_dim = 2, 2, 4, 8
        xq = torch.randn(batch_size, n_heads, seq_len, head_dim)
        xk = torch.randn(batch_size, n_heads, seq_len, head_dim)

        rope = RotaryPositionalEmbeddingFrequencies(head_dim=head_dim, max_seq_len=16)
        cos, sin = rope(seq_len)

        xq_rot, xk_rot = apply_rotary_positional_embeddings(xq, xk, cos, sin)

        print("\n[RoPE Position Zero Identity Check]")
        print(f"  cos[0, 0, 0, :4]: {[round(v, 4) for v in cos[0, 0, 0, :4].tolist()]} (expect 1.0)")
        print(f"  sin[0, 0, 0, :4]: {[round(v, 4) for v in sin[0, 0, 0, :4].tolist()]} (expect 0.0)")
        print(f"  xq at pos 0 diff after RoPE: {(xq_rot[:, :, 0, :] - xq[:, :, 0, :]).abs().max().item():.2e}")
        print(f"  xq at pos 1 diff after RoPE: {(xq_rot[:, :, 1, :] - xq[:, :, 1, :]).abs().max().item():.4f}")

        # Position 0 should match unchanged
        self.assertTrue(torch.allclose(xq_rot[:, :, 0, :], xq[:, :, 0, :], atol=1e-6))
        # Position > 0 should be modified by non-zero angles
        self.assertFalse(torch.allclose(xq_rot[:, :, 1, :], xq[:, :, 1, :], atol=1e-3))

    def test_rope_gradient_flow(self):
        """Ensure gradients propagate back through RoPE into projection layers without NaNs."""
        dim, n_heads, head_dim = 32, 2, 16
        seq_len, batch_size = 4, 2

        wq = torch.nn.Linear(dim, dim, bias=False)
        x = torch.randn(batch_size, seq_len, dim)

        xq = wq(x).view(batch_size, seq_len, n_heads, head_dim).transpose(1, 2)
        rope = RotaryPositionalEmbeddingFrequencies(head_dim=head_dim, max_seq_len=16)
        cos, sin = rope(seq_len)

        xq_rot, _ = apply_rotary_positional_embeddings(xq, xq, cos, sin)
        loss = xq_rot.sum()
        loss.backward()

        print("\n[RoPE Gradient Flow]")
        print(f"  wq.weight gradient shape: {tuple(wq.weight.grad.shape)}")
        print(f"  wq.weight gradient norm : {wq.weight.grad.norm().item():.4f}")

        self.assertIsNotNone(wq.weight.grad)
        self.assertFalse(torch.isnan(wq.weight.grad).any())
        self.assertTrue((wq.weight.grad != 0).any())


class TestLlamaTransfomerDecoderBlock(unittest.TestCase):
    def test_decoder_block_forward(self):
        """Test LlamaTransfomerDecoderBlock output shape matches (B, T, d_model)."""
        cfg = LlamaConfig(
            vocab_size=100,
            block_size=16,
            n_layer=2,
            n_head=4,
            d_model=64,
            dropout=0.0,
        )
        block = LlamaTransfomerDecoderBlock(cfg)
        x = torch.randn(2, 8, cfg.d_model)
        out = block(x)

        print("\n[Llama Transformer Decoder Block Forward]")
        print(f"  Config       : n_head={cfg.n_head}, d_model={cfg.d_model}, head_dim={cfg.d_model // cfg.n_head}")
        print(f"  Input shape  : {tuple(x.shape)} (batch_size, seq_len, d_model)")
        print(f"  Output shape : {tuple(out.shape)} (batch_size, seq_len, d_model)")
        print(f"  Output stats : mean={out.mean().item():.4f}, std={out.std().item():.4f}")

        self.assertEqual(out.shape, (2, 8, cfg.d_model))


class TestLlamaModel(unittest.TestCase):
    def test_llama_model_forward_and_loss(self):
        """Test LlamaModel full forward pass with token inputs and targets."""
        cfg = LlamaConfig(
            vocab_size=100,
            block_size=16,
            n_layer=2,
            n_head=2,
            d_model=32,
            dropout=0.0,
        )
        model = LlamaModel(cfg)

        batch_size = 2
        x = torch.randint(0, cfg.vocab_size, (batch_size, cfg.block_size))
        y = torch.randint(0, cfg.vocab_size, (batch_size, cfg.block_size))

        logits, loss = model(x, y)

        print("\n[Llama Full Model Forward Pass & Loss]")
        print(f"  Input tokens x shape   : {tuple(x.shape)} (B, block_size)")
        print(f"  Target tokens y shape  : {tuple(y.shape)} (B, block_size)")
        print(f"  Sample x token row 0   : {x[0, :8].tolist()} ...")
        print(f"  Output logits shape    : {tuple(logits.shape)} (B, block_size, vocab_size)")
        print(f"  Calculated loss value  : {loss.item():.4f}")

        self.assertEqual(logits.shape, (batch_size, cfg.block_size, cfg.vocab_size))
        self.assertIsNotNone(loss)
        self.assertTrue(loss.item() > 0)


if __name__ == "__main__":
    unittest.main()
