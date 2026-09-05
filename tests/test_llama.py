import unittest
import torch
import torch.nn.functional as F

from mainrun.llama_model import (
    LlamaConfig,
    LlamaDecoderBlock,
    LlamaModel,
    LlamaMLP,
    SwiGLU,
    RotaryPositionalEmbeddingFrequencies,
    apply_rotary_positional_embeddings,
)


class TestRMSNorm(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_rmsnorm_forward_shape_3d(self):
        """Test output shape matches (B, T, C) input shape."""
        batch_size, seq_len, d_model = 4, 16, 64
        norm = torch.nn.RMSNorm(d_model)
        x = torch.randn(batch_size, seq_len, d_model)
        out = norm(x)

        self.assertEqual(out.shape, (batch_size, seq_len, d_model))

    def test_rmsnorm_forward_shape_2d(self):
        """Test output shape matches (B, C) input shape."""
        batch_size, d_model = 8, 32
        norm = torch.nn.RMSNorm(d_model)
        x = torch.randn(batch_size, d_model)
        out = norm(x)

        self.assertEqual(out.shape, (batch_size, d_model))

    def test_rmsnorm_numerical_formula(self):
        """Verify output matches manual RMSNorm formula."""
        d_model = 16
        eps = 1e-5
        norm = torch.nn.RMSNorm(d_model, eps=eps)
        # Set custom non-trivial weights to test scaling
        with torch.no_grad():
            norm.weight.copy_(torch.linspace(0.5, 1.5, d_model))

        x = torch.randn(2, 4, d_model)
        out = norm(x)

        # Manual computation
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        expected = x * torch.rsqrt(variance + eps) * norm.weight

        self.assertTrue(torch.allclose(out, expected, atol=1e-5))

    def test_rmsnorm_unit_rms(self):
        """Verify RMS is approximately 1 along the feature dimension with unit weights."""
        d_model = 128
        norm = torch.nn.RMSNorm(d_model, eps=1e-8)
        x = torch.randn(4, 8, d_model) * 5.0 + 2.0
        out = norm(x)

        rms = torch.sqrt(out.pow(2).mean(dim=-1))
        # With unit weights, RMS of output should be close to 1.0
        expected_rms = torch.ones_like(rms)
        self.assertTrue(torch.allclose(rms, expected_rms, atol=1e-3))

    def test_rmsnorm_gradient_flow(self):
        """Verify gradients flow to both weights and input tensor without NaNs."""
        d_model = 32
        norm = torch.nn.RMSNorm(d_model)
        x = torch.randn(2, 4, d_model, requires_grad=True)

        out = norm(x)
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(norm.weight.grad)
        self.assertFalse(torch.isnan(norm.weight.grad).any())
        self.assertTrue((norm.weight.grad != 0).any())

        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad).any())
        self.assertTrue((x.grad != 0).any())


class TestSwiGLU(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_swiglu_forward_shape(self):
        """Test output shape matches (B, T, d_model) after projection."""
        batch_size, seq_len, d_model, hidden_dim = 2, 8, 64, 172
        swiglu = SwiGLU(d_model=d_model, hidden_dim=hidden_dim)
        x = torch.randn(batch_size, seq_len, d_model)
        out = swiglu(x)

        self.assertEqual(out.shape, (batch_size, seq_len, d_model))


    def test_swiglu_zero_input(self):
        """With zero input and no biases, output should be strictly zero."""
        d_model, hidden_dim = 32, 64
        swiglu = SwiGLU(d_model=d_model, hidden_dim=hidden_dim)
        x = torch.zeros(2, 4, d_model)
        out = swiglu(x)

        self.assertTrue(torch.allclose(out, torch.zeros_like(out), atol=1e-7))

    def test_swiglu_non_linearity(self):
        """Verify SwiGLU is non-linear: f(a + b) != f(a) + f(b)."""
        d_model, hidden_dim = 32, 64
        swiglu = SwiGLU(d_model=d_model, hidden_dim=hidden_dim)
        a = torch.randn(2, 4, d_model)
        b = torch.randn(2, 4, d_model)

        out_sum = swiglu(a + b)
        sum_out = swiglu(a) + swiglu(b)

        self.assertFalse(torch.allclose(out_sum, sum_out, atol=1e-3))


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
        self.assertTrue(torch.allclose(norm_before, norm_after, atol=1e-5))

    def test_rope_position_zero_identity(self):
        """At position t=0, rotation angle is 0 (cos=1, sin=0), so output equals input."""
        batch_size, n_heads, seq_len, head_dim = 2, 2, 4, 8
        xq = torch.randn(batch_size, n_heads, seq_len, head_dim)
        xk = torch.randn(batch_size, n_heads, seq_len, head_dim)

        rope = RotaryPositionalEmbeddingFrequencies(head_dim=head_dim, max_seq_len=16)
        cos, sin = rope(seq_len)

        xq_rot, xk_rot = apply_rotary_positional_embeddings(xq, xk, cos, sin)

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

        self.assertIsNotNone(wq.weight.grad)
        self.assertFalse(torch.isnan(wq.weight.grad).any())
        self.assertTrue((wq.weight.grad != 0).any())


class TestLlamaDecoderBlock(unittest.TestCase):
    def test_decoder_block_forward(self):
        """Test LlamaDecoderBlock output shape matches (B, T, d_model)."""
        cfg = LlamaConfig(
            vocab_size=100,
            block_size=16,
            n_layer=2,
            n_head=4,
            d_model=64,
            dropout=0.0,
        )
        block = LlamaDecoderBlock(cfg)
        x = torch.randn(2, 8, cfg.d_model)
        out = block(x)
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
        self.assertEqual(logits.shape, (batch_size, cfg.block_size, cfg.vocab_size))
        self.assertIsNotNone(loss)
        self.assertTrue(loss.item() > 0)

    def test_residual_projection_scaling(self):
        """Verify residual output projections have scaled std = 0.02 / sqrt(2 * n_layer)."""
        cfg = LlamaConfig(
            vocab_size=100,
            block_size=16,
            n_layer=4,
            n_head=2,
            d_model=64,
            dropout=0.0,
        )
        model = LlamaModel(cfg)
        expected_scale = (2 * cfg.n_layer) ** -0.5  # 1 / sqrt(8) approx 0.3535
        expected_std = 0.02 * expected_scale       # approx 0.00707

        for block in model.blocks:
            attn_proj_std = block.attn.proj.weight.std().item()
            down_proj_std = block.mlp.swiglu.down_proj.weight.std().item()

            # Std should be close to expected_std (within sample variance tolerance)
            self.assertAlmostEqual(attn_proj_std, expected_std, delta=0.003)
            self.assertAlmostEqual(down_proj_std, expected_std, delta=0.003)


if __name__ == "__main__":
    unittest.main()
