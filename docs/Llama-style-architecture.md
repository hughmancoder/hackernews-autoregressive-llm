# Llama Style Transformer Architecture

[Llama Architecture](https://blog.winerva.com/llama-transformer-architecture/)

```
Layer i: LlamaDecoderLayer
├── input: x_i
│
├── input_layernorm: x_norm1 = RMSNorm(x_i)
│
├── self_attn:
│   ├── q_proj: q = Linear(x_norm1)
│   ├── k_proj: k = Linear(x_norm1)
│   ├── v_proj: v = Linear(x_norm1)
│   └── o_proj: attn_out = Linear(attn(q,k,v))
│
├── residual connection: x_attn = x_i + attn_out
│
├── post_attention_layernorm: x_norm2 = RMSNorm(x_attn)
│
├── mlp:
│   ├── gate_proj: g = Linear(x_norm2)
│   ├── up_proj: u = Linear(x_norm2)
│   ├── activation: a = SiLU(g) * u
│   └── down_proj: mlp_out = Linear(a)
│
├── residual connection: x_i+1 = x_attn + mlp_out
└── output: x_i+1 (LayerNorm)


LlamaForCausalLM
 ├─ model: LlamaModel
 │    ├─ embed_tokens
 │    ├─ layers (LlamaDecoderLayer x16)
 │    ├─ norm
 │    └─ rotary_emb
 └─ lm_head
```
## References



[LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/pdf/2302.13971)

[ROFORMER: ENHANCED TRANSFORMER WITH ROTARY
POSITION EMBEDDING](https://arxiv.org/pdf/2104.09864)

[llama architecture guide](https://pub.towardsai.net/build-your-own-llama-3-architecture-from-scratch-using-pytorch-2ce1ecaa901c)

[rope clearly explained](https://towardsdatascience.com/rope-clearly-explained/)

[pytorch Module](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html)

[Swiglu Activation Function](https://www.byhand.ai/p/swiglu-the-activation-function-behind)


[RMSNorm](https://arxiv.org/pdf/1910.07467.pdf)
