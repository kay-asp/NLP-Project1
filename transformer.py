"""Induction-head transformer for the in-context symbol task.

Implements the architecture used by Singh et al. (induction-head ICL setup,
github.com/aadityasingh/icl-dynamics); written independently for this project.
A 2-layer causal transformer with rotary positions, a learned projection for
exemplar vectors and an embedding table for labels, interleaved into one
sequence with the label read out at each exemplar position.

Operates on a single sequence; vmap over the batch in the train step.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jax.nn import gelu, softmax


def rope(x):
    # x: [heads, seq, head_dim]; rotate-half rotary embedding
    seq, hd = x.shape[-2], x.shape[-1]
    inv = 1.0 / (10000.0 ** (jnp.arange(0, hd, 2) / hd))
    ang = jnp.arange(seq)[:, None] * inv[None, :]
    cos = jnp.concatenate([jnp.cos(ang), jnp.cos(ang)], axis=-1)
    sin = jnp.concatenate([jnp.sin(ang), jnp.sin(ang)], axis=-1)
    x1, x2 = jnp.split(x, 2, axis=-1)
    return x * cos + jnp.concatenate([-x2, x1], axis=-1) * sin


class Attention(eqx.Module):
    num_heads: int
    head_dim: int
    use_rope: bool
    q: eqx.nn.Linear
    k: eqx.nn.Linear
    v: eqx.nn.Linear
    out: eqx.nn.Linear

    def __init__(self, d_model, num_heads, use_rope, *, key):
        kq, kk, kv, ko = jr.split(key, 4)
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_rope = use_rope
        self.q = eqx.nn.Linear(d_model, d_model, use_bias=False, key=kq)
        self.k = eqx.nn.Linear(d_model, d_model, use_bias=False, key=kk)
        self.v = eqx.nn.Linear(d_model, d_model, use_bias=False, key=kv)
        self.out = eqx.nn.Linear(d_model, d_model, use_bias=False, key=ko)

    def __call__(self, x, ablate_heads=()):
        # x: [seq, d_model]; returns (out [seq, d_model], attn [heads, seq, seq])
        S, D = x.shape
        H, hd = self.num_heads, self.head_dim
        to_heads = lambda proj: jax.vmap(proj)(x).reshape(S, H, hd).transpose(1, 0, 2)
        q, k, v = to_heads(self.q), to_heads(self.k), to_heads(self.v)
        if self.use_rope:
            q, k = rope(q), rope(k)
        scores = (q @ k.transpose(0, 2, 1)) / jnp.sqrt(hd)
        causal = jnp.tril(jnp.ones((S, S), bool))
        attn = softmax(jnp.where(causal[None], scores, -1e20), axis=-1)
        heads = attn @ v
        if ablate_heads:
            mask = jnp.ones((H, 1, 1)).at[jnp.array(ablate_heads)].set(0.0)
            heads = heads * mask
        merged = heads.transpose(1, 0, 2).reshape(S, D)
        return jax.vmap(self.out)(merged), attn


class MLP(eqx.Module):
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear

    def __init__(self, d_model, hidden, *, key):
        k1, k2 = jr.split(key, 2)
        self.fc1 = eqx.nn.Linear(d_model, hidden, key=k1)
        self.fc2 = eqx.nn.Linear(hidden, d_model, key=k2)

    def __call__(self, x):
        return jax.vmap(self.fc2)(gelu(jax.vmap(self.fc1)(x)))


class Block(eqx.Module):
    norm1: eqx.nn.LayerNorm
    attn: Attention
    norm2: eqx.nn.LayerNorm
    mlp: MLP

    def __init__(self, d_model, num_heads, mlp_ratio, use_rope, *, key):
        ka, km = jr.split(key, 2)
        self.norm1 = eqx.nn.LayerNorm(d_model)
        self.attn = Attention(d_model, num_heads, use_rope, key=ka)
        self.norm2 = eqx.nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, int(d_model * mlp_ratio), key=km)

    def __call__(self, x, ablate_heads=()):
        a, attn = self.attn(jax.vmap(self.norm1)(x), ablate_heads)
        x = x + a
        x = x + self.mlp(jax.vmap(self.norm2)(x))
        return x, attn


class InductionTransformer(eqx.Module):
    """Pre-norm causal transformer over interleaved (exemplar, label) tokens."""
    example_proj: eqx.nn.Linear
    label_embed: eqx.nn.Embedding
    blocks: list
    final_norm: eqx.nn.LayerNorm
    readout: eqx.nn.Linear

    def __init__(self, feature_dim, n_labels, d_model=64, depth=2, num_heads=8,
                 mlp_ratio=4.0, use_rope=True, *, key):
        ke, kl, kr, *kb = jr.split(key, depth + 3)
        self.example_proj = eqx.nn.Linear(feature_dim, d_model, key=ke)
        self.label_embed = eqx.nn.Embedding(n_labels, d_model, key=kl)
        self.blocks = [Block(d_model, num_heads, mlp_ratio, use_rope, key=kb[i])
                       for i in range(depth)]
        self.final_norm = eqx.nn.LayerNorm(d_model)
        self.readout = eqx.nn.Linear(d_model, n_labels, key=kr)

    def __call__(self, examples, labels, *, record=False, ablate=()):
        # examples: [P, feature_dim], labels: [P] (labels[-1] is the query target).
        # Interleave as [e0, l0, e1, l1, ..., e_query]; readout at exemplar positions.
        ex = jax.vmap(self.example_proj)(examples)
        lab = jax.vmap(self.label_embed)(labels[:-1])
        P, d = ex.shape
        tokens = jnp.zeros((2 * P - 1, d)).at[0::2].set(ex).at[1::2].set(lab)

        per_layer = [[] for _ in self.blocks]
        for layer, head in ablate:
            per_layer[layer].append(head)

        x, attns = tokens, []
        for blk, heads in zip(self.blocks, per_layer):
            x, attn = blk(x, tuple(heads))
            attns.append(attn)

        x = jax.vmap(self.final_norm)(x)
        logits = jax.vmap(self.readout)(x[0::2])
        if record:
            return logits, {'attn': attns, 'resid': x}
        return logits
