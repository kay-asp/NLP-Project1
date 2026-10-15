"""Post-hoc metrics and interpretability for the collapse experiments.

Operates on trained / loaded models (import into the analysis notebook).
Depends on collapse.py for the config, samplers and in-loop accuracy; the
transformer's record=True path supplies attention weights and the residual
stream, and its ablate=((layer, head),...) argument supplies head ablation.
"""

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrandom

from transformer import InductionTransformer
from collapse import accuracy
from data import next_batch


# --------------------------------------------------------------------------- #
# Post-hoc distributional metrics (collapse signatures)
# --------------------------------------------------------------------------- #

def label_entropy(model, sampler, key, n_batches):
    """Entropy of the model's predicted query-label distribution (drops under collapse)."""
    ...


def symbol_distribution(synth_examples, embeddings):
    """Histogram over which symbols appear in extended-model synthetic data."""
    ...


# --------------------------------------------------------------------------- #
# Activation extraction
# --------------------------------------------------------------------------- #

def get_activations(model, batch):
    """vmap model(..., record=True) over the batch; return stacked aux
    ({'attn': [B, H, S, S] per layer, 'resid': [B, S, d]})."""
    ...


# --------------------------------------------------------------------------- #
# Circuit probes
# --------------------------------------------------------------------------- #

def attention_maps(aux, layer, head):
    """Return [B, S, S] attention scores for one layer/head."""
    ...


def prev_token_score(aux, layer):
    """Mean attention mass on the immediately-preceding token (layer-0 PT head)."""
    ...


def induction_strength(aux, class_ids, layer, head):
    """Mean query attention mass on the matching in-context exemplar (layer-1 IH).
    Uses class_ids to locate the position whose class equals the query class."""
    ...


def ablate_head(model, layer, head):
    """Return call(examples, labels) -> logits with the given head zeroed."""
    ...


def head_ablation_sweep(cfg, model, sampler, key):
    """Query-label accuracy when each head is individually ablated; identifies the circuit."""
    ...


def symbol_embeddings(model):
    """Return the learned example-projection weights for PCA / t-SNE."""
    ...
