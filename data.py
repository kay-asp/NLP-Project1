"""Data sampling for the in-context symbol task.

Same task and ICL-only logic as Singh et al.'s induction-head setup
(github.com/aadityasingh/icl-dynamics); written independently for this project.
Each sequence is a context of (exemplar, label) pairs followed by a query
exemplar. The query's class appears in the context, and labels are reassigned
per sequence, so the mapping can only be recovered in-context.
"""

from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr


def _sample_one(key, embeddings, class_pool, context_len, n_labels, burstiness):
    n_pool = class_pool.shape[0]
    n_exemplars = embeddings.shape[1]
    kcls, klab, kord, kex = jr.split(key, 4)

    n_distract = context_len - burstiness
    # Distinct classes for this sequence: slot 0 is the query class.
    chosen = class_pool[jr.permutation(kcls, n_pool)[: 1 + n_distract]]
    # One distinct label per distinct class (random per sequence -> forces ICL).
    chosen_labels = jr.permutation(klab, n_labels)[: 1 + n_distract]

    # Context positions: `burstiness` query copies + each distractor once, shuffled.
    ctx_slots = jnp.concatenate([jnp.zeros(burstiness, int),
                                 jnp.arange(1, 1 + n_distract)])
    ctx_slots = jr.permutation(kord, ctx_slots)
    slots = jnp.concatenate([ctx_slots, jnp.array([0])])  # append query

    seq_classes = chosen[slots]
    seq_labels = chosen_labels[slots]
    ex_idx = jr.randint(kex, (context_len + 1,), 0, n_exemplars)
    examples = embeddings[seq_classes, ex_idx]
    return {'examples': examples, 'labels': seq_labels, 'class_ids': seq_classes}


def sample_sequences(key, embeddings, class_pool, batch_size,
                     context_len, n_labels, burstiness=1):
    """Batched sampler. Returns dict of examples [B, P, D], labels/class_ids [B, P]."""
    keys = jr.split(key, batch_size)
    return jax.vmap(partial(_sample_one, embeddings=embeddings, class_pool=class_pool,
                            context_len=context_len, n_labels=n_labels,
                            burstiness=burstiness))(keys)


def make_sampler(embeddings, class_pool, batch_size, context_len, n_labels, burstiness=1):
    """Return sample(key) -> batch, fixing everything but the PRNG key."""
    def sample(key):
        return sample_sequences(key, embeddings, class_pool, batch_size,
                                context_len, n_labels, burstiness)
    return sample


# --------------------------------------------------------------------------- #
# Dataset construction and split wiring (cfg is duck-typed; no Config import)
# --------------------------------------------------------------------------- #

def make_embeddings(key, n_classes, n_exemplars, feature_dim):
    """Return a [n_classes, n_exemplars, feature_dim] array of fixed symbol vectors."""
    ...


def split_classes(cfg):
    """Return (train_idxs, val_idxs, test_idxs) arrays into the class axis."""
    ...


def build_sampler(cfg, embeddings, class_pool):
    """Return sample(key) -> batch, via make_sampler over this split's classes."""
    ...


def next_batch(sampler, key):
    """Draw one {'examples', 'labels', 'class_ids'} batch."""
    ...
