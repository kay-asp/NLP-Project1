"""Model-collapse pipeline for the Singh et al. induction-head task.

Produces results: data, models, training, the collapse loop, in-loop metrics
(loss / accuracy / perplexity) and checkpoint I/O. Post-hoc metrics and
interpretability live in analysis.py. Fill in the method bodies; checkpoint
helpers are implemented.

Base model: predict the query label. Extended model: also predict the next
exemplar and its label (more prone to collapse).
"""

import os
import pickle
from dataclasses import dataclass

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx
import optax

from transformer import InductionTransformer
from data import (make_embeddings, split_classes, build_sampler, next_batch,
                  sample_sequences)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

@dataclass
class Config:
    # data
    data_file: str = 'omniglot_resnet18_randomized_order_s0.h5'
    feature_dim: int = 512        # ResNet-18 feature dim of the Omniglot file
    n_train_classes: int = 50
    n_val_classes: int = 20
    n_test_classes: int = 30
    n_exemplars: int = 5
    n_labels: int = 5
    context_len: int = 2
    burstiness: int = 1
    noise_scale: float = 0.0
    # model
    d_model: int = 64
    depth: int = 2
    num_heads: int = 8
    mlp_ratio: float = 4.0
    use_rope: bool = True
    # training
    batch_size: int = 32
    lr: float = 1e-5
    train_iters: int = 100_000
    eval_every: int = 5_000
    eval_iters: int = 1_000
    # collapse
    mode: str = 'base'            # 'base' | 'extended'
    n_generations: int = 8
    synth_size: int = 10_000
    synth_ratio: float = 1.0      # fraction of each generation's data that is synthetic
    accumulate: bool = False      # keep previous generations' synthetic data
    finetune_from_prev: bool = False
    # io
    ckpt_dir: str = 'checkpoints'
    run_name: str = 'run'
    # seeds
    init_seed: int = 0
    train_seed: int = 1
    eval_seed: int = 2


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #

def build_model(cfg, key):
    """Base InductionTransformer."""
    return InductionTransformer(cfg.feature_dim, cfg.n_labels, d_model=cfg.d_model,
                                depth=cfg.depth, num_heads=cfg.num_heads,
                                mlp_ratio=cfg.mlp_ratio, use_rope=cfg.use_rope, key=key)


class ExtendedSequenceClassifier(eqx.Module):
    """Predicts query label + next exemplar + next label. Wraps a transformer."""
    base: InductionTransformer
    symbol_head: eqx.Module

    def __init__(self, cfg, key):
        ...

    def __call__(self, examples, labels, *, record=False, ablate=()):
        ...


def build_extended_model(cfg, key):
    ...


# --------------------------------------------------------------------------- #
# Losses and steps
# --------------------------------------------------------------------------- #

def _query_logits(model, batch):
    # logits at every exemplar position; the query is the last one.
    logits = jax.vmap(lambda e, l: model(e, l))(batch['examples'], batch['labels'])
    return logits[:, -1, :]


def base_loss(model, batch):
    """Cross-entropy on the query-position label."""
    logits = _query_logits(model, batch)
    targets = batch['labels'][:, -1]
    return optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()


def extended_loss(model, batch):
    """Query-label CE + next-label CE + next-symbol loss."""
    ...


def loss_fn(cfg):
    """Return the loss function for the configured mode."""
    return base_loss if cfg.mode == 'base' else extended_loss


@eqx.filter_jit
def train_step(model, opt_state, batch, optimizer, loss):
    """One optimiser update; return (model, opt_state, loss_value)."""
    loss_value, grads = eqx.filter_value_and_grad(loss)(model, batch)
    params = eqx.filter(model, eqx.is_inexact_array)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_value


@eqx.filter_jit
def eval_step(model, batch, loss):
    """Return (loss_value, query-label predictions) without updating."""
    loss_value = loss(model, batch)
    preds = jnp.argmax(_query_logits(model, batch), axis=-1)
    return loss_value, preds


# --------------------------------------------------------------------------- #
# In-loop metrics
# --------------------------------------------------------------------------- #

def accuracy(model, sampler, key, n_batches):
    """Query-label accuracy averaged over n_batches drawn from sampler."""
    correct = total = 0
    for i in range(n_batches):
        batch = next_batch(sampler, jrandom.fold_in(key, i))
        preds = jnp.argmax(_query_logits(model, batch), axis=-1)
        correct += int(jnp.sum(preds == batch['labels'][:, -1]))
        total += int(preds.shape[0])
    return correct / total


def perplexity(mean_loss):
    return float(jnp.exp(mean_loss))


# --------------------------------------------------------------------------- #
# Training (one generation)
# --------------------------------------------------------------------------- #

def train_generation(cfg, model, train_sampler, val_sampler, key):
    """Train one model; return (model, history) logging loss/accuracy at eval points."""
    optimizer = optax.adam(cfg.lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    loss = loss_fn(cfg)
    n_eval = max(1, cfg.eval_iters // cfg.batch_size)
    history = {'step': [], 'train_loss': [], 'val_loss': [], 'val_acc': []}

    for step in range(1, cfg.train_iters + 1):
        batch = next_batch(train_sampler, jrandom.fold_in(key, step))
        model, opt_state, train_loss = train_step(model, opt_state, batch, optimizer, loss)
        if step == 1 or step % cfg.eval_every == 0:
            vbatch = next_batch(val_sampler, jrandom.fold_in(key, 10_000_000 + step))
            val_loss = loss(model, vbatch)
            val_acc = accuracy(model, val_sampler,
                               jrandom.fold_in(key, 20_000_000 + step), n_eval)
            history['step'].append(step)
            history['train_loss'].append(float(train_loss))
            history['val_loss'].append(float(val_loss))
            history['val_acc'].append(float(val_acc))
            print(f'step {step:>6}/{cfg.train_iters}  train_loss {float(train_loss):.4f}  '
                  f'val_loss {float(val_loss):.4f}  val_acc {val_acc:.3f}', flush=True)
    return model, history


# --------------------------------------------------------------------------- #
# Collapse pipeline
# --------------------------------------------------------------------------- #

def generate_synthetic(cfg, model, embeddings, class_pool, key):
    """Run the trained model to produce a synthetic dataset for the next generation.
    Base: replace query labels with model predictions.
    Extended: autoregressively roll out exemplars + labels."""
    ...


def mix_data(cfg, real_sampler, synth):
    """Combine real and synthetic data per synth_ratio / accumulate; return a sampler."""
    ...


def collapse_pipeline(cfg):
    """Gen 0 trains on real data; later gens train on (mostly) synthetic data from the
    previous model. Each generation's model and history are checkpointed under
    {ckpt_dir}/{run_name}/gen{g}/. Returns (models, histories)."""
    key = jrandom.PRNGKey(cfg.init_seed)
    n_classes = cfg.n_train_classes + cfg.n_val_classes + cfg.n_test_classes
    embeddings = make_embeddings(cfg.data_file, n_classes, cfg.n_exemplars)
    train_cls, val_cls, test_cls = split_classes(cfg)
    val_sampler = build_sampler(cfg, embeddings, val_cls)
    build = build_extended_model if cfg.mode == 'extended' else build_model

    models, histories = [], []
    for g in range(cfg.n_generations + 1):
        gkey = jrandom.fold_in(key, g)
        if g == 0:
            train_sampler = build_sampler(cfg, embeddings, train_cls)
        else:
            synth = generate_synthetic(cfg, models[-1], embeddings, train_cls, gkey)
            train_sampler = mix_data(cfg, build_sampler(cfg, embeddings, train_cls), synth)

        start = models[-1] if (cfg.finetune_from_prev and g > 0) else build(cfg, gkey)
        model, history = train_generation(cfg, start, train_sampler, val_sampler, gkey)

        gdir = os.path.join(cfg.ckpt_dir, cfg.run_name, f'gen{g}')
        save_checkpoint(os.path.join(gdir, 'model.eqx'), model)
        save_pickle(os.path.join(gdir, 'history.pkl'), history)
        models.append(model)
        histories.append(history)

    save_pickle(os.path.join(cfg.ckpt_dir, cfg.run_name, 'histories.pkl'), histories)
    return models, histories


# --------------------------------------------------------------------------- #
# Checkpoint I/O
# --------------------------------------------------------------------------- #

def save_checkpoint(path, model):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    eqx.tree_serialise_leaves(path, model)


def load_checkpoint(cfg, path, kind='base', key=None):
    """Rebuild the model skeleton from cfg, then load saved leaves into it."""
    key = jrandom.PRNGKey(cfg.init_seed) if key is None else key
    skeleton = build_extended_model(cfg, key) if kind == 'extended' else build_model(cfg, key)
    return eqx.tree_deserialise_leaves(path, skeleton)


def save_pickle(path, obj):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
