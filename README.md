# Interpretability of Model Collapse

Studying how **model collapse** (training recursively on a model's own outputs)
degrades the **induction-head** circuits behind in-context learning, using the
synthetic symbol task of Singh et al., *What needs to go right for an induction
head?* (https://github.com/aadityasingh/icl-dynamics).

## Files

| File | Contents |
|------|----------|
| `transformer.py` | The model: a 2-layer causal transformer (attention, RoPE, MLP) with activation recording and head ablation. |
| `data.py` | Sampler for the in-context symbol task (context of exemplar–label pairs + query; labels reassigned per sequence). |
| `collapse.py` | Config, data/model builders, losses, training, the multi-generation collapse pipeline, in-loop metrics, checkpoint I/O. |
| `analysis.py` | Post-hoc metrics (label entropy, symbol distribution) and interpretability (attention maps, previous-token / induction strength, head-ablation sweep, embeddings). |
| `run.ipynb` | Driver notebook: install → config → train → checkpoint → reload → analyse. |

## Setup

Python 3.10+.

```bash
pip install -r requirements.txt
```

Default config uses randomly generated symbol embeddings, so no dataset download
is required. For a GPU, install the CUDA build of JAX first (see `requirements.txt`).

## Running

Open `run.ipynb` and run the cells top to bottom:

1. Config.
2. **Generation 0** — train one model and confirm it learns the task.
3. **Collapse pipeline** — `collapse.collapse_pipeline(cfg)` trains generations 0..N,
   checkpointing each to `checkpoints/{run_name}/gen{g}/`.
4. **Analysis** — reload from checkpoints and compute metrics / circuit probes
   across generations.

Set `cfg.mode` to `'base'` (predict the query label only) or `'extended'`
(also predict the next exemplar and its label).

## Acknowledgements

The transformer architecture and the symbol-task data setup follow Singh et al.
(https://github.com/aadityasingh/icl-dynamics); both are reimplemented here.
