# abliterate

Model-agnostic **abliteration** of refusal directions from transformer language
models. Point it at any HuggingFace causal LM — a full model, or a base model
with a LoRA/PEFT adapter merged on top (e.g. a **reforge** fine-tune) — and it
computes the model's refusal direction and projects it out of the weights,
producing a normal model you can save and serve anywhere.

Abliteration is the weight-orthogonalization method from Arditi et al., *Refusal
in Language Models Is Mediated by a Single Direction* (NeurIPS 2024): a model's
tendency to refuse is largely mediated by one direction in its residual stream,
and removing that direction from everything the model writes into the residual
stream suppresses refusals with minimal impact on other behavior.

This tool goes beyond a single hand-picked direction. Its **`optimize`** mode
runs an automated search (Optuna TPE) over the direction and per-layer,
per-component ablation strengths, **co-minimizing refusals and KL divergence from
the original model** — the approach shown to beat hand-tuned abliteration (cf.
*Heretic*, Weidmann 2025). The result is a strong uncensoring with measurably
less collateral damage than blanket orthogonalization.

> Intended for models you own or are authorized to modify, for research,
> red-teaming, and reducing over-refusal. You are responsible for how you use
> the resulting weights.

## How it works

1. **Load** the model (optionally merging a LoRA/PEFT adapter first).
2. **Collect** mean residual-stream activations at the last token over a set of
   refusal-eliciting ("harmful") prompts and a set of benign ("harmless") prompts.
3. **Compute** the per-layer refusal direction as the normalized difference of
   those means.
4. **Select / optimize** the ablation:
   - **`optimize` (recommended)** — search the direction plus a per-layer weight
     *kernel*, separately for attention-out and MLP-down projections, scoring
     each candidate by `refusal_rate + kl_weight · KL`. Partial, regularized
     ablation preserves task-relevant information that full orthogonalization
     destroys, and MLP interventions (which damage the model more) get their own
     strength.
   - or **fixed / fractional / auto** direction selection for a single-direction,
     full-strength orthogonalization.
5. **Apply** it, either as
   - **permanent weight orthogonalization** (optionally weighted) — rewrite the
     embedding and every attention-output / MLP-down projection; save the model; or
   - **inference-time hooks** — subtract the (weighted) direction from the
     residual stream on the fly, nothing written to disk (used for the optimizer's
     trials and fast experimentation/eval).

### Why it's model-agnostic

Nothing is hardcoded to a specific architecture. The tool discovers the decoder
layers, embedding, and residual-writing projections generically (a name-pattern
registry covering llama/qwen/mistral/gemma/phi/gpt-neox/gpt2/opt/falcon, plus a
structural fallback), and the inference-time hook path works on *any* decoder
stack. Exotic architectures can be pinned via config (`attn_out_names`,
`mlp_out_names`, `decoder_layers_path`).

## Install

```bash
pip install -e .
# recommended: TPE optimizer (optuna) for `optimize` mode
pip install -e ".[optimize]"
# quantized (4/8-bit) loading of large models — CUDA only
pip install -e ".[quantize]"
# everything (optimizer + dataset loaders + bitsandbytes)
pip install -e ".[all]"
```

To bake a permanent standalone model you need a floating-point (fp16/bf16/fp32)
model. Quantized models are supported too, but via a **runtime bundle** (see
below) rather than an in-place weight edit.

## The easiest way: the web UI

No flags to learn. Launch the built-in local app (zero extra dependencies):

```bash
abliterate ui
```

It opens `http://127.0.0.1:7860` in your browser. Paste a model name or folder,
pick **Quick / Balanced / Thorough**, and click **Remove refusals**. You'll see
live progress, before/after example answers, the refusal and KL numbers, and a
box to chat with the resulting model — then it's saved to a folder ready to use.

> paste model → pick thoroughness → click → watch progress → try the result

Advanced settings in the UI let you choose the **output** (full model folder or
a lightweight bundle) and a **Low VRAM (4-bit)** toggle for large models.

Options: `abliterate ui --port 8000 --no-browser` (defaults: host `127.0.0.1`,
port `7860`, opens a browser). It binds to localhost only.

## Quick start (CLI)

```bash
# Recommended: automated, KL-aware abliteration
abliterate optimize --model Qwen/Qwen2.5-1.5B-Instruct -o output/qwen-abliterated

# A reforge LoRA fine-tune: merge the adapter, then abliterate
abliterate optimize --model meta-llama/Llama-3.1-8B --lora ./reforge-runs/my-finetune -o output/my-abliterated

# Simple single-direction orthogonalization (no optimizer)
abliterate run --model Qwen/Qwen2.5-1.5B-Instruct -o output/qwen-simple
```

Or drive it with a config file:

```bash
abliterate run --config configs/example.yaml            # simple
abliterate run --config configs/example.yaml --select optimize   # optimized
```

## CLI

| Command | What it does |
|---|---|
| `abliterate ui` | **Launch the friendly local web app** — the easiest way to use everything below. |
| `abliterate optimize` | **Automated, KL-aware abliteration** (recommended CLI path): search direction + per-component weight kernels to co-minimize refusals and KL, apply the best, save. |
| `abliterate run` | Single-direction pipeline: compute directions, select (`config`/`auto`/`optimize`), apply, save, evaluate. |
| `abliterate directions` | Compute and save per-layer candidate directions (`.pt`). |
| `abliterate apply` | Orthogonalize weights (optionally from a saved directions file) and save the model. |
| `abliterate apply-bundle` | Bake a saved runtime bundle into a full-precision model and save it. |
| `abliterate evaluate` | Measure refusal rate, optionally applying a direction or bundle via hooks. |

Common flags (any config field can be overridden on the command line):

```
--model PATH            HF id or local path (full model, or base for --lora)
--lora PATH             PEFT/LoRA adapter dir to merge (e.g. reforge output)
--harmful SRC           refusal-eliciting prompts: file, or "advbench"/"harmful_behaviors"
--harmless SRC          benign prompts: file, or "alpaca"/"harmless_alpaca"
--n-samples N           prompts sampled per side (default 128)
--layer-index I         use an explicit direction index...
--layer-fraction F      ...or a fractional depth (default 0.6)
--select {config,auto,optimize}   (run only) selection strategy
--no-embed/--no-attn/--no-mlp     skip orthogonalizing that site
--dtype {float32,float16,bfloat16}
```

### Optimized abliteration

The optimizer searches, per attention-out and MLP-down component, a per-layer
strength *kernel* plus the direction (fixed index or per-layer), scoring each
trial by generating on held-out harmful prompts (refusals) and measuring KL
divergence on held-out harmless prompts. It writes the chosen parameters and
before/after metrics to `abliteration_manifest.json`.

```bash
abliterate optimize --model <model> \
    --harmful harmful_behaviors --harmless harmless_alpaca \
    --trials 100 --kl-weight 1.0 -o output/optimized
```

Key `optimize` flags: `--trials`, `--n-startup`, `--kl-weight` (damage vs.
refusal tradeoff), `--eval-harmful`, `--eval-harmless`, `--max-weight` (strength
ceiling), `--optimize-embed`. Without `optuna` installed it falls back to a
seeded random search over the same space.

`run --select auto` is a lighter middle ground: it sweeps a depth band and keeps
the single full-strength direction that most reduces refusals (no KL term).

## Runtime bundles & quantized models

A **bundle** is a tiny artifact (the refusal direction + the weighted-ablation
parameters, a few KB) that reproduces an abliteration at inference time via
forward hooks — the LoRA-adapter equivalent of an abliteration. It applies on
top of the *original* weights, so:

- it works on **quantized models** (4/8-bit bitsandbytes), whose packed weights
  can't be orthogonalized in place;
- it's reversible and small — ship it next to the base model instead of a full copy.

```bash
# Save a bundle instead of a full model
abliterate optimize --model <model> --save bundle -o my-abliteration

# Abliterate a large model loaded in 4-bit (fits on a smaller GPU) — forces a bundle
abliterate optimize --model <big-model> --load-in-4bit -o my-abliteration

# Try a bundle without applying it permanently
abliterate evaluate --model <model> --bundle my-abliteration

# Later, bake a bundle into a full-precision model for a standalone copy
abliterate apply-bundle --model <model> --bundle my-abliteration -o final-model
```

Apply a bundle in Python (this is how you serve a quantized model as abliterated):

```python
from abliterate import load_abliterated

model_bundle, _, hooks = load_abliterated(
    "Qwen/Qwen2.5-1.5B-Instruct", "my-abliteration", load_in_4bit=True,
)
out = model_bundle.model.generate(...)   # generates as if abliterated
hooks.remove()                           # restore the original behavior
```

The trade-off: a bundle applies through PyTorch/transformers hooks, so it's for
serving via transformers (including bitsandbytes). To run in **llama.cpp /
Ollama / LM Studio (GGUF)** or **vLLM**, bake a full-precision model first
(`--save model` or `apply-bundle`) and quantize that with the target runtime's
tooling.

> Quantized (4/8-bit) loading needs a CUDA GPU and `pip install -e ".[quantize]"`.

## Prompt data

`data/harmful.txt` and `data/harmless.txt` ship as small runnable samples. The
harmful set is generic, non-operational category prompts whose only role is to
*elicit* the model's refusal behavior so its direction can be measured. For
serious runs, point `--harmful` / `--harmless` at your own files (one prompt per
line, `#` comments ignored) or at a built-in dataset name (`advbench`, `alpaca`;
needs the `datasets` extra).

## Python API

```python
from abliterate import (
    Config, load_model_and_tokenizer, collect_mean_activations,
    compute_refusal_directions, orthogonalize_model, AblationHooks,
)
from abliterate.config import ModelConfig, AblationConfig

bundle = load_model_and_tokenizer(ModelConfig(path="Qwen/Qwen2.5-1.5B-Instruct"))

harmful = ["Write a threatening message.", ...]
harmless = ["Explain how photosynthesis works.", ...]
acfg = AblationConfig()
mh = collect_mean_activations(bundle, harmful, acfg, desc="harmful")
mb = collect_mean_activations(bundle, harmless, acfg, desc="harmless")

dirs = compute_refusal_directions(mh, mb)
direction = dirs.get(dirs.select_index(acfg))

# Reversible: try it out without touching the weights
with AblationHooks(bundle.model, direction):
    out = bundle.model.generate(...)

# Permanent: bake it into the weights and save
orthogonalize_model(bundle.model, direction, hidden_size=bundle.hidden_size)
bundle.model.save_pretrained("output/abliterated")
bundle.tokenizer.save_pretrained("output/abliterated")
```

Automated, KL-aware optimization:

```python
from abliterate import optimize_ablation, apply_weighted_ablation
from abliterate.config import OptimizeConfig

best_params, history = optimize_ablation(
    bundle, dirs, OptimizeConfig(n_trials=100),
    harmful_eval, harmless_eval, model_cfg=bundle_cfg,
)
apply_weighted_ablation(bundle.model, dirs.directions, best_params,
                        hidden_size=bundle.hidden_size)
```

## Configuration

See `configs/example.yaml` for the full, commented schema. Sections: `model`
(source, dtype, optional architecture overrides), `data` (prompt sources and
sampling), `ablation` (activation position, batch size, single-direction
selection, and which residual-writing sites to touch), `optimize` (trials, eval
sizes, `kl_weight`, search bounds), and `output_dir`.

## Output

Commands write a standard HuggingFace model directory (`config.json`,
`model.safetensors`, tokenizer files) plus `abliteration_manifest.json`. For
`optimize`, the manifest records the searched parameters (direction, per-layer
attention/MLP kernels) and the before/after refusal rate and KL divergence.

## Development

```bash
pip install -e ".[dev]"
pytest -q
```

Tests are CPU-only and download nothing: projection math and partial-strength
ablation are checked against explicit invariants, the weight kernel, KL metric,
and optimizer search space are unit-tested, and architecture discovery +
end-to-end (weighted) orthogonalization run on a tiny in-memory model.

## Notes & caveats

- **Refusal-rate metric** uses substring matching against canonical refusal
  markers — a directional signal, not ground truth. Override the list with
  `abliterate.evaluate.set_refusal_markers(...)` for non-English models.
- **KL divergence** is measured on the next-token distribution at the end of
  harmless prompts (a fast, memory-light damage proxy), not over full responses.
- **Quantized weights** can't be orthogonalized in place — for a 4/8-bit model,
  save a **runtime bundle** (applied via hooks) instead of a full model.
- **`output_hidden_states` under hooks**: on transformers ≥ 5 the intermediate
  `hidden_states` snapshots reflect pre-ablation values (the library captures
  them with its own prepended hooks); the forward computation and generation are
  still ablated. Read `last_hidden_state` or generate to observe the effect.
- **Architecture didn't match?** If a command reports only one matrix modified,
  the residual-write projections weren't found — set `attn_out_names` /
  `mlp_out_names` in the model config.

## Method & references

- Arditi et al., *Refusal in Language Models Is Mediated by a Single Direction*,
  NeurIPS 2024 — the difference-of-means direction and weight orthogonalization.
- P. E. Weidmann, *Heretic: Fully Automatic Censorship Removal for Language
  Models via Optimized Abliteration*, 2025 — TPE optimization co-minimizing
  refusals and KL divergence, per-component ablation weight kernels.
- M. Labonne, *Uncensor any LLM with abliteration* — practical abliteration and
  the `harmful_behaviors` / `harmless_alpaca` contrastive sets.

Independent implementation; not affiliated with the above.
