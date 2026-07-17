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

> Intended for models you own or are authorized to modify, for research,
> red-teaming, and reducing over-refusal. You are responsible for how you use
> the resulting weights.

## How it works

1. **Load** the model (optionally merging a LoRA/PEFT adapter first).
2. **Collect** mean residual-stream activations at the last token over a set of
   refusal-eliciting ("harmful") prompts and a set of benign ("harmless") prompts.
3. **Compute** the per-layer refusal direction as the normalized difference of
   those means.
4. **Select** a direction (by fractional depth, an explicit index, or an
   automatic sweep that picks the layer that best suppresses refusals).
5. **Apply** it, either as
   - **permanent weight orthogonalization** — rewrite the embedding and every
     attention-output / MLP-down projection so their output can't have a
     component along the refusal direction; save the resulting model; or
   - **inference-time hooks** — subtract the direction from the residual stream
     on the fly, nothing written to disk (used for fast experimentation/eval).

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
# optional: load standard research datasets (AdvBench / Alpaca) by name
pip install -e ".[datasets]"
```

Requires a floating-point (not quantized) model, since abliteration edits raw
weights.

## Quick start

```bash
# Full pipeline: directions -> select -> orthogonalize -> save (+ before/after eval)
abliterate run --model Qwen/Qwen2.5-1.5B-Instruct -o output/qwen-abliterated

# A reforge LoRA fine-tune: merge the adapter, then abliterate
abliterate run --model meta-llama/Llama-3.1-8B --lora ./reforge-runs/my-finetune -o output/my-abliterated
```

Or drive it with a config file:

```bash
abliterate run --config configs/example.yaml
```

## CLI

| Command | What it does |
|---|---|
| `abliterate run` | Full pipeline: compute directions, select, orthogonalize, save, evaluate. |
| `abliterate directions` | Compute and save per-layer candidate directions (`.pt`). |
| `abliterate apply` | Orthogonalize weights (optionally from a saved directions file) and save the model. |
| `abliterate evaluate` | Measure refusal rate, optionally applying a direction via hooks. |

Common flags (any config field can be overridden on the command line):

```
--model PATH            HF id or local path (full model, or base for --lora)
--lora PATH             PEFT/LoRA adapter dir to merge (e.g. reforge output)
--harmful SRC           refusal-eliciting prompts: file, or "advbench"
--harmless SRC          benign prompts: file, or "alpaca"
--n-samples N           prompts sampled per side (default 128)
--layer-index I         use an explicit direction index...
--layer-fraction F      ...or a fractional depth (default 0.6)
--select {config,auto}  (run only) direction selection strategy
--no-embed/--no-attn/--no-mlp   skip orthogonalizing that site
--dtype {float32,float16,bfloat16}
```

Automatic layer selection sweeps a depth band and keeps the direction that most
reduces refusal rate on held-out prompts:

```bash
abliterate run --model <model> --select auto -o output/auto
```

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

## Configuration

See `configs/example.yaml` for the full, commented schema. Sections: `model`
(source, dtype, optional architecture overrides), `data` (prompt sources and
sampling), `ablation` (activation position, batch size, direction selection, and
which residual-writing sites to touch), and `output_dir`.

## Output

`abliterate run` / `apply` write a standard HuggingFace model directory
(`config.json`, `model.safetensors`, tokenizer files) plus
`abliteration_manifest.json` recording the source model, the direction index,
how many matrices were modified, and the ablation settings.

## Development

```bash
pip install -e ".[dev]"
pytest -q
```

Tests are CPU-only and download nothing: projection math is checked against
explicit orthogonality invariants, and architecture discovery + end-to-end
orthogonalization run on a tiny in-memory model.

## Notes & caveats

- **Refusal-rate metric** uses substring matching against canonical refusal
  openers — a directional signal, not ground truth.
- **Quantized weights** are not supported; load in fp16/bf16/fp32.
- **`output_hidden_states` under hooks**: on transformers ≥ 5 the intermediate
  `hidden_states` snapshots reflect pre-ablation values (the library captures
  them with its own prepended hooks); the forward computation and generation are
  still ablated. Read `last_hidden_state` or generate to observe the effect.
- **Architecture didn't match?** If `run`/`apply` reports only one matrix
  modified, the residual-write projections weren't found — set
  `attn_out_names` / `mlp_out_names` in the model config.
