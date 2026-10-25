"""Loading and architecture-agnostic introspection of causal LMs.

Abliteration only needs three things from a model, regardless of architecture:

    1. the token embedding (writes to the residual stream),
    2. the ordered list of decoder layers (to read per-layer activations), and
    3. the per-layer projections that write attention/MLP output back into the
       residual stream (attention-output and MLP-down projections).

We locate these with a small registry of leaf-name patterns covering the common
HF families, plus a structural fallback (the longest uniform ``nn.ModuleList``
is the decoder stack). Anything exotic can be pinned via ``ModelConfig``.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import torch
import transformers
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from .config import ModelConfig
from .progress import ProgressCallback, emit

logger = logging.getLogger("abliterate")

# transformers >= 5 renamed the `torch_dtype` load kwarg to `dtype`.
try:
    _TF_MAJOR = int(transformers.__version__.split(".")[0])
except (ValueError, AttributeError):  # pragma: no cover - defensive
    _TF_MAJOR = 4
_DTYPE_KWARG = "dtype" if _TF_MAJOR >= 5 else "torch_dtype"

_DTYPES = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}

# Leaf module names, by role, across common architectures. Order matters only
# for readability; matching is by membership.
_ATTN_OUT_NAMES = {
    "o_proj",       # llama, mistral, qwen2, gemma, phi3
    "out_proj",     # opt, bart-style
    "dense",        # gpt-neox, falcon, phi
    "c_proj",       # gpt2 (Conv1D)
    "wo",           # some t5-ish / custom
}
_MLP_OUT_NAMES = {
    "down_proj",       # llama, mistral, qwen2, gemma, phi3
    "dense_4h_to_h",   # gpt-neox, falcon
    "c_proj",          # gpt2 (Conv1D) -- disambiguated by parent module
    "fc2",             # opt, phi
    "w2",              # some moe/mlp variants
    "down",            # misc
}

# Common attribute paths to the decoder ModuleList, tried before the structural
# fallback.
_DECODER_PATHS = (
    "model.layers",
    "transformer.h",
    "gpt_neox.layers",
    "model.decoder.layers",
    "transformer.blocks",
    "model.transformer.h",
)


@dataclass
class ModelBundle:
    """A loaded model with its tokenizer and resolved architecture handles."""

    model: nn.Module
    tokenizer: PreTrainedTokenizerBase
    hidden_size: int
    num_layers: int


def resolve_dtype(name: str) -> torch.dtype:
    key = name.lower()
    if key not in _DTYPES:
        raise ValueError(f"unknown dtype {name!r}; choose from {sorted(_DTYPES)}")
    return _DTYPES[key]


def _is_local_path(path: str) -> bool:
    """Check if a path refers to a local directory or file."""
    p = Path(path)
    return p.exists() or (p.parent / p.name).exists()


def load_model_and_tokenizer(
    cfg: ModelConfig, *, on_progress: Optional[ProgressCallback] = None
) -> ModelBundle:
    """Load a causal LM (optionally merging a LoRA/PEFT adapter) + tokenizer.

    Weights are loaded in a floating dtype: abliteration edits raw weights, so
    quantized (int8/int4) loading is intentionally not supported here.

    ``on_progress(phase, fraction)`` is called at each sub-step so a UI can show
    motion during the (otherwise opaque) load. ``fraction`` is ``None`` for
    steps with no measurable progress (e.g. a weight download), signalling the
    caller to show an indeterminate spinner.
    """
    dtype = resolve_dtype(cfg.dtype)
    tok_path = cfg.tokenizer_path or cfg.path

    emit(on_progress, "Resolving tokenizer", 0.0)
    logger.info("loading tokenizer from %s", tok_path)
    use_local = _is_local_path(tok_path)
    emit(on_progress, "Loading tokenizer", 0.05)
    tokenizer = AutoTokenizer.from_pretrained(
        tok_path,
        trust_remote_code=cfg.trust_remote_code,
        local_files_only=use_local,
    )
    # Left padding keeps the final real token at position -1 for every sequence,
    # which is where we read activations.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = _build_quantization_config(cfg)
    logger.info("loading model from %s (dtype=%s%s)", cfg.path, cfg.dtype,
                ", 4-bit" if cfg.load_in_4bit else (", 8-bit" if cfg.load_in_8bit else ""))
    load_kwargs = {
        "device_map": cfg.device_map,
        "trust_remote_code": cfg.trust_remote_code,
        _DTYPE_KWARG: dtype,
    }
    if quant_config is not None:
        load_kwargs["quantization_config"] = quant_config
    use_local = _is_local_path(cfg.path)
    # Downloading/reading the shards is unbounded from here, so report it as an
    # indeterminate step (the UI shows a spinner rather than a frozen bar).
    emit(on_progress, "Reading weights from disk" if use_local else "Downloading weights")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.path, **{**load_kwargs, "local_files_only": use_local}
    )
    emit(on_progress, "Loading into memory", 0.7)

    if cfg.lora_adapter:
        if quant_config is not None:
            raise ValueError(
                "cannot merge a LoRA adapter into a quantized model; load in "
                "full precision to merge, or apply the adapter separately"
            )
        emit(on_progress, "Merging LoRA adapter", 0.8)
        model = _merge_lora(model, cfg.lora_adapter)

    model.eval()
    model.requires_grad_(False)

    emit(on_progress, "Resolving architecture", 0.9)
    hidden_size = _infer_hidden_size(model)
    layers = get_decoder_layers(model, cfg)
    logger.info("resolved architecture: hidden_size=%d, num_layers=%d",
                hidden_size, len(layers))
    emit(on_progress, "Model ready", 1.0)
    return ModelBundle(
        model=model,
        tokenizer=tokenizer,
        hidden_size=hidden_size,
        num_layers=len(layers),
    )


# Generation parameters that only take effect while sampling, paired with the
# value that means "left at its default". transformers validates a
# GenerationConfig before writing it (`GenerationConfig.save_pretrained` calls
# `validate(strict=True)`) and refuses to save one that sets any of these while
# `do_sample` is not enabled:
#
#     ValueError: GenerationConfig is invalid:
#     - `temperature`: `do_sample` is set not to set `True`. However,
#       `temperature` is set to `0.9` -- this flag is only used in sample-based
#       generation modes. You should set `do_sample=True` or unset `temperature`.
#
# Real checkpoints do this -- Llama-2 derivatives such as
# `NousResearch/Nous-Hermes-Llama2-13b` ship `temperature: 0.9` and
# `top_p: 0.6` with no `do_sample`, so an otherwise-successful abliteration dies
# at the very last step. Well-formed checkpoints (e.g. Qwen2.5) pair the same
# values with `do_sample: true`, so enabling sampling restores that shape and
# keeps the checkpoint's recommended decoding values instead of discarding them.
_SAMPLING_ONLY_GENERATION_PARAMS = (
    ("temperature", 1.0),
    ("top_p", 1.0),
    ("min_p", None),
    ("top_h", None),
    ("typical_p", 1.0),
    ("top_k", 50),
    ("epsilon_cutoff", 0.0),
    ("eta_cutoff", 0.0),
)


def make_generation_config_saveable(model: nn.Module) -> list[str]:
    """Make ``model``'s generation config legal to save; return the fields fixed.

    Some checkpoints pair sampling-only parameters with no ``do_sample``, which
    trips the strict validation ``save_pretrained`` performs. Enable sampling in
    that case (a no-op for configs that are already valid, including ones with
    ``do_sample: true``), so the run can finish and the checkpoint keeps its
    recommended decoding values.

    Returns the names of the parameters that required the change; empty when
    the config was already saveable.
    """
    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is None or gen_cfg.do_sample is True:
        return []

    overridden = [
        name
        for name, default in _SAMPLING_ONLY_GENERATION_PARAMS
        if (value := getattr(gen_cfg, name, None)) is not None and value != default
    ]
    if not overridden:
        return []

    gen_cfg.do_sample = True
    logger.info(
        "generation config sets %s without do_sample; enabling sampling so it "
        "can be saved",
        ", ".join(overridden),
    )
    return overridden


def _save_pretrained_in_model_namespace(model: nn.Module, out) -> None:
    """``save_pretrained`` with the checkpoint key mapping left unreversed.

    Some architectures declare an on-disk naming convention that differs from
    the in-memory module tree. Qwen3.5 stores text weights as
    ``model.language_model.*`` and its load-time mapping rewrites them to
    ``model.*``. By default ``save_pretrained`` *reverses* that rewrite, emitting
    keys that name modules the loaded model does not have. That is fatal as soon
    as one tensor lives on the meta device (any ``device_map`` that offloads),
    because materializing it resolves the module by name::

        AttributeError: Qwen3_5TextModel has no attribute `language_model`

    Saving in the model's own namespace (``save_original_format=False``) keeps
    every key resolvable; the load-time mapping then simply does not match.
    That argument postdates our transformers floor, so pass it only when the
    signature accepts it.
    """
    kwargs: dict[str, object] = {}
    try:
        if "save_original_format" in inspect.signature(model.save_pretrained).parameters:
            kwargs["save_original_format"] = False
    except (TypeError, ValueError):  # pragma: no cover - defensive
        pass
    model.save_pretrained(out, **kwargs)


def save_model_and_tokenizer(bundle: "ModelBundle", out) -> None:
    """Save an abliterated model + tokenizer, repairing the generation config.

    Always save through this rather than calling ``save_pretrained`` directly:
    a checkpoint whose generation config is invalid (see
    ``make_generation_config_saveable``) otherwise fails here, after all of the
    expensive work has already been done. Models loaded through a ``device_map``
    that offloads are handled too - see
    ``_save_pretrained_in_model_namespace``.
    """
    make_generation_config_saveable(bundle.model)
    _save_pretrained_in_model_namespace(bundle.model, out)
    bundle.tokenizer.save_pretrained(out)


def _build_quantization_config(cfg: ModelConfig):
    """Return a BitsAndBytesConfig for 4/8-bit loading, or None."""
    if not (cfg.load_in_4bit or cfg.load_in_8bit):
        return None
    try:
        from transformers import BitsAndBytesConfig
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError("quantized loading needs transformers with bitsandbytes") from exc
    if cfg.load_in_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=resolve_dtype(cfg.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=True,
        )
    return BitsAndBytesConfig(load_in_8bit=True)


def is_quantized(model: nn.Module) -> bool:
    """True if the model was loaded with bitsandbytes quantization."""
    if getattr(getattr(model, "config", None), "quantization_config", None) is not None:
        return True
    return any(type(m).__name__ in ("Linear4bit", "Linear8bitLt") for m in model.modules())


def _merge_lora(model: nn.Module, adapter_path: str) -> nn.Module:
    """Attach a PEFT adapter and bake it into the base weights.

    This is the path used for reforge fine-tunes, but it accepts any PEFT
    adapter directory. Merging is required: abliteration operates on the
    effective (post-merge) weight matrices.
    """
    try:
        from peft import PeftModel
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "peft is required to merge a LoRA adapter; `pip install peft`"
        ) from exc

    logger.info("merging LoRA adapter from %s", adapter_path)
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    return model


def _infer_hidden_size(model: nn.Module) -> int:
    cfg = getattr(model, "config", None)
    for attr in ("hidden_size", "n_embd", "d_model", "hidden_dim"):
        val = getattr(cfg, attr, None)
        if isinstance(val, int):
            return val
    # Fall back to the embedding matrix width.
    emb = get_embedding_module(model)
    return emb.weight.shape[-1]


def _getattr_path(root: object, path: str) -> Optional[object]:
    cur = root
    for part in path.split("."):
        cur = getattr(cur, part, None)
        if cur is None:
            return None
    return cur


def get_decoder_layers(model: nn.Module, cfg: Optional[ModelConfig] = None) -> nn.ModuleList:
    """Return the ordered ModuleList of decoder blocks."""
    if cfg is not None and cfg.decoder_layers_path:
        found = _getattr_path(model, cfg.decoder_layers_path)
        if isinstance(found, nn.ModuleList):
            return found
        raise ValueError(
            f"decoder_layers_path {cfg.decoder_layers_path!r} did not resolve "
            f"to a ModuleList"
        )

    for path in _DECODER_PATHS:
        found = _getattr_path(model, path)
        if isinstance(found, nn.ModuleList) and len(found) > 0:
            return found

    # Structural fallback: the longest ModuleList of uniform block type.
    best: Optional[nn.ModuleList] = None
    for module in model.modules():
        if isinstance(module, nn.ModuleList) and len(module) > 0:
            types = {type(m).__name__ for m in module}
            if len(types) == 1 and (best is None or len(module) > len(best)):
                best = module
    if best is not None:
        return best
    raise RuntimeError(
        "could not locate decoder layers; set model.decoder_layers_path in config"
    )


def get_base_model(model: nn.Module) -> nn.Module:
    """Return the base transformer, skipping the LM head.

    Running the base module for ``output_hidden_states`` avoids the (large)
    vocabulary projection that ``ForCausalLM.forward`` always computes. Uses the
    model's own ``base_model_prefix`` (``model``, ``transformer``, ...), falling
    back to the model itself when it is already a base module.
    """
    prefix = getattr(model, "base_model_prefix", None)
    if prefix and hasattr(model, prefix):
        base = getattr(model, prefix)
        if isinstance(base, nn.Module):
            return base
    return model


def get_embedding_module(model: nn.Module) -> nn.Embedding:
    emb = model.get_input_embeddings()
    if not isinstance(emb, nn.Embedding):
        raise RuntimeError("input embedding is not an nn.Embedding")
    return emb


def _leaf_name(qualified: str) -> str:
    return qualified.rsplit(".", 1)[-1]


def iter_residual_write_modules(
    layer: nn.Module,
    hidden_size: int,
    cfg: Optional[ModelConfig] = None,
    include_attn: bool = True,
    include_mlp: bool = True,
) -> Iterator[tuple[str, nn.Module]]:
    """Yield (role, module) for projections that write into the residual stream.

    ``role`` is either ``"attn_out"`` or ``"mlp_out"``. Modules are matched by
    leaf name (with config overrides) and validated to actually output a
    hidden-size vector, which filters out same-named layers of the wrong shape.
    """
    attn_names = set(cfg.attn_out_names) if (cfg and cfg.attn_out_names) else _ATTN_OUT_NAMES
    mlp_names = set(cfg.mlp_out_names) if (cfg and cfg.mlp_out_names) else _MLP_OUT_NAMES

    for name, module in layer.named_modules():
        if not _writes_hidden_vector(module, hidden_size):
            continue
        leaf = _leaf_name(name)
        # Disambiguate the gpt2 "c_proj" collision by parent module name.
        parent = name.rsplit(".", 2)[0] if "." in name else ""
        is_mlp_ctx = "mlp" in parent.lower() or "feed" in parent.lower()
        is_attn_ctx = "attn" in parent.lower() or "attention" in parent.lower()

        if include_mlp and leaf in mlp_names and (is_mlp_ctx or leaf != "c_proj"):
            yield "mlp_out", module
        elif include_attn and leaf in attn_names and (is_attn_ctx or leaf != "c_proj"):
            yield "attn_out", module


def _writes_hidden_vector(module: nn.Module, hidden_size: int) -> bool:
    """True if the module's output dimension equals hidden_size."""
    if isinstance(module, nn.Linear):
        return module.out_features == hidden_size
    # transformers Conv1D (gpt2): weight is (in, out); output is last dim.
    if type(module).__name__ == "Conv1D":
        weight = getattr(module, "weight", None)
        return weight is not None and weight.shape[-1] == hidden_size
    return False
