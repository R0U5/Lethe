"""Saving must survive a checkpoint whose generation config is invalid.

Llama-2 derivatives -- e.g. ``NousResearch/Nous-Hermes-Llama2-13b`` -- ship a
generation config with ``temperature``/``top_p`` set but no ``do_sample``.
transformers validates the config when saving it and refuses:

    ValueError: GenerationConfig is invalid: `temperature`: `do_sample` is set
    not to set `True`. However, `temperature` is set to `0.9` ...

so an otherwise-successful abliteration used to die at the final save, after
all of the expensive work was already done.
"""

from __future__ import annotations

import json

import pytest
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from abliterate.model_utils import (
    ModelBundle,
    make_generation_config_saveable,
    save_model_and_tokenizer,
)


def _tiny_model() -> LlamaForCausalLM:
    """A 2-layer in-memory Llama: no download, no GPU, no tokenizer files."""
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    return LlamaForCausalLM(config)


def _tiny_tokenizer() -> PreTrainedTokenizerFast:
    """A minimal word-level tokenizer built from scratch (no network)."""
    from tokenizers import Tokenizer, models

    return PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            models.WordLevel({"hello": 0, "world": 1}, unk_token="hello")
        )
    )


def _llama2_style_generation_config(model: LlamaForCausalLM) -> None:
    """Mimic Nous-Hermes-Llama2-13b: sampling values, no ``do_sample``."""
    model.generation_config.do_sample = None
    model.generation_config.temperature = 0.9
    model.generation_config.top_p = 0.6


def test_invalid_generation_config_breaks_a_plain_save(tmp_path):
    """The regression itself: transformers refuses to write such a config."""
    model = _tiny_model()
    _llama2_style_generation_config(model)

    with pytest.raises(ValueError, match="GenerationConfig is invalid"):
        model.save_pretrained(tmp_path / "unfixed")


def test_repair_enables_sampling_and_keeps_the_checkpoint_values(tmp_path):
    model = _tiny_model()
    _llama2_style_generation_config(model)

    assert make_generation_config_saveable(model) == ["temperature", "top_p"]
    assert model.generation_config.do_sample is True

    out = tmp_path / "fixed"
    model.save_pretrained(out)
    saved = json.loads((out / "generation_config.json").read_text(encoding="utf-8"))
    assert saved["do_sample"] is True
    # The checkpoint's own recommended values survive the repair.
    assert saved["temperature"] == 0.9
    assert saved["top_p"] == 0.6


def test_repair_is_idempotent():
    model = _tiny_model()
    _llama2_style_generation_config(model)

    assert make_generation_config_saveable(model) == ["temperature", "top_p"]
    # A second pass has nothing left to fix.
    assert make_generation_config_saveable(model) == []


def test_already_valid_sampling_config_is_untouched():
    """A config that already enables sampling (Qwen-style) is left alone."""
    model = _tiny_model()
    model.generation_config.do_sample = True
    model.generation_config.temperature = 0.7
    model.generation_config.top_p = 0.8

    assert make_generation_config_saveable(model) == []
    assert model.generation_config.temperature == 0.7
    assert model.generation_config.top_p == 0.8


def test_default_config_is_untouched():
    """Nothing to repair when only defaults are set -- ``do_sample`` stays unset."""
    model = _tiny_model()

    assert make_generation_config_saveable(model) == []
    assert model.generation_config.do_sample is None


def test_save_model_and_tokenizer_writes_a_usable_model(tmp_path):
    model = _tiny_model()
    _llama2_style_generation_config(model)
    bundle = ModelBundle(
        model=model, tokenizer=_tiny_tokenizer(), hidden_size=16, num_layers=2
    )

    out = tmp_path / "saved"
    save_model_and_tokenizer(bundle, out)  # must not raise

    assert (out / "config.json").is_file()
    assert (out / "tokenizer_config.json").is_file()
    saved = json.loads((out / "generation_config.json").read_text(encoding="utf-8"))
    assert saved["do_sample"] is True
