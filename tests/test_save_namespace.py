"""Saving must use the model's own parameter namespace.

Some architectures declare an on-disk naming convention that differs from the
in-memory module tree. Qwen3.5 stores text weights as ``model.language_model.*``
and its load-time mapping rewrites them to ``model.*``. By default
``save_pretrained`` *reverses* that rewrite, emitting keys that name modules the
model does not have.

That is invisible for a fully-resident model, but fatal as soon as one tensor
lives on the meta device (any ``device_map`` that offloads), because
materializing it resolves the module by name::

    AttributeError: Qwen3_5TextModel has no attribute `language_model`

``save_model_and_tokenizer`` therefore saves with
``save_original_format=False`` when the installed transformers supports it.
"""

from __future__ import annotations

import json
import os

import pytest
import torch
from torch import nn

from abliterate.model_utils import (
    _save_pretrained_in_model_namespace,
    save_model_and_tokenizer,
)


class _NewStyleModel(nn.Module):
    """Stands in for transformers >= 5.5, whose save_pretrained takes the flag."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.calls: list[dict] = []

    def save_pretrained(self, out, save_original_format: bool = True, **kwargs):
        self.calls.append({"save_original_format": save_original_format})
        return out


class _OldStyleModel(nn.Module):
    """Stands in for transformers whose save_pretrained has no such flag."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.calls: list[dict] = []

    def save_pretrained(self, out, **kwargs):
        if "save_original_format" in kwargs:
            raise TypeError("save_pretrained() got an unexpected keyword argument")
        self.calls.append(dict(kwargs))
        return out


def test_flag_passed_when_supported():
    model = _NewStyleModel()
    _save_pretrained_in_model_namespace(model, "out")
    assert model.calls == [{"save_original_format": False}]


def test_flag_omitted_when_unsupported():
    model = _OldStyleModel()
    _save_pretrained_in_model_namespace(model, "out")
    assert model.calls == [{}]


def test_save_helper_still_repairs_generation_config():
    """The namespace fix must not drop the generation-config repair."""

    class _WithGen(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(1))
            self.generation_config = type(
                "GC", (), {"do_sample": False, "temperature": 0.9, "top_p": 0.6}
            )()

        def save_pretrained(self, out, save_original_format: bool = True, **kwargs):
            return out

    class _Tok:
        def save_pretrained(self, out):
            return out

    class _Bundle:
        model = _WithGen()
        tokenizer = _Tok()

    save_model_and_tokenizer(_Bundle(), "out")
    assert _Bundle.model.generation_config.do_sample is True


def test_tiny_qwen35_saves_in_its_own_namespace(tmp_path):
    """End-to-end: saved keys must equal the model's own state_dict keys.

    Before the fix these keys carried an inserted ``language_model`` component.
    Skipped when the architecture is unavailable or too costly to build.
    """
    try:
        from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    except ImportError:
        pytest.skip("transformers has no Qwen3.5 support")

    try:
        cfg = Qwen3_5TextConfig(
            vocab_size=64,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            intermediate_size=32,
        )
    except Exception:  # pragma: no cover - config shape varies by version
        pytest.skip("could not build a tiny Qwen3_5TextConfig")

    try:
        model = Qwen3_5ForCausalLM(cfg)
    except Exception:  # pragma: no cover - arch needs extra deps/fields
        pytest.skip("could not instantiate a tiny Qwen3_5ForCausalLM")

    out = str(tmp_path / "saved")
    _save_pretrained_in_model_namespace(model, out)

    with open(os.path.join(out, "model.safetensors"), "rb"):
        pass
    from safetensors import safe_open

    with safe_open(os.path.join(out, "model.safetensors"), framework="pt") as fh:
        saved = set(fh.keys())
    own = set(model.state_dict().keys())

    assert saved == own, (
        f"namespace drift: extra={sorted(saved - own)[:3]} missing={sorted(own - saved)[:3]}"
    )
    assert not any("language_model" in k for k in saved)
