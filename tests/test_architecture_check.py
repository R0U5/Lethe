"""An unrecognized ``model_type`` must fail with a message that names the fix.

Regression guard for a failure mode that costs real time: a checkpoint whose
architecture postdates the installed transformers. The library's own error
("...does not recognize this architecture") reads like a corrupt download, so
it sends you off inspecting the checkpoint and trying load flags that cannot
possibly help. Nothing in it says "upgrade transformers".

These tests never touch the network: they point the loader's pre-flight check at
a local directory holding a hand-written ``config.json``.
"""

import json

import pytest

from abliterate import model_utils
from abliterate.config import ModelConfig

UNKNOWN_TYPE = "definitely_not_a_real_architecture"


def _checkpoint(tmp_path, payload):
    """A directory shaped like a checkpoint, with the given config.json body."""
    (tmp_path / "config.json").write_text(json.dumps(payload), encoding="utf-8")
    return ModelConfig(path=str(tmp_path))


def test_known_model_type_is_accepted(tmp_path):
    cfg = _checkpoint(tmp_path, {"model_type": "llama"})
    model_utils.assert_architecture_supported(cfg)  # must not raise


@pytest.mark.parametrize("model_type", ["qwen3", "llama", "gemma3", "mistral"])
def test_current_architectures_pass_the_guard(tmp_path, model_type):
    """The guard must not reject anything transformers legitimately knows."""
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

    if model_type not in CONFIG_MAPPING_NAMES:
        pytest.skip(f"this transformers predates {model_type}")
    model_utils.assert_architecture_supported(_checkpoint(tmp_path, {"model_type": model_type}))


def test_unknown_model_type_names_the_type_and_the_upgrade(tmp_path):
    cfg = _checkpoint(tmp_path, {"model_type": UNKNOWN_TYPE})
    with pytest.raises(model_utils.UnsupportedArchitectureError) as excinfo:
        model_utils.assert_architecture_supported(cfg)

    message = str(excinfo.value)
    assert UNKNOWN_TYPE in message, "must say which model_type is missing"
    # The message repr()s the path, so compare on the unique directory name
    # rather than the raw path (whose separators get escaped in the repr).
    assert tmp_path.name in message, "must say which checkpoint"
    assert "pip install -U transformers" in message, "must say how to fix it"


def test_error_remains_a_value_error(tmp_path):
    """Callers that guard a load with ``except ValueError`` keep working."""
    cfg = _checkpoint(tmp_path, {"model_type": UNKNOWN_TYPE})
    with pytest.raises(ValueError):
        model_utils.assert_architecture_supported(cfg)


def test_missing_config_defers_to_transformers(tmp_path):
    """Nothing to inspect: stay silent rather than invent an error."""
    model_utils.assert_architecture_supported(ModelConfig(path=str(tmp_path)))


def test_unparseable_config_defers_to_transformers(tmp_path):
    (tmp_path / "config.json").write_text("{ not json", encoding="utf-8")
    model_utils.assert_architecture_supported(ModelConfig(path=str(tmp_path)))


def test_config_without_model_type_defers_to_transformers(tmp_path):
    model_utils.assert_architecture_supported(_checkpoint(tmp_path, {"architectures": ["Foo"]}))


def test_read_model_type_reports_local_checkpoint_type(tmp_path):
    cfg = _checkpoint(tmp_path, {"model_type": "llama"})
    assert model_utils.read_model_type(cfg.path) == "llama"
