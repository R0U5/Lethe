"""Prompt loading is memoized: each distinct source is read once per process."""

from __future__ import annotations

import logging

import pytest

from abliterate import data


@pytest.fixture(autouse=True)
def _clear_prompt_cache():
    data._clear_prompt_cache()
    yield
    data._clear_prompt_cache()


def test_load_prompts_reads_file_once(tmp_path, monkeypatch, caplog):
    f = tmp_path / "prompts.txt"
    f.write_text("alpha\nbeta\n# comment\n\ngamma\n", encoding="utf-8")

    calls = {"n": 0}
    real_read = data._read_text_file

    def counting_read(path):
        calls["n"] += 1
        return real_read(path)

    monkeypatch.setattr(data, "_read_text_file", counting_read)

    with caplog.at_level(logging.INFO, logger="abliterate"):
        first = data.load_prompts(str(f), default_file="harmful.txt")
        second = data.load_prompts(str(f), default_file="harmful.txt")
        third = data.load_prompts(str(f), default_file="harmful.txt")

    assert first == second == third == ["alpha", "beta", "gamma"]
    # The file is read exactly once despite three calls...
    assert calls["n"] == 1
    # ...and the "loaded N prompts" line is emitted exactly once.
    loaded = [r for r in caplog.records if "loaded" in r.getMessage()]
    assert len(loaded) == 1


def test_load_prompts_returns_independent_copies(tmp_path):
    f = tmp_path / "p.txt"
    f.write_text("x\ny\n", encoding="utf-8")

    first = data.load_prompts(str(f), default_file="harmful.txt")
    first.append("mutated")  # a caller mutating its result must not poison the cache

    second = data.load_prompts(str(f), default_file="harmful.txt")
    assert second == ["x", "y"]


def test_load_prompts_distinct_sources_are_cached_separately(tmp_path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("one\n", encoding="utf-8")
    b.write_text("two\n", encoding="utf-8")

    assert data.load_prompts(str(a), default_file="harmful.txt") == ["one"]
    assert data.load_prompts(str(b), default_file="harmful.txt") == ["two"]
    # Re-reading the first still returns its own cached content.
    assert data.load_prompts(str(a), default_file="harmful.txt") == ["one"]
