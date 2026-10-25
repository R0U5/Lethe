"""Prompt formatting helpers.

Instruction models expect their chat template; base models usually have none.
``format_prompt`` applies the tokenizer's chat template when present and falls
back to the raw string otherwise, so the same pipeline works for both.
"""

from __future__ import annotations

from typing import Optional


def _supports_thinking_toggle(tokenizer) -> bool:
    """True when the chat template accepts an ``enable_thinking`` kwarg."""
    template = getattr(tokenizer, "chat_template", None) or ""
    return "enable_thinking" in template


def format_prompt(
    tokenizer,
    instruction: str,
    system: Optional[str] = None,
    *,
    disable_thinking: bool = True,
) -> str:
    """Render a single instruction into the model's expected input string.

    Reasoning models open a thinking block in the assistant turn. Completions
    are then dominated by deliberation, so a small ``max_new_tokens`` truncates
    before the model states its actual answer. Refusal scoring would then read
    the deliberation instead of the answer -- or, worse, an empty string that
    looks compliant -- and the optimizer would happily settle on a near-no-op
    ablation. Turning thinking off for measurement makes completions state the
    answer directly and comparably across models.
    """
    if getattr(tokenizer, "chat_template", None):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": instruction})
        kwargs: dict[str, object] = {}
        if disable_thinking and _supports_thinking_toggle(tokenizer):
            kwargs["enable_thinking"] = False
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **kwargs
            )
        except TypeError:
            # Template does not accept the kwarg despite advertising it.
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
    # No chat template (base model): use the instruction as-is.
    return instruction


def format_batch(
    tokenizer,
    instructions,
    system: Optional[str] = None,
    *,
    disable_thinking: bool = True,
) -> list[str]:
    return [
        format_prompt(tokenizer, ins, system=system, disable_thinking=disable_thinking)
        for ins in instructions
    ]
