"""Prompt formatting helpers.

Instruction models expect their chat template; base models usually have none.
``format_prompt`` applies the tokenizer's chat template when present and falls
back to the raw string otherwise, so the same pipeline works for both.
"""

from __future__ import annotations

from typing import Optional


def format_prompt(tokenizer, instruction: str, system: Optional[str] = None) -> str:
    """Render a single instruction into the model's expected input string."""
    if getattr(tokenizer, "chat_template", None):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": instruction})
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    # No chat template (base model): use the instruction as-is.
    return instruction


def format_batch(tokenizer, instructions, system: Optional[str] = None) -> list[str]:
    return [format_prompt(tokenizer, ins, system) for ins in instructions]
