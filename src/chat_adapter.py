"""
chat_adapter.py

Family-aware layer sitting directly on top of inference_engine. Given an
engine and a list of chat messages, it applies the correct chat template
for the model's `family` and splits the raw output into (final_text,
reasoning).

Everything above this layer operates on messages and clean text. Harmony
channels and <think> blocks never leak upward.

Scope is local HF and vLLM backends (families qwen, deepseek, gemma,
gptoss, llama, ...). The API backends (openai, anthropic) follow a
different contract and are wrapped separately.

Note on gpt-oss. Channel splitting needs the harmony markers to survive
decoding, so the engine must decode that family with
skip_special_tokens=False. If the markers are stripped, _strip_harmony
degrades gracefully, returning the whole text as the answer and empty
reasoning rather than crashing.
"""


from __future__ import annotations

import re
from typing import Any

Message = dict[str, str]


# ----------------------------------------------------------------------
# templating helpers
# ----------------------------------------------------------------------

def _fold_system(messages: list[Message]) -> list[Message]:
    """Merge a leading system turn into the first user turn.

    Used when a chat template rejects a separate system role (e.g. Gemma).
    """
    sys = messages[0]["content"]
    out: list[Message] = []
    injected = False
    for m in messages[1:]:
        if not injected and m["role"] == "user":
            out.append({"role": "user", "content": f"{sys}\n\n{m['content']}"})
            injected = True
        else:
            out.append(m)
    return out if injected else [{"role": "user", "content": sys}] + list(messages[1:])


# ----------------------------------------------------------------------
# output-parsing helpers
# ----------------------------------------------------------------------

def _strip_harmony(text: str) -> tuple[str, str]:
    """gpt-oss: reasoning in the analysis channel, answer in the final channel."""
    final_marker = "<|channel|>final<|message|>"
    final = text.split(final_marker, 1)[1] if final_marker in text else text
    # take the analysis channel content up to the next control token
    am = re.search(
        r"<\|channel\|>analysis<\|message\|>(.*?)(?:<\|end\|>|<\|start\|>|<\|channel\|>|$)",
        text,
        flags=re.DOTALL,
    )
    reasoning = am.group(1) if am else ""
    clean = lambda s: re.sub(r"<\|[^|]*\|>", " ", s).strip()
    return clean(final), clean(reasoning)


def _strip_think(text: str) -> tuple[str, str]:
    """qwen / deepseek reasoning models: CoT wrapped in <think>...</think>."""
    m = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    if m:
        return text[m.end():].strip(), m.group(1).strip()
    if "</think>" in text:                       # opening tag consumed by the template
        reasoning, _, final = text.partition("</think>")
        return final.strip(), reasoning.replace("<think>", "").strip()
    return text.strip(), ""


# ----------------------------------------------------------------------
# adapter
# ----------------------------------------------------------------------

class ChatAdapter:
    """Wraps a BaseInferenceEngine and speaks chat messages."""

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.tokenizer = engine.tokenizer
        cfg = getattr(engine, "model_cfg", {}) or {}
        self.family = str(cfg.get("family", "")).strip().lower()
        self.reasoning_effort = cfg.get("reasoning_effort")
        self.enable_thinking = bool(cfg.get("enable_thinking", False))

    def _template(self, messages: list[Message]) -> str:
        extra: dict[str, Any] = {}
        if self.family == "qwen":
            extra["enable_thinking"] = self.enable_thinking
        elif self.family == "gptoss" and self.reasoning_effort:
            extra["reasoning_effort"] = self.reasoning_effort
        kwargs = dict(tokenize=False, add_generation_prompt=True, **extra)
        try:
            return self.tokenizer.apply_chat_template(messages, **kwargs)
        except Exception:
            if messages and messages[0]["role"] == "system":
                return self.tokenizer.apply_chat_template(_fold_system(messages), **kwargs)
            raise

    def _split(self, text: str) -> tuple[str, str]:
        if self.family == "gptoss":
            return _strip_harmony(text)
        if self.family in {"qwen", "deepseek"}:
            return _strip_think(text)
        return text.strip(), ""

    def chat(self, messages: list[Message]) -> tuple[str, str]:
        """Generate one turn. Returns (final_text, reasoning)."""
        prompt = self._template(messages)
        out = self.engine.generate([prompt])
        raw = out[0] if out else ""
        return self._split(raw)