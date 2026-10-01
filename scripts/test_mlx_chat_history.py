"""Check interactive prompt history and context-window rollover."""
from __future__ import annotations

from pathlib import Path
import sys

import tiktoken

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mlx_hobbylm.chat import serialize_prompt


def main() -> None:
    tokenizer = tiktoken.get_encoding("gpt2")
    history = [("What is Solana?", "It is a renewable-energy company.")]
    prompt = tokenizer.decode(
        serialize_prompt(
            "No, I mean the Web3 network.",
            system="Be concise.",
            history=history,
            tokenizer=tokenizer,
            context=1024,
        )
    )
    assert prompt == (
        "SYSTEM: Be concise.\n"
        "USER: What is Solana?\n"
        "ASSISTANT: It is a renewable-energy company.\n"
        "USER: No, I mean the Web3 network.\n"
        "ASSISTANT:"
    )
    short_prompt = tokenizer.decode(
        serialize_prompt(
            "New question",
            system="",
            history=history,
            tokenizer=tokenizer,
            context=15,
        )
    )
    assert short_prompt == "USER: New question\nASSISTANT:"
    print("PASS: interactive turns retain recent context and drop old turns")


if __name__ == "__main__":
    main()
