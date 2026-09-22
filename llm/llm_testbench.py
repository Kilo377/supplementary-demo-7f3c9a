"""
File: llm_testbench.py

Description:
Minimal testbench for testing LLM generation pipeline.
It sends a simple prompt to APIManager and prints structured
debug output using print_prompt utility.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
ROOT_STR = str(ROOT_DIR)

if ROOT_STR not in sys.path:
    sys.path.insert(0, ROOT_STR)

from llm.api_manager import APIManager, SUPPORTED_PROVIDER_NAMES
from llm.print_prompt import print_prompt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal LLM provider testbench.")
    parser.add_argument("--provider", default="ollama", choices=SUPPORTED_PROVIDER_NAMES)
    parser.add_argument("--model", default=None)
    parser.add_argument("--prompt", default="请只回复：ok")
    return parser


def run_llm_test(*, provider_name: str, model: str | None, prompt: str) -> str:
    """
    Run a minimal LLM generation test.

    INPUT
      None
    OUTPUT
      None
    EXAMPLE OUTPUT
      === TASK: llm_test
      --- character ---------------------------------------------
      test_user

      --- provider / model --------------------------------------
       ollama / qwen2.5:1.5b

      --- prompts -------------------------------------------------
      你好啊

      --- output -------------------------------------------------
      你好！很高兴见到你。
    """

    api = APIManager(provider_name=provider_name)
    response = api.generate(prompt, model=model)

    print_prompt(
        task_name="llm_test",
        character_name="test_user",
        model_name=model or f"{api.provider_name}:default",
        prompt=prompt,
        output=response,
        extra_info={"provider": provider_name},
    )

    print("Final response:", response)
    return response


if __name__ == "__main__":
    args = build_parser().parse_args()
    print("==== LLM Testbench Start ====")
    run_llm_test(provider_name=args.provider, model=args.model, prompt=args.prompt)
    print("==== LLM Testbench End ====")
