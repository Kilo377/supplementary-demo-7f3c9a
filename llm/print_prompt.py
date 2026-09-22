"""
File: print_prompt.py
Description:
Utility function for printing LLM plan_prompts and outputs when debug mode is enabled.
This module is used for inspecting prompts construction and model responses
during development. It should not affect core logic.
"""
from __future__ import annotations

import os


DEBUG = os.getenv("LLM_DEBUG", "").lower() in {"1", "true", "yes", "on"}


def print_prompt(task_name=None,
                 character_name=None,
                 model_name=None,
                 prompt=None,
                 output=None,
                 extra_info=None):

    if not DEBUG:
        return

    print(f"=== TASK: {task_name}")
    print("--- character ---------------------------------------------")
    print(character_name, "\n")

    print("--- provider / model --------------------------------------")
    print(f" {model_name}", "\n")

    if extra_info:
        print("--- extra_info --------------------------------------------")
        print(extra_info, "\n")

    print("--- prompts -------------------------------------------------")
    print(prompt, "\n")

    print("--- output -------------------------------------------------")
    print(output, "\n")

    print("=== END ====================================================")
    print("\n\n")
