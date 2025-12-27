"""
Build medical triage program with concepts injected.

Usage:
    python build.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from canto_core import CantoBuilder, PromptGenerator, configure_dspy
from concepts import patient, triage_level, emergency_symptoms, diagnosis

load_dotenv()

model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
configure_dspy(model)


def build():
    """Build the triage program with concepts injected."""
    builder = CantoBuilder()

    builder.register_concept(patient)
    builder.register_concept(triage_level)
    builder.register_concept(emergency_symptoms)
    builder.register_concept(diagnosis)

    canto_file = Path(__file__).parent / "triage.canto"
    return builder.build(str(canto_file))


def generate_prompt():
    """Generate prompt from the built program."""
    result = build()
    generator = PromptGenerator(result)
    return generator.generate()


if __name__ == "__main__":
    print("=" * 60)
    print("CONCEPTS EXAMPLE: Medical Triage")
    print("=" * 60)

    # Build and generate prompt
    prompt = generate_prompt()

    print("\n--- Generated Prompt ---")
    print(prompt)
