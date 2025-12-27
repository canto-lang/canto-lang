"""
Build food ontology program with concepts injected.

Usage:
    python build.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from canto_core import CantoBuilder, PromptGenerator, configure_dspy
from concepts import (
    food_item,
    drying_terms,
    freezing_terms,
    canning_terms,
    cherry_variety,
    organic_terms,
    dop_terms,
)

load_dotenv()

model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
configure_dspy(model)


def build():
    """Build the food ontology program with concepts injected."""
    builder = CantoBuilder()

    builder.register_concept(food_item)
    builder.register_concept(drying_terms)
    builder.register_concept(freezing_terms)
    builder.register_concept(canning_terms)
    builder.register_concept(cherry_variety)
    builder.register_concept(organic_terms)
    builder.register_concept(dop_terms)

    canto_file = Path(__file__).parent / "ontology.canto"
    return builder.build(str(canto_file))


def generate_prompt():
    """Generate prompt from the built program."""
    result = build()
    generator = PromptGenerator(result)
    return generator.generate()


if __name__ == "__main__":
    print("=" * 60)
    print("CONCEPTS EXAMPLE: Food Ontology")
    print("=" * 60)

    prompt = generate_prompt()

    print("\n--- Generated Prompt ---")
    print(prompt)
