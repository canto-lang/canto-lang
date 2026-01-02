"""
LLM-Based Prompt Logic Extractor.

Uses an LLM to extract logical structure from natural language prompts.
This is the uncertain step in verification - Z3 catches any discrepancies.
"""

import json
from typing import Dict, Any

import dspy

from .extracted_logic import (
    ExtractedLogic,
    ExtractedRule,
    ExtractedCondition,
    ExtractedPrecedence,
)
from .canto_fol import CantoFOL


class ExtractPromptLogicSignature(dspy.Signature):
    """
    Extract the logical rules encoded in a natural language prompt.

    Given a prompt that instructs an LLM how to classify/analyze input,
    extract the implicit logical rules as structured data.

    For each rule, identify:
    - What variable is being set
    - What value it's being set to
    - Under what conditions
    - Any precedence/override relationships
    """

    prompt_text = dspy.InputField(
        desc="The natural language prompt template to analyze"
    )

    known_variables = dspy.InputField(
        desc="JSON object mapping variable names to their types. Boolean variables can ONLY have values true or false."
    )

    known_categories = dspy.InputField(
        desc="JSON dict of semantic category names to example patterns"
    )

    extracted_logic = dspy.OutputField(
        desc='''JSON object with extracted logical structure:
{
    "rules": [
        {
            "variable": "variable_name",
            "value": "the_value",
            "conditions": [
                {"type": "is_like", "args": ["category_name"]},
                {"type": "is", "args": ["other_var", "value"]},
                {"type": "has", "args": ["property_name"]}
            ],
            "is_default": false
        }
    ],
    "precedence": [
        {
            "higher_value": "val1",
            "lower_value": "val2",
            "variable": "var",
            "reason": "explanation"
        }
    ],
    "mutual_exclusivity": [
        {"variable": "var_name", "values": ["val1", "val2"]}
    ],
    "mentioned_variables": ["var1", "var2"],
    "mentioned_categories": ["cat1", "cat2"]
}

CRITICAL RULES:

1. ONLY USE PROVIDED VARIABLES:
   - You MUST only use variable names from known_variables
   - NEVER invent new variables like "subject_is_truthful" or "target_person_equals_base_person"
   - If the prompt describes a concept not in known_variables, DO NOT create a rule for it

2. TYPE AND VALUE RULES:
   - For boolean variables (type: "bool"), values MUST be exactly true or false (JSON booleans)
   - NEVER use null, None, or any other value for boolean variables - ONLY true or false
   - For string variables with "values" field, you MUST use one of the listed values exactly
   - Example: if answer has values ["Yes", "No"], use "Yes" or "No", NOT true/false
   - In conditions, use {"type": "is", "args": ["var_name", value]} where var_name MUST be in known_variables

Example for known_variables = {"is_valid": {"type": "bool"}, "answer": {"type": "string", "values": ["Yes", "No"]}}:
  CORRECT: {"variable": "answer", "value": "Yes", "conditions": [{"type": "is", "args": ["is_valid", true]}]}
  WRONG: {"variable": "answer", "value": true, ...}  // answer is string with values ["Yes", "No"], not boolean!
  WRONG: {"variable": "is_valid", "value": "True", ...}  // is_valid is bool, use true not "True"
  WRONG: {"variable": "is_valid", "value": null, ...}  // NEVER use null/None for booleans!

Extract ALL rules mentioned in the prompt:
- Explicit: "Set X to Y when..."
- Implicit: "X is Y if..."
- Default: "By default, X is Y"
- Override: "Even if A, still B" or "A takes precedence"
'''
    )


class PromptLogicExtractor:
    """
    Extracts logical structure from prompts using LLM.

    The extraction may not be perfect - this is inherent to
    parsing natural language. Z3 verification catches discrepancies
    between extracted logic and the source DSL.
    """

    def __init__(self):
        self._extractor = dspy.ChainOfThought(ExtractPromptLogicSignature)

    def extract(self, prompt: str, fol: CantoFOL) -> ExtractedLogic:
        """
        Extract logical structure from prompt.

        Args:
            prompt: The natural language prompt
            fol: The source FOL IR (provides context about known variables/categories)

        Returns:
            ExtractedLogic with rules, precedence, etc.
        """
        # Prepare context from FOL - include variable types and possible values
        known_vars = {}
        for name, var in fol.variables.items():
            if var.is_bool():
                known_vars[name] = {"type": "bool"}
            elif var.possible_values:
                known_vars[name] = {"type": "string", "values": var.possible_values}
            else:
                known_vars[name] = {"type": "string"}
        known_cats = {
            name: cat.patterns[:5]
            for name, cat in fol.categories.items()
        }

        # Call LLM
        result = self._extractor(
            prompt_text=prompt,
            known_variables=json.dumps(known_vars),
            known_categories=json.dumps(known_cats)
        )

        # Parse result
        data = json.loads(result.extracted_logic)
        return self._parse_result(data)

    def _parse_result(self, data: Dict[str, Any]) -> ExtractedLogic:
        """Parse LLM JSON output into ExtractedLogic."""
        rules = []
        for r in data.get("rules", []):
            conditions = [
                ExtractedCondition(
                    type=c.get("type", ""),
                    args=c.get("args", [])
                )
                for c in r.get("conditions", [])
            ]
            rules.append(ExtractedRule(
                variable=r.get("variable", ""),
                value=r.get("value", ""),
                conditions=conditions,
                is_default=r.get("is_default", False)
            ))

        precedence = [
            ExtractedPrecedence(
                higher_value=p.get("higher_value", ""),
                lower_value=p.get("lower_value", ""),
                variable=p.get("variable", ""),
                reason=p.get("reason", "")
            )
            for p in data.get("precedence", [])
        ]

        return ExtractedLogic(
            rules=rules,
            precedence=precedence,
            mutual_exclusivity=data.get("mutual_exclusivity", []),
            mentioned_variables=data.get("mentioned_variables", []),
            mentioned_categories=data.get("mentioned_categories", [])
        )
