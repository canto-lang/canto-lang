"""
Extracted Logic Data Structures.

These structures represent the logical content extracted from
natural language prompts. They mirror the Canto FOL structures
but are populated by LLM extraction rather than AST translation.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from .types import (
    FOLFormula,
    FOLPredicate,
    FOLEquals,
    FOLImplies,
    FOLNot,
    FOLVar,
    FOLConstant,
    FOLFunctionApp,
    FOLSort,
    make_and,
)


def _to_fol_constant(value: Any) -> FOLConstant:
    """
    Convert a value to an appropriate FOLConstant.

    Detects boolean values (Python bool or string "true"/"false")
    and creates FOLSort.BOOL constants for them.
    """
    # Already a Python bool
    if isinstance(value, bool):
        return FOLConstant(value, FOLSort.BOOL)

    # String representation of boolean
    if isinstance(value, str):
        lower = value.lower()
        if lower == "true":
            return FOLConstant(True, FOLSort.BOOL)
        elif lower == "false":
            return FOLConstant(False, FOLSort.BOOL)

    # Default: string value
    return FOLConstant(str(value), FOLSort.VALUE)


@dataclass
class ExtractedCondition:
    """
    A condition extracted from a prompt.

    Types:
    - "is_like": semantic similarity (args: [input_var, category])
    - "is": equality check (args: [variable, value])
    - "has": property check (args: [property])
    - "not": negation (args: [inner_type, ...inner_args])
    """
    type: str
    args: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type, "args": self.args}

    def to_fol(self, x: FOLVar) -> Optional[FOLFormula]:
        """Convert to FOL formula."""
        if self.type == "is_like" and len(self.args) >= 1:
            category = self.args[-1]
            return FOLPredicate(
                "is_like",
                [x, FOLConstant(category, FOLSort.CATEGORY)]
            )

        elif self.type == "is" and len(self.args) >= 2:
            var_name = self.args[0]
            value = self.args[1]
            var_app = FOLFunctionApp(var_name, [x])
            return FOLEquals(
                var_app,
                _to_fol_constant(value)
            )

        elif self.type == "has" and len(self.args) >= 1:
            prop = self.args[0]
            return FOLPredicate(
                "has",
                [x, FOLConstant(prop, FOLSort.VALUE)]
            )

        elif self.type == "not" and len(self.args) >= 1:
            inner_type = self.args[0] if self.args else ""
            inner_args = self.args[1:] if len(self.args) > 1 else []
            inner = ExtractedCondition(type=inner_type, args=inner_args).to_fol(x)
            if inner:
                return FOLNot(inner)

        return None


@dataclass
class ExtractedRule:
    """
    A rule extracted from a prompt.

    Represents: variable = value when conditions
    """
    variable: str
    value: Any
    conditions: List[ExtractedCondition] = field(default_factory=list)
    is_default: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "variable": self.variable,
            "value": self.value,
            "conditions": [c.to_dict() for c in self.conditions],
            "is_default": self.is_default
        }

    def to_fol(self, x: FOLVar) -> FOLFormula:
        """Convert to FOL formula."""
        # Build conditions
        cond_formulas = []
        for cond in self.conditions:
            fol_cond = cond.to_fol(x)
            if fol_cond:
                cond_formulas.append(fol_cond)

        # Build conclusion - use helper to detect boolean values
        var_app = FOLFunctionApp(self.variable, [x])
        conclusion = FOLEquals(
            var_app,
            _to_fol_constant(self.value)
        )

        # Build rule formula
        if cond_formulas:
            condition = make_and(cond_formulas)
            return FOLImplies(condition, conclusion)
        else:
            return conclusion


@dataclass
class ExtractedPrecedence:
    """
    A precedence relationship extracted from a prompt.

    Represents: higher_value takes precedence over lower_value for variable
    """
    higher_value: str
    lower_value: str
    variable: str
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "higher_value": self.higher_value,
            "lower_value": self.lower_value,
            "variable": self.variable,
            "reason": self.reason
        }


@dataclass
class ExtractedLogic:
    """
    Complete logical structure extracted from a prompt.

    This is the Prompt_FOL representation that gets compared
    against the DSL_FOL for verification.
    """
    rules: List[ExtractedRule] = field(default_factory=list)
    precedence: List[ExtractedPrecedence] = field(default_factory=list)
    mutual_exclusivity: List[Dict[str, Any]] = field(default_factory=list)
    mentioned_variables: List[str] = field(default_factory=list)
    mentioned_categories: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rules": [r.to_dict() for r in self.rules],
            "precedence": [p.to_dict() for p in self.precedence],
            "mutual_exclusivity": self.mutual_exclusivity,
            "mentioned_variables": self.mentioned_variables,
            "mentioned_categories": self.mentioned_categories
        }

    def to_fol_formulas(self) -> List[FOLFormula]:
        """Convert all extracted rules to FOL formulas."""
        x = FOLVar("x")
        return [rule.to_fol(x) for rule in self.rules]

    def get_variables_with_rules(self) -> List[str]:
        """Get list of variables that have extracted rules."""
        return list(set(r.variable for r in self.rules))

    def get_values_for_variable(self, var_name: str) -> List[Any]:
        """Get all values mentioned for a variable."""
        return [r.value for r in self.rules if r.variable == var_name]

    def __repr__(self):
        return (
            f"ExtractedLogic(rules={len(self.rules)}, "
            f"precedence={len(self.precedence)}, "
            f"vars={self.mentioned_variables})"
        )
