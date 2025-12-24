"""
Pydantic models for DeLP query results.

These models handle automatic normalization of Prolog values
(e.g., 'true'/'false' atoms to Python booleans).
"""

from typing import Any, Optional, List
from pydantic import BaseModel, field_validator, model_validator


def normalize_prolog_value(v: Any) -> Any:
    """Normalize Prolog atoms to Python types."""
    if isinstance(v, str):
        if v == 'true':
            return True
        elif v == 'false':
            return False
    elif isinstance(v, dict):
        return {k: normalize_prolog_value(val) for k, val in v.items()}
    elif isinstance(v, list):
        return [normalize_prolog_value(item) for item in v]
    return v


class PrologModel(BaseModel):
    """Base model with Prolog value normalization."""

    model_config = {"extra": "allow"}

    @model_validator(mode='before')
    @classmethod
    def normalize_all(cls, data: Any) -> Any:
        """Normalize all Prolog values before validation."""
        return normalize_prolog_value(data)


class Argument(PrologModel):
    """An argument in a dialectical tree."""
    goal: Optional[str] = None
    goal_args: Optional[List[Any]] = None
    rule_id: Optional[str] = None
    premises: Optional[List[Any]] = None
    specificity: Optional[float] = None


class DialecticalTree(PrologModel):
    """A dialectical tree node."""
    type: Optional[str] = None
    argument: Optional[Argument] = None
    status: Optional[str] = None
    defeaters: Optional[List['DialecticalTree']] = None
    goal: Optional[str] = None  # For 'no_arguments' type


class DeLPQueryResult(PrologModel):
    """Result of a DeLP query."""
    goal: str
    status: str
    tree: Optional[DialecticalTree] = None
    error: Optional[str] = None
    trace: Optional[List[Any]] = None

    @classmethod
    def from_prolog(cls, goal: str, result: Optional[dict]) -> 'DeLPQueryResult':
        """Create from raw Prolog query result."""
        if result:
            return cls(
                goal=goal,
                status=result.get('Status', 'undecided'),
                tree=result.get('TreeDict')
            )
        return cls(goal=goal, status='undecided', tree=None)


# =============================================================================
# DeLPProgram Models - for storing normalized program data
# =============================================================================

class DeLPRuleSource(PrologModel):
    """
    Normalized source rule data.

    Wraps the original AST Rule with normalized values.
    """
    head_variable: str
    head_value: Any  # auto-normalizes 'true' -> True
    conditions: List[Any] = []
    exceptions: List[Any] = []
    priority: str = "strict"
    override_target: Optional[str] = None

    @classmethod
    def from_ast(cls, rule) -> 'DeLPRuleSource':
        """Create from AST Rule object."""
        return cls(
            head_variable=rule.head_variable,
            head_value=rule.head_value,
            conditions=list(rule.conditions) if rule.conditions else [],
            exceptions=list(rule.exceptions) if rule.exceptions else [],
            priority=rule.priority.value if hasattr(rule.priority, 'value') else str(rule.priority),
            override_target=rule.override_target.value if rule.override_target and hasattr(rule.override_target, 'value') else None
        )


class DeLPRule(PrologModel):
    """A rule in the DeLP program with normalized values."""
    id: str
    head: str
    body: List[str] = []
    source: Optional[DeLPRuleSource] = None


class DeLPDeclaration(PrologModel):
    """A variable declaration with normalized values."""
    name: str
    description: Optional[str] = None
    values_from: Optional[List[Any]] = None  # auto-normalizes 'true'/'false'

    @classmethod
    def from_ast(cls, decl) -> 'DeLPDeclaration':
        """Create from AST VariableDeclaration object."""
        return cls(
            name=decl.name,
            description=decl.description,
            values_from=list(decl.values_from) if decl.values_from else None
        )
