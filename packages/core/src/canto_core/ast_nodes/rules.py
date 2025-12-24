"""
AST nodes for rules, conditions, and predicates
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union, Any
from enum import Enum


class RulePriority(Enum):
    """Rule priority levels"""
    STRICT = "strict"  # Regular rules (no normally keyword)
    NORMAL = "normal"  # Defeasible rules (with normally keyword)


class OverrideTarget(Enum):
    """Override targets"""
    ALL = "all"
    NORMAL = "normal"
    NONE = "none"


@dataclass
class Predicate:
    """
    Represents a symbolic predicate (e.g., matches($var1, $var2))
    These remain symbolic and are not evaluated during parsing
    """
    name: str  # Predicate name (e.g., "matches", "has")
    args: List[Any]  # Arguments (variables, values, etc.)

    def __repr__(self):
        args_str = ", ".join(str(arg) for arg in self.args)
        return f"{self.name}({args_str})"


@dataclass
class Condition:
    """
    Represents a condition in a rule
    Examples:
    - ?vaccine_flag is true
    - ?patient_query is like ?vaccine_terms
    - not ?vaccine_flag is true
    - not warranted ?query_intent is "prevention"
    """
    operator: str  # IS, IS_LIKE, HAS, AND, OR, NOT, NOT_WARRANTED
    left: Optional[Union[str, 'Condition']] = None  # Variable or nested condition
    right: Optional[Union[str, List[str], bool, 'Condition']] = None  # Value, pattern list, or nested condition

    def to_predicate(self) -> Predicate:
        """Convert condition to a symbolic predicate"""
        if self.operator == "NOT":
            # not condition (Negation as Failure)
            inner_pred = self.right.to_predicate() if isinstance(self.right, Condition) else self.right
            return Predicate("not", [inner_pred])
        elif self.operator == "NOT_WARRANTED":
            # not warranted condition (warrant-based negation)
            inner_pred = self.right.to_predicate() if isinstance(self.right, Condition) else self.right
            return Predicate("not_warranted", [inner_pred])
        elif self.operator in ["AND", "OR"]:
            # Logical operators
            left_pred = self.left.to_predicate() if isinstance(self.left, Condition) else self.left
            right_pred = self.right.to_predicate() if isinstance(self.right, Condition) else self.right
            return Predicate(self.operator.lower(), [left_pred, right_pred])
        elif self.operator == "IS_LIKE":
            return Predicate("is_like", [self.left, self.right])
        elif self.operator == "HAS":
            return Predicate("has", [self.left, self.right])
        elif self.operator == "IS":
            return Predicate("is", [self.left, self.right])
        else:
            return Predicate(self.operator.lower(), [self.left, self.right])

    def __repr__(self):
        if self.operator == "NOT":
            return f"not {self.right}"
        elif self.operator == "NOT_WARRANTED":
            return f"not warranted {self.right}"
        elif self.operator in ["AND", "OR"]:
            return f"({self.left} {self.operator.lower()} {self.right})"
        else:
            return f"{self.left} {self.operator.lower()} {self.right}"


@dataclass
class Rule:
    """
    Represents a rule in the DSL
    Example: ?vaccine_flag becomes true when ?patient_query is like ?vaccine_terms overriding all
    Example (qualified): ?claims_truth of ?claims becomes true when ...
    Example (nested): ?claims_truth of (?claims of ?puzzle) becomes true when ...
    """
    head_variable: str  # Variable being assigned (without ?)
    head_value: Union[str, bool, int, float]  # Value being assigned
    conditions: Optional[List[Condition]] = None  # when conditions
    exceptions: Optional[List[Condition]] = None  # unless conditions
    priority: RulePriority = RulePriority.STRICT
    override_target: OverrideTarget = OverrideTarget.NONE
    head_parent: Optional[Union[str, tuple]] = None  # Parent for qualified variables, can be nested tuple

    def __post_init__(self):
        # Ensure head_variable doesn't have ? prefix
        if self.head_variable.startswith('?'):
            self.head_variable = self.head_variable[1:]
        # Normalize head_parent (can be string or tuple for nested)
        self.head_parent = self._normalize_parent(self.head_parent)

        # Initialize empty lists if None
        if self.conditions is None:
            self.conditions = []
        if self.exceptions is None:
            self.exceptions = []

    def _normalize_parent(self, parent):
        """Normalize parent reference, handling both simple and nested cases."""
        if parent is None:
            return None
        if isinstance(parent, str):
            return parent.lstrip('?')
        if isinstance(parent, tuple) and len(parent) >= 2:
            # Nested: ('?claims', '?puzzle') or ('?claims', ('?inner', '?outer'))
            return (
                parent[0].lstrip('?') if isinstance(parent[0], str) else parent[0],
                self._normalize_parent(parent[1])
            )
        return parent

    def get_all_conditions(self) -> List[Condition]:
        """Get all conditions (both when and unless)"""
        return self.conditions + self.exceptions

    def is_strict(self) -> bool:
        """Check if this is a strict rule"""
        return self.priority == RulePriority.STRICT

    def is_defeasible(self) -> bool:
        """Check if this is a defeasible rule"""
        return self.priority == RulePriority.NORMAL

    def __repr__(self):
        priority_str = "normally " if self.is_defeasible() else ""
        conditions_str = ""
        if self.conditions:
            conditions_str = f" when {self.conditions}"
        exceptions_str = ""
        if self.exceptions:
            exceptions_str = f" unless {self.exceptions}"
        override_str = ""
        if self.override_target != OverrideTarget.NONE:
            override_str = f" overriding {self.override_target.value}"
        return f"Rule({priority_str}{self.head_variable} = {self.head_value}{conditions_str}{exceptions_str}{override_str})"
