"""
First-Order Logic Types for Canto.

This module defines the base FOL types that serve as the foundation
for the canonical intermediate representation. These types are
backend-agnostic and can be encoded to Z3, translated to Prolog, etc.
"""

from dataclasses import dataclass, field
from typing import List, Set, Any, Union, Dict
from enum import Enum
from abc import ABC, abstractmethod


class FOLSort(Enum):
    """
    Sorts (types) in our FOL system.

    These correspond to the semantic domains in Canto:
    - INPUT: The user input domain (abstract)
    - VALUE: Variable values (strings, bools, numbers)
    - CATEGORY: Semantic category names
    - BOOL: Boolean values
    """
    INPUT = "input"
    VALUE = "value"
    CATEGORY = "category"
    BOOL = "bool"


@dataclass(frozen=True)
class FOLVariable:
    """
    A logical variable in FOL (not a Canto DSL variable).

    This represents a bound or free variable in quantified formulas.
    For example, in ∀x. P(x), 'x' is a FOLVariable.
    """
    name: str
    sort: FOLSort

    def __repr__(self):
        return f"{self.name}:{self.sort.value}"


# =============================================================================
# Terms
# =============================================================================

class FOLTerm(ABC):
    """
    Base class for FOL terms.

    Terms are expressions that denote objects in the domain:
    - Constants: specific values like 'true', 'vaccine_terms'
    - Variables: logical variables like x, y
    - Function applications: f(t1, t2, ...)
    """

    @abstractmethod
    def free_variables(self) -> Set[str]:
        """Return set of free variable names in this term."""
        pass

    @abstractmethod
    def substitute(self, var_name: str, replacement: 'FOLTerm') -> 'FOLTerm':
        """Substitute a variable with a term."""
        pass


@dataclass(frozen=True)
class FOLConstant(FOLTerm):
    """
    A constant value in FOL.

    Examples:
    - FOLConstant('true', FOLSort.VALUE)
    - FOLConstant('vaccine_terms', FOLSort.CATEGORY)
    """
    value: Any
    sort: FOLSort

    def free_variables(self) -> Set[str]:
        return set()

    def substitute(self, var_name: str, replacement: FOLTerm) -> FOLTerm:
        return self  # Constants don't contain variables

    def __repr__(self):
        if self.sort == FOLSort.VALUE:
            if isinstance(self.value, str):
                return f"'{self.value}'"
            return str(self.value)
        elif self.sort == FOLSort.CATEGORY:
            return f"@{self.value}"
        return str(self.value)


@dataclass(frozen=True)
class FOLVar(FOLTerm):
    """
    A logical variable reference in FOL.

    This is a reference to a variable bound by a quantifier,
    or a free variable in an open formula.
    """
    name: str

    def free_variables(self) -> Set[str]:
        return {self.name}

    def substitute(self, var_name: str, replacement: FOLTerm) -> FOLTerm:
        if self.name == var_name:
            return replacement
        return self

    def __repr__(self):
        return self.name


@dataclass(frozen=True)
class FOLFunctionApp(FOLTerm):
    """
    Function application: f(t1, t2, ...).

    In Canto, this represents variable value access:
    - vaccine_flag(x) means "the value of vaccine_flag for input x"
    """
    function: str
    args: tuple  # Using tuple for hashability

    def __init__(self, function: str, args: List[FOLTerm]):
        object.__setattr__(self, 'function', function)
        object.__setattr__(self, 'args', tuple(args))

    def free_variables(self) -> Set[str]:
        result = set()
        for arg in self.args:
            result |= arg.free_variables()
        return result

    def substitute(self, var_name: str, replacement: FOLTerm) -> FOLTerm:
        new_args = [arg.substitute(var_name, replacement) for arg in self.args]
        return FOLFunctionApp(self.function, new_args)

    def __repr__(self):
        args_str = ", ".join(str(a) for a in self.args)
        return f"{self.function}({args_str})"


# =============================================================================
# Formulas
# =============================================================================

class FOLFormula(ABC):
    """
    Base class for FOL formulas.

    Formulas are expressions that denote truth values:
    - Predicates: P(t1, t2, ...)
    - Equality: t1 = t2
    - Logical connectives: ¬, ∧, ∨, →
    - Quantifiers: ∀, ∃
    """

    @abstractmethod
    def free_variables(self) -> Set[str]:
        """Return set of free variable names in this formula."""
        pass

    @abstractmethod
    def substitute(self, var_name: str, replacement: FOLTerm) -> 'FOLFormula':
        """Substitute a variable with a term."""
        pass


@dataclass
class FOLPredicate(FOLFormula):
    """
    Atomic predicate: P(t1, t2, ...).

    Examples:
    - is_like(x, @vaccine_terms)
    - has(x, 'fever')
    - fires('rule_1')

    Attributes:
        name: Predicate name
        args: List of FOL terms as arguments
        metadata: Optional dict for backend-specific info (e.g., source_var for Prolog)
    """
    name: str
    args: List[FOLTerm] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def free_variables(self) -> Set[str]:
        result = set()
        for arg in self.args:
            result |= arg.free_variables()
        return result

    def substitute(self, var_name: str, replacement: FOLTerm) -> FOLFormula:
        new_args = [arg.substitute(var_name, replacement) for arg in self.args]
        return FOLPredicate(self.name, new_args, self.metadata)

    def __repr__(self):
        if not self.args:
            return self.name
        args_str = ", ".join(str(a) for a in self.args)
        return f"{self.name}({args_str})"


@dataclass
class FOLEquals(FOLFormula):
    """
    Equality formula: t1 = t2.

    Example: vaccine_flag(x) = 'true'
    """
    left: FOLTerm
    right: FOLTerm

    def free_variables(self) -> Set[str]:
        return self.left.free_variables() | self.right.free_variables()

    def substitute(self, var_name: str, replacement: FOLTerm) -> FOLFormula:
        return FOLEquals(
            self.left.substitute(var_name, replacement),
            self.right.substitute(var_name, replacement)
        )

    def __repr__(self):
        return f"{self.left} = {self.right}"


@dataclass
class FOLNot(FOLFormula):
    """
    Negation: ¬φ.
    """
    formula: FOLFormula

    def free_variables(self) -> Set[str]:
        return self.formula.free_variables()

    def substitute(self, var_name: str, replacement: FOLTerm) -> FOLFormula:
        return FOLNot(self.formula.substitute(var_name, replacement))

    def __repr__(self):
        return f"¬({self.formula})"


@dataclass
class FOLAnd(FOLFormula):
    """
    Conjunction: φ1 ∧ φ2 ∧ ... ∧ φn.

    Supports n-ary conjunction for convenience.
    """
    conjuncts: List[FOLFormula] = field(default_factory=list)

    def free_variables(self) -> Set[str]:
        result = set()
        for c in self.conjuncts:
            result |= c.free_variables()
        return result

    def substitute(self, var_name: str, replacement: FOLTerm) -> FOLFormula:
        return FOLAnd([c.substitute(var_name, replacement) for c in self.conjuncts])

    def __repr__(self):
        if not self.conjuncts:
            return "⊤"  # True
        if len(self.conjuncts) == 1:
            return str(self.conjuncts[0])
        return " ∧ ".join(f"({c})" for c in self.conjuncts)


@dataclass
class FOLOr(FOLFormula):
    """
    Disjunction: φ1 ∨ φ2 ∨ ... ∨ φn.

    Supports n-ary disjunction for convenience.
    """
    disjuncts: List[FOLFormula] = field(default_factory=list)

    def free_variables(self) -> Set[str]:
        result = set()
        for d in self.disjuncts:
            result |= d.free_variables()
        return result

    def substitute(self, var_name: str, replacement: FOLTerm) -> FOLFormula:
        return FOLOr([d.substitute(var_name, replacement) for d in self.disjuncts])

    def __repr__(self):
        if not self.disjuncts:
            return "⊥"  # False
        if len(self.disjuncts) == 1:
            return str(self.disjuncts[0])
        return " ∨ ".join(f"({d})" for d in self.disjuncts)


@dataclass
class FOLImplies(FOLFormula):
    """
    Implication: φ → ψ.
    """
    antecedent: FOLFormula
    consequent: FOLFormula

    def free_variables(self) -> Set[str]:
        return self.antecedent.free_variables() | self.consequent.free_variables()

    def substitute(self, var_name: str, replacement: FOLTerm) -> FOLFormula:
        return FOLImplies(
            self.antecedent.substitute(var_name, replacement),
            self.consequent.substitute(var_name, replacement)
        )

    def __repr__(self):
        return f"({self.antecedent}) → ({self.consequent})"


@dataclass
class FOLForall(FOLFormula):
    """
    Universal quantification: ∀x. φ.
    """
    variable: FOLVariable
    formula: FOLFormula

    def free_variables(self) -> Set[str]:
        return self.formula.free_variables() - {self.variable.name}

    def substitute(self, var_name: str, replacement: FOLTerm) -> FOLFormula:
        if var_name == self.variable.name:
            # Variable is bound, don't substitute in body
            return self
        # Check for variable capture
        if self.variable.name in replacement.free_variables():
            # Need to rename bound variable to avoid capture
            # For simplicity, we'll raise an error; proper implementation
            # would do alpha-conversion
            raise ValueError(f"Variable capture: {self.variable.name}")
        return FOLForall(
            self.variable,
            self.formula.substitute(var_name, replacement)
        )

    def __repr__(self):
        return f"∀{self.variable.name}. ({self.formula})"


@dataclass
class FOLExists(FOLFormula):
    """
    Existential quantification: ∃x. φ.
    """
    variable: FOLVariable
    formula: FOLFormula

    def free_variables(self) -> Set[str]:
        return self.formula.free_variables() - {self.variable.name}

    def substitute(self, var_name: str, replacement: FOLTerm) -> FOLFormula:
        if var_name == self.variable.name:
            return self
        if self.variable.name in replacement.free_variables():
            raise ValueError(f"Variable capture: {self.variable.name}")
        return FOLExists(
            self.variable,
            self.formula.substitute(var_name, replacement)
        )

    def __repr__(self):
        return f"∃{self.variable.name}. ({self.formula})"


# =============================================================================
# Utility Functions
# =============================================================================

def make_and(formulas: List[FOLFormula]) -> FOLFormula:
    """Create conjunction, handling empty and singleton cases."""
    if not formulas:
        return FOLPredicate("true", [])  # ⊤
    if len(formulas) == 1:
        return formulas[0]
    return FOLAnd(formulas)


def make_or(formulas: List[FOLFormula]) -> FOLFormula:
    """Create disjunction, handling empty and singleton cases."""
    if not formulas:
        return FOLPredicate("false", [])  # ⊥
    if len(formulas) == 1:
        return formulas[0]
    return FOLOr(formulas)


def make_implies(antecedent: FOLFormula, consequent: FOLFormula) -> FOLFormula:
    """Create implication."""
    return FOLImplies(antecedent, consequent)


def make_forall(var_name: str, sort: FOLSort, formula: FOLFormula) -> FOLFormula:
    """Create universal quantification."""
    return FOLForall(FOLVariable(var_name, sort), formula)


def make_exists(var_name: str, sort: FOLSort, formula: FOLFormula) -> FOLFormula:
    """Create existential quantification."""
    return FOLExists(FOLVariable(var_name, sort), formula)
