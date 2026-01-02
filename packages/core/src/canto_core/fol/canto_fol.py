"""
Canto-specific FOL structures.

This module defines the canonical intermediate representation (IR) for
Canto programs using FOL. All downstream transformations (Prolog, ASP,
verification) start from this representation.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from enum import Enum

from .types import (
    FOLFormula,
    FOLPredicate,
    FOLEquals,
    FOLImplies,
    FOLAnd,
    FOLOr,
    FOLNot,
    FOLForall,
    FOLVar,
    FOLConstant,
    FOLFunctionApp,
    FOLSort,
    FOLVariable,
    make_and,
)


class RuleType(Enum):
    """Type of rule in Canto/DeLP."""
    STRICT = "strict"
    DEFEASIBLE = "defeasible"


class ValueType(Enum):
    """Type of values a Canto variable can hold."""
    STRING = "string"
    BOOL = "bool"


@dataclass
class CantoVariable:
    """
    A Canto variable declaration in FOL representation.

    This represents a variable that can take values, optionally
    with a constrained domain (can be "a", "b", "c").
    """
    name: str
    value_type: ValueType = ValueType.STRING  # Explicit type
    description: Optional[str] = None
    possible_values: List[Any] = field(default_factory=list)
    source: Optional[str] = None  # For extraction: 'from ?source'

    def to_fol_function_name(self) -> str:
        """Returns the FOL function name for this variable."""
        return self.name

    def is_bool(self) -> bool:
        """Check if this variable is boolean."""
        return self.value_type == ValueType.BOOL

    @property
    def values_from(self) -> Optional[List[Any]]:
        """Alias for possible_values for DeLPDeclaration compatibility."""
        return self.possible_values if self.possible_values else None

    def __repr__(self):
        type_str = "bool" if self.is_bool() else "string"
        if self.possible_values:
            return f"CantoVariable({self.name}:{type_str}, values={self.possible_values})"
        return f"CantoVariable({self.name}:{type_str})"


@dataclass
class CantoCategory:
    """
    A semantic category in FOL representation.

    Semantic categories define prototype-based pattern matching,
    e.g., "vaccine_terms" resembles ["vaccine", "vaccination", "shot"].
    """
    name: str
    patterns: List[str]
    description: Optional[str] = None

    def to_fol_predicates(self) -> List[FOLFormula]:
        """
        Convert to FOL pattern predicates.

        Generates: pattern(category_name, example) for each example.
        """
        predicates = []
        for pattern in self.patterns:
            predicates.append(
                FOLPredicate(
                    "pattern",
                    [
                        FOLConstant(self.name, FOLSort.CATEGORY),
                        FOLConstant(pattern, FOLSort.VALUE)
                    ]
                )
            )
        return predicates

    def __repr__(self):
        return f"CantoCategory({self.name}, {len(self.patterns)} patterns)"


@dataclass
class CantoRule:
    """
    A Canto rule in FOL form.

    Represents both strict and defeasible rules with their conditions.
    """
    id: str
    head_variable: str
    head_value: Any
    conditions: Optional[FOLFormula]  # The 'when' clause as FOL (None = unconditional)
    rule_type: RuleType
    original_ast: Any = None  # Reference to original AST node for metadata

    @property
    def body(self) -> List[str]:
        """
        Get the rule body as a list of Prolog predicate strings.

        This provides compatibility with the DeLPRule interface.
        """
        if not self.conditions:
            return []

        from .backends.prolog import PrologBackend
        # Create a minimal backend just for formula translation
        backend = PrologBackend.__new__(PrologBackend)
        return backend._get_body_predicates(self.conditions)

    @property
    def head(self) -> str:
        """
        Get the rule head as a Prolog string.

        This provides compatibility with the DeLPRule interface.
        """
        from .backends.prolog import normalize_prolog_atom
        value = self.head_value
        if isinstance(value, bool):
            formatted = str(value).lower()
        elif isinstance(value, (int, float)):
            formatted = str(value)
        elif isinstance(value, str):
            escaped = value.replace("\\", "\\\\").replace("'", "\\'")
            formatted = f"'{escaped}'"
        else:
            formatted = normalize_prolog_atom(value)
        return f"{self.head_variable}({formatted})"

    @property
    def source(self) -> Optional[Any]:
        """
        Get the source AST rule.

        This provides compatibility with the DeLPRule interface.
        Returns the original_ast wrapped in a compatible format.
        """
        if self.original_ast:
            # Return a simple namespace with the expected attributes
            class RuleSource:
                def __init__(self, ast):
                    self.head_variable = ast.head_variable
                    self.head_value = ast.head_value
                    self.conditions = list(ast.conditions) if ast.conditions else []
                    self.exceptions = list(ast.exceptions) if ast.exceptions else []
                    self.priority = ast.priority.value if hasattr(ast.priority, 'value') else str(ast.priority)
                    self.override_target = ast.override_target.value if ast.override_target and hasattr(ast.override_target, 'value') else None
            return RuleSource(self.original_ast)
        return None

    def to_fol(self, input_var: FOLVariable) -> FOLFormula:
        """
        Convert rule to FOL formula.

        Returns: ∀input. conditions(input) → var(input) = value

        For unconditional rules, returns: ∀input. var(input) = value
        """
        x = FOLVar(input_var.name)

        # Build conclusion: var(input) = value
        var_app = FOLFunctionApp(self.head_variable, [x])
        value_const = self._value_to_constant(self.head_value)
        conclusion = FOLEquals(var_app, value_const)

        # Build implication: conditions → conclusion
        if self.conditions:
            implication = FOLImplies(self.conditions, conclusion)
        else:
            # Unconditional rule - just the conclusion
            implication = conclusion

        # Wrap in universal quantifier
        return FOLForall(input_var, implication)

    def to_firing_condition(self, input_var: FOLVariable) -> FOLFormula:
        """
        Get the formula representing when this rule fires.

        Returns the conditions (or True if unconditional).
        """
        if self.conditions:
            return self.conditions
        return FOLPredicate("true", [])

    def _value_to_constant(self, value: Any) -> FOLConstant:
        """Convert a value to FOL constant."""
        if isinstance(value, bool):
            return FOLConstant(value, FOLSort.BOOL)
        # Handle string booleans
        if isinstance(value, str) and value.lower() in ("true", "false"):
            return FOLConstant(value.lower() == "true", FOLSort.BOOL)
        return FOLConstant(str(value), FOLSort.VALUE)

    def __repr__(self):
        type_str = "S" if self.rule_type == RuleType.STRICT else "D"
        return f"CantoRule({self.id}[{type_str}]: {self.head_variable}={self.head_value})"


@dataclass
class CantoSuperiority:
    """
    Superiority relation between rules.

    In DeLP, sup(A, B) means rule A defeats rule B when both fire
    with contradicting conclusions.

    Provides dict-like access for DeLPProgram compatibility.
    """
    superior: str  # Rule ID
    inferior: str  # Rule ID

    def __getitem__(self, key: str) -> str:
        """Dict-like access for DeLPProgram compatibility."""
        if key == 'superior':
            return self.superior
        elif key == 'inferior':
            return self.inferior
        raise KeyError(key)

    def to_fol(self) -> FOLFormula:
        """
        Convert to FOL formula.

        Semantics: If both rules fire with contradicting conclusions,
        the superior rule's conclusion wins.

        fires(superior) ∧ fires(inferior) ∧ conflict(superior, inferior)
            → wins(superior) ∧ ¬wins(inferior)
        """
        fires_sup = FOLPredicate("fires", [FOLConstant(self.superior, FOLSort.VALUE)])
        fires_inf = FOLPredicate("fires", [FOLConstant(self.inferior, FOLSort.VALUE)])
        conflict = FOLPredicate(
            "conflicts",
            [FOLConstant(self.superior, FOLSort.VALUE),
             FOLConstant(self.inferior, FOLSort.VALUE)]
        )
        wins_sup = FOLPredicate("wins", [FOLConstant(self.superior, FOLSort.VALUE)])
        wins_inf = FOLPredicate("wins", [FOLConstant(self.inferior, FOLSort.VALUE)])

        return FOLImplies(
            FOLAnd([fires_sup, fires_inf, conflict]),
            FOLAnd([wins_sup, FOLNot(wins_inf)])
        )

    def __repr__(self):
        return f"sup({self.superior} > {self.inferior})"


@dataclass
class CantoHasRelationship:
    """
    A 'has' relationship between variables.

    Represents structured data: ?patient has a ?diagnosis
    or collection data: ?patient has a list of ?symptoms
    """
    parent: str
    child: str
    is_list: bool
    description: Optional[str] = None

    def __repr__(self):
        list_str = "list of " if self.is_list else ""
        return f"has({self.parent}, {list_str}{self.child})"


@dataclass
class CantoFOL:
    """
    Complete Canto program in FOL representation.

    This is the canonical intermediate representation (IR) for Canto.
    All downstream transformations start from here:
    - Prolog backend generation
    - ASP backend generation (future)
    - Z3 verification
    - Prompt verification

    Provides compatibility with DeLPProgram interface through property aliases.
    """

    # Declarations
    variables: Dict[str, CantoVariable] = field(default_factory=dict)
    categories: Dict[str, CantoCategory] = field(default_factory=dict)

    # Rules
    strict_rules: List[CantoRule] = field(default_factory=list)
    defeasible_rules: List[CantoRule] = field(default_factory=list)

    # Superiority relations
    superiority: List[CantoSuperiority] = field(default_factory=list)

    # Has relationships (for nested structures)
    has_relationships: Dict[str, CantoHasRelationship] = field(default_factory=dict)

    # Metadata
    source_file: Optional[str] = None

    # =========================================================================
    # DeLPProgram compatibility aliases
    # =========================================================================

    @property
    def declarations(self) -> Dict[str, CantoVariable]:
        """Alias for variables (DeLPProgram compatibility)."""
        return self.variables

    @property
    def semantic_categories(self) -> Dict[str, CantoCategory]:
        """Alias for categories (DeLPProgram compatibility)."""
        return self.categories

    def to_prolog_string(self) -> str:
        """
        Generate Prolog code for this program.

        This provides compatibility with DeLPProgram.to_prolog_string().
        """
        from .backends.prolog import PrologBackend
        backend = PrologBackend(self)
        return backend.generate()

    # =========================================================================
    # Core methods
    # =========================================================================

    def all_rules(self) -> List[CantoRule]:
        """Get all rules (strict + defeasible)."""
        return self.strict_rules + self.defeasible_rules

    def get_rule(self, rule_id: str) -> Optional[CantoRule]:
        """Get rule by ID."""
        for rule in self.all_rules():
            if rule.id == rule_id:
                return rule
        return None

    def get_rules_for_variable(self, var_name: str) -> List[CantoRule]:
        """Get all rules that conclude about a variable."""
        return [r for r in self.all_rules() if r.head_variable == var_name]

    def get_values_for_variable(self, var_name: str) -> Set[Any]:
        """Get all possible values for a variable from rules."""
        values = set()
        # From rules
        for rule in self.all_rules():
            if rule.head_variable == var_name:
                values.add(rule.head_value)
        # From declaration
        var = self.variables.get(var_name)
        if var and var.possible_values:
            values.update(var.possible_values)
        return values

    def get_superiority_for_rule(self, rule_id: str) -> List[CantoSuperiority]:
        """Get all superiority relations where this rule is superior."""
        return [s for s in self.superiority if s.superior == rule_id]

    def get_defeated_by(self, rule_id: str) -> List[CantoSuperiority]:
        """Get all superiority relations where this rule is inferior."""
        return [s for s in self.superiority if s.inferior == rule_id]

    def get_variable_for_function(self, func_name: str) -> Optional[CantoVariable]:
        """
        Resolve a function name to its base variable.

        Handles both direct variable names and path-based names like
        "base_is_truthful_of_puzzle" which map to the child variable
        in a has_relationship.

        Args:
            func_name: Function name (e.g., "answer" or "base_is_truthful_of_puzzle")

        Returns:
            The CantoVariable if found, None otherwise
        """
        # Direct variable lookup
        if func_name in self.variables:
            return self.variables[func_name]

        # Check has_relationships for path-based names
        for rel in self.has_relationships.values():
            expected_name = f"{rel.child}_of_{rel.parent}"
            if func_name == expected_name:
                return self.variables.get(rel.child)

        return None

    def to_fol_formulas(self) -> List[FOLFormula]:
        """
        Convert entire program to list of FOL formulas.

        This is the core transformation that produces the canonical FOL
        representation for verification.
        """
        formulas = []
        input_var = FOLVariable("x", FOLSort.INPUT)

        # 1. Add category patterns
        for category in self.categories.values():
            formulas.extend(category.to_fol_predicates())

        # 2. Add rules
        for rule in self.all_rules():
            formulas.append(rule.to_fol(input_var))

        # 3. Add superiority relations
        for sup in self.superiority:
            formulas.append(sup.to_fol())

        # 4. Add mutual exclusivity constraints
        formulas.extend(self._mutual_exclusivity_constraints(input_var))

        # 5. Add rule type axioms (strict defeats defeasible by default)
        formulas.extend(self._rule_type_axioms())

        return formulas

    def _mutual_exclusivity_constraints(self, input_var: FOLVariable) -> List[FOLFormula]:
        """
        Generate mutual exclusivity constraints.

        For each variable with multiple possible values:
        ∀x. var(x) = v1 → var(x) ≠ v2 (for all v1 ≠ v2)
        """
        formulas = []
        x = FOLVar(input_var.name)

        for var_name in self.variables:
            var = self.variables[var_name]
            values = list(self.get_values_for_variable(var_name))
            if len(values) <= 1:
                continue

            var_app = FOLFunctionApp(var_name, [x])
            is_bool_var = var.is_bool()

            # Normalize values for boolean variables
            if is_bool_var:
                # Convert to canonical boolean set {True, False}
                normalized = set()
                for v in values:
                    if v in (True, "true"):
                        normalized.add(True)
                    elif v in (False, "false"):
                        normalized.add(False)
                    else:
                        normalized.add(v)
                values = list(normalized)

            # Pairwise exclusivity
            for i, v1 in enumerate(values):
                for v2 in values[i + 1:]:
                    # ¬(var(x) = v1 ∧ var(x) = v2)
                    if is_bool_var:
                        eq1 = FOLEquals(var_app, FOLConstant(v1, FOLSort.BOOL))
                        eq2 = FOLEquals(var_app, FOLConstant(v2, FOLSort.BOOL))
                    else:
                        eq1 = FOLEquals(var_app, FOLConstant(str(v1), FOLSort.VALUE))
                        eq2 = FOLEquals(var_app, FOLConstant(str(v2), FOLSort.VALUE))
                    constraint = FOLNot(FOLAnd([eq1, eq2]))
                    formulas.append(FOLForall(input_var, constraint))

        return formulas

    def _rule_type_axioms(self) -> List[FOLFormula]:
        """
        Generate axioms for rule type precedence.

        Strict rules defeat defeasible rules by default
        (unless explicit superiority says otherwise).
        """
        formulas = []

        # For each strict rule
        for strict in self.strict_rules:
            # Check each defeasible rule for the same variable
            for defeasible in self.defeasible_rules:
                if strict.head_variable != defeasible.head_variable:
                    continue
                if strict.head_value == defeasible.head_value:
                    continue  # Same conclusion, no conflict

                # Check if there's already an explicit superiority
                already_has_sup = any(
                    s.superior == strict.id and s.inferior == defeasible.id
                    for s in self.superiority
                )
                if already_has_sup:
                    continue

                # Add implicit superiority: strict beats defeasible
                formulas.append(
                    FOLPredicate(
                        "implicit_sup",
                        [
                            FOLConstant(strict.id, FOLSort.VALUE),
                            FOLConstant(defeasible.id, FOLSort.VALUE)
                        ]
                    )
                )

        return formulas

    def pretty_print(self) -> str:
        """Pretty print the FOL representation."""
        lines = ["=" * 60, "CANTO FOL IR", "=" * 60]

        if self.source_file:
            lines.append(f"Source: {self.source_file}")
        lines.append("")

        # Variables
        lines.append("VARIABLES:")
        for name, var in self.variables.items():
            desc = f' // "{var.description}"' if var.description else ""
            values = self.get_values_for_variable(name)
            if values:
                lines.append(f"  {name}: {{{', '.join(str(v) for v in values)}}}{desc}")
            else:
                lines.append(f"  {name}{desc}")

        # Categories
        if self.categories:
            lines.append("\nCATEGORIES:")
            for name, cat in self.categories.items():
                examples = cat.patterns[:3]
                more = f"... (+{len(cat.patterns) - 3})" if len(cat.patterns) > 3 else ""
                lines.append(f"  {name}: [{', '.join(examples)}{more}]")

        # Strict Rules
        if self.strict_rules:
            lines.append("\nSTRICT RULES:")
            for rule in self.strict_rules:
                lines.append(f"  {rule.id}: {rule.head_variable} = {rule.head_value}")
                if rule.conditions:
                    lines.append(f"    when: {rule.conditions}")

        # Defeasible Rules
        if self.defeasible_rules:
            lines.append("\nDEFEASIBLE RULES:")
            for rule in self.defeasible_rules:
                lines.append(f"  {rule.id}: {rule.head_variable} = {rule.head_value}")
                if rule.conditions:
                    lines.append(f"    when: {rule.conditions}")

        # Superiority
        if self.superiority:
            lines.append("\nSUPERIORITY:")
            for sup in self.superiority:
                lines.append(f"  {sup.superior} > {sup.inferior}")

        # Has relationships
        if self.has_relationships:
            lines.append("\nSTRUCTURE:")
            for key, rel in self.has_relationships.items():
                list_str = "list of " if rel.is_list else ""
                lines.append(f"  {rel.parent} has {list_str}{rel.child}")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)

    def __repr__(self):
        return (
            f"CantoFOL(vars={len(self.variables)}, "
            f"cats={len(self.categories)}, "
            f"strict={len(self.strict_rules)}, "
            f"defeasible={len(self.defeasible_rules)}, "
            f"sup={len(self.superiority)})"
        )
