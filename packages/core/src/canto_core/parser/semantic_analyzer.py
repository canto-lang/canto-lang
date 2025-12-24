"""
Semantic analyzer for Canto DSL

Validates that all predicates used in rules are properly declared.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Union, Optional
from enum import Enum

from ..ast_nodes import (
    VariableDeclaration,
    SemanticCategory,
    HasDeclaration,
    Rule,
    Condition,
)


class PredicateKind(Enum):
    """Kind of predicate declaration"""
    INPUT = "input"           # can be, meaning - runtime input
    CATEGORY = "category"     # resembles - semantic category/pattern
    DERIVED = "derived"       # Rule head - derived from other predicates
    PROPERTY = "property"     # has a / has a list of - structural property


@dataclass
class DeclaredPredicate:
    """Information about a declared predicate"""
    name: str
    kind: PredicateKind
    source: str  # Description of where it was declared
    parent: Optional[str] = None  # Parent predicate if nested in a 'with' block


@dataclass
class SemanticError:
    """Represents a semantic error"""
    message: str
    predicate: str
    context: Optional[str] = None

    def __str__(self):
        if self.context:
            return f"{self.message}: ?{self.predicate} (in {self.context})"
        return f"{self.message}: ?{self.predicate}"


@dataclass
class SemanticAnalysisResult:
    """Result of semantic analysis"""
    errors: List[SemanticError] = field(default_factory=list)
    warnings: List[SemanticError] = field(default_factory=list)
    declared_predicates: Dict[str, DeclaredPredicate] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def __str__(self):
        lines = []
        if self.errors:
            lines.append(f"Errors ({len(self.errors)}):")
            for error in self.errors:
                lines.append(f"  - {error}")
        if self.warnings:
            lines.append(f"Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                lines.append(f"  - {warning}")
        if not lines:
            lines.append("No errors or warnings")
        return "\n".join(lines)


class SemanticAnalyzer:
    """
    Analyzes CLX AST for semantic errors.

    Validates:
    - All predicates used in conditions are declared
    - MATCHES uses input predicate on left and category on right
    """

    def __init__(self):
        self.declared: Dict[str, DeclaredPredicate] = {}
        self.errors: List[SemanticError] = []
        self.warnings: List[SemanticError] = []

    def analyze(self, ast: List[Union[VariableDeclaration, SemanticCategory, HasDeclaration, Rule]]) -> SemanticAnalysisResult:
        """
        Analyze the AST and return validation result.
        """
        self.declared = {}
        self.errors = []
        self.warnings = []

        # Pass 1: Collect all declarations
        self._collect_declarations(ast)

        # Pass 2: Validate all predicate references
        self._validate_references(ast)

        return SemanticAnalysisResult(
            errors=self.errors,
            warnings=self.warnings,
            declared_predicates=self.declared
        )

    def _collect_declarations(self, ast: List[Union[VariableDeclaration, SemanticCategory, HasDeclaration, Rule]], parent: Optional[str] = None):
        """Collect all predicate declarations from AST

        Args:
            ast: List of AST nodes to process
            parent: Optional parent predicate name if processing nested 'with' block children
        """
        for node in ast:
            if isinstance(node, VariableDeclaration):
                name = self._normalize_name(node.name)
                self.declared[name] = DeclaredPredicate(
                    name=name,
                    kind=PredicateKind.INPUT,
                    source="can be" if node.values_from else "meaning",
                    parent=parent
                )
                # Recursively collect children from 'with' blocks, passing this as parent
                if hasattr(node, 'children') and node.children:
                    self._collect_declarations(node.children, parent=name)
            elif isinstance(node, SemanticCategory):
                name = self._normalize_name(node.name)
                self.declared[name] = DeclaredPredicate(
                    name=name,
                    kind=PredicateKind.CATEGORY,
                    source="resembles",
                    parent=parent
                )
            elif isinstance(node, HasDeclaration):
                # has a / has a list of creates a property relationship
                # The child becomes a declared predicate
                child_name = self._normalize_name(node.child)
                cardinality = "has a list of" if node.is_list else "has a"
                parent_name = self._normalize_name(node.parent)
                self.declared[child_name] = DeclaredPredicate(
                    name=child_name,
                    kind=PredicateKind.PROPERTY,
                    source=f"{cardinality} (child of {node.parent})",
                    parent=parent_name
                )
                # Parent should also be declared (or will be auto-declared)
                if parent_name not in self.declared:
                    self.declared[parent_name] = DeclaredPredicate(
                        name=parent_name,
                        kind=PredicateKind.INPUT,
                        source=f"has parent (inferred from {cardinality})",
                        parent=parent
                    )
                # Handle 'from ?source' clause
                if hasattr(node, 'source') and node.source:
                    source_name = self._normalize_name(node.source)
                    if source_name not in self.declared:
                        self.declared[source_name] = DeclaredPredicate(
                            name=source_name,
                            kind=PredicateKind.INPUT,
                            source=f"extraction source (from clause)"
                        )
                # Recursively collect children from 'with' blocks
                if hasattr(node, 'children') and node.children:
                    self._collect_declarations(node.children, parent=child_name)
            elif isinstance(node, Rule):
                # Rule heads are derived predicates
                name = self._normalize_name(node.head_variable)
                if name not in self.declared:
                    self.declared[name] = DeclaredPredicate(
                        name=name,
                        kind=PredicateKind.DERIVED,
                        source="rule head"
                    )

    def _validate_references(self, ast: List[Union[VariableDeclaration, SemanticCategory, HasDeclaration, Rule]]):
        """Validate all predicate references in rules"""
        for node in ast:
            if isinstance(node, Rule):
                context = f"rule: ?{node.head_variable} becomes {node.head_value}"

                # Validate when conditions
                for condition in node.conditions:
                    self._validate_condition(condition, context)

                # Validate unless conditions
                for condition in node.exceptions:
                    self._validate_condition(condition, context)

    def _validate_condition(self, condition: Condition, context: str):
        """Recursively validate a condition"""
        if condition.operator in ["AND", "OR"]:
            # Binary logical operators - validate both sides
            if isinstance(condition.left, Condition):
                self._validate_condition(condition.left, context)
            if isinstance(condition.right, Condition):
                self._validate_condition(condition.right, context)

        elif condition.operator in ["NOT", "NOT_WARRANTED"]:
            # Unary operators - validate the inner condition
            if isinstance(condition.right, Condition):
                self._validate_condition(condition.right, context)

        elif condition.operator == "IS_LIKE":
            # is like: left should be input, right should be category
            left_name = self._normalize_name(condition.left)
            right_name = self._normalize_name(condition.right)

            # Check left side (input predicate)
            if left_name not in self.declared:
                self.errors.append(SemanticError(
                    message="Undefined predicate",
                    predicate=left_name,
                    context=context
                ))
            elif self.declared[left_name].kind == PredicateKind.CATEGORY:
                self.warnings.append(SemanticError(
                    message="is like left side should be input, not category",
                    predicate=left_name,
                    context=context
                ))

            # Check parent if qualified variable
            parent_name = self._get_parent_from_qualified(condition.left)
            if parent_name and parent_name not in self.declared:
                self.errors.append(SemanticError(
                    message="Undefined predicate",
                    predicate=parent_name,
                    context=context
                ))

            # Check if nested predicate is used without qualification
            self._check_nested_qualification(condition.left, context)

            # Check right side (category)
            if right_name not in self.declared:
                self.errors.append(SemanticError(
                    message="Undefined predicate",
                    predicate=right_name,
                    context=context
                ))
            elif self.declared[right_name].kind != PredicateKind.CATEGORY:
                self.warnings.append(SemanticError(
                    message="is like right side should be a category (resembles)",
                    predicate=right_name,
                    context=context
                ))

        elif condition.operator == "IS":
            # is: left should be declared
            left_name = self._normalize_name(condition.left)
            if left_name and left_name not in self.declared:
                self.errors.append(SemanticError(
                    message="Undefined predicate",
                    predicate=left_name,
                    context=context
                ))
            # Check parent if qualified variable
            parent_name = self._get_parent_from_qualified(condition.left)
            if parent_name and parent_name not in self.declared:
                self.errors.append(SemanticError(
                    message="Undefined predicate",
                    predicate=parent_name,
                    context=context
                ))
            # Check if nested predicate is used without qualification
            self._check_nested_qualification(condition.left, context)

        elif condition.operator == "HAS":
            # has: left should be declared (typically a list property)
            left_name = self._normalize_name(condition.left)
            if left_name and left_name not in self.declared:
                self.errors.append(SemanticError(
                    message="Undefined predicate",
                    predicate=left_name,
                    context=context
                ))
            # Check parent if qualified variable
            parent_name = self._get_parent_from_qualified(condition.left)
            if parent_name and parent_name not in self.declared:
                self.errors.append(SemanticError(
                    message="Undefined predicate",
                    predicate=parent_name,
                    context=context
                ))
            # Check if nested predicate is used without qualification
            self._check_nested_qualification(condition.left, context)
            # Right side can be a value or a variable
            right_name = self._normalize_name(condition.right)
            if right_name and isinstance(condition.right, str) and condition.right.startswith('?'):
                # It's a variable reference, check if declared
                if right_name not in self.declared:
                    self.errors.append(SemanticError(
                        message="Undefined predicate",
                        predicate=right_name,
                        context=context
                    ))

        elif condition.operator.startswith("HAS_") and "_IS_LIKE" in condition.operator:
            # Quantified has with is like: has any/all/none that is like ?category
            # Left should be declared (list property)
            left_name = self._normalize_name(condition.left)
            if left_name and left_name not in self.declared:
                self.errors.append(SemanticError(
                    message="Undefined predicate",
                    predicate=left_name,
                    context=context
                ))
            # Check parent if qualified variable
            parent_name = self._get_parent_from_qualified(condition.left)
            if parent_name and parent_name not in self.declared:
                self.errors.append(SemanticError(
                    message="Undefined predicate",
                    predicate=parent_name,
                    context=context
                ))
            # Check if nested predicate is used without qualification
            self._check_nested_qualification(condition.left, context)
            # Right should be a category
            right_name = self._normalize_name(condition.right)
            if right_name and right_name not in self.declared:
                self.errors.append(SemanticError(
                    message="Undefined predicate",
                    predicate=right_name,
                    context=context
                ))
            elif right_name and self.declared[right_name].kind != PredicateKind.CATEGORY:
                self.warnings.append(SemanticError(
                    message="is like right side should be a category (resembles)",
                    predicate=right_name,
                    context=context
                ))

        elif condition.operator.startswith("HAS_") and ("_IS_NOT" in condition.operator or "_IS" in condition.operator):
            # Quantified has with is/is not: has any/all/none that is/is not "value"
            # Left should be declared (list property)
            left_name = self._normalize_name(condition.left)
            if left_name and left_name not in self.declared:
                self.errors.append(SemanticError(
                    message="Undefined predicate",
                    predicate=left_name,
                    context=context
                ))
            # Check parent if qualified variable
            parent_name = self._get_parent_from_qualified(condition.left)
            if parent_name and parent_name not in self.declared:
                self.errors.append(SemanticError(
                    message="Undefined predicate",
                    predicate=parent_name,
                    context=context
                ))
            # Check if nested predicate is used without qualification
            self._check_nested_qualification(condition.left, context)
            # Right side is a value (string/bool/number), no need to validate

    def _normalize_name(self, name) -> Optional[str]:
        """Normalize predicate name (remove ? prefix if present)

        Handles both simple variables (?var) and qualified variables (?child of ?parent).
        For qualified variables, returns the child name (the property being accessed).
        """
        if name is None:
            return None
        if isinstance(name, dict) and 'child' in name:
            # Qualified variable: return the child name
            return name['child'].lstrip('?')
        if isinstance(name, str):
            return name.lstrip('?')
        return None

    def _get_parent_from_qualified(self, name) -> Optional[str]:
        """Get immediate parent name from qualified variable reference.

        For nested qualifications like ?a of (?b of ?c), returns 'b' (the immediate parent).
        """
        if isinstance(name, dict) and 'parent' in name:
            parent = name['parent']
            # Handle nested qualification: parent could be a tuple or string
            if isinstance(parent, tuple):
                # Nested: ?a of (?b of ?c) -> parent is ('?b', '?c'), return 'b'
                return parent[0].lstrip('?')
            elif isinstance(parent, str):
                return parent.lstrip('?')
        return None

    def _get_all_parents_from_qualified(self, name) -> List[str]:
        """Get all parent names from a qualified variable reference.

        For nested qualifications like ?a of (?b of ?c), returns ['b', 'c'].
        """
        parents = []
        if isinstance(name, dict) and 'parent' in name:
            parent = name['parent']
            if isinstance(parent, tuple):
                # Recursively extract all parents from the chain
                self._extract_parents_from_tuple(parent, parents)
            elif isinstance(parent, str):
                parents.append(parent.lstrip('?'))
        return parents

    def _extract_parents_from_tuple(self, tup, parents: List[str]):
        """Recursively extract parent names from nested tuple structure."""
        if isinstance(tup, tuple) and len(tup) >= 2:
            # First element is the variable name
            parents.append(tup[0].lstrip('?'))
            # Second element could be another tuple or a string
            if isinstance(tup[1], tuple):
                self._extract_parents_from_tuple(tup[1], parents)
            elif isinstance(tup[1], str):
                parents.append(tup[1].lstrip('?'))

    def _is_qualified_reference(self, name) -> bool:
        """Check if the reference is qualified (?child of ?parent)"""
        return isinstance(name, dict) and 'child' in name and 'parent' in name

    def _check_nested_qualification(self, name, context: str):
        """Check if a nested predicate is referenced without qualification.

        Issues a warning if a predicate that was declared inside a 'with' block
        is used without explicit parent qualification (?child of ?parent).
        """
        normalized = self._normalize_name(name)
        if normalized and normalized in self.declared:
            decl = self.declared[normalized]
            # If the predicate has a parent (nested), and the reference is not qualified
            if decl.parent and not self._is_qualified_reference(name):
                self.warnings.append(SemanticError(
                    message=f"Nested predicate should be qualified with parent (use ?{normalized} of ?{decl.parent})",
                    predicate=normalized,
                    context=context
                ))


def analyze(ast: List[Union[VariableDeclaration, SemanticCategory, HasDeclaration, Rule]]) -> SemanticAnalysisResult:
    """Convenience function to analyze AST"""
    analyzer = SemanticAnalyzer()
    return analyzer.analyze(ast)
