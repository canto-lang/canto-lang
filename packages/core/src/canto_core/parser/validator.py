"""
Semantic validation for Canto DSL
"""

from typing import List, Set, Union
from ..ast_nodes import VariableDeclaration, SemanticCategory, Rule


class ValidationError(Exception):
    """Raised when validation fails"""
    pass


class CantoValidator:
    """
    Validates semantic correctness of Canto DSL programs
    """

    def __init__(self):
        self.declared_variables: Set[str] = set()
        self.semantic_categories: Set[str] = set()

    def validate(self, ast: List[Union[VariableDeclaration, SemanticCategory, Rule]]) -> bool:
        """
        Validate an AST
        Returns True if valid, raises ValidationError otherwise
        """
        # First pass: collect all declarations
        for node in ast:
            if isinstance(node, VariableDeclaration):
                self._register_variable(node)
            elif isinstance(node, SemanticCategory):
                self._register_semantic_category(node)

        # Second pass: validate rules
        for node in ast:
            if isinstance(node, Rule):
                self._validate_rule(node)

        return True

    def _register_variable(self, var_decl: VariableDeclaration):
        """Register a variable declaration"""
        name = var_decl.name
        if name in self.declared_variables:
            raise ValidationError(f"Variable ?{name} declared multiple times")

        self.declared_variables.add(name)

    def _register_semantic_category(self, sem_cat: SemanticCategory):
        """Register a semantic category"""
        name = sem_cat.name
        if name in self.declared_variables:
            raise ValidationError(
                f"Semantic category ?{name} conflicts with variable declaration"
            )
        if name in self.semantic_categories:
            raise ValidationError(f"Semantic category ?{name} declared multiple times")

        self.semantic_categories.add(name)

    def _validate_rule(self, rule: Rule):
        """Validate a rule"""
        # Check head variable is declared
        head_var = rule.head_variable
        if head_var not in self.declared_variables and head_var not in self.semantic_categories:
            raise ValidationError(
                f"Rule references undeclared variable ?{head_var}"
            )

        # Validate all conditions
        for condition in rule.conditions:
            self._validate_condition(condition)

        # Validate all exceptions
        for exception in rule.exceptions:
            self._validate_condition(exception)

    def _validate_condition(self, condition):
        """Validate a condition"""
        from ast_nodes.rules import Condition

        if not isinstance(condition, Condition):
            return

        # Recursively validate nested conditions
        if condition.operator in ["AND", "OR"]:
            if isinstance(condition.left, Condition):
                self._validate_condition(condition.left)
            if isinstance(condition.right, Condition):
                self._validate_condition(condition.right)
            return

        if condition.operator == "NOT":
            if isinstance(condition.right, Condition):
                self._validate_condition(condition.right)
            return

        # Validate variable references
        left = condition.left
        if isinstance(left, str) and left.startswith('?'):
            var_name = left[1:]
            if var_name not in self.declared_variables and var_name not in self.semantic_categories:
                raise ValidationError(
                    f"Condition references undeclared variable ?{var_name}"
                )

        right = condition.right
        if isinstance(right, str) and right.startswith('?'):
            var_name = right[1:]
            if var_name not in self.declared_variables and var_name not in self.semantic_categories:
                raise ValidationError(
                    f"Condition references undeclared variable ?{var_name}"
                )


def validate_ast(ast: List[Union[VariableDeclaration, SemanticCategory, Rule]]) -> bool:
    """
    Convenience function to validate an AST
    """
    validator = CantoValidator()
    return validator.validate(ast)
