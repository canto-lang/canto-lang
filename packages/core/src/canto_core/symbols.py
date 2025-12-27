"""
Symbol Table for Canto - Resolution phase for name binding.

Provides explicit resolution of variable references to their declarations,
whether from parsed Canto source or injected Python concepts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .concept import Concept
    from .ast_nodes import VariableDeclaration, SemanticCategory, HasDeclaration


class SymbolKind(Enum):
    """The kind/origin of a symbol."""
    VARIABLE = "variable"      # VariableDeclaration from Canto
    CATEGORY = "category"      # SemanticCategory from Canto
    HAS = "has"                # HasDeclaration from Canto
    CONCEPT = "concept"        # Injected from Python Concept


@dataclass
class Symbol:
    """
    A resolved symbol in the symbol table.

    Tracks the name, kind, and source of a declaration.
    """
    name: str
    kind: SymbolKind
    source: Union['VariableDeclaration', 'SemanticCategory', 'HasDeclaration', 'Concept']

    def __repr__(self) -> str:
        return f"Symbol({self.name!r}, {self.kind.value})"


class ResolutionError(Exception):
    """Base class for resolution errors."""
    pass


class UnresolvedReferenceError(ResolutionError):
    """Raised when a variable reference cannot be resolved."""

    def __init__(self, name: str, location: str | None = None):
        self.name = name
        self.location = location
        msg = f"Unresolved reference: '{name}'"
        if location:
            msg += f" at {location}"
        super().__init__(msg)


class DuplicateDeclarationError(ResolutionError):
    """Raised when a symbol is declared multiple times."""

    def __init__(self, name: str, existing: Symbol, new_source: object):
        self.name = name
        self.existing = existing
        self.new_source = new_source
        msg = f"Duplicate declaration: '{name}' already declared as {existing.kind.value}"
        super().__init__(msg)


class SymbolTable:
    """
    Symbol table for resolving variable references.

    Collects declarations from:
    - Parsed Canto AST (VariableDeclaration, SemanticCategory, HasDeclaration)
    - Injected Python Concepts

    Then validates that all references in the AST can be resolved.
    """

    def __init__(self):
        self._symbols: dict[str, Symbol] = {}
        self._references: set[str] = set()

    def declare(self, name: str, kind: SymbolKind, source: object) -> Symbol:
        """
        Declare a symbol in the table.

        Args:
            name: The symbol name (without ? prefix)
            kind: The kind of symbol
            source: The source object (AST node or Concept)

        Returns:
            The created Symbol

        Raises:
            DuplicateDeclarationError: If the symbol is already declared
        """
        # Normalize name (remove ? prefix if present)
        name = name.lstrip('?')

        if name in self._symbols:
            raise DuplicateDeclarationError(name, self._symbols[name], source)

        symbol = Symbol(name=name, kind=kind, source=source)
        self._symbols[name] = symbol
        return symbol

    def declare_from_ast(self, node: object) -> Symbol | None:
        """
        Declare a symbol from an AST node.

        Automatically determines the symbol kind from the node type.

        Args:
            node: An AST node (VariableDeclaration, SemanticCategory, or HasDeclaration)

        Returns:
            The created Symbol, or None if the node type is not a declaration
        """
        from .ast_nodes import VariableDeclaration, SemanticCategory, HasDeclaration

        if isinstance(node, VariableDeclaration):
            return self.declare(node.name, SymbolKind.VARIABLE, node)
        elif isinstance(node, SemanticCategory):
            return self.declare(node.name, SymbolKind.CATEGORY, node)
        elif isinstance(node, HasDeclaration):
            # HasDeclaration declares the child variable
            return self.declare(node.child, SymbolKind.HAS, node)

        return None

    def declare_from_concept(self, concept: 'Concept') -> Symbol:
        """
        Declare a symbol from a Python Concept.

        Args:
            concept: The Concept to declare

        Returns:
            The created Symbol
        """
        return self.declare(concept.name, SymbolKind.CONCEPT, concept)

    def add_reference(self, name: str) -> None:
        """
        Record a variable reference for later resolution checking.

        Args:
            name: The referenced variable name (without ? prefix)
        """
        name = name.lstrip('?')
        self._references.add(name)

    def resolve(self, name: str) -> Symbol:
        """
        Resolve a variable reference to its declaration.

        Args:
            name: The variable name to resolve (without ? prefix)

        Returns:
            The resolved Symbol

        Raises:
            UnresolvedReferenceError: If the reference cannot be resolved
        """
        name = name.lstrip('?')

        if name not in self._symbols:
            raise UnresolvedReferenceError(name)

        return self._symbols[name]

    def get(self, name: str) -> Symbol | None:
        """
        Get a symbol by name, or None if not found.

        Args:
            name: The variable name (without ? prefix)

        Returns:
            The Symbol if found, None otherwise
        """
        name = name.lstrip('?')
        return self._symbols.get(name)

    def validate_references(self) -> list[UnresolvedReferenceError]:
        """
        Validate that all recorded references can be resolved.

        Returns:
            List of UnresolvedReferenceError for any unresolved references
        """
        errors = []
        for ref in sorted(self._references):
            if ref not in self._symbols:
                errors.append(UnresolvedReferenceError(ref))
        return errors

    def get_unused_symbols(self) -> list[Symbol]:
        """
        Get symbols that were declared but never referenced.

        Useful for warnings about unused concepts.

        Returns:
            List of unused Symbols
        """
        unused = []
        for name, symbol in self._symbols.items():
            if name not in self._references:
                unused.append(symbol)
        return unused

    def __contains__(self, name: str) -> bool:
        """Check if a symbol is declared."""
        name = name.lstrip('?')
        return name in self._symbols

    def __len__(self) -> int:
        """Return the number of declared symbols."""
        return len(self._symbols)

    def __iter__(self):
        """Iterate over symbol names."""
        return iter(self._symbols)

    def items(self):
        """Iterate over (name, symbol) pairs."""
        return self._symbols.items()

    def values(self):
        """Iterate over symbols."""
        return self._symbols.values()


def collect_declarations(ast: list) -> list[tuple[str, object]]:
    """
    Collect all declarations from an AST.

    Walks the AST and extracts all variable/category declarations.

    Args:
        ast: List of AST nodes

    Returns:
        List of (name, node) tuples for all declarations
    """
    from .ast_nodes import VariableDeclaration, SemanticCategory, HasDeclaration, Rule

    declarations = []

    def visit(node):
        if isinstance(node, VariableDeclaration):
            declarations.append((node.name, node))
            # Recursively collect from children
            for child in node.children:
                visit(child)
        elif isinstance(node, SemanticCategory):
            declarations.append((node.name, node))
        elif isinstance(node, HasDeclaration):
            declarations.append((node.child, node))
            # Recursively collect from children
            for child in node.children:
                visit(child)
        elif isinstance(node, Rule):
            # Rules don't declare new variables, they reference them
            pass

    for node in ast:
        visit(node)

    return declarations


def collect_references(ast: list) -> set[str]:
    """
    Collect all variable references from an AST.

    Walks the AST and extracts all variable names that are referenced
    (in rules, conditions, etc.).

    Args:
        ast: List of AST nodes

    Returns:
        Set of variable names that are referenced
    """
    from .ast_nodes import VariableDeclaration, SemanticCategory, HasDeclaration, Rule, Condition

    references = set()

    def extract_from_condition(cond: Condition):
        """Extract variable references from a condition."""
        if cond is None:
            return

        # Left side
        if isinstance(cond.left, str) and cond.left.startswith('?'):
            references.add(cond.left.lstrip('?'))
        elif isinstance(cond.left, dict):
            # Qualified variable: {"child": "?x", "parent": "?y"}
            # For property access like ?child of ?parent, only parent needs to be declared
            # The child is a property accessor, not a standalone reference
            if 'parent' in cond.left:
                parent = cond.left['parent']
                if isinstance(parent, str) and parent.startswith('?'):
                    references.add(parent.lstrip('?'))
            # List reference for length conditions
            if 'list' in cond.left:
                list_ref = cond.left['list']
                if isinstance(list_ref, str) and list_ref.startswith('?'):
                    references.add(list_ref.lstrip('?'))
                elif isinstance(list_ref, dict):
                    if 'child' in list_ref:
                        references.add(list_ref['child'].lstrip('?'))
                    if 'parent' in list_ref:
                        references.add(list_ref['parent'].lstrip('?'))
            if 'property' in cond.left:
                prop = cond.left['property']
                if isinstance(prop, str) and prop.startswith('?'):
                    references.add(prop.lstrip('?'))
        elif isinstance(cond.left, Condition):
            extract_from_condition(cond.left)

        # Right side
        if isinstance(cond.right, str) and cond.right.startswith('?'):
            references.add(cond.right.lstrip('?'))
        elif isinstance(cond.right, dict):
            # Let binding: {"binding": "?x", "conditions": [...]}
            if 'binding' in cond.right:
                binding = cond.right['binding']
                if isinstance(binding, str) and binding.startswith('?'):
                    references.add(binding.lstrip('?'))
            if 'conditions' in cond.right:
                for bound_cond in cond.right['conditions']:
                    if isinstance(bound_cond, dict):
                        if 'left' in bound_cond:
                            left = bound_cond['left']
                            if isinstance(left, str) and left.startswith('?'):
                                references.add(left.lstrip('?'))
                            elif isinstance(left, tuple):
                                # Qualified: (child, parent)
                                for part in left:
                                    if isinstance(part, str) and part.startswith('?'):
                                        references.add(part.lstrip('?'))
                        if 'right' in bound_cond:
                            right = bound_cond['right']
                            if isinstance(right, str) and right.startswith('?'):
                                references.add(right.lstrip('?'))
            # Property-based condition
            if 'property' in cond.right:
                prop = cond.right['property']
                if isinstance(prop, str) and prop.startswith('?'):
                    references.add(prop.lstrip('?'))
            if 'value' in cond.right:
                val = cond.right['value']
                if isinstance(val, str) and val.startswith('?'):
                    references.add(val.lstrip('?'))
        elif isinstance(cond.right, Condition):
            extract_from_condition(cond.right)

    def visit(node):
        if isinstance(node, VariableDeclaration):
            # Source reference
            if node.source:
                references.add(node.source.lstrip('?'))
            # Recursively visit children
            for child in node.children:
                visit(child)
        elif isinstance(node, HasDeclaration):
            # Parent is a reference
            references.add(node.parent.lstrip('?'))
            if node.source:
                references.add(node.source.lstrip('?'))
            for child in node.children:
                visit(child)
        elif isinstance(node, Rule):
            # Head parent is a reference
            if node.head_parent:
                if isinstance(node.head_parent, str):
                    references.add(node.head_parent.lstrip('?'))
                elif isinstance(node.head_parent, tuple):
                    # Nested: (child, parent) or deeper
                    def extract_from_tuple(t):
                        if isinstance(t, str):
                            references.add(t.lstrip('?'))
                        elif isinstance(t, tuple):
                            for part in t:
                                extract_from_tuple(part)
                    extract_from_tuple(node.head_parent)

            # Conditions are references
            for cond in node.conditions:
                extract_from_condition(cond)
            for cond in node.exceptions:
                extract_from_condition(cond)

    for node in ast:
        visit(node)

    return references


def build_symbol_table(
    ast: list,
    concepts: dict | None = None,
    validate: bool = True
) -> tuple[SymbolTable, list[ResolutionError]]:
    """
    Build a symbol table from an AST and optional concepts.

    Args:
        ast: List of AST nodes from parsing
        concepts: Optional dict of {name: Concept} for injected concepts
        validate: Whether to validate references (default True)

    Returns:
        Tuple of (SymbolTable, list of errors)
        If validate=True and there are unresolved references, errors will be populated
    """
    symbols = SymbolTable()
    errors = []

    # First, declare all concepts (they take precedence)
    if concepts:
        for concept in concepts.values():
            try:
                symbols.declare_from_concept(concept)
            except DuplicateDeclarationError as e:
                errors.append(e)

    # Then, declare all AST declarations
    for name, node in collect_declarations(ast):
        try:
            symbols.declare_from_ast(node)
        except DuplicateDeclarationError as e:
            errors.append(e)

    # Collect and record all references
    if validate:
        refs = collect_references(ast)
        for ref in refs:
            symbols.add_reference(ref)

        # Validate references
        errors.extend(symbols.validate_references())

    return symbols, errors
