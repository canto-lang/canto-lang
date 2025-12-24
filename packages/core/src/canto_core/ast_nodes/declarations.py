"""
AST nodes for variable declarations and semantic categories
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from ast_nodes.rules import Rule


@dataclass
class ImportDeclaration:
    """
    Represents an import statement in the DSL
    Example: import food_categories
    """
    name: str  # Module name (without .canto extension)

    def __repr__(self):
        return f"ImportDeclaration({self.name})"


@dataclass
class VariableDeclaration:
    """
    Represents a variable declaration in the DSL
    Example: ?patient_query meaning "the text provided by the patient"
    Example with extraction source:
        ?entities meaning "extracted entities" from ?text with
            ?companies has a list of ?company
    Example with nested structure:
        ?patient meaning "patient info" with
            ?name meaning "patient name"
            ?age meaning "patient age"
    """
    name: str  # Variable name (without ?)
    description: Optional[str] = None
    values_from: Optional[List[str]] = None  # For can be "a", "b", "c"
    source: Optional[str] = None  # For 'from ?source' extraction clause (inherited by children)
    children: List[Union['VariableDeclaration', 'Rule']] = field(default_factory=list)  # For 'with' blocks

    def __post_init__(self):
        # Ensure name doesn't have ? prefix
        if self.name.startswith('?'):
            self.name = self.name[1:]
        if self.source and self.source.startswith('?'):
            self.source = self.source[1:]

    def __repr__(self):
        if self.children:
            return f"VariableDeclaration({self.name}, children={len(self.children)})"
        return f"VariableDeclaration({self.name})"


@dataclass
class SemanticCategory:
    """
    Represents a semantic category (resembles pattern)
    Example: ?vaccine_terms resembles "vaccine", "vaccines", "vaccination"
    """
    name: str  # Variable name (without ?)
    patterns: List[str]  # List of pattern strings
    description: Optional[str] = None

    def __post_init__(self):
        # Ensure name doesn't have ? prefix
        if self.name.startswith('?'):
            self.name = self.name[1:]

    def __repr__(self):
        return f"SemanticCategory({self.name} ~ {len(self.patterns)} patterns)"


@dataclass
class HasDeclaration:
    """
    Represents a structural relationship (has declaration)

    Examples:
    - ?patient has a ?diagnosis meaning "the diagnosis assigned"
    - ?patient has a list of ?medications meaning "current medications"
    - ?entities has a list of ?company from ?text

    This establishes parent-child relationships between variables,
    enabling nested data structures in the domain model.
    """
    parent: str  # Parent variable name (without ?)
    child: str   # Child variable name (without ?)
    is_list: bool  # True if "has a list of", False if "has a"
    description: Optional[str] = None
    source: Optional[str] = None  # For 'from ?source' extraction clause
    children: List[Union['VariableDeclaration', 'Rule']] = field(default_factory=list)  # For 'with' blocks

    def __post_init__(self):
        # Ensure names don't have ? prefix
        if self.parent.startswith('?'):
            self.parent = self.parent[1:]
        if self.child.startswith('?'):
            self.child = self.child[1:]
        if self.source and self.source.startswith('?'):
            self.source = self.source[1:]

    def __repr__(self):
        list_str = "list of " if self.is_list else ""
        from_str = f" from {self.source}" if self.source else ""
        return f"HasDeclaration({self.parent} has a {list_str}{self.child}{from_str})"
