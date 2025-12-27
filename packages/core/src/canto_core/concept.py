"""
Concept - Python-Canto interop for semantic structures.

Allows defining semantic concepts in Python that can be injected
into Canto programs at build time.
"""

from __future__ import annotations

import re
from typing import Self, Union

from .ast_nodes import VariableDeclaration, SemanticCategory


class Concept:
    """
    A semantic concept that can be shared with Canto programs.

    Concepts support:
    - meaning: semantic description
    - resembles: similar terms for semantic matching
    - can_be: possible values/classifications
    - has: nested sub-concepts

    Example:
        patient = (
            Concept("patient")
            .meaning("extracted patient information")
            .has(
                Concept("name").meaning("patient's full name"),
                Concept("age").meaning("patient's age in years"),
            )
        )
    """

    _NAME_PATTERN = re.compile(r"^[a-z_][a-z0-9_]*$")

    def __init__(self, name: str):
        if not isinstance(name, str):
            raise TypeError(f"Concept name must be a string, got {type(name).__name__}")

        if not name:
            raise ValueError("Concept name cannot be empty")

        if not self._NAME_PATTERN.match(name):
            raise ValueError(
                f"Invalid concept name '{name}': must start with lowercase letter or underscore, "
                "contain only lowercase letters, digits, and underscores"
            )

        self.name = name
        self._meaning: str | None = None
        self._resembles: list[str] = []
        self._can_be: list[str | bool] = []
        self._children: list[Concept] = []

    def meaning(self, description: str) -> Self:
        """Set the semantic meaning of this concept."""
        self._meaning = description
        return self

    def resembles(self, *terms: str) -> Self:
        """Add similar terms for semantic matching."""
        self._resembles.extend(terms)
        return self

    def can_be(self, *values: str | bool) -> Self:
        """Define possible values/classifications."""
        self._can_be.extend(values)
        return self

    def has(self, *children: Concept) -> Self:
        """Add nested sub-concepts."""
        self._children.extend(children)
        return self

    def to_ast(self) -> Union[VariableDeclaration, SemanticCategory]:
        """
        Convert this concept to an AST node for direct injection.

        Returns:
            SemanticCategory if concept has resembles patterns,
            VariableDeclaration otherwise.
        """
        if self._resembles:
            return SemanticCategory(
                name=self.name,
                patterns=self._resembles.copy(),
                description=self._meaning,
            )

        # Convert can_be values to strings (booleans become "true"/"false")
        values_from = None
        if self._can_be:
            values_from = [
                str(v).lower() if isinstance(v, bool) else v
                for v in self._can_be
            ]

        # Recursively convert children
        children = [child.to_ast() for child in self._children]

        return VariableDeclaration(
            name=self.name,
            description=self._meaning,
            values_from=values_from,
            children=children,
        )

    def to_dict(self) -> dict:
        """Export as dictionary for serialization."""
        d = {"name": self.name}

        if self._meaning is not None:
            d["meaning"] = self._meaning

        if self._resembles:
            d["resembles"] = self._resembles.copy()

        if self._can_be:
            d["can_be"] = self._can_be.copy()

        if self._children:
            d["has"] = [child.to_dict() for child in self._children]

        return d

    @classmethod
    def from_dict(cls, data: dict) -> Concept:
        """Create concept from dictionary."""
        concept = cls(data["name"])

        if "meaning" in data:
            concept.meaning(data["meaning"])

        if "resembles" in data:
            concept.resembles(*data["resembles"])

        if "can_be" in data:
            concept.can_be(*data["can_be"])

        if "has" in data:
            children = [cls.from_dict(child_data) for child_data in data["has"]]
            concept.has(*children)

        return concept

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Concept):
            return NotImplemented

        return (
            self.name == other.name
            and self._meaning == other._meaning
            and self._resembles == other._resembles
            and self._can_be == other._can_be
            and self._children == other._children
        )

    def __hash__(self) -> int:
        return hash((self.name, self._meaning, tuple(self._resembles), tuple(self._can_be)))

    def __repr__(self) -> str:
        return f"Concept({self.name!r})"
