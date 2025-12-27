"""
CantoBuilder - Build-time concept injection for Canto programs.

Allows registering Python-defined concepts and resolving them
against Canto programs at build time using a symbol table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from .concept import Concept
from .parser import CantoParser, ParseResult
from .symbols import (
    SymbolTable,
    ResolutionError,
    build_symbol_table,
)


@dataclass
class BuildResult:
    """
    Result of building a Canto program with concepts.

    Attributes:
        ast: List of AST nodes from the Canto source
        symbols: Symbol table with all declarations resolved
        instructions: Optional instructions from triple-quoted string at top of file
        concepts: Dict of registered concepts
    """
    ast: list
    symbols: SymbolTable
    instructions: str | None = None
    concepts: dict[str, Concept] | None = None


class ResolutionErrors(Exception):
    """Raised when there are resolution errors during build."""

    def __init__(self, errors: list[ResolutionError]):
        self.errors = errors
        error_msgs = [str(e) for e in errors]
        msg = f"Resolution failed with {len(errors)} error(s):\n  " + "\n  ".join(error_msgs)
        super().__init__(msg)


class CantoBuilder:
    """
    Builder for Canto programs with concept injection and resolution.

    Concepts registered with the builder are resolved against references
    in the Canto source at build time. The resolution phase validates
    that all references in the source can be satisfied.

    Example:
        builder = CantoBuilder()
        builder.register_concept(patient)
        builder.register_concept(triage_level)

        result = builder.build("medical_triage.canto")

        # Access the symbol table
        patient_symbol = result.symbols.resolve("patient")
    """

    def __init__(self, strict: bool = True):
        """
        Initialize the builder.

        Args:
            strict: If True (default), raise on unresolved references.
                    If False, collect errors but don't raise.
        """
        self._concepts: dict[str, Concept] = {}
        self._parser = CantoParser()
        self._strict = strict

    def register_concept(self, concept: Concept) -> Self:
        """
        Register a concept for build-time resolution.

        Args:
            concept: The Concept to register

        Returns:
            self for method chaining
        """
        self._concepts[concept.name] = concept
        return self

    def build(self, canto_file: str, validate: bool = True) -> BuildResult:
        """
        Build a Canto program with registered concepts resolved.

        1. Parse the Canto file
        2. Build symbol table from concepts and AST declarations
        3. Validate that all references can be resolved

        Args:
            canto_file: Path to the .canto file
            validate: Whether to validate references (default True)

        Returns:
            BuildResult with AST and symbol table

        Raises:
            ResolutionErrors: If strict=True and there are unresolved references
        """
        result = self._parser.parse_file(canto_file)
        return self._build_from_parse_result(result, validate)

    def build_string(self, canto_source: str, validate: bool = True) -> BuildResult:
        """
        Build a Canto program from a string with registered concepts resolved.

        Args:
            canto_source: Canto source code as a string
            validate: Whether to validate references (default True)

        Returns:
            BuildResult with AST and symbol table

        Raises:
            ResolutionErrors: If strict=True and there are unresolved references
        """
        result = self._parser.parse(canto_source)
        return self._build_from_parse_result(result, validate)

    def _build_from_parse_result(
        self,
        parse_result: ParseResult,
        validate: bool
    ) -> BuildResult:
        """
        Build from a ParseResult.

        Args:
            parse_result: The parsed Canto program
            validate: Whether to validate references

        Returns:
            BuildResult with AST and symbol table
        """
        # Build symbol table with resolution
        symbols, errors = build_symbol_table(
            ast=list(parse_result.ast),
            concepts=self._concepts if self._concepts else None,
            validate=validate
        )

        # Handle errors
        if errors and self._strict:
            raise ResolutionErrors(errors)

        return BuildResult(
            ast=list(parse_result.ast),
            symbols=symbols,
            instructions=parse_result.instructions,
            concepts=self._concepts.copy() if self._concepts else None
        )
