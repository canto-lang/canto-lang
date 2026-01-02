"""
PromptGenerator - High-level abstraction for generating prompts from Canto programs.

Wraps the full build pipeline:
1. FOL Translator → FOL IR → Prolog KB
2. LLMPredicateGenerator → predicates for semantic categories
3. DeLPReasoningAnalyzer → reasoning structure
4. PromptTemplateGenerator → final prompt
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from .codegen import (
    LLMPredicateGenerator,
    PromptTemplateGenerator,
)
from .delp import DeLPReasoningAnalyzer
from .fol import translate_to_fol

if TYPE_CHECKING:
    from .builder import BuildResult


class PromptGenerator:
    """
    Generate prompts from built Canto programs.

    Wraps the full build pipeline to provide a simple interface
    for generating prompts from Canto programs with injected concepts.

    Example:
        builder = CantoBuilder()
        builder.register_concept(patient)
        result = builder.build("triage.canto")

        generator = PromptGenerator(result)
        prompt = generator.generate()
    """

    def __init__(self, build_result: 'BuildResult', max_retries: int = 3):
        """
        Initialize the prompt generator.

        Args:
            build_result: The built Canto program (from CantoBuilder.build())
            max_retries: Max retries for LLM-based generation steps
        """
        self._result = build_result
        self._max_retries = max_retries

        # Build combined AST from concepts + parsed AST
        self._ast = self._build_combined_ast()

        # Lazy-initialized components
        self._program = None
        self._llm_predicates = None
        self._reasoning_structure = None

    def _build_combined_ast(self) -> list:
        """Build AST combining concepts and parsed nodes."""
        ast = []

        # Add concept AST nodes first
        if self._result.concepts:
            for concept in self._result.concepts.values():
                ast.append(concept.to_ast())

        # Add parsed AST nodes
        ast.extend(self._result.ast)

        return ast

    @property
    def program(self):
        """Lazily translate to FOL IR."""
        if self._program is None:
            self._program = translate_to_fol(self._ast)
        return self._program

    @property
    def llm_predicates(self) -> str:
        """Lazily generate LLM predicates for semantic categories."""
        if self._llm_predicates is None:
            generator = LLMPredicateGenerator(max_retries=self._max_retries)
            predicates = []
            for category in self.program.semantic_categories.values():
                pred = generator.generate_from_category(category)
                predicates.append(pred)
            self._llm_predicates = "\n\n".join(predicates)
        return self._llm_predicates

    @property
    def reasoning_structure(self) -> dict:
        """Lazily analyze reasoning structure."""
        if self._reasoning_structure is None:
            analyzer = DeLPReasoningAnalyzer(
                self.program,
                extra_prolog=self.llm_predicates
            )
            self._reasoning_structure = analyzer.analyze()
        return self._reasoning_structure

    def generate(self, instructions: Optional[str] = None) -> str:
        """
        Generate a prompt from the built program.

        Args:
            instructions: Override instructions (uses DSL instructions if None)

        Returns:
            Generated prompt string
        """
        prompt_gen = PromptTemplateGenerator(max_retries=self._max_retries)

        # Use provided instructions, DSL instructions, or default
        instr = instructions or self._result.instructions or ""

        prompt = prompt_gen.generate_from_structure(
            reasoning_structure=self.reasoning_structure,
            dsl_instructions=instr,
            program=self.program
        )

        return prompt
