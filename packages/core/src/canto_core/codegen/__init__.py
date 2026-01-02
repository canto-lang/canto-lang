"""
Code generation module for Canto DSL

This module handles the compilation of .canto DSL files into Prolog knowledge bases
through a mixed approach:
1. FOL translation via translate_to_fol (../fol/translator.py)
2. LLM-based predicate generation (llm_predicates.py)
3. Prompt optimization via PDO (pdo/)

For DSL to Prolog translation, use translate_to_fol() from canto_core.fol.
"""

from ..fol import translate_to_fol
from ..delp.program import DeLPProgram
from .llm_predicates import LLMPredicateGenerator, configure_dspy
from .prolog_verifier import PrologSyntaxVerifier, verify_prolog_code
from .prompt_generator import PromptTemplateGenerator
from .pdo import CantoPDO, PDOConfig, format_reasoning_context

# Backwards compatibility alias
translate_to_delp = translate_to_fol

__all__ = [
    'translate_to_fol',
    'translate_to_delp',  # Deprecated alias
    'DeLPProgram',
    'LLMPredicateGenerator',
    'configure_dspy',
    'PrologSyntaxVerifier',
    'verify_prolog_code',
    'PromptTemplateGenerator',
    'CantoPDO',
    'PDOConfig',
    'format_reasoning_context',
]
