"""
Code generation module for Canto DSL

This module handles the compilation of .canto DSL files into Prolog knowledge bases
through a mixed approach:
1. Deterministic translation (translator.py)
2. LLM-based predicate generation (llm_predicates.py)
3. Prompt optimization via PDO (pdo/)
"""

from .translator import translate_to_delp, DeLPTranslator
from ..delp.program import DeLPProgram
from .llm_predicates import LLMPredicateGenerator, configure_dspy
from .prolog_verifier import PrologSyntaxVerifier, verify_prolog_code
from .prompt_generator import PromptTemplateGenerator
from .pdo import CantoPDO, PDOConfig, format_reasoning_context

__all__ = [
    'translate_to_delp',
    'DeLPProgram',
    'DeLPTranslator',
    'LLMPredicateGenerator',
    'configure_dspy',
    'PrologSyntaxVerifier',
    'verify_prolog_code',
    'PromptTemplateGenerator',
    'CantoPDO',
    'PDOConfig',
    'format_reasoning_context',
]
