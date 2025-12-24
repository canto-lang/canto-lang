"""
Canto Core - Parser, codegen, and reasoning engine for Canto DSL
"""

from .parser import CantoParser, parse_file, parse_string, ParseResult
from .codegen import (
    translate_to_delp,
    DeLPProgram,
    DeLPTranslator,
    LLMPredicateGenerator,
    configure_dspy,
    PrologSyntaxVerifier,
    verify_prolog_code,
    PromptTemplateGenerator,
    CantoPDO,
    PDOConfig,
    format_reasoning_context,
)

# Optional DeLP imports (requires janus-swi)
try:
    from .delp import JanusDeLP, create_janus_engine, DeLPReasoningAnalyzer, ReasoningPattern
except ImportError:
    pass

__all__ = [
    # Parser
    'CantoParser',
    'parse_file',
    'parse_string',
    'ParseResult',
    # Codegen
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
    # DeLP
    'JanusDeLP',
    'create_janus_engine',
    'DeLPReasoningAnalyzer',
    'ReasoningPattern',
]
