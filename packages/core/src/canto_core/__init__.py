"""
Canto Core - Parser, codegen, and reasoning engine for Canto DSL
"""

from .parser import CantoParser, parse_file, parse_string, ParseResult
from .concept import Concept
from .builder import CantoBuilder, BuildResult, ResolutionErrors
from .symbols import (
    Symbol,
    SymbolTable,
    SymbolKind,
    ResolutionError,
    UnresolvedReferenceError,
    DuplicateDeclarationError,
    build_symbol_table,
    collect_declarations,
    collect_references,
)
from .prompt import PromptGenerator
from .codegen import (
    translate_to_delp,
    DeLPProgram,
    LLMPredicateGenerator,
    configure_dspy,
    PrologSyntaxVerifier,
    verify_prolog_code,
    PromptTemplateGenerator,
    CantoPDO,
    PDOConfig,
    format_reasoning_context,
)
from .fol import translate_to_fol, CantoFOL

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
    # Concepts & Builder
    'Concept',
    'CantoBuilder',
    'BuildResult',
    'ResolutionErrors',
    'PromptGenerator',
    # Symbols
    'Symbol',
    'SymbolTable',
    'SymbolKind',
    'ResolutionError',
    'UnresolvedReferenceError',
    'DuplicateDeclarationError',
    'build_symbol_table',
    'collect_declarations',
    'collect_references',
    # Codegen
    'translate_to_delp',
    'DeLPProgram',
    'LLMPredicateGenerator',
    # FOL
    'translate_to_fol',
    'CantoFOL',
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
