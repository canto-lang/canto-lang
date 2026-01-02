"""
First-Order Logic Intermediate Representation for Canto.

The FOL IR serves as the canonical representation for Canto programs,
enabling formal verification via Z3 and multiple backend generation.

Architecture:
    DSL (AST) → FOL IR → Z3 (verification)
                      → Prolog (backend)
                      → ASP (future)
"""

from .types import (
    FOLSort,
    FOLVariable,
    FOLTerm,
    FOLConstant,
    FOLVar,
    FOLFunctionApp,
    FOLFormula,
    FOLPredicate,
    FOLEquals,
    FOLNot,
    FOLAnd,
    FOLOr,
    FOLImplies,
    FOLForall,
    FOLExists,
)

from .canto_fol import (
    RuleType,
    ValueType,
    CantoVariable,
    CantoCategory,
    CantoRule,
    CantoSuperiority,
    CantoFOL,
)

from .translator import ASTToFOLTranslator, translate_to_fol

from .z3_encoder import Z3Encoder, Z3Verifier

from .equivalence import EquivalenceVerifier, VerificationResult

from .extracted_logic import (
    ExtractedLogic,
    ExtractedRule,
    ExtractedCondition,
    ExtractedPrecedence,
)

from .prompt_extractor import PromptLogicExtractor

from .verified_builder import (
    VerifiedCantoBuilder,
    VerifiedBuildResult,
    StaticVerificationResult,
    verify_canto_file,
    verify_canto_string,
)

from .backends.prolog import PrologBackend

__all__ = [
    # Base FOL types
    "FOLSort",
    "FOLVariable",
    "FOLTerm",
    "FOLConstant",
    "FOLVar",
    "FOLFunctionApp",
    "FOLFormula",
    "FOLPredicate",
    "FOLEquals",
    "FOLNot",
    "FOLAnd",
    "FOLOr",
    "FOLImplies",
    "FOLForall",
    "FOLExists",
    # Canto FOL
    "RuleType",
    "ValueType",
    "CantoVariable",
    "CantoCategory",
    "CantoRule",
    "CantoSuperiority",
    "CantoFOL",
    # Translator
    "ASTToFOLTranslator",
    "translate_to_fol",
    # Z3 Encoder and Verifier
    "Z3Encoder",
    "Z3Verifier",
    # Equivalence Verification
    "EquivalenceVerifier",
    "VerificationResult",
    # Extracted Logic
    "ExtractedLogic",
    "ExtractedRule",
    "ExtractedCondition",
    "ExtractedPrecedence",
    "PromptLogicExtractor",
    # Verified Builder
    "VerifiedCantoBuilder",
    "VerifiedBuildResult",
    "StaticVerificationResult",
    "verify_canto_file",
    "verify_canto_string",
    # Backends
    "PrologBackend",
]
