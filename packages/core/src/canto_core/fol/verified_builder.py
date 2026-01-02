"""
Verified Builder - Complete FOL verification pipeline for Canto.

Orchestrates the full verification flow:
  DSL → FOL → Z3 verification → Backend generation

Provides both static verification (at compile time) and
equivalence verification (comparing generated prompts to DSL).
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path

from ..parser import CantoParser
from ..builder import CantoBuilder, BuildResult
from ..concept import Concept

from .canto_fol import CantoFOL
from .translator import ASTToFOLTranslator
from .z3_encoder import Z3Encoder, Z3Verifier
from .equivalence import EquivalenceVerifier, VerificationResult
from .backends.prolog import PrologBackend


@dataclass
class StaticVerificationResult:
    """Result of static verification checks."""

    passed: bool
    satisfiability: tuple[bool, Optional[Any]]
    no_contradictions: tuple[bool, Optional[Dict]]
    acyclicity: tuple[bool, Optional[List[str]]]
    determinism: tuple[bool, Optional[Dict]]

    @property
    def summary(self) -> str:
        """Get a summary of verification results."""
        if self.passed:
            return "All static verification checks passed"

        issues = []
        if not self.satisfiability[0]:
            issues.append("Program is unsatisfiable")
        if not self.no_contradictions[0]:
            issues.append(f"Contradiction found: {self.no_contradictions[1]}")
        if not self.acyclicity[0]:
            issues.append(f"Cycle in superiority: {self.acyclicity[1]}")
        if not self.determinism[0]:
            issues.append(f"Non-determinism: {self.determinism[1]}")

        return "; ".join(issues)


@dataclass
class VerifiedBuildResult:
    """
    Result of building with full FOL verification.

    Contains the original build result plus FOL IR and verification results.
    """

    # Original build result
    build: BuildResult

    # FOL intermediate representation
    fol: CantoFOL

    # Static verification
    static_verification: StaticVerificationResult

    # Generated backends
    prolog_code: Optional[str] = None

    # Prompt equivalence (populated by verify_prompt)
    prompt_verification: Optional[VerificationResult] = None

    @property
    def passed(self) -> bool:
        """Check if all verifications passed."""
        return self.static_verification.passed

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "source_file": self.fol.source_file,
            "static_verification": {
                "passed": self.static_verification.passed,
                "summary": self.static_verification.summary,
            },
            "fol_summary": {
                "variables": list(self.fol.variables.keys()),
                "categories": list(self.fol.categories.keys()),
                "rules_count": len(list(self.fol.all_rules())),
                "superiority_count": len(self.fol.superiority),
            },
            "prompt_verification": (
                {
                    "equivalent": self.prompt_verification.equivalent,
                    "message": self.prompt_verification.message,
                }
                if self.prompt_verification
                else None
            ),
        }


class VerifiedCantoBuilder:
    """
    Builder with full FOL verification pipeline.

    Extends the standard CantoBuilder with:
    - FOL IR translation
    - Z3 static verification
    - Backend generation from verified FOL
    - Prompt equivalence verification

    Example:
        builder = VerifiedCantoBuilder()
        result = builder.build("program.canto")

        if result.passed:
            print("All verifications passed!")
            print(result.prolog_code)
        else:
            print(f"Issues: {result.static_verification.summary}")
    """

    def __init__(self, strict: bool = True):
        """
        Initialize verified builder.

        Args:
            strict: If True, raise on verification failures
        """
        self._base_builder = CantoBuilder(strict=strict)
        self._strict = strict

    def register_concept(self, concept: Concept) -> "VerifiedCantoBuilder":
        """Register a concept for build-time resolution."""
        self._base_builder.register_concept(concept)
        return self

    def build(
        self,
        canto_file: str,
        generate_prolog: bool = True,
        skip_verification: bool = False,
    ) -> VerifiedBuildResult:
        """
        Build with full verification pipeline.

        Args:
            canto_file: Path to .canto file
            generate_prolog: Whether to generate Prolog backend
            skip_verification: Skip Z3 verification (for speed)

        Returns:
            VerifiedBuildResult with FOL IR and verification results
        """
        # Step 1: Standard build
        build_result = self._base_builder.build(canto_file, validate=True)

        # Step 2: Translate to FOL
        translator = ASTToFOLTranslator()
        fol = translator.translate(build_result.ast, source_file=canto_file)

        # Step 3: Static verification
        if skip_verification:
            static_result = StaticVerificationResult(
                passed=True,
                satisfiability=(True, None),
                no_contradictions=(True, None),
                acyclicity=(True, None),
                determinism=(True, None),
            )
        else:
            static_result = self._run_static_verification(fol)

        # Step 4: Generate backends
        prolog_code = None
        if generate_prolog:
            backend = PrologBackend(fol)
            prolog_code = backend.generate()

        return VerifiedBuildResult(
            build=build_result,
            fol=fol,
            static_verification=static_result,
            prolog_code=prolog_code,
        )

    def build_string(
        self,
        canto_source: str,
        generate_prolog: bool = True,
        skip_verification: bool = False,
    ) -> VerifiedBuildResult:
        """
        Build from string with full verification pipeline.

        Args:
            canto_source: Canto source code
            generate_prolog: Whether to generate Prolog backend
            skip_verification: Skip Z3 verification (for speed)

        Returns:
            VerifiedBuildResult with FOL IR and verification results
        """
        # Step 1: Standard build
        build_result = self._base_builder.build_string(canto_source, validate=True)

        # Step 2: Translate to FOL
        translator = ASTToFOLTranslator()
        fol = translator.translate(build_result.ast, source_file="<string>")

        # Step 3: Static verification
        if skip_verification:
            static_result = StaticVerificationResult(
                passed=True,
                satisfiability=(True, None),
                no_contradictions=(True, None),
                acyclicity=(True, None),
                determinism=(True, None),
            )
        else:
            static_result = self._run_static_verification(fol)

        # Step 4: Generate backends
        prolog_code = None
        if generate_prolog:
            backend = PrologBackend(fol)
            prolog_code = backend.generate()

        return VerifiedBuildResult(
            build=build_result,
            fol=fol,
            static_verification=static_result,
            prolog_code=prolog_code,
        )

    def verify_prompt(
        self,
        result: VerifiedBuildResult,
        prompt: str,
    ) -> VerificationResult:
        """
        Verify a generated prompt against the DSL.

        Uses LLM extraction + Z3 to check equivalence.

        Args:
            result: The build result containing FOL IR
            prompt: The generated prompt to verify

        Returns:
            VerificationResult with equivalence analysis
        """
        verifier = EquivalenceVerifier(result.fol)
        verification = verifier.verify(prompt)

        # Store in result
        result.prompt_verification = verification

        return verification

    def quick_constraint_check(
        self,
        result: VerifiedBuildResult,
        prompt: str,
    ) -> List[str]:
        """
        Quick constraint check without full Z3 verification.

        Checks that all DSL elements are mentioned in the prompt.

        Args:
            result: The build result containing FOL IR
            prompt: The generated prompt to check

        Returns:
            List of constraint violations (empty if all pass)
        """
        verifier = EquivalenceVerifier(result.fol)
        return verifier.verify_constraints(prompt)

    def _run_static_verification(self, fol: CantoFOL) -> StaticVerificationResult:
        """Run all static verification checks."""
        verifier = Z3Verifier(fol)
        results = verifier.verify_all()

        passed = all(r[0] for r in results.values())

        return StaticVerificationResult(
            passed=passed,
            satisfiability=results["satisfiability"],
            no_contradictions=results["no_contradictions"],
            acyclicity=results["acyclicity"],
            determinism=results["determinism"],
        )


def verify_canto_file(
    canto_file: str,
    generate_prolog: bool = True,
) -> VerifiedBuildResult:
    """
    Convenience function to verify a Canto file.

    Args:
        canto_file: Path to .canto file
        generate_prolog: Whether to generate Prolog

    Returns:
        VerifiedBuildResult
    """
    builder = VerifiedCantoBuilder()
    return builder.build(canto_file, generate_prolog=generate_prolog)


def verify_canto_string(
    canto_source: str,
    generate_prolog: bool = True,
) -> VerifiedBuildResult:
    """
    Convenience function to verify Canto source code.

    Args:
        canto_source: Canto source code
        generate_prolog: Whether to generate Prolog

    Returns:
        VerifiedBuildResult
    """
    builder = VerifiedCantoBuilder()
    return builder.build_string(canto_source, generate_prolog=generate_prolog)
