"""
Equivalence Verifier.

Verifies that a generated prompt is semantically equivalent to its source DSL.
Uses Z3 to check bidirectional implication between DSL_FOL and Prompt_FOL.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

from z3 import Solver, Not, sat, unsat

from .canto_fol import CantoFOL
from .z3_encoder import Z3Encoder
from .prompt_extractor import PromptLogicExtractor
from .extracted_logic import ExtractedLogic
from .types import make_and


@dataclass
class VerificationResult:
    """Result of DSL ↔ Prompt verification."""

    # Overall result
    equivalent: bool

    # Bidirectional checks
    dsl_implies_prompt: bool  # DSL ⊆ Prompt
    prompt_implies_dsl: bool  # Prompt ⊆ DSL

    # Detailed issues
    counterexamples: List[Dict[str, Any]] = field(default_factory=list)
    missing_in_prompt: List[str] = field(default_factory=list)
    extra_in_prompt: List[str] = field(default_factory=list)

    # Summary
    message: str = ""

    def __repr__(self):
        status = "EQUIVALENT" if self.equivalent else "NOT EQUIVALENT"
        return f"VerificationResult({status}: {self.message})"


class EquivalenceVerifier:
    """
    Verifies that a generated prompt is equivalent to its source DSL.

    Verification strategy:
    1. Extract logic from prompt using LLM → Prompt_FOL
    2. Encode DSL to FOL → DSL_FOL
    3. Check DSL_FOL → Prompt_FOL (prompt covers all DSL behaviors)
    4. Check Prompt_FOL → DSL_FOL (prompt doesn't add behaviors)
    5. Report counterexamples if verification fails
    """

    def __init__(self, fol: CantoFOL):
        self.fol = fol
        self.encoder = Z3Encoder()
        self.extractor = PromptLogicExtractor()

        # Pre-compute DSL Z3 formula
        self._dsl_z3 = self.encoder.encode(fol)

    def verify(self, prompt: str, debug: bool = False) -> VerificationResult:
        """
        Verify prompt against DSL.

        Args:
            prompt: The natural language prompt to verify
            debug: If True, print extracted logic for debugging

        Returns:
            VerificationResult with detailed verification info
        """
        counterexamples = []
        missing = []
        extra = []

        # Step 1: Extract logic from prompt
        extracted = self.extractor.extract(prompt, self.fol)

        if debug:
            print(f"\n[EXTRACTED LOGIC] {extracted}")
            print(f"  Rules ({len(extracted.rules)}):")
            for i, rule in enumerate(extracted.rules):
                print(f"    [{i}] {rule.variable} = {repr(rule.value)} (type: {type(rule.value).__name__})")
                print(f"        conditions: {[c.to_dict() for c in rule.conditions]}")
                print(f"        is_default: {rule.is_default}")
            print(f"  Mentioned vars: {extracted.mentioned_variables}")
            print(f"  Mentioned cats: {extracted.mentioned_categories}")

        # Step 2: Convert extracted logic to FOL
        prompt_formulas = extracted.to_fol_formulas()

        if debug:
            print(f"  FOL formulas ({len(prompt_formulas)}):")
            for i, f in enumerate(prompt_formulas):
                print(f"    [{i}] {f}")

        # Step 3: Encode prompt FOL to Z3
        if prompt_formulas:
            prompt_z3 = self.encoder.encode_formula(make_and(prompt_formulas))
        else:
            prompt_z3 = None

        # Step 4a: Check DSL → Prompt
        dsl_implies_prompt, ce1 = self._check_implication(
            self._dsl_z3,
            prompt_z3,
            "DSL → Prompt"
        )
        if ce1:
            counterexamples.append(ce1)

        # Step 4b: Check Prompt → DSL
        prompt_implies_dsl, ce2 = self._check_implication(
            prompt_z3,
            self._dsl_z3,
            "Prompt → DSL"
        )
        if ce2:
            counterexamples.append(ce2)

        # Step 5: Detailed comparison
        missing.extend(self._find_missing(extracted))
        extra.extend(self._find_extra(extracted))

        # Build result
        equivalent = (
            dsl_implies_prompt and
            prompt_implies_dsl and
            not missing and
            not extra
        )

        if equivalent:
            message = "Prompt is semantically equivalent to DSL"
        else:
            issues = []
            if not dsl_implies_prompt:
                issues.append("Prompt may miss DSL behaviors")
            if not prompt_implies_dsl:
                issues.append("Prompt may add extra behaviors")
            if missing:
                issues.append(f"{len(missing)} missing elements")
            if extra:
                issues.append(f"{len(extra)} extra elements")
            message = "; ".join(issues)

        return VerificationResult(
            equivalent=equivalent,
            dsl_implies_prompt=dsl_implies_prompt,
            prompt_implies_dsl=prompt_implies_dsl,
            counterexamples=counterexamples,
            missing_in_prompt=missing,
            extra_in_prompt=extra,
            message=message
        )

    def _check_implication(
        self,
        antecedent,
        consequent,
        name: str
    ) -> Tuple[bool, Optional[Dict]]:
        """
        Check if antecedent → consequent is valid.

        Valid iff there's no input where antecedent holds but consequent doesn't.
        """
        s = Solver()

        if antecedent is not None:
            s.add(antecedent)
        if consequent is not None:
            s.add(Not(consequent))
        else:
            # No consequent means anything goes
            return (True, None)

        if s.check() == unsat:
            return (True, None)
        else:
            return (False, {
                'check': name,
                'counterexample': str(s.model())
            })

    def _find_missing(self, extracted: ExtractedLogic) -> List[str]:
        """Find DSL elements not represented in extracted prompt logic."""
        missing = []

        # Check each DSL variable is mentioned
        extracted_vars = set(extracted.mentioned_variables)
        extracted_vars.update(r.variable for r in extracted.rules)

        for var_name in self.fol.variables:
            # Skip variables with no rules
            if not self.fol.get_rules_for_variable(var_name):
                continue
            if var_name not in extracted_vars:
                missing.append(f"Variable '{var_name}' not found in prompt")

        # Check categories are mentioned
        extracted_cats = set(extracted.mentioned_categories)
        for rule in extracted.rules:
            for cond in rule.conditions:
                if cond.type == "is_like" and cond.args:
                    extracted_cats.add(cond.args[-1])

        for cat_name in self.fol.categories:
            # Check if category is used in any rule
            used_in_rule = False
            for rule in self.fol.all_rules():
                if rule.conditions and cat_name in str(rule.conditions):
                    used_in_rule = True
                    break

            if used_in_rule and cat_name not in extracted_cats:
                missing.append(f"Category '{cat_name}' not found in prompt")

        return missing

    def _find_extra(self, extracted: ExtractedLogic) -> List[str]:
        """Find extracted elements not in DSL."""
        extra = []

        dsl_vars = set(self.fol.variables.keys())
        dsl_cats = set(self.fol.categories.keys())

        # Check for unknown variables
        for rule in extracted.rules:
            if rule.variable and rule.variable not in dsl_vars:
                extra.append(f"Unknown variable '{rule.variable}' in prompt")

        # Check for unknown categories
        for rule in extracted.rules:
            for cond in rule.conditions:
                if cond.type == "is_like" and cond.args:
                    cat = cond.args[-1]
                    if cat not in dsl_cats:
                        extra.append(f"Unknown category '{cat}' in prompt")

        return extra

    def verify_constraints(self, prompt: str) -> List[str]:
        """
        Quick verification of structural constraints only.

        Checks that all DSL variables and categories are mentioned
        in the prompt without full Z3 verification.

        Returns:
            List of constraint violations (empty if all pass)
        """
        violations = []
        prompt_lower = prompt.lower()

        # Check variables
        for var_name in self.fol.variables:
            if not self.fol.get_rules_for_variable(var_name):
                continue

            variations = [
                var_name.lower(),
                var_name.replace("_", " ").lower(),
                var_name.replace("_", "-").lower(),
            ]
            if not any(v in prompt_lower for v in variations):
                violations.append(f"Variable '{var_name}' not mentioned in prompt")

        # Check categories used in rules
        for cat_name, cat in self.fol.categories.items():
            # Check if any pattern is mentioned
            patterns_found = sum(
                1 for p in cat.patterns
                if p.lower() in prompt_lower
            )
            if patterns_found == 0:
                # Check category name itself
                if cat_name.lower() not in prompt_lower:
                    violations.append(
                        f"Category '{cat_name}' (no patterns found) not in prompt"
                    )

        # Check superiority relations are reflected
        for sup in self.fol.superiority:
            superior = self.fol.get_rule(sup.superior)
            inferior = self.fol.get_rule(sup.inferior)

            if superior and inferior:
                # Look for precedence language
                precedence_terms = [
                    "takes precedence",
                    "overrides",
                    "even if",
                    "regardless",
                    "important",
                    "always",
                ]
                has_precedence = any(t in prompt_lower for t in precedence_terms)

                if not has_precedence:
                    violations.append(
                        f"Override {sup.superior} > {sup.inferior} may not be explicit"
                    )

        return violations
