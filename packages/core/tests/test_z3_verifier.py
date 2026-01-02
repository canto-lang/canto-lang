"""
Tests for Z3 encoder and static verifier.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from canto_core.parser.dsl_parser import parse_string
from canto_core.fol import (
    translate_to_fol,
    Z3Encoder,
    Z3Verifier,
    FOLPredicate,
    FOLEquals,
    FOLAnd,
    FOLOr,
    FOLNot,
    FOLImplies,
    FOLVar,
    FOLConstant,
    FOLFunctionApp,
    FOLSort,
)


class TestZ3Encoder:
    """Test Z3 encoding of FOL formulas."""

    def test_encoder_creation(self):
        """Test that encoder can be created."""
        encoder = Z3Encoder()
        assert encoder is not None
        assert encoder.InputSort is not None

    def test_encode_predicate(self):
        """Test encoding a predicate."""
        encoder = Z3Encoder()

        x = FOLVar("x")
        cat = FOLConstant("urgent", FOLSort.CATEGORY)
        pred = FOLPredicate("is_like", [x, cat])

        z3_formula = encoder.encode_formula(pred)
        assert z3_formula is not None

    def test_encode_equals(self):
        """Test encoding equality."""
        encoder = Z3Encoder()

        var_app = FOLFunctionApp("priority", [FOLVar("x")])
        val = FOLConstant("high", FOLSort.VALUE)
        eq = FOLEquals(var_app, val)

        z3_formula = encoder.encode_formula(eq)
        assert z3_formula is not None

    def test_encode_and(self):
        """Test encoding conjunction."""
        encoder = Z3Encoder()

        p1 = FOLPredicate("true", [])
        p2 = FOLPredicate("true", [])
        conj = FOLAnd([p1, p2])

        z3_formula = encoder.encode_formula(conj)
        assert z3_formula is not None

    def test_encode_or(self):
        """Test encoding disjunction."""
        encoder = Z3Encoder()

        p1 = FOLPredicate("true", [])
        p2 = FOLPredicate("false", [])
        disj = FOLOr([p1, p2])

        z3_formula = encoder.encode_formula(disj)
        assert z3_formula is not None

    def test_encode_not(self):
        """Test encoding negation."""
        encoder = Z3Encoder()

        p = FOLPredicate("true", [])
        neg = FOLNot(p)

        z3_formula = encoder.encode_formula(neg)
        assert z3_formula is not None

    def test_encode_implies(self):
        """Test encoding implication."""
        encoder = Z3Encoder()

        p1 = FOLPredicate("true", [])
        p2 = FOLPredicate("true", [])
        impl = FOLImplies(p1, p2)

        z3_formula = encoder.encode_formula(impl)
        assert z3_formula is not None

    def test_encode_complete_fol(self):
        """Test encoding a complete CantoFOL program."""
        dsl = """
        ?priority can be "high", "low"
        ?input meaning "input"
        ?urgent resembles "urgent"

        ?priority becomes "high" when ?input is like ?urgent
        """
        ast = parse_string(dsl)
        fol = translate_to_fol(ast)

        encoder = Z3Encoder()
        z3_formula = encoder.encode(fol)
        assert z3_formula is not None


class TestZ3Verifier:
    """Test Z3 static verification."""

    def test_verifier_creation(self):
        """Test that verifier can be created."""
        dsl = """?flag can be true, false"""
        ast = parse_string(dsl)
        fol = translate_to_fol(ast)

        verifier = Z3Verifier(fol)
        assert verifier is not None

    def test_check_satisfiability_simple(self):
        """Test satisfiability check on simple program."""
        dsl = """
        ?flag can be true, false
        ?flag becomes true
        """
        ast = parse_string(dsl)
        fol = translate_to_fol(ast)

        verifier = Z3Verifier(fol)
        is_sat, model = verifier.check_satisfiability()

        # Should be satisfiable
        assert is_sat is True

    def test_verify_no_contradictions_detects_potential_conflict(self):
        """Test that Z3 correctly detects potential contradictions.

        Two strict rules with different values for the same variable
        can both fire if their conditions are both satisfiable.
        Z3 correctly identifies this as a potential contradiction.
        """
        dsl = """
        ?priority can be "high", "low"
        ?input meaning "input"
        ?urgent resembles "urgent"
        ?normal resembles "normal"

        ?priority becomes "high" when ?input is like ?urgent
        ?priority becomes "low" when ?input is like ?normal
        """
        ast = parse_string(dsl)
        fol = translate_to_fol(ast)

        verifier = Z3Verifier(fol)
        no_contradictions, details = verifier.verify_no_contradictions()

        # Z3 correctly detects that both conditions could be satisfied
        assert no_contradictions is False
        assert details is not None
        assert details['type'] == 'contradiction'

    def test_verify_acyclicity_clean(self):
        """Test acyclicity on program without cycles."""
        dsl = """
        ?flag can be true, false

        normally ?flag becomes false
        ?flag becomes true overriding all
        """
        ast = parse_string(dsl)
        fol = translate_to_fol(ast)

        verifier = Z3Verifier(fol)
        is_acyclic, cycle = verifier.verify_acyclicity()

        # Should be acyclic
        assert is_acyclic is True
        assert cycle is None

    def test_verify_all(self):
        """Test running all verifications."""
        dsl = """
        ?priority can be "high", "low"
        ?input meaning "input"
        ?urgent resembles "urgent"

        ?priority becomes "high" when ?input is like ?urgent
        normally ?priority becomes "low"
        """
        ast = parse_string(dsl)
        fol = translate_to_fol(ast)

        verifier = Z3Verifier(fol)
        results = verifier.verify_all()

        assert "satisfiability" in results
        assert "no_contradictions" in results
        assert "acyclicity" in results
        assert "determinism" in results

    def test_verify_determinism_with_resolution(self):
        """Test determinism when conflicts are resolved by superiority."""
        dsl = """
        ?flag can be true, false

        normally ?flag becomes false
        ?flag becomes true overriding all
        """
        ast = parse_string(dsl)
        fol = translate_to_fol(ast)

        verifier = Z3Verifier(fol)
        is_deterministic, details = verifier.verify_determinism()

        # Should be deterministic (strict overrides defeasible)
        assert is_deterministic is True


class TestZ3VerifierEdgeCases:
    """Test edge cases in Z3 verification."""

    def test_empty_program(self):
        """Test verification of empty program."""
        dsl = """?flag can be true, false"""
        ast = parse_string(dsl)
        fol = translate_to_fol(ast)

        verifier = Z3Verifier(fol)
        results = verifier.verify_all()

        # Empty program should pass all checks
        assert all(r[0] for r in results.values())

    def test_single_rule(self):
        """Test verification of single rule."""
        dsl = """
        ?flag can be true, false
        ?flag becomes true
        """
        ast = parse_string(dsl)
        fol = translate_to_fol(ast)

        verifier = Z3Verifier(fol)
        results = verifier.verify_all()

        assert all(r[0] for r in results.values())

    def test_multiple_variables(self):
        """Test verification with multiple variables."""
        dsl = """
        ?a can be true, false
        ?b can be true, false
        ?c can be true, false

        ?a becomes true
        ?b becomes true when ?a is true
        ?c becomes true when ?b is true
        """
        ast = parse_string(dsl)
        fol = translate_to_fol(ast)

        verifier = Z3Verifier(fol)
        results = verifier.verify_all()

        assert results["satisfiability"][0] is True


class TestZ3ImplicationChecking:
    """Test implication checking for equivalence verification."""

    def test_check_implication_valid(self):
        """Test checking a valid implication."""
        dsl = """
        ?flag can be true, false
        ?flag becomes true
        """
        ast = parse_string(dsl)
        fol = translate_to_fol(ast)

        verifier = Z3Verifier(fol)

        # true → true should be valid
        p_true = FOLPredicate("true", [])
        is_valid, counter = verifier.check_implication(p_true, p_true)

        assert is_valid is True
        assert counter is None

    def test_check_implication_invalid(self):
        """Test checking an invalid implication."""
        dsl = """
        ?flag can be true, false
        ?flag becomes true
        """
        ast = parse_string(dsl)
        fol = translate_to_fol(ast)

        verifier = Z3Verifier(fol)

        # true → false should be invalid
        p_true = FOLPredicate("true", [])
        p_false = FOLPredicate("false", [])
        is_valid, counter = verifier.check_implication(p_true, p_false)

        assert is_valid is False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
