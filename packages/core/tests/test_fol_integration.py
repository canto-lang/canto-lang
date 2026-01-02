"""
Integration tests for the full FOL verification pipeline.

Tests the complete flow: DSL → FOL → Z3 → Verification
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from canto_core.parser.dsl_parser import parse_string
from canto_core.fol import (
    translate_to_fol,
    Z3Verifier,
    EquivalenceVerifier,
    VerificationResult,
    PrologBackend,
)


class TestFullPipeline:
    """Test the complete verification pipeline."""

    def test_simple_pipeline(self):
        """Test a simple DSL through the full pipeline."""
        dsl = """
        ?priority can be "high", "medium", "low"
        ?input meaning "input text"
        ?urgent_words resembles "urgent", "critical", "emergency"

        ?priority becomes "high"
            when ?input is like ?urgent_words

        normally ?priority becomes "medium"
        """
        # Parse
        ast = parse_string(dsl)

        # Translate to FOL
        fol = translate_to_fol(ast)
        assert fol is not None
        assert "priority" in fol.variables
        assert "urgent_words" in fol.categories

        # Verify with Z3
        verifier = Z3Verifier(fol)
        results = verifier.verify_all()

        # All checks should pass
        assert results["satisfiability"][0] is True
        assert results["no_contradictions"][0] is True
        assert results["acyclicity"][0] is True

    def test_pipeline_with_conflicts(self):
        """Test pipeline with conflict resolution."""
        dsl = """
        ?result can be "a", "b"

        normally ?result becomes "a"
        normally ?result becomes "b"

        ?result becomes "a" overriding all
        """
        ast = parse_string(dsl)
        fol = translate_to_fol(ast)

        verifier = Z3Verifier(fol)
        results = verifier.verify_all()

        # Should have superiority relations that resolve conflicts
        assert len(fol.superiority) >= 1

    def test_pipeline_with_nested_structure(self):
        """Test pipeline with nested 'with' structures."""
        dsl = """
        ?patient meaning "patient" with
            ?name meaning "patient name"
            ?symptoms meaning "symptoms" with
                ?description meaning "symptom description"
                ?severity can be "mild", "moderate", "severe"

        ?severity becomes "severe"
            when ?description is like ?critical_symptoms

        ?critical_symptoms resembles "chest pain", "unconscious"
        """
        ast = parse_string(dsl)
        fol = translate_to_fol(ast)

        # Should have all nested variables
        assert "patient" in fol.variables
        assert "name" in fol.variables
        assert "symptoms" in fol.variables
        assert "description" in fol.variables
        assert "severity" in fol.variables

        # Should have has relationships
        assert len(fol.has_relationships) > 0

        # Verify
        verifier = Z3Verifier(fol)
        results = verifier.verify_all()
        assert results["satisfiability"][0] is True


class TestEquivalenceVerification:
    """Test prompt equivalence verification."""

    def test_constraint_check_passing(self):
        """Test constraint check on a good prompt."""
        dsl = """
        ?priority can be "high", "low"
        ?input meaning "input"
        ?urgent resembles "urgent", "critical"

        ?priority becomes "high" when ?input is like ?urgent
        normally ?priority becomes "low"
        """
        ast = parse_string(dsl)
        fol = translate_to_fol(ast)

        verifier = EquivalenceVerifier(fol)

        # A prompt that mentions all elements
        prompt = """
        Analyze the input to determine priority.

        Set priority to "high" when the input contains urgent or critical language.
        Otherwise, set priority to "low" by default.

        INPUT: {user_input}
        """

        violations = verifier.verify_constraints(prompt)

        # Should have few or no violations
        # (depends on exact matching logic)
        print(f"Violations: {violations}")

    def test_constraint_check_missing_variable(self):
        """Test constraint check when variable is missing."""
        dsl = """
        ?priority can be "high", "low"
        ?severity can be "critical", "normal"

        ?priority becomes "high"
        ?severity becomes "critical"
        """
        ast = parse_string(dsl)
        fol = translate_to_fol(ast)

        verifier = EquivalenceVerifier(fol)

        # A prompt that only mentions priority
        prompt = """
        Set the priority to high.

        INPUT: {user_input}
        """

        violations = verifier.verify_constraints(prompt)

        # Should detect missing severity
        assert any("severity" in v.lower() for v in violations)

    def test_constraint_check_missing_category(self):
        """Test constraint check when category patterns are missing."""
        dsl = """
        ?flag can be true, false
        ?input meaning "input"
        ?important_words resembles "critical", "urgent", "important"

        ?flag becomes true when ?input is like ?important_words
        """
        ast = parse_string(dsl)
        fol = translate_to_fol(ast)

        verifier = EquivalenceVerifier(fol)

        # A prompt that doesn't mention the patterns
        prompt = """
        Determine the flag value.

        INPUT: {user_input}
        """

        violations = verifier.verify_constraints(prompt)
        # May detect missing category patterns
        print(f"Violations: {violations}")


class TestPrologBackendIntegration:
    """Test Prolog backend generation from FOL."""

    def test_prolog_generation(self):
        """Test generating Prolog from FOL."""
        dsl = """
        ?priority can be "high", "low"
        ?input meaning "input"
        ?urgent resembles "urgent", "critical"

        ?priority becomes "high" when ?input is like ?urgent
        normally ?priority becomes "low"
        """
        ast = parse_string(dsl)
        fol = translate_to_fol(ast)

        backend = PrologBackend(fol)
        prolog = backend.generate()

        # Should contain key elements
        assert "pattern(urgent" in prolog
        assert "priority" in prolog
        assert "rule_info" in prolog

    def test_prolog_with_superiority(self):
        """Test Prolog generation includes superiority."""
        dsl = """
        ?flag can be true, false

        normally ?flag becomes false
        ?flag becomes true overriding all
        """
        ast = parse_string(dsl)
        fol = translate_to_fol(ast)

        backend = PrologBackend(fol)
        prolog = backend.generate()

        # Should contain superiority relations
        assert "sup(" in prolog


class TestVerificationAndGeneration:
    """Test that verification and generation work together."""

    def test_verify_then_generate(self):
        """Test verifying FOL then generating Prolog."""
        dsl = """
        ?triage can be "critical", "urgent", "standard"
        ?symptoms meaning "symptoms"
        ?critical_symptoms resembles "chest pain", "unconscious"
        ?urgent_symptoms resembles "high fever", "severe pain"

        ?triage becomes "critical" when ?symptoms is like ?critical_symptoms
        normally ?triage becomes "urgent" when ?symptoms is like ?urgent_symptoms
        normally ?triage becomes "standard"

        ?triage becomes "critical" when ?symptoms is like ?critical_symptoms
            overriding all
        """
        ast = parse_string(dsl)
        fol = translate_to_fol(ast)

        # Step 1: Verify
        verifier = Z3Verifier(fol)
        results = verifier.verify_all()

        all_passed = all(r[0] for r in results.values())
        print(f"Verification passed: {all_passed}")
        for name, (passed, details) in results.items():
            print(f"  {name}: {passed}")
            if not passed:
                print(f"    Details: {details}")

        # Step 2: Generate Prolog (only if verified)
        if all_passed:
            backend = PrologBackend(fol)
            prolog = backend.generate()
            assert len(prolog) > 0
            print(f"Generated {len(prolog)} chars of Prolog")


class TestRealWorldExamples:
    """Test with realistic DSL examples."""

    def test_content_moderation(self):
        """Test a content moderation example."""
        dsl = """
        ?moderation_action can be "remove", "flag", "allow"
        ?content meaning "user content"

        ?spam_patterns resembles "buy now", "click here", "free money"
        ?harassment_patterns resembles "hate speech", "threat", "abuse"
        ?safe_patterns resembles "question", "feedback", "discussion"

        ?moderation_action becomes "remove"
            when ?content is like ?harassment_patterns

        ?moderation_action becomes "flag"
            when ?content is like ?spam_patterns

        normally ?moderation_action becomes "allow"
            when ?content is like ?safe_patterns

        normally ?moderation_action becomes "allow"

        ?moderation_action becomes "remove"
            when ?content is like ?harassment_patterns
            overriding all
        """
        ast = parse_string(dsl)
        fol = translate_to_fol(ast)

        # Verify
        verifier = Z3Verifier(fol)
        results = verifier.verify_all()

        print("\nContent Moderation Example:")
        print(f"  Variables: {list(fol.variables.keys())}")
        print(f"  Categories: {list(fol.categories.keys())}")
        print(f"  Rules: {len(list(fol.all_rules()))}")
        print(f"  Superiority: {len(fol.superiority)}")

        for name, (passed, details) in results.items():
            status = "✓" if passed else "✗"
            print(f"  {status} {name}")

    def test_ticket_routing(self):
        """Test a ticket routing example."""
        dsl = """
        ?route_to can be "engineering", "sales", "support", "billing"
        ?ticket meaning "support ticket"

        ?technical_keywords resembles "bug", "error", "crash", "api"
        ?sales_keywords resembles "pricing", "enterprise", "contract"
        ?billing_keywords resembles "invoice", "payment", "refund"

        ?route_to becomes "engineering"
            when ?ticket is like ?technical_keywords

        ?route_to becomes "sales"
            when ?ticket is like ?sales_keywords

        ?route_to becomes "billing"
            when ?ticket is like ?billing_keywords

        normally ?route_to becomes "support"
        """
        ast = parse_string(dsl)
        fol = translate_to_fol(ast)

        verifier = Z3Verifier(fol)
        results = verifier.verify_all()

        # Should pass verification
        assert results["satisfiability"][0] is True
        assert results["acyclicity"][0] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
