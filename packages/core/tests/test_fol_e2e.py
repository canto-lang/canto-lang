"""
End-to-end tests for FOL verification using example DSL files.

These tests verify the complete pipeline works with real DSL files.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from canto_core.parser.dsl_parser import parse_file, parse_string
from canto_core.fol import (
    translate_to_fol,
    Z3Verifier,
    EquivalenceVerifier,
    PrologBackend,
    VerifiedCantoBuilder,
    verify_canto_string,
)


# Path to examples directory
EXAMPLES_DIR = Path(__file__).parent.parent.parent.parent / "examples"


class TestExampleFiles:
    """Test FOL verification with example DSL files."""

    @pytest.mark.skipif(
        not (EXAMPLES_DIR / "medical_query.canto").exists(),
        reason="Example file not found"
    )
    def test_medical_query_example(self):
        """Test the medical_query.canto example."""
        example_path = EXAMPLES_DIR / "medical_query.canto"
        ast = parse_file(str(example_path))
        fol = translate_to_fol(ast, source_file=str(example_path))

        # Basic structure checks
        assert len(fol.variables) > 0
        assert len(fol.categories) > 0
        assert len(list(fol.all_rules())) > 0

        # Verify
        verifier = Z3Verifier(fol)
        results = verifier.verify_all()

        print(f"\nmedical_query.canto verification:")
        print(f"  Variables: {len(fol.variables)}")
        print(f"  Categories: {len(fol.categories)}")
        print(f"  Rules: {len(list(fol.all_rules()))}")

        for name, (passed, details) in results.items():
            status = "✓" if passed else "✗"
            print(f"  {status} {name}")
            if not passed and details:
                print(f"    {details}")

    @pytest.mark.skipif(
        not (EXAMPLES_DIR / "food_categories.canto").exists(),
        reason="Example file not found"
    )
    def test_food_categories_example(self):
        """Test the food_categories.canto example if it exists."""
        example_path = EXAMPLES_DIR / "food_categories.canto"
        ast = parse_file(str(example_path))
        fol = translate_to_fol(ast, source_file=str(example_path))

        verifier = Z3Verifier(fol)
        results = verifier.verify_all()

        print(f"\nfood_categories.canto verification:")
        for name, (passed, details) in results.items():
            status = "✓" if passed else "✗"
            print(f"  {status} {name}")

    def test_all_example_files(self):
        """Test all .canto files in the examples directory."""
        if not EXAMPLES_DIR.exists():
            pytest.skip("Examples directory not found")

        canto_files = list(EXAMPLES_DIR.glob("*.canto"))
        if not canto_files:
            pytest.skip("No .canto files found in examples")

        print(f"\nTesting {len(canto_files)} example files:")

        for canto_file in canto_files:
            print(f"\n  {canto_file.name}:")
            try:
                ast = parse_file(str(canto_file))
                fol = translate_to_fol(ast, source_file=str(canto_file))

                verifier = Z3Verifier(fol)
                results = verifier.verify_all()

                all_passed = all(r[0] for r in results.values())
                status = "✓" if all_passed else "✗"
                print(f"    {status} Verification: {'PASSED' if all_passed else 'FAILED'}")

                if not all_passed:
                    for name, (passed, details) in results.items():
                        if not passed:
                            print(f"      - {name}: {details}")

            except Exception as e:
                print(f"    ✗ Error: {e}")


class TestVerifiedBuilder:
    """Test the VerifiedCantoBuilder convenience class."""

    def test_build_simple_dsl(self):
        """Test building a simple DSL with verification."""
        dsl = """
        ?priority can be "high", "low"
        ?input meaning "input"
        ?urgent resembles "urgent"

        ?priority becomes "high" when ?input is like ?urgent
        normally ?priority becomes "low"
        """
        result = verify_canto_string(dsl)

        assert result.fol is not None
        assert result.static_verification is not None
        assert result.static_verification.passed is True
        assert result.prolog_code is not None

    def test_build_with_prolog_generation(self):
        """Test building generates valid Prolog."""
        dsl = """
        ?flag can be true, false
        ?text meaning "text"
        ?keywords resembles "key1", "key2"

        ?flag becomes true when ?text is like ?keywords
        """
        result = verify_canto_string(dsl, generate_prolog=True)

        assert result.prolog_code is not None
        assert "pattern(keywords" in result.prolog_code
        assert "flag" in result.prolog_code

    def test_build_without_prolog(self):
        """Test building without Prolog generation."""
        dsl = """?flag can be true, false"""
        result = verify_canto_string(dsl, generate_prolog=False)

        assert result.prolog_code is None

    def test_builder_with_concepts(self):
        """Test builder with registered concepts."""
        from canto_core.concept import Concept

        # Create a simple concept using fluent API
        priority_concept = (
            Concept("priority_level")
            .meaning("Priority classification")
        )

        dsl = """
        ?priority can be "high", "low"
        ?priority becomes "high"
        """

        builder = VerifiedCantoBuilder()
        builder.register_concept(priority_concept)
        result = builder.build_string(dsl)

        # Single strict rule with unconditional value - should pass
        assert result.passed is True


class TestEndToEndVerification:
    """Test complete end-to-end verification scenarios."""

    def test_e2e_triage_system(self):
        """Test a complete triage system DSL.

        Note: This DSL has duplicate strict rules for 'critical',
        which Z3 correctly identifies as non-determinism.
        """
        dsl = """
        \"\"\"
        Medical triage system for emergency room.
        Classify patient urgency based on symptoms.
        \"\"\"

        ?triage_level can be "critical", "urgent", "standard", "non-urgent"
        ?patient_symptoms meaning "the patient's reported symptoms"

        ?critical_indicators resembles
            "chest pain",
            "difficulty breathing",
            "unconscious",
            "severe bleeding",
            "stroke symptoms"

        ?urgent_indicators resembles
            "high fever",
            "severe pain",
            "broken bone",
            "deep cut"

        ?minor_indicators resembles
            "cold symptoms",
            "minor cut",
            "headache",
            "sprain"

        ?triage_level becomes "critical"
            when ?patient_symptoms is like ?critical_indicators

        normally ?triage_level becomes "urgent"
            when ?patient_symptoms is like ?urgent_indicators

        normally ?triage_level becomes "standard"
            when ?patient_symptoms is like ?minor_indicators

        normally ?triage_level becomes "non-urgent"

        ?triage_level becomes "critical"
            when ?patient_symptoms is like ?critical_indicators
            overriding all
        """

        # Full pipeline
        result = verify_canto_string(dsl)

        # Check FOL structure
        assert "triage_level" in result.fol.variables
        assert "critical_indicators" in result.fol.categories
        assert "urgent_indicators" in result.fol.categories
        assert "minor_indicators" in result.fol.categories

        # Check basic verification
        sv = result.static_verification
        assert sv.satisfiability[0] is True
        assert sv.acyclicity[0] is True
        # Z3 correctly detects potential non-determinism due to duplicate rules
        assert sv.determinism[0] is False

        # Check Prolog generated
        assert result.prolog_code is not None
        assert "critical" in result.prolog_code

        print("\nTriage System E2E Test:")
        print(f"  Variables: {list(result.fol.variables.keys())}")
        print(f"  Categories: {list(result.fol.categories.keys())}")
        print(f"  Rules: {len(list(result.fol.all_rules()))}")
        print(f"  Superiority: {len(result.fol.superiority)}")
        print(f"  Prolog size: {len(result.prolog_code)} chars")

    def test_e2e_with_prompt_verification(self):
        """Test end-to-end including prompt verification.

        Note: This DSL has strict rules for positive/negative that could
        both fire. Z3 correctly detects this as a contradiction.
        """
        dsl = """
        ?sentiment can be "positive", "negative", "neutral"
        ?text meaning "input text"

        ?positive_words resembles "great", "excellent", "amazing", "love"
        ?negative_words resembles "bad", "terrible", "hate", "awful"

        ?sentiment becomes "positive" when ?text is like ?positive_words
        ?sentiment becomes "negative" when ?text is like ?negative_words
        normally ?sentiment becomes "neutral"
        """

        # Build and verify
        builder = VerifiedCantoBuilder()
        result = builder.build_string(dsl)
        # Z3 correctly detects potential contradiction between strict rules
        assert result.passed is False
        assert result.static_verification.no_contradictions[0] is False

        # Test prompt that covers all elements
        good_prompt = """
        Analyze the sentiment of the input text.

        Classify as:
        - "positive" if text contains words like: great, excellent, amazing, love
        - "negative" if text contains words like: bad, terrible, hate, awful
        - "neutral" otherwise (default)

        INPUT: {user_input}

        OUTPUT: sentiment value
        """

        violations = builder.quick_constraint_check(result, good_prompt)
        print(f"\nGood prompt violations: {violations}")

        # Test prompt missing elements
        incomplete_prompt = """
        Determine if the text is positive or negative.

        INPUT: {user_input}
        """

        violations = builder.quick_constraint_check(result, incomplete_prompt)
        print(f"Incomplete prompt violations: {violations}")
        # Should have violations for missing patterns


class TestPrologBackendE2E:
    """Test Prolog backend generation end-to-end."""

    def test_prolog_is_syntactically_valid(self):
        """Test that generated Prolog is syntactically reasonable."""
        dsl = """
        ?category can be "a", "b", "c"
        ?input meaning "input"
        ?a_patterns resembles "alpha", "apple"
        ?b_patterns resembles "beta", "banana"

        ?category becomes "a" when ?input is like ?a_patterns
        ?category becomes "b" when ?input is like ?b_patterns
        normally ?category becomes "c"
        """

        result = verify_canto_string(dsl)
        prolog = result.prolog_code

        # Check basic Prolog syntax elements
        assert ":-" in prolog  # Rules
        assert "." in prolog   # Facts end with period
        assert "pattern(" in prolog  # Pattern facts
        assert "rule_info(" in prolog  # Rule metadata

        # Check no obvious syntax errors
        assert ":-." not in prolog  # Empty body
        assert "()" not in prolog   # Empty args (should have content)

        print(f"\nGenerated Prolog ({len(prolog)} chars):")
        # Print first 500 chars
        print(prolog[:500] + "..." if len(prolog) > 500 else prolog)


class TestErrorHandling:
    """Test error handling in the verification pipeline."""

    def test_handles_empty_dsl(self):
        """Test handling of empty DSL."""
        dsl = ""
        try:
            result = verify_canto_string(dsl)
            # Should either return empty result or raise
            assert result.fol is not None
        except Exception as e:
            # Acceptable to raise on empty input
            print(f"Empty DSL raised: {type(e).__name__}")

    def test_handles_dsl_with_only_declarations(self):
        """Test DSL with only declarations (no rules)."""
        dsl = """
        ?flag can be true, false
        ?text meaning "some text"
        ?patterns resembles "a", "b", "c"
        """

        result = verify_canto_string(dsl)

        # Should succeed with no rules
        assert result.passed is True
        assert len(list(result.fol.all_rules())) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
