"""
Tests for Canto DSL semantic analyzer
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from canto_core.parser import parse_string
from canto_core.parser.semantic_analyzer import analyze, SemanticAnalyzer, PredicateKind


class TestUndefinedPredicates:
    """Tests for detecting undefined predicates"""

    def test_undefined_predicate_in_matches_left(self):
        """Test that undefined predicate on left side of is like is detected"""
        dsl = """
        ?fraud_risk resembles "identity theft", "suspicious activity"

        ?loan_approved becomes false
            when ?applicant_notes is like ?fraud_risk
        """
        ast = parse_string(dsl)
        result = analyze(ast)

        assert not result.is_valid
        assert len(result.errors) == 1
        assert result.errors[0].predicate == "applicant_notes"
        assert "Undefined predicate" in result.errors[0].message

    def test_undefined_predicate_in_matches_right(self):
        """Test that undefined category on right side of is like is detected"""
        dsl = """
        ?applicant_notes meaning "notes from application"

        ?loan_approved becomes false
            when ?applicant_notes is like ?fraud_risk
        """
        ast = parse_string(dsl)
        result = analyze(ast)

        assert not result.is_valid
        assert len(result.errors) == 1
        assert result.errors[0].predicate == "fraud_risk"

    def test_undefined_predicate_in_is_condition(self):
        """Test that undefined predicate in is condition is detected"""
        dsl = """
        ?result becomes true
            when ?undefined_var is "value"
        """
        ast = parse_string(dsl)
        result = analyze(ast)

        assert not result.is_valid
        assert len(result.errors) == 1
        assert result.errors[0].predicate == "undefined_var"

    def test_multiple_undefined_predicates(self):
        """Test that multiple undefined predicates are all detected"""
        dsl = """
        ?result becomes true
            when ?undefined1 is "a" and ?undefined2 is like ?undefined3
        """
        ast = parse_string(dsl)
        result = analyze(ast)

        assert not result.is_valid
        assert len(result.errors) == 3
        predicates = {e.predicate for e in result.errors}
        assert predicates == {"undefined1", "undefined2", "undefined3"}


class TestValidPrograms:
    """Tests for valid programs that should pass semantic analysis"""

    def test_all_predicates_declared(self):
        """Test that a valid program passes analysis"""
        dsl = """
        ?credit_score can be "high", "medium", "low"
        ?income can be "stable", "unstable"
        ?applicant_notes meaning "notes from application"

        ?fraud_risk resembles "identity theft", "suspicious activity"

        ?loan_approved becomes true
            when ?credit_score is "high" and ?income is "stable"

        ?loan_approved becomes false
            when ?applicant_notes is like ?fraud_risk
        """
        ast = parse_string(dsl)
        result = analyze(ast)

        assert result.is_valid
        assert len(result.errors) == 0

    def test_rule_head_can_be_referenced(self):
        """Test that rule heads can be referenced in other rules"""
        dsl = """
        ?a becomes true

        ?b becomes true
            when ?a is true
        """
        ast = parse_string(dsl)
        result = analyze(ast)

        assert result.is_valid

    def test_complex_conditions_with_not(self):
        """Test NOT conditions with defined predicates"""
        dsl = """
        ?flag can be "yes", "no"

        ?result becomes true
            when not ?flag is "no"
        """
        ast = parse_string(dsl)
        result = analyze(ast)

        assert result.is_valid

    def test_unless_clause_validated(self):
        """Test that unless clause predicates are validated"""
        dsl = """
        ?treatment_intent resembles "treat", "treating"
        ?prevention_intent resembles "prevent", "preventing"
        ?patient_query meaning "the patient query"

        ?query_intent becomes "treatment"
            when ?patient_query is like ?treatment_intent
            unless ?patient_query is like ?prevention_intent
        """
        ast = parse_string(dsl)
        result = analyze(ast)

        assert result.is_valid


class TestWarnings:
    """Tests for semantic warnings"""

    def test_warning_matches_left_is_category(self):
        """Test warning when is like left side is a category instead of input"""
        dsl = """
        ?category1 resembles "a", "b"
        ?category2 resembles "c", "d"

        ?result becomes true
            when ?category1 is like ?category2
        """
        ast = parse_string(dsl)
        result = analyze(ast)

        assert result.is_valid  # Still valid, just a warning
        assert len(result.warnings) >= 1
        assert any("left side should be input" in w.message for w in result.warnings)

    def test_warning_matches_right_is_not_category(self):
        """Test warning when is like right side is not a category"""
        dsl = """
        ?input1 meaning "some input"
        ?input2 meaning "another input"

        ?result becomes true
            when ?input1 is like ?input2
        """
        ast = parse_string(dsl)
        result = analyze(ast)

        assert result.is_valid  # Still valid, just a warning
        assert len(result.warnings) >= 1
        assert any("right side should be a category" in w.message for w in result.warnings)


class TestPredicateKinds:
    """Tests for predicate kind detection"""

    def test_input_predicate_kinds(self):
        """Test that can be and meaning create INPUT predicates"""
        dsl = """
        ?enum_var can be "a", "b"
        ?text_var meaning "description"
        """
        ast = parse_string(dsl)
        result = analyze(ast)

        assert result.declared_predicates["enum_var"].kind == PredicateKind.INPUT
        assert result.declared_predicates["text_var"].kind == PredicateKind.INPUT

    def test_category_predicate_kind(self):
        """Test that resembles creates CATEGORY predicates"""
        dsl = """
        ?patterns resembles "a", "b", "c"
        """
        ast = parse_string(dsl)
        result = analyze(ast)

        assert result.declared_predicates["patterns"].kind == PredicateKind.CATEGORY

    def test_derived_predicate_kind(self):
        """Test that rule heads create DERIVED predicates"""
        dsl = """
        ?result becomes true
        """
        ast = parse_string(dsl)
        result = analyze(ast)

        assert result.declared_predicates["result"].kind == PredicateKind.DERIVED


class TestWithBlockVariables:
    """Tests for nested variables in 'with' blocks"""

    def test_nested_with_block_variables_are_declared(self):
        """Test that variables inside 'with' blocks are recognized"""
        dsl = """
        ?patient meaning "patient info" with
            ?name meaning "patient's name"
            ?age meaning "patient's age"

        ?result becomes true
            when ?name is "John"
        """
        ast = parse_string(dsl)
        result = analyze(ast)

        assert result.is_valid, f"Errors: {result.errors}"
        assert "patient" in result.declared_predicates
        assert "name" in result.declared_predicates
        assert "age" in result.declared_predicates

    def test_deeply_nested_with_block_variables(self):
        """Test that deeply nested 'with' block variables are recognized"""
        dsl = """
        ?patient meaning "patient info" with
            ?symptoms with
                ?text meaning "symptom description"
                ?severity can be "mild", "severe"

        ?result becomes true
            when ?text is "headache"
        """
        ast = parse_string(dsl)
        result = analyze(ast)

        assert result.is_valid, f"Errors: {result.errors}"
        assert "patient" in result.declared_predicates
        assert "symptoms" in result.declared_predicates
        assert "text" in result.declared_predicates
        assert "severity" in result.declared_predicates

    def test_with_block_enum_variables(self):
        """Test that enum variables inside 'with' blocks are recognized"""
        dsl = """
        ?patient meaning "patient" with
            ?status can be "active", "inactive"

        ?result becomes true
            when ?status is "active"
        """
        ast = parse_string(dsl)
        result = analyze(ast)

        assert result.is_valid, f"Errors: {result.errors}"
        assert "status" in result.declared_predicates
        assert result.declared_predicates["status"].kind == PredicateKind.INPUT

    def test_undefined_nested_variable_detected(self):
        """Test that undefined variables are still detected even with 'with' blocks"""
        dsl = """
        ?patient meaning "patient info" with
            ?name meaning "patient's name"

        ?result becomes true
            when ?undefined_var is "value"
        """
        ast = parse_string(dsl)
        result = analyze(ast)

        assert not result.is_valid
        assert len(result.errors) == 1
        assert result.errors[0].predicate == "undefined_var"


class TestFromClauseVariables:
    """Tests for variables from 'from ?source' clause"""

    def test_from_source_variable_declared(self):
        """Test that the source variable in 'from' clause is recognized"""
        dsl = """
        ?text meaning "input text"
        ?company resembles "Apple", "Google"

        ?entities has a list of ?company from ?text

        ?result becomes true
            when ?text is "Apple announced"
        """
        ast = parse_string(dsl)
        result = analyze(ast)

        assert result.is_valid, f"Errors: {result.errors}"
        assert "text" in result.declared_predicates
        assert "company" in result.declared_predicates
        assert "entities" in result.declared_predicates


class TestNestedVariableQualification:
    """Tests for nested variable qualification warnings"""

    def test_unqualified_nested_variable_warning(self):
        """Test that using a nested variable without qualification produces a warning"""
        dsl = """
        ?puzzle meaning "the logic puzzle" with
            ?claims_truth can be true, false

        ?result becomes true
            when ?claims_truth is true
        """
        ast = parse_string(dsl)
        result = analyze(ast)

        # Should be valid but with a warning
        assert result.is_valid, f"Errors: {result.errors}"
        assert len(result.warnings) >= 1
        assert any("should be qualified" in w.message for w in result.warnings)
        assert any("claims_truth" in w.predicate for w in result.warnings)

    def test_qualified_nested_variable_no_warning(self):
        """Test that using a nested variable with qualification produces no warning"""
        dsl = """
        ?puzzle meaning "the logic puzzle" with
            ?claims_truth can be true, false

        ?result becomes true
            when ?claims_truth of ?puzzle is true
        """
        ast = parse_string(dsl)
        result = analyze(ast)

        # Should be valid with no qualification warnings
        assert result.is_valid, f"Errors: {result.errors}"
        # Check no warnings about qualification
        qual_warnings = [w for w in result.warnings if "should be qualified" in w.message]
        assert len(qual_warnings) == 0, f"Unexpected qualification warnings: {qual_warnings}"

    def test_deeply_nested_variable_qualification(self):
        """Test warning for deeply nested variables without qualification"""
        dsl = """
        ?patient meaning "patient info" with
            ?symptoms with
                ?severity can be "mild", "severe"

        ?result becomes true
            when ?severity is "severe"
        """
        ast = parse_string(dsl)
        result = analyze(ast)

        # Should be valid but with a warning about unqualified ?severity
        assert result.is_valid, f"Errors: {result.errors}"
        assert len(result.warnings) >= 1
        assert any("severity" in w.predicate for w in result.warnings)

    def test_top_level_variable_no_warning(self):
        """Test that top-level variables produce no qualification warning"""
        dsl = """
        ?credit_score can be "high", "low"

        ?result becomes true
            when ?credit_score is "high"
        """
        ast = parse_string(dsl)
        result = analyze(ast)

        # Should be valid with no warnings
        assert result.is_valid, f"Errors: {result.errors}"
        qual_warnings = [w for w in result.warnings if "should be qualified" in w.message]
        assert len(qual_warnings) == 0, f"Unexpected qualification warnings: {qual_warnings}"


class TestRealExamples:
    """Tests using real example files"""

    def test_medical_query_passes(self):
        """Test that medical_query.canto passes semantic analysis"""
        example_path = Path(__file__).parent.parent / "examples" / "medical_query.canto"
        if example_path.exists():
            from canto_core.parser import parse_file
            ast = parse_file(str(example_path))
            result = analyze(ast)

            assert result.is_valid, f"Errors: {result.errors}"

    def test_fintech_loan_passes(self):
        """Test that fintech_loan.canto passes semantic analysis"""
        example_path = Path(__file__).parent.parent / "examples" / "fintech_loan.canto"
        if example_path.exists():
            from canto_core.parser import parse_file
            ast = parse_file(str(example_path))
            result = analyze(ast)

            assert result.is_valid, f"Errors: {result.errors}"

    def test_medical_triage_has_passes(self):
        """Test that medical triage with nested 'with' blocks passes semantic analysis"""
        dsl = """
        ?emergency_symptoms resembles "chest pain", "difficulty breathing", "severe bleeding"
        ?urgent_symptoms resembles "high fever", "persistent vomiting", "severe pain"
        ?cardiac_terms resembles "heart", "cardiac", "chest pain", "palpitations"

        ?patient meaning "extracted patient information" with
            ?name meaning "patient's full name"
            ?age meaning "patient's age in years"
            ?chief_complaint meaning "primary reason for visit"
            ?symptoms with
                ?text meaning "symptom description"
                ?is_emergency can be true, false
            ?medications with
                ?med_name meaning "medication name"
                ?dosage meaning "current dosage"

        ?is_emergency of (?symptoms of ?patient) becomes true
            when ?text of (?symptoms of ?patient) is like ?emergency_symptoms

        ?urgency can be "emergency", "urgent", "routine"

        ?urgency becomes "emergency"
            when ?symptoms of ?patient has any that ?is_emergency is true
            overriding all

        normally ?urgency becomes "urgent"
            when ?symptoms of ?patient has any that is like ?urgent_symptoms

        normally ?urgency becomes "routine"

        ?referral becomes "cardiology"
            when ?symptoms of ?patient has any that is like ?cardiac_terms
        """
        ast = parse_string(dsl)
        result = analyze(ast)

        assert result.is_valid, f"Errors: {result.errors}"
        # Verify nested variables are recognized
        assert "patient" in result.declared_predicates
        assert "symptoms" in result.declared_predicates
        assert "text" in result.declared_predicates
        assert "is_emergency" in result.declared_predicates

    def test_web_of_lies_passes(self):
        """Test that web of lies puzzle passes semantic analysis"""
        dsl = """
        ?truth_claim resembles "tells the truth", "is truthful", "is honest"
        ?lie_claim resembles "lies", "is lying", "is a liar"

        ?puzzle meaning "the logic puzzle" with
            ?first_person_is_truthful can be true, false
            ?claims with
                ?speaker meaning "who is speaking"
                ?target meaning "who they describe"
                ?assertion meaning "what they claim"
                ?claims_truth can be true, false

        ?claims_truth of (?claims of ?puzzle) becomes true when ?assertion of (?claims of ?puzzle) is like ?truth_claim
        ?claims_truth of (?claims of ?puzzle) becomes false when ?assertion of (?claims of ?puzzle) is like ?lie_claim

        ?answer can be "Yes", "No"

        ?answer becomes "Yes"
            when ?first_person_is_truthful of ?puzzle is true
            and (length of ?claims of ?puzzle where ?claims_truth is false) is 0

        ?answer becomes "Yes"
            when ?first_person_is_truthful of ?puzzle is false
            and (length of ?claims of ?puzzle where ?claims_truth is false) > 0

        normally ?answer becomes "No"
        """
        ast = parse_string(dsl)
        result = analyze(ast)

        assert result.is_valid, f"Errors: {result.errors}"
        # Verify nested variables are recognized
        assert "puzzle" in result.declared_predicates
        assert "claims" in result.declared_predicates
        assert "claims_truth" in result.declared_predicates


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
