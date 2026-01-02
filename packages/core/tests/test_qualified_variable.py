"""
Tests for qualified variable syntax: ?child of ?parent
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from canto_core.parser.dsl_parser import parse_string
from canto_core.ast_nodes import HasDeclaration, VariableDeclaration, Rule, Condition
from canto_core.fol import translate_to_fol
from canto_core.parser.semantic_analyzer import analyze, PredicateKind


class TestQualifiedVariableParsing:
    """Tests for parsing qualified variable references"""

    def test_qualified_variable_in_has_condition(self):
        """Test parsing: when ?symptoms of ?patient has 'value'"""
        dsl = """
        ?patient meaning "patient record"
        ?patient has a list of ?symptoms meaning "reported symptoms"

        ?emergency becomes true
            when ?symptoms of ?patient has "chest pain"
        """
        ast = parse_string(dsl)

        rule = [n for n in ast if isinstance(n, Rule)][0]
        assert len(rule.conditions) == 1
        condition = rule.conditions[0]
        assert condition.operator == "HAS"
        # Left side should be a dict with child/parent
        assert isinstance(condition.left, dict)
        assert condition.left['child'] == '?symptoms'
        assert condition.left['parent'] == '?patient'
        assert condition.right == "chest pain"

    def test_qualified_variable_in_is_like_condition(self):
        """Test parsing: when ?symptom of ?patient is like ?category"""
        dsl = """
        ?patient meaning "patient record"
        ?patient has a ?symptom meaning "main symptom"
        ?emergency_terms resembles "chest pain", "difficulty breathing" meaning "symptoms indicating emergency"

        ?emergency becomes true
            when ?symptom of ?patient is like ?emergency_terms
        """
        ast = parse_string(dsl)

        rule = [n for n in ast if isinstance(n, Rule)][0]
        condition = rule.conditions[0]
        assert condition.operator == "IS_LIKE"
        assert isinstance(condition.left, dict)
        assert condition.left['child'] == '?symptom'
        assert condition.left['parent'] == '?patient'
        assert condition.right == "?emergency_terms"

    def test_qualified_variable_in_is_condition(self):
        """Test parsing: when ?status of ?patient is 'critical'"""
        dsl = """
        ?patient meaning "patient record"
        ?patient has a ?status meaning "patient status"

        ?alert becomes true
            when ?status of ?patient is "critical"
        """
        ast = parse_string(dsl)

        rule = [n for n in ast if isinstance(n, Rule)][0]
        condition = rule.conditions[0]
        assert condition.operator == "IS"
        assert isinstance(condition.left, dict)
        assert condition.left['child'] == '?status'
        assert condition.left['parent'] == '?patient'
        assert condition.right == "critical"

    def test_simple_variable_still_works(self):
        """Test that simple variables still work without 'of'"""
        dsl = """
        ?symptoms meaning "symptoms list"

        ?emergency becomes true
            when ?symptoms has "chest pain"
        """
        ast = parse_string(dsl)

        rule = [n for n in ast if isinstance(n, Rule)][0]
        condition = rule.conditions[0]
        assert condition.operator == "HAS"
        # Left side should be a simple string
        assert condition.left == "?symptoms"


class TestQualifiedVariableTranslation:
    """Tests for translating qualified variables to DeLP"""

    def test_qualified_has_translation(self):
        """Test translation of qualified has condition"""
        dsl = """
        ?patient meaning "patient record"
        ?patient has a list of ?symptoms meaning "symptoms"

        ?emergency becomes true
            when ?symptoms of ?patient has "chest pain"
        """
        ast = parse_string(dsl)
        program = translate_to_fol(ast)

        # Check the generated rule body
        rule = program.strict_rules[0]
        assert len(rule.body) == 1
        # Should be has(symptoms_of_patient, 'chest pain')
        assert "has(symptoms_of_patient," in rule.body[0]

    def test_qualified_is_like_translation(self):
        """Test translation of qualified is like condition"""
        dsl = """
        ?patient meaning "patient record"
        ?patient has a ?complaint meaning "chief complaint"
        ?cardiac_terms resembles "heart", "chest pain" meaning "terms related to cardiac issues"

        ?referral becomes "cardiology"
            when ?complaint of ?patient is like ?cardiac_terms
        """
        ast = parse_string(dsl)
        program = translate_to_fol(ast)

        rule = program.strict_rules[0]
        # Should be is_like(complaint_of_patient, cardiac_terms)
        assert "is_like(complaint_of_patient, cardiac_terms)" in rule.body[0]

    def test_prolog_output_with_qualified(self):
        """Test that Prolog output correctly formats qualified variables"""
        dsl = """
        ?patient meaning "patient record"
        ?patient has a list of ?symptoms meaning "symptoms"

        ?emergency becomes true
            when ?symptoms of ?patient has "chest pain"
        """
        ast = parse_string(dsl)
        program = translate_to_fol(ast)
        prolog_str = program.to_prolog_string()

        # Check that the qualified variable appears correctly in Prolog
        assert "has(symptoms_of_patient," in prolog_str


class TestQualifiedVariableSemantics:
    """Tests for semantic analysis of qualified variables"""

    def test_valid_qualified_reference(self):
        """Test that valid qualified reference passes analysis"""
        dsl = """
        ?patient meaning "patient record"
        ?patient has a list of ?symptoms meaning "symptoms"

        ?emergency becomes true
            when ?symptoms of ?patient has "chest pain"
        """
        ast = parse_string(dsl)
        result = analyze(ast)

        assert result.is_valid
        assert "patient" in result.declared_predicates
        assert "symptoms" in result.declared_predicates

    def test_undefined_child_in_qualified(self):
        """Test that undefined child predicate is caught"""
        dsl = """
        ?patient meaning "patient record"

        ?emergency becomes true
            when ?undefined_prop of ?patient has "value"
        """
        ast = parse_string(dsl)
        result = analyze(ast)

        assert not result.is_valid
        assert any(e.predicate == "undefined_prop" for e in result.errors)

    def test_undefined_parent_in_qualified(self):
        """Test that undefined parent predicate is caught"""
        dsl = """
        ?symptoms meaning "symptoms list"

        ?emergency becomes true
            when ?symptoms of ?undefined_parent has "value"
        """
        ast = parse_string(dsl)
        result = analyze(ast)

        assert not result.is_valid
        assert any(e.predicate == "undefined_parent" for e in result.errors)


class TestQualifiedVariableIntegration:
    """Integration tests combining qualified variables with other features"""

    def test_qualified_with_and(self):
        """Test qualified variable combined with AND"""
        dsl = """
        ?patient meaning "patient record"
        ?patient has a list of ?symptoms meaning "symptoms"
        ?patient has a ?age meaning "age"

        ?emergency becomes true
            when ?symptoms of ?patient has "chest pain"
            and ?age of ?patient is 65
        """
        ast = parse_string(dsl)
        program = translate_to_fol(ast)

        rule = program.strict_rules[0]
        assert len(rule.body) == 2
        assert "has(symptoms_of_patient," in rule.body[0]
        assert "age_of_patient(65)" in rule.body[1]

    def test_qualified_with_or(self):
        """Test qualified variable combined with OR (uses De Morgan)"""
        dsl = """
        ?patient meaning "patient record"
        ?patient has a list of ?symptoms meaning "symptoms"

        ?emergency becomes true
            when ?symptoms of ?patient has "chest pain"
            or ?symptoms of ?patient has "difficulty breathing"
        """
        ast = parse_string(dsl)
        program = translate_to_fol(ast)

        # OR is translated using De Morgan's law: A or B = not((not A) and (not B))
        # This keeps it as a single rule, avoiding cycles with "overriding all"
        rules = program.strict_rules
        assert len(rules) == 1
        # Verify De Morgan pattern in the rule body
        assert '\\+' in rules[0].body[0]

    def test_complete_medical_example_with_qualified(self):
        """Test a complete medical example using qualified variables"""
        dsl = """
        ?patient meaning "extracted patient information"
        ?patient has a ?name meaning "patient name"
        ?patient has a ?age meaning "patient age"
        ?patient has a list of ?symptoms meaning "reported symptoms"
        ?patient has a ?chief_complaint meaning "main complaint"

        ?emergency_symptoms resembles "chest pain", "difficulty breathing" meaning "symptoms indicating a medical emergency"

        ?urgency can be "emergency", "urgent", "routine"

        ?urgency becomes "emergency"
            when ?symptoms of ?patient has "chest pain"
            overriding all

        ?urgency becomes "emergency"
            when ?chief_complaint of ?patient is like ?emergency_symptoms
            overriding all

        normally ?urgency becomes "routine"
        """
        ast = parse_string(dsl)
        result = analyze(ast)

        assert result.is_valid

        program = translate_to_fol(ast)

        # Check has relationships
        assert len(program.has_relationships) == 4

        # Check rules generated
        assert len(program.strict_rules) == 2  # Two emergency rules
        assert len(program.defeasible_rules) == 1  # One routine rule

        # Check superiority relations exist
        assert len(program.superiority) >= 2


class TestQualifiedVariableEdgeCases:
    """Tests for edge cases with qualified variables"""

    def test_qualified_with_boolean_value(self):
        """Test qualified variable with boolean value comparison"""
        dsl = """
        ?patient meaning "patient"
        ?patient has a ?active meaning "is active"

        ?alert becomes true
            when ?active of ?patient is true
        """
        ast = parse_string(dsl)
        rule = [n for n in ast if isinstance(n, Rule)][0]
        condition = rule.conditions[0]
        assert condition.operator == "IS"
        assert isinstance(condition.left, dict)
        assert condition.right is True

    def test_qualified_with_number_value(self):
        """Test qualified variable with numeric value comparison"""
        dsl = """
        ?patient meaning "patient"
        ?patient has a ?age meaning "age"

        ?senior becomes true
            when ?age of ?patient is 65
        """
        ast = parse_string(dsl)
        rule = [n for n in ast if isinstance(n, Rule)][0]
        condition = rule.conditions[0]
        assert condition.operator == "IS"
        assert condition.right == 65

    def test_multiple_qualified_same_parent(self):
        """Test multiple qualified variables with same parent"""
        dsl = """
        ?patient meaning "patient"
        ?patient has a ?name meaning "name"
        ?patient has a ?age meaning "age"
        ?patient has a ?status meaning "status"

        ?test becomes true
            when ?name of ?patient is "John"
            and ?age of ?patient is 65
            and ?status of ?patient is "active"
        """
        ast = parse_string(dsl)
        program = translate_to_fol(ast)

        rule = program.strict_rules[0]
        assert len(rule.body) == 3
        assert all("_of_patient" in body for body in rule.body)

    def test_qualified_in_unless_clause(self):
        """Test qualified variable in unless clause"""
        dsl = """
        ?patient meaning "patient"
        ?patient has a ?status meaning "status"

        ?can_discharge becomes true
            unless ?status of ?patient is "critical"
        """
        ast = parse_string(dsl)
        rule = [n for n in ast if isinstance(n, Rule)][0]
        assert len(rule.exceptions) == 1
        exception = rule.exceptions[0]
        assert isinstance(exception.left, dict)
        assert exception.left['child'] == '?status'
        assert exception.left['parent'] == '?patient'

    def test_qualified_in_defeasible_rule_body(self):
        """Test qualified variable in defeasible (normally) rule body"""
        dsl = """
        ?patient meaning "patient"
        ?patient has a ?status meaning "status"

        normally ?urgency becomes "routine"
            when ?status of ?patient is "stable"
        """
        ast = parse_string(dsl)
        program = translate_to_fol(ast)

        # Should have one defeasible rule
        assert len(program.defeasible_rules) == 1
        rule = program.defeasible_rules[0]
        assert "status_of_patient" in rule.body[0]

    def test_qualified_in_rule_head_supported(self):
        """Test that qualified variables in rule head are supported"""
        dsl = """
        ?patient meaning "patient"
        ?patient has a ?urgency meaning "urgency level"

        normally ?urgency of ?patient becomes "routine"
        """
        # Qualified variables in rule head are now supported
        ast = parse_string(dsl)
        rule = [n for n in ast if isinstance(n, Rule)][0]
        assert rule.head_variable == 'urgency'
        assert rule.head_parent == 'patient'
        assert rule.head_value == 'routine'

    def test_qualified_long_variable_names(self):
        """Test qualified variables with longer names"""
        dsl = """
        ?patient_record meaning "full patient record"
        ?patient_record has a ?primary_diagnosis meaning "main diagnosis"

        ?needs_specialist becomes true
            when ?primary_diagnosis of ?patient_record is like ?cardiac_terms
        ?cardiac_terms resembles "heart disease" meaning "cardiac"
        """
        ast = parse_string(dsl)
        rule = [n for n in ast if isinstance(n, Rule)][0]
        condition = rule.conditions[0]
        assert condition.left['child'] == '?primary_diagnosis'
        assert condition.left['parent'] == '?patient_record'


class TestQualifiedVariablePrologOutput:
    """Tests for Prolog output with qualified variables"""

    def test_prolog_format_underscores(self):
        """Test that Prolog output uses underscores for qualified names"""
        dsl = """
        ?patient meaning "patient"
        ?patient has a ?status meaning "status"

        ?alert becomes true
            when ?status of ?patient is "critical"
        """
        ast = parse_string(dsl)
        program = translate_to_fol(ast)
        prolog_str = program.to_prolog_string()

        # Check that qualified variable is formatted as child_of_parent
        assert "status_of_patient" in prolog_str

    def test_prolog_multiple_qualified_different_parents(self):
        """Test Prolog output with multiple parents"""
        dsl = """
        ?patient meaning "patient"
        ?patient has a ?status meaning "patient status"
        ?doctor meaning "doctor"
        ?doctor has a ?status meaning "doctor status"

        ?both_active becomes true
            when ?status of ?patient is "active"
            and ?status of ?doctor is "on_duty"
        """
        ast = parse_string(dsl)
        program = translate_to_fol(ast)
        prolog_str = program.to_prolog_string()

        assert "status_of_patient" in prolog_str
        assert "status_of_doctor" in prolog_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
