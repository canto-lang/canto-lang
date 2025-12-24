"""
Tests for the 'has' primitive (structural relationships)
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from canto_core.parser.dsl_parser import parse_string
from canto_core.ast_nodes import HasDeclaration, VariableDeclaration, Rule
from canto_core.codegen import DeLPTranslator
from canto_core.parser.semantic_analyzer import analyze, PredicateKind


class TestHasDeclarationParsing:
    """Tests for parsing has declarations"""

    def test_has_single_declaration(self):
        """Test parsing: ?parent has a ?child"""
        dsl = """
        ?patient has a ?diagnosis meaning "the diagnosis assigned"
        """
        ast = parse_string(dsl)

        assert len(ast) == 1
        node = ast[0]
        assert isinstance(node, HasDeclaration)
        assert node.parent == "patient"
        assert node.child == "diagnosis"
        assert node.is_list is False
        assert node.description == "the diagnosis assigned"

    def test_has_list_declaration(self):
        """Test parsing: ?parent has a list of ?children"""
        dsl = """
        ?patient has a list of ?medications meaning "current medications"
        """
        ast = parse_string(dsl)

        assert len(ast) == 1
        node = ast[0]
        assert isinstance(node, HasDeclaration)
        assert node.parent == "patient"
        assert node.child == "medications"
        assert node.is_list is True
        assert node.description == "current medications"

    def test_has_without_description(self):
        """Test parsing has declaration without description"""
        dsl = """
        ?order has a ?total
        ?order has a list of ?items
        """
        ast = parse_string(dsl)

        assert len(ast) == 2

        assert isinstance(ast[0], HasDeclaration)
        assert ast[0].parent == "order"
        assert ast[0].child == "total"
        assert ast[0].is_list is False
        assert ast[0].description is None

        assert isinstance(ast[1], HasDeclaration)
        assert ast[1].parent == "order"
        assert ast[1].child == "items"
        assert ast[1].is_list is True
        assert ast[1].description is None

    def test_multiple_has_declarations(self):
        """Test parsing multiple has declarations"""
        dsl = """
        ?patient has a ?name meaning "patient name"
        ?patient has a ?age meaning "patient age"
        ?patient has a list of ?symptoms meaning "reported symptoms"
        ?patient has a list of ?medications meaning "current medications"
        """
        ast = parse_string(dsl)

        assert len(ast) == 4
        assert all(isinstance(node, HasDeclaration) for node in ast)

        # Check single properties
        name_decl = ast[0]
        assert name_decl.child == "name"
        assert name_decl.is_list is False

        age_decl = ast[1]
        assert age_decl.child == "age"
        assert age_decl.is_list is False

        # Check list properties
        symptoms_decl = ast[2]
        assert symptoms_decl.child == "symptoms"
        assert symptoms_decl.is_list is True

        meds_decl = ast[3]
        assert meds_decl.child == "medications"
        assert meds_decl.is_list is True


class TestHasConditionParsing:
    """Tests for parsing has conditions in rules"""

    def test_has_condition_with_string(self):
        """Test parsing: when ?list has 'value'"""
        dsl = """
        ?symptoms meaning "symptoms list"
        ?emergency becomes true
            when ?symptoms has "chest pain"
        """
        ast = parse_string(dsl)

        rule = [n for n in ast if isinstance(n, Rule)][0]
        assert len(rule.conditions) == 1
        condition = rule.conditions[0]
        assert condition.operator == "HAS"
        assert condition.left == "?symptoms"
        assert condition.right == "chest pain"

    def test_has_condition_with_variable(self):
        """Test parsing: when ?list has ?item"""
        dsl = """
        ?symptoms meaning "symptoms list"
        ?symptom meaning "a symptom"
        ?has_symptom becomes true
            when ?symptoms has ?symptom
        """
        ast = parse_string(dsl)

        rule = [n for n in ast if isinstance(n, Rule)][0]
        condition = rule.conditions[0]
        assert condition.operator == "HAS"
        assert condition.left == "?symptoms"
        assert condition.right == "?symptom"


class TestHasTranslation:
    """Tests for translating has declarations to DeLP"""

    def test_has_single_translation(self):
        """Test translation of single has declaration"""
        dsl = """
        ?patient has a ?diagnosis meaning "the diagnosis"
        """
        ast = parse_string(dsl)
        translator = DeLPTranslator()
        program = translator.translate(ast)

        assert len(program.has_relationships) == 1
        key = "patient_diagnosis"
        assert key in program.has_relationships
        has_decl = program.has_relationships[key]
        assert has_decl.parent == "patient"
        assert has_decl.child == "diagnosis"
        assert has_decl.is_list is False

    def test_has_list_translation(self):
        """Test translation of list has declaration"""
        dsl = """
        ?patient has a list of ?medications meaning "meds"
        """
        ast = parse_string(dsl)
        translator = DeLPTranslator()
        program = translator.translate(ast)

        key = "patient_medications"
        assert key in program.has_relationships
        has_decl = program.has_relationships[key]
        assert has_decl.is_list is True

    def test_prolog_output_contains_has_property(self):
        """Test that Prolog output contains has_property facts"""
        dsl = """
        ?patient has a ?name meaning "patient name"
        ?patient has a list of ?symptoms meaning "symptoms"
        """
        ast = parse_string(dsl)
        translator = DeLPTranslator()
        program = translator.translate(ast)
        prolog_str = program.to_prolog_string()

        assert "has_property(patient, name, single)" in prolog_str
        assert "has_property(patient, symptoms, list)" in prolog_str
        assert ":- dynamic has_property/3" in prolog_str


class TestHasSemanticAnalysis:
    """Tests for semantic analysis of has declarations"""

    def test_has_declares_child_predicate(self):
        """Test that has declaration makes child predicate available"""
        dsl = """
        ?patient has a list of ?symptoms meaning "symptoms"

        ?emergency becomes true
            when ?symptoms has "chest pain"
        """
        ast = parse_string(dsl)
        result = analyze(ast)

        assert result.is_valid
        assert "symptoms" in result.declared_predicates
        assert result.declared_predicates["symptoms"].kind == PredicateKind.PROPERTY

    def test_has_infers_parent_predicate(self):
        """Test that has declaration infers parent if not declared"""
        dsl = """
        ?patient has a ?name meaning "name"
        """
        ast = parse_string(dsl)
        result = analyze(ast)

        assert result.is_valid
        # Both parent and child should be declared
        assert "patient" in result.declared_predicates
        assert "name" in result.declared_predicates

    def test_undefined_predicate_in_has_condition(self):
        """Test that undefined predicate in has condition is caught"""
        dsl = """
        ?emergency becomes true
            when ?undefined_list has "value"
        """
        ast = parse_string(dsl)
        result = analyze(ast)

        assert not result.is_valid
        assert len(result.errors) == 1
        assert result.errors[0].predicate == "undefined_list"


class TestHasIntegration:
    """Integration tests combining has with other features"""

    def test_has_with_is_like(self):
        """Test combining has with is like"""
        dsl = """
        ?patient has a list of ?symptoms meaning "symptoms"
        ?emergency_terms resembles "chest pain", "difficulty breathing"

        ?emergency becomes true
            when ?symptoms has ?symptom
            and ?symptom is like ?emergency_terms
        """
        ast = parse_string(dsl)
        result = analyze(ast)

        # Should be valid - all predicates are declared
        # Note: ?symptom is used as a variable in the condition, not declared
        # This is actually a pattern that might need refinement
        assert "symptoms" in result.declared_predicates
        assert "emergency_terms" in result.declared_predicates

    def test_nested_has_declarations(self):
        """Test nested has declarations"""
        dsl = """
        ?hospital has a list of ?patients meaning "patients"
        ?patient has a ?name meaning "patient name"
        ?patient has a list of ?visits meaning "visits"
        ?visit has a ?date meaning "visit date"
        ?visit has a ?diagnosis meaning "diagnosis"
        """
        ast = parse_string(dsl)
        result = analyze(ast)

        assert result.is_valid
        assert "patients" in result.declared_predicates
        assert "name" in result.declared_predicates
        assert "visits" in result.declared_predicates
        assert "date" in result.declared_predicates
        assert "diagnosis" in result.declared_predicates

    def test_complete_medical_triage_example(self):
        """Test a complete medical triage example with has"""
        dsl = """
        ?intake_form meaning "patient intake form"

        ?patient has a ?name meaning "patient name"
        ?patient has a ?age meaning "patient age"
        ?patient has a list of ?symptoms meaning "reported symptoms"
        ?patient has a ?chief_complaint meaning "main complaint"

        ?emergency_symptoms resembles "chest pain", "difficulty breathing"

        ?urgency can be "emergency", "urgent", "routine"

        ?urgency becomes "emergency"
            when ?symptoms has "chest pain"
            overriding all

        normally ?urgency becomes "routine"
        """
        ast = parse_string(dsl)
        result = analyze(ast)

        assert result.is_valid

        # Translate and check
        translator = DeLPTranslator()
        program = translator.translate(ast)

        # Should have has relationships
        assert len(program.has_relationships) == 4

        # Should have rules
        assert len(program.strict_rules) + len(program.defeasible_rules) == 2


class TestHasEdgeCases:
    """Tests for edge cases and boundary conditions"""

    def test_has_with_variable_prefix_in_name(self):
        """Test has declaration with ? in child name parses correctly"""
        dsl = """
        ?patient has a ?name meaning "name"
        """
        ast = parse_string(dsl)

        assert len(ast) == 1
        node = ast[0]
        assert isinstance(node, HasDeclaration)
        # The parser should strip the ? prefix
        assert node.child == "name"

    def test_has_condition_with_boolean(self):
        """Test has condition with boolean value"""
        dsl = """
        ?patient has a list of ?flags meaning "flags"

        ?has_true_flag becomes true
            when ?flags has true
        """
        ast = parse_string(dsl)
        rule = [n for n in ast if isinstance(n, Rule)][0]
        condition = rule.conditions[0]
        assert condition.operator == "HAS"
        assert condition.right is True

    def test_has_condition_with_number(self):
        """Test has condition with numeric value"""
        dsl = """
        ?order has a list of ?quantities meaning "quantities"

        ?has_zero becomes true
            when ?quantities has 0
        """
        ast = parse_string(dsl)
        rule = [n for n in ast if isinstance(n, Rule)][0]
        condition = rule.conditions[0]
        assert condition.operator == "HAS"
        assert condition.right == 0

    def test_multiple_has_same_parent(self):
        """Test multiple has declarations with same parent"""
        dsl = """
        ?order has a ?customer meaning "customer"
        ?order has a ?total meaning "total"
        ?order has a list of ?items meaning "items"
        ?order has a ?status meaning "status"
        """
        ast = parse_string(dsl)

        assert len(ast) == 4
        for node in ast:
            assert isinstance(node, HasDeclaration)
            assert node.parent == "order"

    def test_has_declarations_multiple_parents(self):
        """Test has declarations with different parents"""
        dsl = """
        ?patient has a ?name meaning "patient name"
        ?doctor has a ?name meaning "doctor name"
        ?hospital has a ?name meaning "hospital name"
        """
        ast = parse_string(dsl)

        assert len(ast) == 3
        parents = [node.parent for node in ast]
        assert set(parents) == {"patient", "doctor", "hospital"}


class TestHasComplexConditions:
    """Tests for complex conditions involving has"""

    def test_has_with_and(self):
        """Test has condition combined with AND"""
        dsl = """
        ?patient has a list of ?symptoms meaning "symptoms"
        ?patient has a ?age meaning "age"

        ?high_risk becomes true
            when ?symptoms has "chest pain"
            and ?age is 65
        """
        ast = parse_string(dsl)
        rule = [n for n in ast if isinstance(n, Rule)][0]
        assert len(rule.conditions) == 1
        # The conditions are combined into a single tree
        cond = rule.conditions[0]
        assert cond.operator == "AND"

    def test_has_with_or(self):
        """Test has condition combined with OR"""
        dsl = """
        ?patient has a list of ?symptoms meaning "symptoms"

        ?emergency becomes true
            when ?symptoms has "chest pain"
            or ?symptoms has "difficulty breathing"
        """
        ast = parse_string(dsl)
        rule = [n for n in ast if isinstance(n, Rule)][0]
        cond = rule.conditions[0]
        assert cond.operator == "OR"

    def test_has_in_unless_clause(self):
        """Test has condition in unless clause"""
        dsl = """
        ?patient has a list of ?allergies meaning "allergies"

        ?can_prescribe becomes true
            unless ?allergies has "penicillin"
        """
        ast = parse_string(dsl)
        rule = [n for n in ast if isinstance(n, Rule)][0]
        assert len(rule.exceptions) == 1
        exception = rule.exceptions[0]
        assert exception.operator == "HAS"


class TestHasPrologOutput:
    """Tests for Prolog output correctness"""

    def test_prolog_has_predicate_format(self):
        """Test that has conditions generate correct Prolog predicates"""
        dsl = """
        ?symptoms meaning "symptoms"

        ?emergency becomes true
            when ?symptoms has "chest pain"
        """
        ast = parse_string(dsl)
        translator = DeLPTranslator()
        program = translator.translate(ast)
        prolog_str = program.to_prolog_string()

        assert "has(symptoms, 'chest pain')" in prolog_str

    def test_prolog_has_dynamic_declared(self):
        """Test that has/2 is declared as dynamic"""
        dsl = """
        ?symptoms meaning "symptoms"

        ?emergency becomes true
            when ?symptoms has "chest pain"
        """
        ast = parse_string(dsl)
        translator = DeLPTranslator()
        program = translator.translate(ast)
        prolog_str = program.to_prolog_string()

        assert ":- dynamic has/2." in prolog_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
