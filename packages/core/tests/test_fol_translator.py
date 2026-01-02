"""
Unit tests for AST → FOL translation.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from canto_core.parser.dsl_parser import parse_string
from canto_core.fol import (
    ASTToFOLTranslator,
    translate_to_fol,
    CantoFOL,
    RuleType,
    FOLPredicate,
    FOLEquals,
    FOLAnd,
    FOLNot,
)


class TestBasicTranslation:
    """Test basic AST to FOL translation."""

    def test_translate_variable_declaration(self):
        """Test translating a simple variable declaration."""
        dsl = """
        ?priority can be "high", "medium", "low"
        """
        ast = parse_string(dsl)
        fol = translate_to_fol(ast)

        assert "priority" in fol.variables
        var = fol.variables["priority"]
        assert var.name == "priority"

    def test_translate_semantic_category(self):
        """Test translating a semantic category."""
        dsl = """
        ?urgent_words resembles "urgent", "emergency", "critical"
        """
        ast = parse_string(dsl)
        fol = translate_to_fol(ast)

        assert "urgent_words" in fol.categories
        cat = fol.categories["urgent_words"]
        assert "urgent" in cat.patterns
        assert "emergency" in cat.patterns
        assert "critical" in cat.patterns

    def test_translate_simple_rule(self):
        """Test translating a simple strict rule."""
        dsl = """
        ?priority can be "high", "low"
        ?input meaning "input text"
        ?urgent_words resembles "urgent", "critical"

        ?priority becomes "high"
            when ?input is like ?urgent_words
        """
        ast = parse_string(dsl)
        fol = translate_to_fol(ast)

        # Should have one strict rule
        assert len(fol.strict_rules) == 1
        rule = fol.strict_rules[0]
        assert rule.head_variable == "priority"
        assert rule.head_value == "high"
        assert rule.rule_type == RuleType.STRICT

    def test_translate_defeasible_rule(self):
        """Test translating a defeasible rule."""
        dsl = """
        ?priority can be "high", "low"

        normally ?priority becomes "low"
        """
        ast = parse_string(dsl)
        fol = translate_to_fol(ast)

        # Should have one defeasible rule
        assert len(fol.defeasible_rules) == 1
        rule = fol.defeasible_rules[0]
        assert rule.head_variable == "priority"
        assert rule.head_value == "low"
        assert rule.rule_type == RuleType.DEFEASIBLE


class TestConditionTranslation:
    """Test translation of various condition types."""

    def test_is_like_condition(self):
        """Test is_like condition translation."""
        dsl = """
        ?flag can be true, false
        ?text meaning "text"
        ?pattern resembles "word1", "word2"

        ?flag becomes true when ?text is like ?pattern
        """
        ast = parse_string(dsl)
        fol = translate_to_fol(ast)

        rule = fol.strict_rules[0]
        assert rule.conditions is not None
        # Should be an is_like predicate
        assert isinstance(rule.conditions, FOLPredicate)
        assert rule.conditions.name == "is_like"

    def test_is_equals_condition(self):
        """Test is (equals) condition translation."""
        dsl = """
        ?flag can be true, false
        ?status can be "active", "inactive"

        ?flag becomes true when ?status is "active"
        """
        ast = parse_string(dsl)
        fol = translate_to_fol(ast)

        rule = fol.strict_rules[0]
        assert rule.conditions is not None

    def test_and_condition(self):
        """Test AND condition translation."""
        dsl = """
        ?result can be true, false
        ?a can be true, false
        ?b can be true, false

        ?result becomes true when ?a is true and ?b is true
        """
        ast = parse_string(dsl)
        fol = translate_to_fol(ast)

        rule = fol.strict_rules[0]
        assert rule.conditions is not None
        assert isinstance(rule.conditions, FOLAnd)
        assert len(rule.conditions.conjuncts) == 2

    def test_not_condition(self):
        """Test NOT condition (unless) translation."""
        dsl = """
        ?flag can be true, false
        ?text meaning "text"
        ?bad_words resembles "bad"

        ?flag becomes true unless ?text is like ?bad_words
        """
        ast = parse_string(dsl)
        fol = translate_to_fol(ast)

        rule = fol.strict_rules[0]
        assert rule.conditions is not None
        assert isinstance(rule.conditions, FOLNot)


class TestSuperiorityTranslation:
    """Test superiority relation translation."""

    def test_overrides_all(self):
        """Test overriding all clause."""
        dsl = """
        ?flag can be true, false

        normally ?flag becomes false
        ?flag becomes true overriding all
        """
        ast = parse_string(dsl)
        fol = translate_to_fol(ast)

        # Should have superiority relations
        assert len(fol.superiority) >= 1

    def test_overrides_normal(self):
        """Test overriding normal (defeasible only) clause."""
        dsl = """
        ?flag can be true, false

        normally ?flag becomes false
        ?flag becomes true
        ?flag becomes true overriding normal
        """
        ast = parse_string(dsl)
        fol = translate_to_fol(ast)

        # Should have superiority only over defeasible rules
        assert len(fol.superiority) >= 1


class TestFOLConversion:
    """Test conversion of CantoFOL to pure FOL formulas."""

    def test_to_fol_formulas(self):
        """Test converting CantoFOL to list of FOL formulas."""
        dsl = """
        ?priority can be "high", "low"
        ?input meaning "input"
        ?urgent resembles "urgent"

        ?priority becomes "high" when ?input is like ?urgent
        normally ?priority becomes "low"
        """
        ast = parse_string(dsl)
        fol = translate_to_fol(ast)

        formulas = fol.to_fol_formulas()
        assert len(formulas) > 0

    def test_get_rules_for_variable(self):
        """Test getting rules for a specific variable."""
        dsl = """
        ?a can be true, false
        ?b can be true, false

        ?a becomes true
        ?b becomes true
        """
        ast = parse_string(dsl)
        fol = translate_to_fol(ast)

        a_rules = fol.get_rules_for_variable("a")
        assert len(a_rules) == 1
        assert a_rules[0].head_variable == "a"

    def test_get_values_for_variable(self):
        """Test getting possible values for a variable."""
        dsl = """
        ?status can be "active", "pending", "closed"
        ?status becomes "active"
        normally ?status becomes "pending"
        """
        ast = parse_string(dsl)
        fol = translate_to_fol(ast)

        values = list(fol.get_values_for_variable("status"))
        assert "active" in values
        assert "pending" in values


class TestTranslatorClass:
    """Test ASTToFOLTranslator class methods."""

    def test_translator_source_file(self):
        """Test that source file is recorded."""
        dsl = """?flag can be true, false"""
        ast = parse_string(dsl)

        translator = ASTToFOLTranslator()
        fol = translator.translate(ast, source_file="test.canto")

        assert fol.source_file == "test.canto"

    def test_translator_multiple_calls(self):
        """Test that translator can be reused."""
        translator = ASTToFOLTranslator()

        dsl1 = """?a can be true, false"""
        dsl2 = """?b can be true, false"""

        ast1 = parse_string(dsl1)
        ast2 = parse_string(dsl2)

        fol1 = translator.translate(ast1)
        fol2 = translator.translate(ast2)

        assert "a" in fol1.variables
        assert "b" in fol2.variables


class TestNestedStructures:
    """Test translation of nested 'with' structures."""

    def test_with_block_translation(self):
        """Test that 'with' blocks create has relationships."""
        dsl = """
        ?patient meaning "patient" with
            ?name meaning "name"
            ?age meaning "age"
        """
        ast = parse_string(dsl)
        fol = translate_to_fol(ast)

        assert "patient" in fol.variables
        assert "name" in fol.variables
        assert "age" in fol.variables

        # Should have has_relationships
        assert len(fol.has_relationships) >= 2

    def test_deeply_nested_with_block(self):
        """Test deeply nested 'with' blocks."""
        dsl = """
        ?root meaning "root" with
            ?level1 meaning "level1" with
                ?level2 meaning "level2"
        """
        ast = parse_string(dsl)
        fol = translate_to_fol(ast)

        assert "root" in fol.variables
        assert "level1" in fol.variables
        assert "level2" in fol.variables


class TestAllConditionTypes:
    """Test FOL translation of ALL condition types."""

    def test_has_any_is_not(self):
        """Test HAS_ANY_IS_NOT translation."""
        dsl = '''
?items meaning "items"
?flag can be true, false
?flag becomes true when ?items has any that is not "approved"
'''
        ast = parse_string(dsl)
        fol = translate_to_fol(ast)

        rule = fol.strict_rules[0]
        assert rule.conditions is not None
        assert rule.conditions.name == "has_any_neq"

    def test_has_all_is_not(self):
        """Test HAS_ALL_IS_NOT translation."""
        dsl = '''
?items meaning "items"
?flag can be true, false
?flag becomes true when ?items has all that is not "invalid"
'''
        ast = parse_string(dsl)
        fol = translate_to_fol(ast)

        rule = fol.strict_rules[0]
        assert rule.conditions is not None
        assert rule.conditions.name == "has_all_neq"

    def test_has_none_is_not(self):
        """Test HAS_NONE_IS_NOT translation (negation of has_any_neq)."""
        dsl = '''
?items meaning "items"
?flag can be true, false
?flag becomes true when ?items has none that is not "valid"
'''
        ast = parse_string(dsl)
        fol = translate_to_fol(ast)

        rule = fol.strict_rules[0]
        assert rule.conditions is not None
        # HAS_NONE_IS_NOT translates to NOT(has_any_neq)
        assert isinstance(rule.conditions, FOLNot)

    def test_has_any_property_is(self):
        """Test HAS_ANY_PROPERTY_IS translation."""
        dsl = '''
?order meaning "order"
?order has a list of ?items meaning "items"
?flag can be true, false
?flag becomes true when ?items of ?order has any that ?status is "shipped"
'''
        ast = parse_string(dsl)
        fol = translate_to_fol(ast)

        rule = fol.strict_rules[0]
        assert rule.conditions is not None
        assert "has_any_prop" in rule.conditions.name

    def test_has_any_property_is_like(self):
        """Test HAS_ANY_PROPERTY_IS_LIKE translation."""
        dsl = '''
?patient meaning "patient"
?patient has a list of ?symptoms meaning "symptoms"
?emergency resembles "chest pain"
?flag can be true, false
?flag becomes true when ?symptoms of ?patient has any that ?text is like ?emergency
'''
        ast = parse_string(dsl)
        fol = translate_to_fol(ast)

        rule = fol.strict_rules[0]
        assert rule.conditions is not None
        assert "prop_like" in rule.conditions.name

    def test_compare_gt(self):
        """Test COMPARE_GT translation."""
        dsl = '''
?count meaning "count"
?flag can be true, false
?flag becomes true when ?count > 5
'''
        ast = parse_string(dsl)
        fol = translate_to_fol(ast)

        rule = fol.strict_rules[0]
        assert rule.conditions is not None
        assert rule.conditions.name == "compare_gt"

    def test_compare_lt(self):
        """Test COMPARE_LT translation."""
        dsl = '''
?count meaning "count"
?flag can be true, false
?flag becomes true when ?count < 10
'''
        ast = parse_string(dsl)
        fol = translate_to_fol(ast)

        rule = fol.strict_rules[0]
        assert rule.conditions is not None
        assert rule.conditions.name == "compare_lt"

    def test_compare_gte(self):
        """Test COMPARE_GTE translation."""
        dsl = '''
?count meaning "count"
?flag can be true, false
?flag becomes true when ?count >= 3
'''
        ast = parse_string(dsl)
        fol = translate_to_fol(ast)

        rule = fol.strict_rules[0]
        assert rule.conditions is not None
        assert rule.conditions.name == "compare_gte"

    def test_compare_lte(self):
        """Test COMPARE_LTE translation."""
        dsl = '''
?count meaning "count"
?flag can be true, false
?flag becomes true when ?count <= 100
'''
        ast = parse_string(dsl)
        fol = translate_to_fol(ast)

        rule = fol.strict_rules[0]
        assert rule.conditions is not None
        assert rule.conditions.name == "compare_lte"

    def test_length_where_is_eq(self):
        """Test LENGTH_WHERE_IS_EQ translation."""
        dsl = '''
?items meaning "items"
?flag can be true, false
?flag becomes true when (length of ?items where ?status is "active") is 3
'''
        ast = parse_string(dsl)
        fol = translate_to_fol(ast)

        rule = fol.strict_rules[0]
        assert rule.conditions is not None
        assert "length_where_is_eq" in rule.conditions.name

    def test_length_where_like_gt(self):
        """Test LENGTH_WHERE_LIKE_GT translation."""
        dsl = '''
?items meaning "items"
?good resembles "good"
?flag can be true, false
?flag becomes true when (length of ?items where ?text is like ?good) > 2
'''
        ast = parse_string(dsl)
        fol = translate_to_fol(ast)

        rule = fol.strict_rules[0]
        assert rule.conditions is not None
        assert "length_where_like_gt" in rule.conditions.name

    def test_let_any_bound(self):
        """Test LET_ANY_BOUND translation."""
        dsl = '''
?chain has a list of ?link meaning "chain"
?link meaning "link" with
    ?speaker meaning "speaker"
?target meaning "target"
?flag can be true, false
?flag becomes true when let ?link be any in ?chain where ?speaker of ?link is ?target
'''
        ast = parse_string(dsl)
        fol = translate_to_fol(ast)

        rule = fol.strict_rules[0]
        assert rule.conditions is not None
        assert "let_any" in rule.conditions.name

    def test_let_all_bound(self):
        """Test LET_ALL_BOUND translation."""
        dsl = '''
?items has a list of ?item meaning "items"
?item meaning "item" with
    ?status can be "ok", "error"
?flag can be true, false
?flag becomes true when let ?item be all in ?items where ?status of ?item is "ok"
'''
        ast = parse_string(dsl)
        fol = translate_to_fol(ast)

        rule = fol.strict_rules[0]
        assert rule.conditions is not None
        assert "let_all" in rule.conditions.name

    def test_let_none_bound(self):
        """Test LET_NONE_BOUND translation."""
        dsl = '''
?items has a list of ?item meaning "items"
?item meaning "item" with
    ?is_error can be true, false
?flag can be true, false
?flag becomes true when let ?item be none in ?items where ?is_error of ?item is true
'''
        ast = parse_string(dsl)
        fol = translate_to_fol(ast)

        rule = fol.strict_rules[0]
        assert rule.conditions is not None
        assert "let_none" in rule.conditions.name


class TestComplexDSL:
    """Test translation of complex DSL examples."""

    def test_medical_triage_example(self):
        """Test a medical triage-like example."""
        dsl = """
        ?triage_level can be "critical", "urgent", "standard"
        ?symptoms meaning "patient symptoms"
        ?critical_symptoms resembles "chest pain", "difficulty breathing", "unconscious"
        ?urgent_symptoms resembles "high fever", "severe pain"

        ?triage_level becomes "critical"
            when ?symptoms is like ?critical_symptoms

        normally ?triage_level becomes "urgent"
            when ?symptoms is like ?urgent_symptoms

        normally ?triage_level becomes "standard"

        ?triage_level becomes "critical"
            when ?symptoms is like ?critical_symptoms
            overriding all
        """
        ast = parse_string(dsl)
        fol = translate_to_fol(ast)

        # Should have the variable
        assert "triage_level" in fol.variables

        # Should have categories
        assert "critical_symptoms" in fol.categories
        assert "urgent_symptoms" in fol.categories

        # Should have multiple rules
        assert len(list(fol.all_rules())) >= 4

        # Should have superiority
        assert len(fol.superiority) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
