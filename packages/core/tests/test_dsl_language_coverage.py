"""
Comprehensive DSL Language Coverage Tests

This test file ensures all DSL language constructs are tested.
It serves as both a test suite and documentation of the language features.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from canto_core.parser.dsl_parser import parse_string
from canto_core.ast_nodes import (
    Rule, Condition, SemanticCategory, VariableDeclaration,
    HasDeclaration, ImportDeclaration
)
from canto_core.ast_nodes.rules import RulePriority, OverrideTarget
from canto_core.fol import translate_to_fol
from canto_core.fol import translate_to_fol


# =============================================================================
# DECLARATIONS
# =============================================================================

class TestDeclarations:
    """Tests for all declaration types."""

    def test_freetext_variable(self):
        """Test: ?var meaning "description" """
        dsl = '?query meaning "the user query"'
        ast = parse_string(dsl)

        assert len(ast) == 1
        var = ast[0]
        assert isinstance(var, VariableDeclaration)
        assert var.name == "query"
        assert var.description == "the user query"
        assert var.values_from is None

    def test_enum_variable_strings(self):
        """Test: ?var can be "a", "b", "c" """
        dsl = '?status can be "active", "pending", "closed"'
        ast = parse_string(dsl)

        var = ast[0]
        assert isinstance(var, VariableDeclaration)
        assert var.name == "status"
        assert var.values_from == ["active", "pending", "closed"]

    def test_enum_variable_booleans(self):
        """Test: ?var can be true, false """
        dsl = '?flag can be true, false'
        ast = parse_string(dsl)

        var = ast[0]
        assert var.values_from == ["true", "false"]

    def test_enum_with_description(self):
        """Test: ?var can be "a", "b" meaning "desc" """
        dsl = '?priority can be "high", "low" meaning "task priority"'
        ast = parse_string(dsl)

        var = ast[0]
        assert var.values_from == ["high", "low"]
        assert var.description == "task priority"

    def test_semantic_category(self):
        """Test: @category resembles "pattern1", "pattern2" """
        dsl = '?urgent_words resembles "urgent", "asap", "critical"'
        ast = parse_string(dsl)

        cat = ast[0]
        assert isinstance(cat, SemanticCategory)
        assert cat.name == "urgent_words"
        assert cat.patterns == ["urgent", "asap", "critical"]

    def test_semantic_category_with_description(self):
        """Test: @category resembles "pattern" meaning "desc" """
        dsl = '?positive resembles "good", "great" meaning "positive sentiment"'
        ast = parse_string(dsl)

        cat = ast[0]
        assert cat.description == "positive sentiment"

    def test_has_single(self):
        """Test: ?parent has a ?child meaning "desc" """
        dsl = '?patient has a ?diagnosis meaning "the diagnosis"'
        ast = parse_string(dsl)

        has = ast[0]
        assert isinstance(has, HasDeclaration)
        assert has.parent == "patient"
        assert has.child == "diagnosis"
        assert has.is_list is False

    def test_has_list(self):
        """Test: ?parent has a list of ?child meaning "desc" """
        dsl = '?patient has a list of ?symptoms meaning "symptoms list"'
        ast = parse_string(dsl)

        has = ast[0]
        assert has.is_list is True

    def test_has_list_from(self):
        """Test: ?parent has a list of ?child from ?source """
        dsl = '?entities has a list of ?company from ?text'
        ast = parse_string(dsl)

        has = ast[0]
        assert has.source == "text"

    def test_with_block_simple(self):
        """Test: ?var meaning "desc" with nested children """
        dsl = '''
?patient meaning "patient info" with
    ?name meaning "patient name"
    ?age meaning "patient age"
'''
        ast = parse_string(dsl)

        var = ast[0]
        assert len(var.children) == 2

    def test_with_block_deeply_nested(self):
        """Test: deeply nested with blocks """
        dsl = '''
?root meaning "root" with
    ?level1 meaning "level 1" with
        ?level2 meaning "level 2" with
            ?level3 meaning "level 3"
'''
        ast = parse_string(dsl)

        root = ast[0]
        assert len(root.children) == 1
        level1 = root.children[0]
        assert len(level1.children) == 1

    def test_variable_from_source(self):
        """Test: ?var meaning "desc" from ?source """
        dsl = '?entities meaning "extracted entities" from ?text'
        ast = parse_string(dsl)

        var = ast[0]
        assert var.source == "text"


# =============================================================================
# RULES
# =============================================================================

class TestRules:
    """Tests for rule constructs."""

    def test_unconditional_rule(self):
        """Test: ?var becomes value """
        dsl = '''
?flag can be true, false
?flag becomes true
'''
        ast = parse_string(dsl)
        rule = [n for n in ast if isinstance(n, Rule)][0]

        assert rule.head_variable == "flag"
        assert rule.head_value is True
        assert rule.conditions == []

    def test_conditional_rule_when(self):
        """Test: ?var becomes value when condition """
        dsl = '''
?flag can be true, false
?status can be "active", "inactive"
?flag becomes true when ?status is "active"
'''
        ast = parse_string(dsl)
        rule = [n for n in ast if isinstance(n, Rule)][0]

        assert len(rule.conditions) == 1

    def test_conditional_rule_unless(self):
        """Test: ?var becomes value unless condition """
        dsl = '''
?flag can be true, false
?text meaning "text"
?bad resembles "bad"
?flag becomes true unless ?text is like ?bad
'''
        ast = parse_string(dsl)
        rule = [n for n in ast if isinstance(n, Rule)][0]

        assert len(rule.exceptions) == 1

    def test_defeasible_rule_normally(self):
        """Test: normally ?var becomes value """
        dsl = '''
?flag can be true, false
normally ?flag becomes false
'''
        ast = parse_string(dsl)
        rule = [n for n in ast if isinstance(n, Rule)][0]

        assert rule.priority == RulePriority.NORMAL
        assert rule.is_defeasible()

    def test_strict_rule(self):
        """Test: strict rule (no normally keyword) """
        dsl = '''
?flag can be true, false
?flag becomes true
'''
        ast = parse_string(dsl)
        rule = [n for n in ast if isinstance(n, Rule)][0]

        assert rule.priority == RulePriority.STRICT
        assert rule.is_strict()

    def test_overriding_all(self):
        """Test: ?var becomes value overriding all """
        dsl = '''
?flag can be true, false
?flag becomes true overriding all
'''
        ast = parse_string(dsl)
        rule = [n for n in ast if isinstance(n, Rule)][0]

        assert rule.override_target == OverrideTarget.ALL

    def test_overriding_normal(self):
        """Test: ?var becomes value overriding normal """
        dsl = '''
?flag can be true, false
?flag becomes true overriding normal
'''
        ast = parse_string(dsl)
        rule = [n for n in ast if isinstance(n, Rule)][0]

        assert rule.override_target == OverrideTarget.NORMAL

    def test_qualified_variable_simple(self):
        """Test: ?child of ?parent becomes value """
        dsl = '''
?parent meaning "parent"
?child can be true, false
?parent has a ?child meaning "child property"
?child of ?parent becomes true
'''
        ast = parse_string(dsl)
        rule = [n for n in ast if isinstance(n, Rule)][0]

        assert rule.head_variable == "child"
        assert rule.head_parent == "parent"

    def test_qualified_variable_nested(self):
        """Test: ?child of (?grandchild of ?grandparent) becomes value """
        dsl = '''
?grandparent meaning "gp"
?grandchild meaning "gc"
?child meaning "c"
?child of (?grandchild of ?grandparent) becomes true
'''
        ast = parse_string(dsl)
        rule = [n for n in ast if isinstance(n, Rule)][0]

        assert rule.head_variable == "child"
        assert isinstance(rule.head_parent, tuple)

    def test_rule_string_value(self):
        """Test: ?var becomes "string value" """
        dsl = '''
?status meaning "status"
?status becomes "active"
'''
        ast = parse_string(dsl)
        rule = [n for n in ast if isinstance(n, Rule)][0]

        assert rule.head_value == "active"

    def test_rule_numeric_value(self):
        """Test: ?var becomes 42 """
        dsl = '''
?count meaning "count"
?count becomes 42
'''
        ast = parse_string(dsl)
        rule = [n for n in ast if isinstance(n, Rule)][0]

        assert rule.head_value == 42


# =============================================================================
# CONDITIONS - BASIC
# =============================================================================

class TestBasicConditions:
    """Tests for basic condition operators."""

    def test_is_string(self):
        """Test: ?var is "value" """
        dsl = '''
?status can be "active", "inactive"
?flag can be true, false
?flag becomes true when ?status is "active"
'''
        ast = parse_string(dsl)
        rule = [n for n in ast if isinstance(n, Rule)][0]
        cond = rule.conditions[0]

        assert cond.operator == "IS"
        assert cond.left == "?status"
        assert cond.right == "active"

    def test_is_boolean(self):
        """Test: ?var is true """
        dsl = '''
?enabled can be true, false
?result can be true, false
?result becomes true when ?enabled is true
'''
        ast = parse_string(dsl)
        rule = [n for n in ast if isinstance(n, Rule)][0]
        cond = rule.conditions[0]

        assert cond.right is True

    def test_is_like(self):
        """Test: ?var is like @category """
        dsl = '''
?text meaning "text"
?positive resembles "good", "great"
?flag can be true, false
?flag becomes true when ?text is like ?positive
'''
        ast = parse_string(dsl)
        rule = [n for n in ast if isinstance(n, Rule)][0]
        cond = rule.conditions[0]

        assert cond.operator == "IS_LIKE"

    def test_not_naf(self):
        """Test: not ?var is value (Negation as Failure) """
        dsl = '''
?flag can be true, false
?result can be true, false
?result becomes true when not ?flag is true
'''
        ast = parse_string(dsl)
        rule = [n for n in ast if isinstance(n, Rule)][0]
        cond = rule.conditions[0]

        assert cond.operator == "NOT"

    def test_not_warranted(self):
        """Test: not warranted ?var is value """
        dsl = '''
?intent meaning "intent"
?result meaning "result"
?result becomes "test" when not warranted ?intent is "prevention"
'''
        ast = parse_string(dsl)
        rule = [n for n in ast if isinstance(n, Rule)][0]
        cond = rule.conditions[0]

        assert cond.operator == "NOT_WARRANTED"

    def test_and_conditions(self):
        """Test: condition1 and condition2 """
        dsl = '''
?a can be true, false
?b can be true, false
?result can be true, false
?result becomes true when ?a is true and ?b is true
'''
        ast = parse_string(dsl)
        rule = [n for n in ast if isinstance(n, Rule)][0]
        cond = rule.conditions[0]

        assert cond.operator == "AND"

    def test_or_conditions(self):
        """Test: condition1 or condition2 """
        dsl = '''
?a can be true, false
?b can be true, false
?result can be true, false
?result becomes true when ?a is true or ?b is true
'''
        ast = parse_string(dsl)
        rule = [n for n in ast if isinstance(n, Rule)][0]
        cond = rule.conditions[0]

        assert cond.operator == "OR"

    def test_parenthesized_conditions(self):
        """Test: (condition1 and condition2) or condition3 """
        dsl = '''
?a can be true, false
?b can be true, false
?c can be true, false
?result can be true, false
?result becomes true when (?a is true and ?b is true) or ?c is true
'''
        ast = parse_string(dsl)
        rule = [n for n in ast if isinstance(n, Rule)][0]

        assert len(rule.conditions) == 1


# =============================================================================
# CONDITIONS - HAS
# =============================================================================

class TestHasConditions:
    """Tests for HAS condition operators."""

    def test_has_simple(self):
        """Test: ?list has "value" """
        dsl = '''
?symptoms meaning "symptoms list"
?flag can be true, false
?flag becomes true when ?symptoms has "chest pain"
'''
        ast = parse_string(dsl)
        rule = [n for n in ast if isinstance(n, Rule)][0]
        cond = rule.conditions[0]

        assert cond.operator == "HAS"

    def test_has_qualified(self):
        """Test: ?child of ?parent has "value" """
        dsl = '''
?patient meaning "patient"
?patient has a list of ?symptoms meaning "symptoms"
?flag can be true, false
?flag becomes true when ?symptoms of ?patient has "pain"
'''
        ast = parse_string(dsl)
        rule = [n for n in ast if isinstance(n, Rule)][0]
        cond = rule.conditions[0]

        assert cond.operator == "HAS"
        assert isinstance(cond.left, dict)  # qualified

    def test_has_any_is(self):
        """Test: ?list has any that is "value" """
        dsl = '''
?list meaning "list"
?flag can be true, false
?flag becomes true when ?list has any that is "active"
'''
        ast = parse_string(dsl)
        rule = [n for n in ast if isinstance(n, Rule)][0]
        cond = rule.conditions[0]

        assert cond.operator == "HAS_ANY_IS"

    def test_has_any_is_like(self):
        """Test: ?list has any that is like @category """
        dsl = '''
?symptoms meaning "symptoms"
?emergency resembles "chest pain"
?flag can be true, false
?flag becomes true when ?symptoms has any that is like ?emergency
'''
        ast = parse_string(dsl)
        rule = [n for n in ast if isinstance(n, Rule)][0]
        cond = rule.conditions[0]

        assert cond.operator == "HAS_ANY_IS_LIKE"

    def test_has_any_is_not(self):
        """Test: ?list has any that is not "value" """
        dsl = '''
?items meaning "items"
?flag can be true, false
?flag becomes true when ?items has any that is not "approved"
'''
        ast = parse_string(dsl)
        rule = [n for n in ast if isinstance(n, Rule)][0]
        cond = rule.conditions[0]

        assert cond.operator == "HAS_ANY_IS_NOT"

    def test_has_all_is(self):
        """Test: ?list has all that is "value" """
        dsl = '''
?items meaning "items"
?flag can be true, false
?flag becomes true when ?items has all that is "complete"
'''
        ast = parse_string(dsl)
        rule = [n for n in ast if isinstance(n, Rule)][0]
        cond = rule.conditions[0]

        assert cond.operator == "HAS_ALL_IS"

    def test_has_all_is_like(self):
        """Test: ?list has all that is like @category """
        dsl = '''
?items meaning "items"
?good resembles "good"
?flag can be true, false
?flag becomes true when ?items has all that is like ?good
'''
        ast = parse_string(dsl)
        rule = [n for n in ast if isinstance(n, Rule)][0]
        cond = rule.conditions[0]

        assert cond.operator == "HAS_ALL_IS_LIKE"

    def test_has_all_is_not(self):
        """Test: ?list has all that is not "value" """
        dsl = '''
?items meaning "items"
?flag can be true, false
?flag becomes true when ?items has all that is not "invalid"
'''
        ast = parse_string(dsl)
        rule = [n for n in ast if isinstance(n, Rule)][0]
        cond = rule.conditions[0]

        assert cond.operator == "HAS_ALL_IS_NOT"

    def test_has_none_is(self):
        """Test: ?list has none that is "value" """
        dsl = '''
?items meaning "items"
?flag can be true, false
?flag becomes true when ?items has none that is "error"
'''
        ast = parse_string(dsl)
        rule = [n for n in ast if isinstance(n, Rule)][0]
        cond = rule.conditions[0]

        assert cond.operator == "HAS_NONE_IS"

    def test_has_none_is_like(self):
        """Test: ?list has none that is like @category """
        dsl = '''
?items meaning "items"
?bad resembles "bad"
?flag can be true, false
?flag becomes true when ?items has none that is like ?bad
'''
        ast = parse_string(dsl)
        rule = [n for n in ast if isinstance(n, Rule)][0]
        cond = rule.conditions[0]

        assert cond.operator == "HAS_NONE_IS_LIKE"

    def test_has_none_is_not(self):
        """Test: ?list has none that is not "value" """
        dsl = '''
?items meaning "items"
?flag can be true, false
?flag becomes true when ?items has none that is not "valid"
'''
        ast = parse_string(dsl)
        rule = [n for n in ast if isinstance(n, Rule)][0]
        cond = rule.conditions[0]

        assert cond.operator == "HAS_NONE_IS_NOT"


# =============================================================================
# CONDITIONS - PROPERTY-BASED QUANTIFIED
# =============================================================================

class TestPropertyQuantifiedConditions:
    """Tests for property-based quantified conditions."""

    def test_has_any_property_is(self):
        """Test: ?list has any that ?property is value """
        dsl = '''
?order meaning "order"
?order has a list of ?items meaning "items"
?flag can be true, false
?flag becomes true when ?items of ?order has any that ?status is "shipped"
'''
        ast = parse_string(dsl)
        rule = [n for n in ast if isinstance(n, Rule)][0]
        cond = rule.conditions[0]

        assert "PROPERTY" in cond.operator or "HAS_ANY" in cond.operator

    def test_has_any_property_is_like(self):
        """Test: ?list has any that ?property is like @category """
        dsl = '''
?patient meaning "patient"
?patient has a list of ?symptoms meaning "symptoms"
?emergency resembles "chest pain"
?flag can be true, false
?flag becomes true when ?symptoms of ?patient has any that ?text is like ?emergency
'''
        ast = parse_string(dsl)
        rule = [n for n in ast if isinstance(n, Rule)][0]

        assert len(rule.conditions) == 1


# =============================================================================
# CONDITIONS - LET BINDING
# =============================================================================

class TestLetBindingConditions:
    """Tests for let binding conditions."""

    def test_let_any_bound(self):
        """Test: let ?x be any in ?collection where conditions """
        dsl = '''
?chain has a list of ?link meaning "chain"
?link meaning "link" with
    ?speaker meaning "speaker"
?target meaning "target"
?result can be true, false
?result becomes true when let ?link be any in ?chain where ?speaker of ?link is ?target
'''
        ast = parse_string(dsl)
        rule = [n for n in ast if isinstance(n, Rule)][0]
        cond = rule.conditions[0]

        assert cond.operator == "LET_ANY_BOUND"

    def test_let_all_bound(self):
        """Test: let ?x be all in ?collection where conditions """
        dsl = '''
?items has a list of ?item meaning "items"
?item meaning "item" with
    ?status can be "ok", "error"
?result can be true, false
?result becomes true when let ?item be all in ?items where ?status of ?item is "ok"
'''
        ast = parse_string(dsl)
        rule = [n for n in ast if isinstance(n, Rule)][0]
        cond = rule.conditions[0]

        assert cond.operator == "LET_ALL_BOUND"

    def test_let_none_bound(self):
        """Test: let ?x be none in ?collection where conditions """
        dsl = '''
?items has a list of ?item meaning "items"
?item meaning "item" with
    ?is_error can be true, false
?result can be true, false
?result becomes true when let ?item be none in ?items where ?is_error of ?item is true
'''
        ast = parse_string(dsl)
        rule = [n for n in ast if isinstance(n, Rule)][0]
        cond = rule.conditions[0]

        assert cond.operator == "LET_NONE_BOUND"

    def test_let_with_compound_conditions(self):
        """Test: let binding with and/or in where clause """
        dsl = '''
?chain has a list of ?link meaning "chain"
?link meaning "link" with
    ?speaker meaning "speaker"
    ?valid can be true, false
?target meaning "target"
?result can be true, false
?result becomes true when let ?link be any in ?chain where ?speaker of ?link is ?target and ?valid of ?link is true
'''
        ast = parse_string(dsl)
        rule = [n for n in ast if isinstance(n, Rule)][0]

        assert len(rule.conditions) == 1


# =============================================================================
# CONDITIONS - LENGTH OF
# =============================================================================

class TestLengthConditions:
    """Tests for length of conditions."""

    def test_length_where_is_eq(self):
        """Test: (length of ?list where ?prop is value) is N """
        dsl = '''
?claims meaning "claims"
?result can be true, false
?result becomes true when (length of ?claims where ?truth is false) is 0
'''
        ast = parse_string(dsl)
        rule = [n for n in ast if isinstance(n, Rule)][0]
        cond = rule.conditions[0]

        assert "LENGTH_WHERE_IS_EQ" in cond.operator

    def test_length_where_is_gt(self):
        """Test: (length of ?list where ?prop is value) > N """
        dsl = '''
?items meaning "items"
?result can be true, false
?result becomes true when (length of ?items where ?status is "active") > 5
'''
        ast = parse_string(dsl)
        rule = [n for n in ast if isinstance(n, Rule)][0]
        cond = rule.conditions[0]

        assert "LENGTH_WHERE_IS_GT" in cond.operator

    def test_length_where_is_lt(self):
        """Test: (length of ?list where ?prop is value) < N """
        dsl = '''
?items meaning "items"
?result can be true, false
?result becomes true when (length of ?items where ?status is "error") < 3
'''
        ast = parse_string(dsl)
        rule = [n for n in ast if isinstance(n, Rule)][0]
        cond = rule.conditions[0]

        assert "LENGTH_WHERE_IS_LT" in cond.operator

    def test_length_where_is_gte(self):
        """Test: (length of ?list where ?prop is value) >= N """
        dsl = '''
?items meaning "items"
?result can be true, false
?result becomes true when (length of ?items where ?status is "ok") >= 2
'''
        ast = parse_string(dsl)
        rule = [n for n in ast if isinstance(n, Rule)][0]
        cond = rule.conditions[0]

        assert "LENGTH_WHERE_IS_GTE" in cond.operator

    def test_length_where_is_lte(self):
        """Test: (length of ?list where ?prop is value) <= N """
        dsl = '''
?items meaning "items"
?result can be true, false
?result becomes true when (length of ?items where ?status is "pending") <= 10
'''
        ast = parse_string(dsl)
        rule = [n for n in ast if isinstance(n, Rule)][0]
        cond = rule.conditions[0]

        assert "LENGTH_WHERE_IS_LTE" in cond.operator

    def test_length_where_like_gt(self):
        """Test: (length of ?list where ?prop is like @cat) > N """
        dsl = '''
?items meaning "items"
?good resembles "good", "great"
?result can be true, false
?result becomes true when (length of ?items where ?text is like ?good) > 3
'''
        ast = parse_string(dsl)
        rule = [n for n in ast if isinstance(n, Rule)][0]
        cond = rule.conditions[0]

        assert "LENGTH_WHERE_LIKE_GT" in cond.operator

    def test_length_qualified_list(self):
        """Test: (length of ?child of ?parent where ...) """
        dsl = '''
?puzzle meaning "puzzle"
?claims meaning "claims"
?result can be true, false
?result becomes true when (length of ?claims of ?puzzle where ?truth is false) is 0
'''
        ast = parse_string(dsl)
        rule = [n for n in ast if isinstance(n, Rule)][0]
        cond = rule.conditions[0]

        # The list reference should be qualified
        assert isinstance(cond.left.get('list'), dict)


# =============================================================================
# CONDITIONS - COMPARISON
# =============================================================================

class TestComparisonConditions:
    """Tests for comparison conditions."""

    def test_compare_gt(self):
        """Test: ?var > N """
        dsl = '''
?count meaning "count"
?result can be true, false
?result becomes true when ?count > 5
'''
        ast = parse_string(dsl)
        rule = [n for n in ast if isinstance(n, Rule)][0]
        cond = rule.conditions[0]

        assert cond.operator == "COMPARE_GT"
        assert cond.right == 5

    def test_compare_lt(self):
        """Test: ?var < N """
        dsl = '''
?count meaning "count"
?result can be true, false
?result becomes true when ?count < 10
'''
        ast = parse_string(dsl)
        rule = [n for n in ast if isinstance(n, Rule)][0]
        cond = rule.conditions[0]

        assert cond.operator == "COMPARE_LT"

    def test_compare_gte(self):
        """Test: ?var >= N """
        dsl = '''
?count meaning "count"
?result can be true, false
?result becomes true when ?count >= 3
'''
        ast = parse_string(dsl)
        rule = [n for n in ast if isinstance(n, Rule)][0]
        cond = rule.conditions[0]

        assert cond.operator == "COMPARE_GTE"

    def test_compare_lte(self):
        """Test: ?var <= N """
        dsl = '''
?count meaning "count"
?result can be true, false
?result becomes true when ?count <= 100
'''
        ast = parse_string(dsl)
        rule = [n for n in ast if isinstance(n, Rule)][0]
        cond = rule.conditions[0]

        assert cond.operator == "COMPARE_LTE"


# =============================================================================
# TRANSLATION TO DELP
# =============================================================================

class TestDeLPTranslation:
    """Tests for DeLPTranslator coverage of all constructs."""

    def test_translate_all_quantified_has(self):
        """Test that all 9 quantified has operators translate correctly."""
        dsl = '''
?p meaning "parent"
?p has a list of ?c meaning "children"
?cat resembles "item"

?r1 can be true, false
?r2 can be true, false
?r3 can be true, false
?r4 can be true, false
?r5 can be true, false
?r6 can be true, false
?r7 can be true, false
?r8 can be true, false
?r9 can be true, false

?r1 becomes true when ?c of ?p has any that is like ?cat
?r2 becomes true when ?c of ?p has any that is "value"
?r3 becomes true when ?c of ?p has any that is not "value"
?r4 becomes true when ?c of ?p has all that is like ?cat
?r5 becomes true when ?c of ?p has all that is "value"
?r6 becomes true when ?c of ?p has all that is not "value"
?r7 becomes true when ?c of ?p has none that is like ?cat
?r8 becomes true when ?c of ?p has none that is "value"
?r9 becomes true when ?c of ?p has none that is not "value"
'''
        ast = parse_string(dsl)
        program = translate_to_fol(ast)
        prolog = program.to_prolog_string()

        # All predicates should be present
        assert "has_any_like" in prolog
        assert "has_any_eq" in prolog
        assert "has_any_neq" in prolog
        assert "has_all_like" in prolog
        assert "has_all_eq" in prolog
        assert "has_all_neq" in prolog
        assert "has_none_like" in prolog
        assert "has_none_eq" in prolog
        assert "has_none_neq" in prolog

    def test_translate_let_binding(self):
        """Test let binding translation."""
        dsl = '''
?items has a list of ?item meaning "items"
?item meaning "item" with
    ?status can be "ok", "error"
?result can be true, false
?result becomes true when let ?item be any in ?items where ?status of ?item is "ok"
'''
        ast = parse_string(dsl)
        program = translate_to_fol(ast)

        rule = program.strict_rules[0]
        # FOL translator uses let_any (not let_any_bound)
        assert "let_any(" in rule.body[0]

    def test_translate_length_where(self):
        """Test length where translation."""
        dsl = '''
?items meaning "items"
?result can be true, false
?result becomes true when (length of ?items where ?status is "active") > 5
'''
        ast = parse_string(dsl)
        program = translate_to_fol(ast)

        rule = program.strict_rules[0]
        assert "length_where_is_gt" in rule.body[0]


# =============================================================================
# TRANSLATION TO FOL
# =============================================================================

class TestFOLTranslation:
    """Tests for FOL translator coverage."""

    def test_fol_basic_is_like(self):
        """Test FOL translation of is_like."""
        dsl = '''
?flag can be true, false
?text meaning "text"
?pattern resembles "word"
?flag becomes true when ?text is like ?pattern
'''
        ast = parse_string(dsl)
        fol = translate_to_fol(ast)

        rule = fol.strict_rules[0]
        assert rule.conditions is not None
        assert rule.conditions.name == "is_like"

    def test_fol_and_condition(self):
        """Test FOL translation of AND."""
        dsl = '''
?a can be true, false
?b can be true, false
?result can be true, false
?result becomes true when ?a is true and ?b is true
'''
        ast = parse_string(dsl)
        fol = translate_to_fol(ast)

        from canto_core.fol import FOLAnd
        rule = fol.strict_rules[0]
        assert isinstance(rule.conditions, FOLAnd)

    def test_fol_not_condition(self):
        """Test FOL translation of NOT."""
        dsl = '''
?flag can be true, false
?result can be true, false
?result becomes true when not ?flag is true
'''
        ast = parse_string(dsl)
        fol = translate_to_fol(ast)

        from canto_core.fol import FOLNot
        rule = fol.strict_rules[0]
        assert isinstance(rule.conditions, FOLNot)


# =============================================================================
# INTEGRATION - COMPLETE EXAMPLES
# =============================================================================

class TestCompleteExamples:
    """Integration tests with complete DSL programs."""

    def test_medical_triage_complete(self):
        """Test complete medical triage example."""
        dsl = '''
?emergency_symptoms resembles "chest pain", "difficulty breathing"
?urgent_symptoms resembles "high fever", "severe pain"

?patient meaning "patient" with
    ?symptoms with
        ?text meaning "symptom text"
        ?is_emergency can be true, false

?is_emergency becomes true when ?text is like ?emergency_symptoms

?urgency can be "emergency", "urgent", "routine"

?urgency becomes "emergency"
    when ?symptoms of ?patient has any that ?is_emergency is true
    overriding all

normally ?urgency becomes "urgent"
    when ?symptoms of ?patient has any that is like ?urgent_symptoms

normally ?urgency becomes "routine"
'''
        ast = parse_string(dsl)

        categories = [n for n in ast if isinstance(n, SemanticCategory)]
        assert len(categories) == 2

        rules = [n for n in ast if isinstance(n, Rule)]
        assert len(rules) >= 4

        # Should translate without errors
        program = translate_to_fol(ast)

        assert len(program.strict_rules) >= 2
        assert len(program.defeasible_rules) >= 2

    def test_web_of_lies_complete(self):
        """Test complete web of lies example."""
        dsl = '''
?truth_claim resembles "tells the truth", "is truthful"
?lie_claim resembles "lies", "is a liar"

?puzzle meaning "the puzzle" with
    ?first_person_is_truthful can be true, false
    ?claims with
        ?speaker meaning "who speaks"
        ?assertion meaning "what they say"
        ?claims_truth can be true, false

?claims_truth of ?claims becomes true when ?assertion of ?claims is like ?truth_claim
?claims_truth of ?claims becomes false when ?assertion of ?claims is like ?lie_claim

?answer can be "Yes", "No"

?answer becomes "Yes"
    when ?first_person_is_truthful of ?puzzle is true
    and (length of ?claims of ?puzzle where ?claims_truth is false) is 0

?answer becomes "Yes"
    when ?first_person_is_truthful of ?puzzle is false
    and (length of ?claims of ?puzzle where ?claims_truth is false) > 0

normally ?answer becomes "No"
'''
        ast = parse_string(dsl)

        categories = [n for n in ast if isinstance(n, SemanticCategory)]
        assert len(categories) == 2

        rules = [n for n in ast if isinstance(n, Rule)]
        assert len(rules) >= 5

        program = translate_to_fol(ast)
        prolog = program.to_prolog_string()

        # Check key Prolog constructs
        assert "length_where_is_eq" in prolog
        assert "length_where_is_gt" in prolog


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
