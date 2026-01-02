"""
Tests for FOL translator (formerly DeLP translator)
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from canto_core.parser.dsl_parser import parse_string, parse_file
from canto_core.fol import translate_to_fol


# Helper function for backwards compatibility
def translate_to_delp(ast):
    """Wrapper that uses translate_to_fol"""
    return translate_to_fol(ast)


def test_simple_rule_translation():
    """Test translating a simple rule"""
    dsl = """
    ?vaccine_flag can be true, false
    ?patient_query meaning "patient query text"
    ?vaccine_terms resembles "vaccine", "vaccination"

    ?vaccine_flag becomes true
        when ?patient_query is like ?vaccine_terms
    """
    ast = parse_string(dsl)
    delp = translate_to_delp(ast)

    # Should have declarations
    assert len(delp.declarations) == 2
    assert "vaccine_flag" in delp.declarations
    assert "patient_query" in delp.declarations

    # Should have semantic category
    assert len(delp.semantic_categories) == 1
    assert "vaccine_terms" in delp.semantic_categories

    # Should have one strict rule
    assert len(delp.strict_rules) == 1
    rule = delp.strict_rules[0]
    assert rule.head == "vaccine_flag(true)"
    assert "is_like(patient_query, vaccine_terms)" in rule.body


def test_defeasible_rule_translation():
    """Test translating a defeasible rule"""
    dsl = """
    ?vaccine_flag can be true, false
    ?query_intent meaning "query intent"

    normally ?vaccine_flag becomes true
        when ?query_intent is "prevention"
    """
    ast = parse_string(dsl)
    delp = translate_to_delp(ast)

    # Should have one defeasible rule
    assert len(delp.defeasible_rules) == 1
    rule = delp.defeasible_rules[0]
    assert rule.head == "vaccine_flag(true)"
    assert "query_intent('prevention')" in rule.body


def test_unless_clause_translation():
    """Test translating unless clause"""
    dsl = """
    ?query_intent meaning "query intent"
    ?patient_query meaning "patient query"
    ?treatment_intent resembles "treat", "treating"
    ?prevention_intent resembles "prevent"

    ?query_intent becomes "treatment"
        when ?patient_query is like ?treatment_intent
        unless ?patient_query is like ?prevention_intent
    """
    ast = parse_string(dsl)
    delp = translate_to_delp(ast)

    # Should have one strict rule with negated condition
    assert len(delp.strict_rules) == 1
    rule = delp.strict_rules[0]
    assert rule.head == "query_intent('treatment')"
    assert any("is_like(patient_query, treatment_intent)" in pred for pred in rule.body)
    assert any("\\+" in pred and "prevention_intent" in pred for pred in rule.body)


def test_overrides_all():
    """Test overriding all clause"""
    dsl = """
    ?vaccine_flag can be true, false
    ?patient_query meaning "patient query"
    ?vaccine_terms resembles "vaccine"

    normally ?vaccine_flag becomes true
        when ?patient_query is like ?vaccine_terms

    ?vaccine_flag becomes true
        when ?patient_query is like ?vaccine_terms
        overriding all
    """
    ast = parse_string(dsl)
    delp = translate_to_delp(ast)

    # Should have superiority relations
    assert len(delp.superiority) > 0


def test_prolog_output():
    """Test generating Prolog string"""
    dsl = """
    ?vaccine_flag can be true, false
    ?patient_query meaning "patient query"
    ?vaccine_terms resembles "vaccine", "vaccination"

    ?vaccine_flag becomes true
        when ?patient_query is like ?vaccine_terms
    """
    ast = parse_string(dsl)
    delp = translate_to_delp(ast)

    prolog_str = delp.to_prolog_string()

    # Should contain declarations
    assert "DECLARATIONS" in prolog_str
    assert "vaccine_flag" in prolog_str

    # Should contain semantic categories
    assert "SEMANTIC CATEGORIES" in prolog_str
    assert "pattern(vaccine_terms" in prolog_str

    # Should contain strict rule
    assert "STRICT RULES" in prolog_str
    assert "vaccine_flag(true) :-" in prolog_str
    assert "is_like(patient_query, vaccine_terms)" in prolog_str


def test_translate_medical_query_file():
    """Test translating the full medical_query.canto file"""
    example_path = Path(__file__).parent.parent / "examples" / "medical_query.canto"
    if example_path.exists():
        ast = parse_file(str(example_path))
        delp = translate_to_delp(ast)

        # Should have declarations
        assert len(delp.declarations) > 0

        # Should have semantic categories
        assert len(delp.semantic_categories) > 0

        # Should have rules
        assert len(delp.strict_rules) + len(delp.defeasible_rules) > 0

        # Should have superiority relations (from overriding clauses)
        assert len(delp.superiority) > 0

        print(f"\nTranslated medical_query.canto:")
        print(f"  Declarations: {len(delp.declarations)}")
        print(f"  Semantic Categories: {len(delp.semantic_categories)}")
        print(f"  Strict Rules: {len(delp.strict_rules)}")
        print(f"  Defeasible Rules: {len(delp.defeasible_rules)}")
        print(f"  Superiority Relations: {len(delp.superiority)}")

        # Generate Prolog output
        prolog = delp.to_prolog_string()
        print(f"\nGenerated Prolog length: {len(prolog)} characters")


def test_rule_info_metadata():
    """Test that rule_info/4 metadata is generated correctly"""
    dsl = """
    ?flag can be true, false
    ?flag becomes true
    normally ?flag becomes false
    """
    ast = parse_string(dsl)
    delp = translate_to_delp(ast)
    prolog_str = delp.to_prolog_string()

    # Should have rule_info declarations
    assert "rule_info(flag_1, flag(true), strict, [])" in prolog_str
    assert "rule_info(flag_2, flag(false), defeasible, [])" in prolog_str


def test_superiority_order_independence():
    """Test that overriding works regardless of rule definition order"""
    # Test 1: overriding rule defined FIRST
    dsl1 = """
    ?flag can be true, false
    ?flag becomes true overriding all
    normally ?flag becomes false
    """
    ast1 = parse_string(dsl1)
    delp1 = translate_to_delp(ast1)
    assert len(delp1.superiority) >= 1

    # Test 2: overriding rule defined SECOND
    dsl2 = """
    ?flag can be true, false
    normally ?flag becomes false
    ?flag becomes true overriding all
    """
    ast2 = parse_string(dsl2)
    delp2 = translate_to_delp(ast2)
    assert len(delp2.superiority) >= 1

    # Both should have superiority relations
    # (actual rule IDs may differ due to ordering)
    assert len(delp1.superiority) == len(delp2.superiority)


def test_overrides_normal_only():
    """Test overriding normal only affects defeasible rules"""
    dsl = """
    ?flag can be true, false
    normally ?flag becomes false
    ?flag becomes true
    ?flag becomes true overriding normal
    """
    ast = parse_string(dsl)
    delp = translate_to_delp(ast)

    # Should have superiority only over defeasible rule
    # flag_3 should override flag_1 (defeasible), but not flag_2 (strict)
    assert len(delp.superiority) == 1


def test_multiple_variables():
    """Test translation with multiple variables"""
    dsl = """
    ?var1 can be true, false
    ?var2 meaning "text value"
    ?var3 meaning "numeric value"

    ?var1 becomes true
    ?var2 becomes "test"
    ?var3 becomes 42
    """
    ast = parse_string(dsl)
    delp = translate_to_delp(ast)

    assert len(delp.declarations) == 3
    assert len(delp.strict_rules) == 3


def test_complex_body():
    """Test translation of complex rule bodies"""
    dsl = """
    ?a can be true, false
    ?b can be true, false
    ?result can be true, false

    ?result becomes true
        when ?a is true and ?b is true
    """
    ast = parse_string(dsl)
    delp = translate_to_delp(ast)

    rule = delp.strict_rules[0]
    # Should have multiple conditions in body
    assert len(rule.body) == 2


def test_dynamic_predicates():
    """Test that dynamic predicates are declared"""
    dsl = """
    ?flag can be true, false
    ?flag becomes true
    """
    ast = parse_string(dsl)
    delp = translate_to_delp(ast)
    prolog_str = delp.to_prolog_string()

    # Should declare dynamic predicates
    assert ":- dynamic is_like/2" in prolog_str
    assert ":- dynamic rule_info/4" in prolog_str
    assert ":- dynamic flag/1" in prolog_str


def test_pattern_facts():
    """Test that semantic category patterns are translated"""
    dsl = """
    ?terms resembles "pattern1", "pattern2", "pattern3"
    """
    ast = parse_string(dsl)
    delp = translate_to_delp(ast)
    prolog_str = delp.to_prolog_string()

    # Should have pattern facts (unquoted lowercase atoms)
    assert "pattern(terms, pattern1)." in prolog_str
    assert "pattern(terms, pattern2)." in prolog_str
    assert "pattern(terms, pattern3)." in prolog_str


def test_no_conditions_rule():
    """Test rule without conditions"""
    dsl = """
    ?flag can be true, false
    ?flag becomes true
    """
    ast = parse_string(dsl)
    delp = translate_to_delp(ast)

    rule = delp.strict_rules[0]
    assert rule.body == []

    prolog_str = delp.to_prolog_string()
    # Should generate: flag(true) :- true.
    assert "flag(true) :- true" in prolog_str


def test_multiple_overrides():
    """Test multiple overriding clauses"""
    dsl = """
    ?flag can be true, false
    normally ?flag becomes false
    ?flag becomes true overriding all
    ?flag becomes false overriding normal
    """
    ast = parse_string(dsl)
    delp = translate_to_delp(ast)

    # Should have multiple superiority relations
    assert len(delp.superiority) >= 1


def test_nested_with_block_declarations():
    """Test that nested 'with' block variables are collected by translator"""
    dsl = """
    ?patient meaning "patient info" with
        ?name meaning "patient's name"
        ?symptoms with
            ?text meaning "symptom description"
            ?is_emergency can be true, false
    """
    ast = parse_string(dsl)
    delp = translate_to_delp(ast)

    # All nested variables should be collected
    assert "patient" in delp.declarations
    assert "name" in delp.declarations
    assert "symptoms" in delp.declarations
    assert "text" in delp.declarations
    assert "is_emergency" in delp.declarations
    assert len(delp.declarations) == 5


def test_deeply_nested_with_block_declarations():
    """Test that deeply nested 'with' blocks are properly collected"""
    dsl = """
    ?root meaning "root" with
        ?level1 meaning "level 1" with
            ?level2 meaning "level 2" with
                ?level3 meaning "level 3"
    """
    ast = parse_string(dsl)
    delp = translate_to_delp(ast)

    assert "root" in delp.declarations
    assert "level1" in delp.declarations
    assert "level2" in delp.declarations
    assert "level3" in delp.declarations
    assert len(delp.declarations) == 4


def test_with_block_prolog_output():
    """Test that nested declarations are included in Prolog output"""
    dsl = """
    ?patient meaning "patient info" with
        ?name meaning "patient's name"
        ?age meaning "patient's age"
    """
    ast = parse_string(dsl)
    delp = translate_to_delp(ast)
    prolog_str = delp.to_prolog_string()

    # All variables should be declared as dynamic
    assert ":- dynamic patient/1" in prolog_str
    assert ":- dynamic name/1" in prolog_str
    assert ":- dynamic age/1" in prolog_str


def test_with_block_generates_has_property():
    """Test that 'with' blocks generate has_property/3 facts like explicit 'has' declarations"""
    dsl = """
    ?patient meaning "patient info" with
        ?name meaning "patient's name"
        ?symptoms with
            ?text meaning "symptom description"
            ?is_emergency can be true, false
    """
    ast = parse_string(dsl)
    delp = translate_to_delp(ast)

    # Should have has_relationships for the nested structure
    assert len(delp.has_relationships) == 4
    assert "patient_name" in delp.has_relationships
    assert "patient_symptoms" in delp.has_relationships
    assert "symptoms_text" in delp.has_relationships
    assert "symptoms_is_emergency" in delp.has_relationships

    # Check the relationships are correct
    assert delp.has_relationships["patient_name"].parent == "patient"
    assert delp.has_relationships["patient_name"].child == "name"
    assert delp.has_relationships["symptoms_text"].parent == "symptoms"
    assert delp.has_relationships["symptoms_text"].child == "text"

    # Check Prolog output
    prolog_str = delp.to_prolog_string()
    assert "has_property(patient, name, single)" in prolog_str
    assert "has_property(patient, symptoms, single)" in prolog_str
    assert "has_property(symptoms, text, single)" in prolog_str
    assert "has_property(symptoms, is_emergency, single)" in prolog_str


def test_with_block_same_as_explicit_has():
    """Test that 'with' block generates same structure as explicit 'has' declaration"""
    # Using 'with' block
    dsl_with = """
    ?patient meaning "patient" with
        ?name meaning "name"
    """

    # Using explicit 'has'
    dsl_has = """
    ?patient meaning "patient"
    ?patient has a ?name meaning "name"
    """

    ast_with = parse_string(dsl_with)
    ast_has = parse_string(dsl_has)

    delp_with = translate_to_delp(ast_with)
    delp_has = translate_to_delp(ast_has)

    # Both should have the same has_property in Prolog output
    prolog_with = delp_with.to_prolog_string()
    prolog_has = delp_has.to_prolog_string()

    assert "has_property(patient, name, single)" in prolog_with
    assert "has_property(patient, name, single)" in prolog_has


def test_let_binding_translation():
    """Test that let binding translates to correct Prolog"""
    dsl = """
?chain has a list of ?link meaning "the chain"
?link meaning "a link" with
    ?speaker meaning "the speaker"
    ?speaker_is_truthful can be true, false

?target meaning "target person"

?result can be "Yes", "No"

?result becomes "Yes"
    when let ?link be any in ?chain where
        ?speaker of ?link is ?target
        and ?speaker_is_truthful of ?link is true
"""
    ast = parse_string(dsl)
    delp = translate_to_delp(ast)

    # Should have one strict rule
    assert len(delp.strict_rules) == 1
    rule = delp.strict_rules[0]

    # Check the rule head
    assert rule.head == "result('Yes')"

    # Check the rule body contains let_any with correct structure
    assert len(rule.body) == 1
    body_pred = rule.body[0]
    # FOL translator uses let_any (not let_any_bound)
    assert "let_any(" in body_pred
    assert "chain" in body_pred
    assert "link" in body_pred


def test_let_all_translation():
    """Test let all binding translates correctly"""
    dsl = """
?items has a list of ?item meaning "items"
?item meaning "an item" with
    ?status can be "active", "inactive"

?all_active can be true, false

?all_active becomes true
    when let ?item be all in ?items where ?status of ?item is "active"
"""
    ast = parse_string(dsl)
    delp = translate_to_delp(ast)

    rule = delp.strict_rules[0]
    # FOL translator uses let_all (not let_all_bound)
    assert "let_all(" in rule.body[0]


def test_let_none_translation():
    """Test let none binding translates correctly"""
    dsl = """
?items has a list of ?item meaning "items"
?item meaning "an item" with
    ?is_error can be true, false

?no_errors can be true, false

?no_errors becomes true
    when let ?item be none in ?items where ?is_error of ?item is true
"""
    ast = parse_string(dsl)
    delp = translate_to_delp(ast)

    rule = delp.strict_rules[0]
    # FOL translator uses let_none (not let_none_bound)
    assert "let_none(" in rule.body[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
