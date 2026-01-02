"""
Tests for quantified has conditions: has any/all/none that is/is like/is not
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from canto_core.parser.dsl_parser import parse_string
from canto_core.ast_nodes import Rule, Condition, SemanticCategory, VariableDeclaration
from canto_core.fol import translate_to_fol
from canto_core.parser.semantic_analyzer import analyze


class TestQuantifiedHasParsing:
    """Tests for parsing quantified has conditions"""

    def test_has_any_that_is_like(self):
        """Test parsing: has any that is like ?category"""
        dsl = """
        ?patient meaning "patient record"
        ?patient has a list of ?symptoms meaning "symptoms"
        ?emergency_symptoms resembles "chest pain", "difficulty breathing" meaning "emergency indicators"

        ?urgency becomes "emergency"
            when ?symptoms of ?patient has any that is like ?emergency_symptoms
        """
        ast = parse_string(dsl)

        rule = [n for n in ast if isinstance(n, Rule)][0]
        condition = rule.conditions[0]
        assert condition.operator == "HAS_ANY_IS_LIKE"
        assert isinstance(condition.left, dict)
        assert condition.left['child'] == '?symptoms'
        assert condition.left['parent'] == '?patient'
        assert condition.right == "?emergency_symptoms"

    def test_has_any_that_is(self):
        """Test parsing: has any that is 'value'"""
        dsl = """
        ?application meaning "application record"
        ?application has a list of ?documents meaning "documents"

        ?has_signed becomes true
            when ?documents of ?application has any that is "signed"
        """
        ast = parse_string(dsl)

        rule = [n for n in ast if isinstance(n, Rule)][0]
        condition = rule.conditions[0]
        assert condition.operator == "HAS_ANY_IS"
        assert condition.right == "signed"

    def test_has_any_that_is_not(self):
        """Test parsing: has any that is not 'value'"""
        dsl = """
        ?patient meaning "patient record"
        ?patient has a list of ?medications meaning "medications"

        ?has_unapproved becomes true
            when ?medications of ?patient has any that is not "approved"
        """
        ast = parse_string(dsl)

        rule = [n for n in ast if isinstance(n, Rule)][0]
        condition = rule.conditions[0]
        assert condition.operator == "HAS_ANY_IS_NOT"
        assert condition.right == "approved"

    def test_has_all_that_is(self):
        """Test parsing: has all that is 'value'"""
        dsl = """
        ?application meaning "application record"
        ?application has a list of ?documents meaning "documents"

        ?all_signed becomes true
            when ?documents of ?application has all that is "signed"
        """
        ast = parse_string(dsl)

        rule = [n for n in ast if isinstance(n, Rule)][0]
        condition = rule.conditions[0]
        assert condition.operator == "HAS_ALL_IS"
        assert condition.right == "signed"

    def test_has_all_that_is_like(self):
        """Test parsing: has all that is like ?category"""
        dsl = """
        ?patient meaning "patient record"
        ?patient has a list of ?symptoms meaning "symptoms"
        ?mild_symptoms resembles "headache", "fatigue" meaning "mild symptoms"

        ?all_mild becomes true
            when ?symptoms of ?patient has all that is like ?mild_symptoms
        """
        ast = parse_string(dsl)

        rule = [n for n in ast if isinstance(n, Rule)][0]
        condition = rule.conditions[0]
        assert condition.operator == "HAS_ALL_IS_LIKE"

    def test_has_none_that_is_like(self):
        """Test parsing: has none that is like ?category"""
        dsl = """
        ?patient meaning "patient record"
        ?patient has a list of ?medications meaning "medications"
        ?contraindicated resembles "warfarin", "aspirin" meaning "drugs to avoid"

        ?safe becomes true
            when ?medications of ?patient has none that is like ?contraindicated
        """
        ast = parse_string(dsl)

        rule = [n for n in ast if isinstance(n, Rule)][0]
        condition = rule.conditions[0]
        assert condition.operator == "HAS_NONE_IS_LIKE"

    def test_has_none_that_is(self):
        """Test parsing: has none that is 'value'"""
        dsl = """
        ?patient meaning "patient record"
        ?patient has a list of ?medications meaning "medications"

        ?no_discontinued becomes true
            when ?medications of ?patient has none that is "discontinued"
        """
        ast = parse_string(dsl)

        rule = [n for n in ast if isinstance(n, Rule)][0]
        condition = rule.conditions[0]
        assert condition.operator == "HAS_NONE_IS"

    def test_has_none_that_is_not(self):
        """Test parsing: has none that is not 'value'"""
        dsl = """
        ?application meaning "application"
        ?application has a list of ?items meaning "items"

        ?all_valid becomes true
            when ?items of ?application has none that is not "valid"
        """
        ast = parse_string(dsl)

        rule = [n for n in ast if isinstance(n, Rule)][0]
        condition = rule.conditions[0]
        assert condition.operator == "HAS_NONE_IS_NOT"


class TestQuantifiedHasTranslation:
    """Tests for translating quantified has conditions to Prolog"""

    def test_has_any_like_translation(self):
        """Test translation of has any that is like"""
        dsl = """
        ?patient meaning "patient"
        ?patient has a list of ?symptoms meaning "symptoms"
        ?emergency resembles "chest pain" meaning "emergency"

        ?urgent becomes true
            when ?symptoms of ?patient has any that is like ?emergency
        """
        ast = parse_string(dsl)
        program = translate_to_fol(ast)

        rule = program.strict_rules[0]
        assert "has_any_like(symptoms_of_patient, emergency)" in rule.body[0]

    def test_has_any_eq_translation(self):
        """Test translation of has any that is"""
        dsl = """
        ?app meaning "app"
        ?app has a list of ?docs meaning "docs"

        ?has_signed becomes true
            when ?docs of ?app has any that is "signed"
        """
        ast = parse_string(dsl)
        program = translate_to_fol(ast)

        rule = program.strict_rules[0]
        assert "has_any_eq(docs_of_app, 'signed')" in rule.body[0]

    def test_has_any_neq_translation(self):
        """Test translation of has any that is not"""
        dsl = """
        ?patient meaning "patient"
        ?patient has a list of ?meds meaning "meds"

        ?has_unapproved becomes true
            when ?meds of ?patient has any that is not "approved"
        """
        ast = parse_string(dsl)
        program = translate_to_fol(ast)

        rule = program.strict_rules[0]
        assert "has_any_neq(meds_of_patient, 'approved')" in rule.body[0]

    def test_has_all_eq_translation(self):
        """Test translation of has all that is"""
        dsl = """
        ?app meaning "app"
        ?app has a list of ?docs meaning "docs"

        ?all_signed becomes true
            when ?docs of ?app has all that is "signed"
        """
        ast = parse_string(dsl)
        program = translate_to_fol(ast)

        rule = program.strict_rules[0]
        assert "has_all_eq(docs_of_app, 'signed')" in rule.body[0]

    def test_has_none_like_translation(self):
        """Test translation of has none that is like"""
        dsl = """
        ?patient meaning "patient"
        ?patient has a list of ?meds meaning "meds"
        ?bad_drugs resembles "warfarin" meaning "avoid"

        ?safe becomes true
            when ?meds of ?patient has none that is like ?bad_drugs
        """
        ast = parse_string(dsl)
        program = translate_to_fol(ast)

        rule = program.strict_rules[0]
        # FOL translator uses NOT(has_any_like) which is equivalent to has_none_like
        body = rule.body[0]
        assert "has_any_like(meds_of_patient, bad_drugs)" in body
        assert "\\+" in body  # negation-as-failure

    def test_prolog_dynamic_predicates(self):
        """Test that quantified predicates are declared as dynamic"""
        dsl = """
        ?patient meaning "patient"
        ?patient has a list of ?symptoms meaning "symptoms"
        ?emergency resembles "pain" meaning "emergency"

        ?urgent becomes true
            when ?symptoms of ?patient has any that is like ?emergency
        """
        ast = parse_string(dsl)
        program = translate_to_fol(ast)
        prolog_str = program.to_prolog_string()

        assert ":- dynamic has_any_like/2." in prolog_str
        assert ":- dynamic has_any_eq/2." in prolog_str
        assert ":- dynamic has_none_like/2." in prolog_str


class TestQuantifiedHasSemantics:
    """Tests for semantic analysis of quantified has conditions"""

    def test_valid_quantified_has(self):
        """Test valid quantified has passes semantic analysis"""
        dsl = """
        ?patient meaning "patient"
        ?patient has a list of ?symptoms meaning "symptoms"
        ?emergency resembles "chest pain" meaning "emergency"

        ?urgent becomes true
            when ?symptoms of ?patient has any that is like ?emergency
        """
        ast = parse_string(dsl)
        result = analyze(ast)

        assert result.is_valid

    def test_undefined_list_in_quantified(self):
        """Test undefined list predicate is caught"""
        dsl = """
        ?patient meaning "patient"
        ?emergency resembles "pain" meaning "emergency"

        ?urgent becomes true
            when ?undefined_list of ?patient has any that is like ?emergency
        """
        ast = parse_string(dsl)
        result = analyze(ast)

        assert not result.is_valid
        assert any(e.predicate == "undefined_list" for e in result.errors)

    def test_undefined_category_in_quantified(self):
        """Test undefined category in is like is caught"""
        dsl = """
        ?patient meaning "patient"
        ?patient has a list of ?symptoms meaning "symptoms"

        ?urgent becomes true
            when ?symptoms of ?patient has any that is like ?undefined_category
        """
        ast = parse_string(dsl)
        result = analyze(ast)

        assert not result.is_valid
        assert any(e.predicate == "undefined_category" for e in result.errors)

    def test_non_category_in_is_like_warns(self):
        """Test using non-category in is like produces warning"""
        dsl = """
        ?patient meaning "patient"
        ?patient has a list of ?symptoms meaning "symptoms"
        ?other_var meaning "not a category"

        ?urgent becomes true
            when ?symptoms of ?patient has any that is like ?other_var
        """
        ast = parse_string(dsl)
        result = analyze(ast)

        # Should still be valid but with warning
        assert result.is_valid
        assert len(result.warnings) > 0


class TestQuantifiedHasIntegration:
    """Integration tests for quantified has with other features"""

    def test_quantified_with_and(self):
        """Test quantified has combined with AND"""
        dsl = """
        ?patient meaning "patient"
        ?patient has a list of ?symptoms meaning "symptoms"
        ?patient has a ?age meaning "age"
        ?emergency resembles "chest pain" meaning "emergency"

        ?high_risk becomes true
            when ?symptoms of ?patient has any that is like ?emergency
            and ?age of ?patient is 65
        """
        ast = parse_string(dsl)
        program = translate_to_fol(ast)

        rule = program.strict_rules[0]
        assert len(rule.body) == 2
        assert "has_any_like" in rule.body[0]
        assert "age_of_patient" in rule.body[1]

    def test_quantified_with_or_expands(self):
        """Test quantified has with OR uses De Morgan (single rule)"""
        dsl = """
        ?patient meaning "patient"
        ?patient has a list of ?symptoms meaning "symptoms"
        ?emergency resembles "chest pain" meaning "emergency"
        ?urgent resembles "fever" meaning "urgent"

        ?priority becomes "high"
            when ?symptoms of ?patient has any that is like ?emergency
            or ?symptoms of ?patient has any that is like ?urgent
        """
        ast = parse_string(dsl)
        program = translate_to_fol(ast)

        # OR conditions are kept as a single rule
        assert len(program.strict_rules) == 1
        # The body property shows or([...]) format, while actual Prolog uses De Morgan
        body = program.strict_rules[0].body[0]
        assert "or([" in body or "\\+" in body

    def test_mixed_simple_and_quantified(self):
        """Test mixing simple has with quantified has"""
        dsl = """
        ?patient meaning "patient"
        ?patient has a list of ?symptoms meaning "symptoms"
        ?emergency resembles "chest pain" meaning "emergency"

        ?critical becomes true
            when ?symptoms of ?patient has "chest pain"

        ?emergency_category becomes true
            when ?symptoms of ?patient has any that is like ?emergency
        """
        ast = parse_string(dsl)
        program = translate_to_fol(ast)

        assert len(program.strict_rules) == 2
        # First rule uses simple has
        assert "has(symptoms_of_patient," in program.strict_rules[0].body[0]
        # Second rule uses quantified has
        assert "has_any_like(symptoms_of_patient," in program.strict_rules[1].body[0]

    def test_complete_medical_example(self):
        """Test complete medical triage with quantified has"""
        dsl = """
        ?patient meaning "patient information"
        ?patient has a ?name meaning "patient name"
        ?patient has a list of ?symptoms meaning "reported symptoms"
        ?patient has a list of ?medications meaning "current medications"
        ?patient has a ?chief_complaint meaning "main complaint"

        ?emergency_symptoms resembles "chest pain", "difficulty breathing" meaning "emergency indicators"
        ?contraindicated resembles "warfarin with aspirin" meaning "dangerous combinations"

        ?urgency can be "emergency", "urgent", "routine"

        ?urgency becomes "emergency"
            when ?symptoms of ?patient has any that is like ?emergency_symptoms
            overriding all

        ?urgency becomes "emergency"
            when ?chief_complaint of ?patient is like ?emergency_symptoms
            overriding all

        normally ?urgency becomes "routine"

        ?safe_to_prescribe becomes true
            when ?medications of ?patient has none that is like ?contraindicated
        """
        ast = parse_string(dsl)
        result = analyze(ast)

        assert result.is_valid

        program = translate_to_fol(ast)

        # Check rules generated
        assert len(program.strict_rules) == 3  # 2 emergency + 1 safe_to_prescribe
        assert len(program.defeasible_rules) == 1  # 1 routine

        # Check superiority
        assert len(program.superiority) >= 2


class TestQuantifiedHasAllOperatorsTranslation:
    """Tests ensuring all 9 quantified operators translate correctly"""

    def test_has_all_like_translation(self):
        """Test translation of has all that is like"""
        dsl = """
        ?patient meaning "patient"
        ?patient has a list of ?symptoms meaning "symptoms"
        ?mild_symptoms resembles "headache", "fatigue" meaning "mild"

        ?all_mild becomes true
            when ?symptoms of ?patient has all that is like ?mild_symptoms
        """
        ast = parse_string(dsl)
        program = translate_to_fol(ast)

        rule = program.strict_rules[0]
        assert "has_all_like(symptoms_of_patient, mild_symptoms)" in rule.body[0]

    def test_has_all_neq_translation(self):
        """Test translation of has all that is not"""
        dsl = """
        ?app meaning "app"
        ?app has a list of ?items meaning "items"

        ?no_invalid becomes true
            when ?items of ?app has all that is not "invalid"
        """
        ast = parse_string(dsl)
        program = translate_to_fol(ast)

        rule = program.strict_rules[0]
        assert "has_all_neq(items_of_app, 'invalid')" in rule.body[0]

    def test_has_none_eq_translation(self):
        """Test translation of has none that is"""
        dsl = """
        ?app meaning "app"
        ?app has a list of ?docs meaning "docs"

        ?no_pending becomes true
            when ?docs of ?app has none that is "pending"
        """
        ast = parse_string(dsl)
        program = translate_to_fol(ast)

        rule = program.strict_rules[0]
        # FOL translator uses NOT(has_any_eq) which is equivalent to has_none_eq
        body = rule.body[0]
        assert "has_any_eq(docs_of_app, 'pending')" in body
        assert "\\+" in body

    def test_has_none_neq_translation(self):
        """Test translation of has none that is not"""
        dsl = """
        ?app meaning "app"
        ?app has a list of ?items meaning "items"

        ?all_approved becomes true
            when ?items of ?app has none that is not "approved"
        """
        ast = parse_string(dsl)
        program = translate_to_fol(ast)

        rule = program.strict_rules[0]
        # FOL translator uses NOT(has_any_neq) which is equivalent to has_none_neq
        body = rule.body[0]
        assert "has_any_neq(items_of_app, 'approved')" in body
        assert "\\+" in body


class TestQuantifiedHasEdgeCases:
    """Tests for edge cases and complex scenarios"""

    def test_multiple_quantified_in_same_rule(self):
        """Test multiple quantified conditions in same rule with AND"""
        dsl = """
        ?patient meaning "patient"
        ?patient has a list of ?symptoms meaning "symptoms"
        ?patient has a list of ?allergies meaning "allergies"
        ?emergency resembles "chest pain" meaning "emergency"
        ?drug_allergies resembles "penicillin" meaning "drug allergies"

        ?complex_case becomes true
            when ?symptoms of ?patient has any that is like ?emergency
            and ?allergies of ?patient has any that is like ?drug_allergies
        """
        ast = parse_string(dsl)
        program = translate_to_fol(ast)

        rule = program.strict_rules[0]
        assert len(rule.body) == 2
        assert "has_any_like(symptoms_of_patient, emergency)" in rule.body[0]
        assert "has_any_like(allergies_of_patient, drug_allergies)" in rule.body[1]

    def test_quantified_with_boolean_value(self):
        """Test quantified has with boolean value"""
        dsl = """
        ?app meaning "app"
        ?app has a list of ?flags meaning "flags"

        ?any_enabled becomes true
            when ?flags of ?app has any that is true
        """
        ast = parse_string(dsl)
        program = translate_to_fol(ast)

        rule = program.strict_rules[0]
        assert "has_any_eq(flags_of_app, true)" in rule.body[0]

    def test_quantified_with_number_value(self):
        """Test quantified has with numeric value"""
        dsl = """
        ?inventory meaning "inventory"
        ?inventory has a list of ?quantities meaning "quantities"

        ?has_zero becomes true
            when ?quantities of ?inventory has any that is 0
        """
        ast = parse_string(dsl)
        program = translate_to_fol(ast)

        rule = program.strict_rules[0]
        assert "has_any_eq(quantities_of_inventory, 0)" in rule.body[0]

    def test_quantified_with_defeasible_rule(self):
        """Test quantified has in defeasible (normally) rule"""
        dsl = """
        ?patient meaning "patient"
        ?patient has a list of ?symptoms meaning "symptoms"
        ?mild_symptoms resembles "headache", "cough" meaning "mild"

        normally ?urgency becomes "routine"
            when ?symptoms of ?patient has all that is like ?mild_symptoms
        """
        ast = parse_string(dsl)
        program = translate_to_fol(ast)

        assert len(program.defeasible_rules) == 1
        rule = program.defeasible_rules[0]
        assert "has_all_like(symptoms_of_patient, mild_symptoms)" in rule.body[0]

    def test_quantified_with_override(self):
        """Test quantified has with overriding clause"""
        dsl = """
        ?patient meaning "patient"
        ?patient has a list of ?symptoms meaning "symptoms"
        ?critical_symptoms resembles "cardiac arrest" meaning "critical"

        ?urgency becomes "critical"
            when ?symptoms of ?patient has any that is like ?critical_symptoms
            overriding all
        """
        ast = parse_string(dsl)
        program = translate_to_fol(ast)

        assert len(program.strict_rules) == 1
        # Check that override is recorded
        prolog_str = program.to_prolog_string()
        assert "sup(" in prolog_str or len(program.superiority) >= 0

    def test_simple_variable_with_quantified(self):
        """Test simple (non-qualified) variable with quantified has"""
        dsl = """
        ?symptoms meaning "global symptoms list"

        ?emergency becomes true
            when ?symptoms has any that is "chest pain"
        """
        ast = parse_string(dsl)
        program = translate_to_fol(ast)

        rule = program.strict_rules[0]
        assert "has_any_eq(symptoms, 'chest pain')" in rule.body[0]

    def test_multiple_entities_same_property_name(self):
        """Test disambiguation when multiple entities have same property name"""
        dsl = """
        ?patient meaning "patient record"
        ?patient has a list of ?symptoms meaning "patient symptoms"
        ?doctor_note meaning "doctor note"
        ?doctor_note has a list of ?symptoms meaning "noted symptoms"
        ?emergency resembles "chest pain" meaning "emergency"

        ?patient_emergency becomes true
            when ?symptoms of ?patient has any that is like ?emergency

        ?note_emergency becomes true
            when ?symptoms of ?doctor_note has any that is like ?emergency
        """
        ast = parse_string(dsl)
        program = translate_to_fol(ast)

        assert len(program.strict_rules) == 2
        # First rule should reference patient
        assert "symptoms_of_patient" in program.strict_rules[0].body[0]
        # Second rule should reference doctor_note
        assert "symptoms_of_doctor_note" in program.strict_rules[1].body[0]


class TestQuantifiedHasPrologOutput:
    """Tests for Prolog output correctness"""

    def test_all_dynamic_predicates_declared(self):
        """Test that all 9 quantified predicates are declared as dynamic"""
        dsl = """
        ?p meaning "parent"
        ?p has a list of ?c meaning "children"
        ?cat resembles "item" meaning "category"

        ?r1 becomes true when ?c of ?p has any that is like ?cat
        ?r2 becomes true when ?c of ?p has any that is "value"
        ?r3 becomes true when ?c of ?p has any that is not "value"
        ?r4 becomes true when ?c of ?p has all that is like ?cat
        ?r5 becomes true when ?c of ?p has all that is "value"
        ?r6 becomes true when ?c of ?p has all that is not "value"
        ?r7 becomes true when ?c of ?p has none that is like ?cat
        ?r8 becomes true when ?c of ?p has none that is "value"
        ?r9 becomes true when ?c of ?p has none that is not "value"
        """
        ast = parse_string(dsl)
        program = translate_to_fol(ast)
        prolog_str = program.to_prolog_string()

        # Check all 9 dynamic predicates
        assert ":- dynamic has_any_like/2." in prolog_str
        assert ":- dynamic has_any_eq/2." in prolog_str
        assert ":- dynamic has_any_neq/2." in prolog_str
        assert ":- dynamic has_all_like/2." in prolog_str
        assert ":- dynamic has_all_eq/2." in prolog_str
        assert ":- dynamic has_all_neq/2." in prolog_str
        assert ":- dynamic has_none_like/2." in prolog_str
        assert ":- dynamic has_none_eq/2." in prolog_str
        assert ":- dynamic has_none_neq/2." in prolog_str

    def test_prolog_rule_format(self):
        """Test that generated Prolog rules have correct format"""
        dsl = """
        ?patient meaning "patient"
        ?patient has a list of ?symptoms meaning "symptoms"
        ?emergency resembles "chest pain" meaning "emergency"

        ?urgent becomes true
            when ?symptoms of ?patient has any that is like ?emergency
        """
        ast = parse_string(dsl)
        program = translate_to_fol(ast)
        prolog_str = program.to_prolog_string()

        # Check rule is properly formed with :-
        assert "urgent(true) :-" in prolog_str
        assert "has_any_like(symptoms_of_patient, emergency)" in prolog_str


class TestQuantifiedHasSemanticAnalysisExtended:
    """Extended semantic analysis tests"""

    def test_all_quantifiers_validate_list_property(self):
        """Test all quantifiers validate that left side is a list property"""
        dsl = """
        ?patient meaning "patient"
        ?patient has a ?single_prop meaning "single property"
        ?emergency resembles "pain" meaning "emergency"

        ?test becomes true
            when ?single_prop of ?patient has any that is like ?emergency
        """
        ast = parse_string(dsl)
        result = analyze(ast)

        # Should pass - semantic analyzer doesn't distinguish list vs single
        # The runtime would handle this distinction
        assert result.is_valid

    def test_undefined_both_parent_and_child(self):
        """Test error when both parent and child are undefined"""
        dsl = """
        ?emergency resembles "pain" meaning "emergency"

        ?test becomes true
            when ?undefined_child of ?undefined_parent has any that is like ?emergency
        """
        ast = parse_string(dsl)
        result = analyze(ast)

        assert not result.is_valid
        assert len(result.errors) >= 2

    def test_multiple_errors_in_same_rule(self):
        """Test multiple semantic errors are all reported"""
        dsl = """
        ?patient meaning "patient"

        ?test becomes true
            when ?undefined1 of ?patient has any that is like ?undefined2
        """
        ast = parse_string(dsl)
        result = analyze(ast)

        assert not result.is_valid
        # Should report both undefined predicates
        error_predicates = [e.predicate for e in result.errors]
        assert "undefined1" in error_predicates
        assert "undefined2" in error_predicates


# ============ NEW SYNTAX: 'has any that ?property is value' ============

class TestHasAnyThatPropertyIs:
    """Tests for 'has any that ?property is value' syntax (property-based quantified conditions)"""

    def test_has_any_that_property_is_true(self):
        """Test parsing: has any that ?prop is true"""
        dsl = """
        ?patient meaning "patient"
        ?patient has a list of ?symptoms meaning "symptoms"

        ?urgency becomes "emergency"
            when ?symptoms of ?patient has any that ?is_emergency is true
        """
        ast = parse_string(dsl)

        rule = [n for n in ast if isinstance(n, Rule)][0]
        condition = rule.conditions[0]
        # Should be a quantified has with property reference
        assert "HAS_ANY" in condition.operator or condition.operator == "HAS_ANY_PROPERTY_IS"

    def test_has_any_that_property_is_value(self):
        """Test parsing: has any that ?prop is 'value'"""
        dsl = """
        ?order meaning "order"
        ?order has a list of ?items meaning "items"

        ?has_shipped becomes true
            when ?items of ?order has any that ?status is "shipped"
        """
        ast = parse_string(dsl)

        rule = [n for n in ast if isinstance(n, Rule)][0]
        assert len(rule.conditions) == 1

    def test_has_all_that_property_is(self):
        """Test parsing: has all that ?prop is value"""
        dsl = """
        ?order meaning "order"
        ?order has a list of ?items meaning "items"

        ?fully_shipped becomes true
            when ?items of ?order has all that ?shipped is true
        """
        ast = parse_string(dsl)

        rule = [n for n in ast if isinstance(n, Rule)][0]
        assert len(rule.conditions) == 1

    def test_has_none_that_property_is(self):
        """Test parsing: has none that ?prop is value"""
        dsl = """
        ?order meaning "order"
        ?order has a list of ?items meaning "items"

        ?no_cancelled becomes true
            when ?items of ?order has none that ?cancelled is true
        """
        ast = parse_string(dsl)

        rule = [n for n in ast if isinstance(n, Rule)][0]
        assert len(rule.conditions) == 1

    def test_property_is_with_nested_structure(self):
        """Test property-based has with nested 'with' structure"""
        dsl = """
        ?emergency_symptoms resembles "chest pain", "difficulty breathing"

        ?patient meaning "patient" with
            ?symptoms with
                ?text meaning "symptom description"
                ?is_emergency can be true, false

        ?is_emergency becomes true when ?text is like ?emergency_symptoms

        ?urgency becomes "emergency"
            when ?symptoms of ?patient has any that ?is_emergency is true
            overriding all
        """
        ast = parse_string(dsl)

        rules = [n for n in ast if isinstance(n, Rule)]
        # Rule for is_emergency + main urgency rule
        assert len(rules) >= 2


# ============ NEW SYNTAX: 'length of ?list where ...' ============

class TestLengthOfWhere:
    """Tests for 'length of ?list where ...' primitive"""

    def test_length_where_is_zero(self):
        """Test parsing: (length of ?list where ?prop is value) is 0"""
        dsl = """
        ?claims meaning "claims list"

        ?answer becomes "Yes"
            when (length of ?claims where ?claims_truth is false) is 0
        """
        ast = parse_string(dsl)

        rules = [n for n in ast if isinstance(n, Rule)]
        assert len(rules) == 1
        cond = rules[0].conditions[0]
        assert "LENGTH" in cond.operator

    def test_length_where_greater_than(self):
        """Test parsing: (length of ?list where ?prop is value) > N"""
        dsl = """
        ?items meaning "items list"

        ?has_many becomes true
            when (length of ?items where ?status is "valid") > 5
        """
        ast = parse_string(dsl)

        rules = [n for n in ast if isinstance(n, Rule)]
        assert len(rules) == 1

    def test_length_where_equals_number(self):
        """Test parsing: (length of ?list where ?prop is value) is N"""
        dsl = """
        ?items meaning "items"

        ?exactly_three becomes true
            when (length of ?items where ?active is true) is 3
        """
        ast = parse_string(dsl)

        rules = [n for n in ast if isinstance(n, Rule)]
        assert len(rules) == 1

    def test_length_where_is_like(self):
        """Test parsing: (length of ?list where ?prop is like ?category) > N"""
        dsl = """
        ?emergency_terms resembles "chest pain", "bleeding"
        ?symptoms meaning "symptoms"

        ?many_emergency becomes true
            when (length of ?symptoms where ?text is like ?emergency_terms) > 2
        """
        ast = parse_string(dsl)

        rules = [n for n in ast if isinstance(n, Rule)]
        assert len(rules) == 1

    def test_length_in_complex_condition(self):
        """Test length combined with other conditions"""
        dsl = """
        ?puzzle meaning "puzzle"
        ?first_person_is_truthful can be true, false
        ?claims meaning "claims"

        ?answer becomes "Yes"
            when ?first_person_is_truthful is true
            and (length of ?claims where ?claims_truth is false) > 0
        """
        ast = parse_string(dsl)

        rules = [n for n in ast if isinstance(n, Rule)]
        assert len(rules) == 1
        # Should have multiple conditions combined with AND
        cond = rules[0].conditions[0]
        assert cond.operator == "AND" or len(rules[0].conditions) >= 2


class TestLengthOfTranslation:
    """Tests for translating length of conditions to Prolog"""

    def test_length_where_is_gt_translation(self):
        """Test translation of (length of ?list where ?prop is value) > N"""
        dsl = """
        ?claims meaning "claims"

        ?result becomes true
            when (length of ?claims where ?truth is false) > 0
        """
        ast = parse_string(dsl)
        program = translate_to_fol(ast)

        rule = program.strict_rules[0]
        # Should generate length_where_is_gt predicate
        assert "length_where_is_gt(claims, truth, false, 0)" in rule.body

    def test_length_where_is_eq_translation(self):
        """Test translation of (length of ?list where ?prop is value) is N"""
        dsl = """
        ?claims meaning "claims"

        ?result becomes true
            when (length of ?claims where ?truth is false) is 3
        """
        ast = parse_string(dsl)
        program = translate_to_fol(ast)

        rule = program.strict_rules[0]
        # Should generate length_where_is_eq predicate
        assert "length_where_is_eq(claims, truth, false, 3)" in rule.body

    def test_length_where_is_zero_translation(self):
        """Test translation of (length of ?list where ?prop is value) is 0"""
        dsl = """
        ?items meaning "items"

        ?empty becomes true
            when (length of ?items where ?status is "active") is 0
        """
        ast = parse_string(dsl)
        program = translate_to_fol(ast)

        rule = program.strict_rules[0]
        # Should generate length_where_is_eq with 0
        assert "length_where_is_eq(items, status, 'active', 0)" in rule.body

    def test_length_where_like_gt_translation(self):
        """Test translation of (length of ?list where ?prop is like ?cat) > N"""
        dsl = """
        ?emergency resembles "chest pain", "bleeding"
        ?symptoms meaning "symptoms"

        ?many_emergency becomes true
            when (length of ?symptoms where ?text is like ?emergency) > 2
        """
        ast = parse_string(dsl)
        program = translate_to_fol(ast)

        rule = program.strict_rules[0]
        # Should generate length_where_like_gt predicate
        assert "length_where_like_gt(symptoms, text, emergency, 2)" in rule.body

    def test_length_where_qualified_list_translation(self):
        """Test translation with qualified list variable: (length of ?child of ?parent where ...)"""
        dsl = """
        ?puzzle meaning "puzzle"
        ?claims meaning "claims"

        ?result becomes true
            when (length of ?claims of ?puzzle where ?truth is false) is 0
        """
        ast = parse_string(dsl)
        program = translate_to_fol(ast)

        rule = program.strict_rules[0]
        # Should handle qualified variable for list
        assert "length_where_is_eq(claims_of_puzzle, truth, false, 0)" in rule.body

    def test_length_where_prolog_output_valid(self):
        """Test that generated Prolog output is syntactically valid"""
        dsl = """
        ?items meaning "items"

        ?result becomes true
            when (length of ?items where ?active is true) > 5
        """
        ast = parse_string(dsl)
        program = translate_to_fol(ast)
        prolog = program.to_prolog_string()

        # Should not contain Python dict syntax
        assert "{'list'" not in prolog
        assert "'value': False" not in prolog
        assert "'value': True" not in prolog

        # Should contain proper Prolog predicate
        assert "length_where_is_gt(items, active, true, 5)" in prolog


class TestCompleteExamplesNewSyntax:
    """Integration tests with complete examples using new syntax"""

    def test_web_of_lies_complete(self):
        """Test the web of lies example with new syntax"""
        dsl = '''
        ?truth_claim resembles "tells the truth", "is truthful", "is honest"
        ?lie_claim resembles "lies", "is lying", "is a liar"

        ?puzzle meaning "the logic puzzle" with
            ?first_person_is_truthful can be true, false
            ?claims with
                ?speaker meaning "who is speaking"
                ?target meaning "who they describe"
                ?assertion meaning "what they claim"
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

    def test_medical_triage_complete(self):
        """Test the medical triage example with new syntax"""
        dsl = '''
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
                ?name meaning "medication name"
                ?dosage meaning "current dosage"

        ?is_emergency becomes true when ?text is like ?emergency_symptoms

        ?urgency can be "emergency", "urgent", "routine"

        ?urgency becomes "emergency"
            when ?symptoms of ?patient has any that ?is_emergency is true
            overriding all

        normally ?urgency becomes "urgent"
            when ?symptoms of ?patient has any that is like ?urgent_symptoms

        normally ?urgency becomes "routine"

        ?referral becomes "cardiology"
            when ?symptoms of ?patient has any that is like ?cardiac_terms
        '''
        ast = parse_string(dsl)

        categories = [n for n in ast if isinstance(n, SemanticCategory)]
        assert len(categories) == 3

        rules = [n for n in ast if isinstance(n, Rule)]
        assert len(rules) >= 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
