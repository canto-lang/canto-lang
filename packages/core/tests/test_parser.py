"""
Tests for Canto DSL parser
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from canto_core.parser import parse_string, parse_file
from canto_core.parser.validator import validate_ast
from canto_core.ast_nodes import VariableDeclaration, SemanticCategory, Rule
from canto_core.ast_nodes.rules import RulePriority, OverrideTarget, Condition


def test_freetext_variable():
    """Test parsing freetext variables (meaning)"""
    dsl = """
    ?patient_query meaning "the text provided by the patient"
    """
    ast = parse_string(dsl)
    assert len(ast) == 1
    assert isinstance(ast[0], VariableDeclaration)
    assert ast[0].name == "patient_query"
    assert ast[0].description == "the text provided by the patient"
    assert ast[0].values_from is None


def test_enum_variable():
    """Test parsing enum variables (can be)"""
    dsl = """
    ?query_intent can be "prevention", "treatment", "information_seeking"
    """
    ast = parse_string(dsl)
    assert len(ast) == 1
    assert isinstance(ast[0], VariableDeclaration)
    assert ast[0].name == "query_intent"
    assert ast[0].values_from == ["prevention", "treatment", "information_seeking"]


def test_enum_variable_with_description():
    """Test parsing enum variable with meaning description"""
    dsl = """
    ?query_intent can be "prevention", "treatment" meaning "the intent behind the query"
    """
    ast = parse_string(dsl)
    assert len(ast) == 1
    assert isinstance(ast[0], VariableDeclaration)
    assert ast[0].name == "query_intent"
    assert ast[0].description == "the intent behind the query"
    assert ast[0].values_from == ["prevention", "treatment"]


def test_boolean_variable():
    """Test parsing boolean variables (can be true, false)"""
    dsl = """
    ?is_valid can be true, false
    """
    ast = parse_string(dsl)
    assert len(ast) == 1
    assert isinstance(ast[0], VariableDeclaration)
    assert ast[0].name == "is_valid"
    assert ast[0].values_from == ["true", "false"]


def test_pattern_declaration():
    """Test parsing pattern declarations (resembles)"""
    dsl = """
    ?vaccine_terms resembles "vaccine", "vaccines", "vaccination"
    """
    ast = parse_string(dsl)
    assert len(ast) == 1
    assert isinstance(ast[0], SemanticCategory)
    assert ast[0].name == "vaccine_terms"
    assert ast[0].patterns == ["vaccine", "vaccines", "vaccination"]


def test_pattern_declaration_with_description():
    """Test parsing pattern declarations with meaning description"""
    dsl = """
    ?vaccine_terms resembles "vaccine", "vaccines", "vaccination"
        meaning "explicit vaccine-related terminology"
    """
    ast = parse_string(dsl)
    assert len(ast) == 1
    assert isinstance(ast[0], SemanticCategory)
    assert ast[0].name == "vaccine_terms"
    assert ast[0].patterns == ["vaccine", "vaccines", "vaccination"]
    assert ast[0].description == "explicit vaccine-related terminology"


def test_strict_rule():
    """Test parsing strict rules"""
    dsl = """
    ?vaccine_terms resembles "vaccine", "vaccination"

    ?vaccine_flag becomes true
        when ?patient_query is like ?vaccine_terms
        overriding all
    """
    ast = parse_string(dsl)
    rules = [node for node in ast if isinstance(node, Rule)]
    assert len(rules) == 1
    assert rules[0].head_variable == "vaccine_flag"
    assert rules[0].head_value == True
    assert rules[0].priority == RulePriority.STRICT
    assert rules[0].override_target == OverrideTarget.ALL


def test_defeasible_rule():
    """Test parsing defeasible rules (normally)"""
    dsl = """
    ?query_intent can be "prevention", "treatment"

    normally ?vaccine_flag becomes true
        when ?query_intent is "prevention"
    """
    ast = parse_string(dsl)
    rules = [node for node in ast if isinstance(node, Rule)]
    assert len(rules) == 1
    assert rules[0].priority == RulePriority.NORMAL


def test_rule_with_unless():
    """Test parsing rules with unless clause"""
    dsl = """
    ?treatment_intent resembles "treat", "treating"
    ?prevention_intent resembles "prevent", "preventing"

    ?query_intent becomes "treatment"
        when ?patient_query is like ?treatment_intent
        unless ?patient_query is like ?prevention_intent
    """
    ast = parse_string(dsl)
    rules = [node for node in ast if isinstance(node, Rule)]
    assert len(rules) == 1
    assert len(rules[0].conditions) > 0
    assert len(rules[0].exceptions) > 0


def test_complex_conditions():
    """Test parsing complex conditions with and/or"""
    dsl = """
    ?query_intent can be "prevention", "treatment"
    ?preventable_conditions resembles "COVID", "flu"

    normally ?vaccine_flag becomes true
        when ?query_intent is "prevention" and ?patient_query is like ?preventable_conditions
    """
    ast = parse_string(dsl)
    rules = [node for node in ast if isinstance(node, Rule)]
    assert len(rules) == 1


def test_parse_medical_query_file():
    """Test parsing the full medical_query.canto example"""
    example_path = Path(__file__).parent.parent / "examples" / "medical_query.canto"
    if example_path.exists():
        ast = parse_file(str(example_path))
        assert len(ast) > 0

        # Count different node types
        var_decls = [n for n in ast if isinstance(n, VariableDeclaration)]
        sem_cats = [n for n in ast if isinstance(n, SemanticCategory)]
        rules = [n for n in ast if isinstance(n, Rule)]

        assert len(sem_cats) > 0
        assert len(rules) > 0

        print(f"\nParsed medical_query.canto:")
        print(f"  Variable declarations: {len(var_decls)}")
        print(f"  Semantic Categories: {len(sem_cats)}")
        print(f"  Rules: {len(rules)}")


def test_boolean_values():
    """Test parsing boolean values"""
    dsl = """
    ?flag1 becomes true
    ?flag2 becomes false
    """
    ast = parse_string(dsl)
    rules = [n for n in ast if isinstance(n, Rule)]
    assert len(rules) == 2
    assert rules[0].head_value is True
    assert rules[1].head_value is False


def test_number_values():
    """Test parsing number values"""
    dsl = """
    ?count becomes 42
    """
    ast = parse_string(dsl)
    rules = [n for n in ast if isinstance(n, Rule)]
    assert rules[0].head_value == 42


def test_string_values():
    """Test parsing string values"""
    dsl = """
    ?status becomes "active"
    """
    ast = parse_string(dsl)
    rules = [n for n in ast if isinstance(n, Rule)]
    assert rules[0].head_value == "active"


def test_overriding_normal():
    """Test parsing overriding normal"""
    dsl = """
    ?flag becomes true
        overriding normal
    """
    ast = parse_string(dsl)
    rules = [n for n in ast if isinstance(n, Rule)]
    assert rules[0].override_target == OverrideTarget.NORMAL


def test_multiple_conditions():
    """Test rules with multiple conditions"""
    dsl = """
    ?result becomes true
        when ?a is true and ?b is true and ?c is true
    """
    ast = parse_string(dsl)
    rules = [n for n in ast if isinstance(n, Rule)]
    assert len(rules[0].conditions) >= 1
    assert rules[0].head_variable == "result"
    assert rules[0].head_value is True


def test_comments_preserved():
    """Test that comments don't interfere with parsing"""
    dsl = """
    # This is a header comment
    ?flag becomes true  # inline comment
        # Another comment
        when ?other is true  # condition comment
    """
    ast = parse_string(dsl)
    assert len(ast) == 1


def test_empty_dsl():
    """Test parsing empty DSL"""
    dsl = ""
    ast = parse_string(dsl)
    assert len(ast) == 0


def test_whitespace_handling():
    """Test that various whitespace patterns work"""
    dsl = """

    ?flag becomes true


    """
    ast = parse_string(dsl)
    assert len(ast) == 1


def test_not_condition():
    """Test parsing not conditions"""
    dsl = """
    ?vaccine_terms resembles "vaccine", "vaccination"

    ?vaccine_flag becomes false
        when not ?patient_query is like ?vaccine_terms
    """
    ast = parse_string(dsl)
    rules = [n for n in ast if isinstance(n, Rule)]
    assert len(rules) == 1
    assert len(rules[0].conditions) == 1
    assert rules[0].conditions[0].operator == "NOT"


# ============ NEW SYNTAX: 'with' blocks ============

def test_with_block_simple():
    """Test parsing simple 'with' block for nested structure"""
    dsl = """
    ?patient meaning "extracted patient information" with
        ?name meaning "patient's full name"
        ?age meaning "patient's age in years"
    """
    ast = parse_string(dsl)

    assert len(ast) >= 1
    parent = ast[0]
    assert isinstance(parent, VariableDeclaration)
    assert parent.name == "patient"
    assert parent.description == "extracted patient information"
    # Should have nested children
    assert hasattr(parent, 'children') and len(parent.children) == 2


def test_with_block_nested():
    """Test parsing nested 'with' blocks"""
    dsl = """
    ?patient meaning "patient info" with
        ?name meaning "patient name"
        ?symptoms with
            ?text meaning "symptom description"
            ?severity meaning "symptom severity"
    """
    ast = parse_string(dsl)

    assert len(ast) >= 1
    parent = ast[0]
    assert parent.name == "patient"
    assert hasattr(parent, 'children')


def test_with_block_with_enum():
    """Test 'with' block containing enum variable (can be)"""
    dsl = """
    ?patient meaning "patient" with
        ?name meaning "name"
        ?status can be "active", "inactive"
    """
    ast = parse_string(dsl)

    assert len(ast) >= 1
    parent = ast[0]
    assert parent.name == "patient"


def test_with_block_with_enum_and_rule_outside():
    """Test 'with' block with enum, rule defined at top level"""
    dsl = """
    ?emergency_symptoms resembles "chest pain", "difficulty breathing"

    ?patient meaning "patient" with
        ?symptoms with
            ?text meaning "symptom description"
            ?is_emergency can be true, false

    ?is_emergency becomes true when ?text is like ?emergency_symptoms
    """
    ast = parse_string(dsl)

    categories = [n for n in ast if isinstance(n, SemanticCategory)]
    assert len(categories) == 1
    assert categories[0].name == "emergency_symptoms"

    rules = [n for n in ast if isinstance(n, Rule)]
    assert len(rules) == 1


# ============ NEW SYNTAX: 'from' clause ============

def test_has_list_from_source():
    """Test parsing: ?parent has a list of ?type from ?source"""
    from canto_core.ast_nodes import HasDeclaration

    dsl = """
    ?text meaning "the input text"
    ?company resembles "Apple", "Google" meaning "company names"

    ?entities has a list of ?company from ?text
    """
    ast = parse_string(dsl)

    has_decls = [n for n in ast if isinstance(n, HasDeclaration)]
    assert len(has_decls) == 1
    has_decl = has_decls[0]
    assert has_decl.parent == "entities"
    assert has_decl.child == "company"
    assert has_decl.is_list is True
    # Should have extraction source
    assert hasattr(has_decl, 'source') and has_decl.source == "text"


def test_multiple_has_from_same_source():
    """Test multiple has declarations from same source"""
    from canto_core.ast_nodes import HasDeclaration

    dsl = """
    ?text meaning "input text"
    ?company resembles "Apple" meaning "company"
    ?person resembles "John" meaning "person"

    ?entities has a list of ?company from ?text
    ?entities has a list of ?person from ?text
    """
    ast = parse_string(dsl)

    has_decls = [n for n in ast if isinstance(n, HasDeclaration)]
    assert len(has_decls) == 2
    for decl in has_decls:
        assert decl.parent == "entities"
        assert decl.is_list is True
        assert hasattr(decl, 'source') and decl.source == "text"


def test_preprocess_with_blocks():
    """Test that the pre-processor correctly converts indentation to end markers"""
    from canto_core.parser.dsl_parser import preprocess_with_blocks

    dsl = """
?patient meaning "info" with
    ?name meaning "name"
    ?symptoms with
        ?text meaning "description"
    ?medications with
        ?med_name meaning "med"
"""

    result = preprocess_with_blocks(dsl)

    # Should have 'end' markers inserted
    assert result.count('end') == 3  # One for symptoms, medications, patient

    # Verify the structure by checking specific patterns
    assert '?symptoms with' in result
    assert '?medications with' in result


def test_indentation_based_with_blocks():
    """Test parsing of indentation-based with blocks"""
    dsl = """
?patient meaning "patient info" with
    ?name meaning "patient name"
    ?symptoms with
        ?text meaning "symptom description"
        ?is_emergency can be true, false
    ?medications with
        ?med_name meaning "medication name"
        ?dosage meaning "dosage"

?is_emergency becomes true when ?text is like ?emergency_symptoms
"""
    ast = parse_string(dsl)

    # Should have 2 top-level nodes: patient declaration and rule
    assert len(ast) == 2

    # First should be patient with nested structure
    patient = ast[0]
    assert isinstance(patient, VariableDeclaration)
    assert patient.name == "patient"
    assert len(patient.children) == 3  # name, symptoms, medications

    # Find symptoms child
    symptoms = None
    for child in patient.children:
        if child.name == "symptoms":
            symptoms = child
            break

    assert symptoms is not None
    assert len(symptoms.children) == 2  # text, is_emergency

    # Find medications child
    medications = None
    for child in patient.children:
        if child.name == "medications":
            medications = child
            break

    assert medications is not None
    assert len(medications.children) == 2  # med_name, dosage


def test_indentation_with_sibling_with_blocks():
    """Test that sibling with blocks at same indentation level are correctly parsed"""
    dsl = """
?root meaning "root" with
    ?child1 with
        ?nested1 meaning "n1"
    ?child2 with
        ?nested2 meaning "n2"
    ?child3 meaning "c3"
"""
    ast = parse_string(dsl)

    root = ast[0]
    assert root.name == "root"
    assert len(root.children) == 3  # child1, child2, child3

    # child1 and child2 should each have one nested child
    child1 = root.children[0]
    assert child1.name == "child1"
    assert len(child1.children) == 1
    assert child1.children[0].name == "nested1"

    child2 = root.children[1]
    assert child2.name == "child2"
    assert len(child2.children) == 1
    assert child2.children[0].name == "nested2"

    # child3 should have no children
    child3 = root.children[2]
    assert child3.name == "child3"
    assert len(child3.children) == 0


def test_import_statement():
    """Test parsing import statements"""
    from canto_core.ast_nodes import ImportDeclaration

    dsl = """
import food_categories

?food_item meaning "the food"
"""
    ast = parse_string(dsl)
    assert len(ast) == 2
    assert isinstance(ast[0], ImportDeclaration)
    assert ast[0].name == "food_categories"
    assert isinstance(ast[1], VariableDeclaration)


def test_multiple_imports():
    """Test parsing multiple import statements"""
    from canto_core.ast_nodes import ImportDeclaration

    dsl = """
import categories
import rules

?item meaning "an item"
"""
    ast = parse_string(dsl)
    assert len(ast) == 3
    assert isinstance(ast[0], ImportDeclaration)
    assert ast[0].name == "categories"
    assert isinstance(ast[1], ImportDeclaration)
    assert ast[1].name == "rules"


def test_import_with_instructions():
    """Test that imports work with instructions block"""
    from canto_core.ast_nodes import ImportDeclaration

    dsl = '''
"""
This is the instructions block.
"""

import my_module

?var meaning "a variable"
'''
    result = parse_string(dsl)
    assert result.instructions == "This is the instructions block."
    assert len(result.ast) == 2
    assert isinstance(result.ast[0], ImportDeclaration)
    assert result.ast[0].name == "my_module"


def test_import_file_resolution(tmp_path):
    """Test that parse_file resolves imports correctly"""
    from canto_core.ast_nodes import ImportDeclaration

    # Create a module file
    module_file = tmp_path / "my_module.canto"
    module_file.write_text('?imported_var meaning "from module"')

    # Create main file that imports the module
    main_file = tmp_path / "main.canto"
    main_file.write_text('''
import my_module

?main_var meaning "from main"
''')

    ast = parse_file(str(main_file))

    # Should have both variables, imported first
    assert len(ast) == 2
    assert ast[0].name == "imported_var"
    assert ast[1].name == "main_var"


def test_import_circular_prevention(tmp_path):
    """Test that circular imports don't cause infinite recursion"""
    # Create two files that import each other
    file_a = tmp_path / "a.canto"
    file_b = tmp_path / "b.canto"

    file_a.write_text('''
import b
?var_a meaning "from a"
''')
    file_b.write_text('''
import a
?var_b meaning "from b"
''')

    # Should not hang or crash
    ast = parse_file(str(file_a))

    # Should have variables from both files
    names = [node.name for node in ast]
    assert "var_a" in names
    assert "var_b" in names


def test_import_file_not_found(tmp_path):
    """Test that missing import raises FileNotFoundError"""
    main_file = tmp_path / "main.canto"
    main_file.write_text('import nonexistent')

    with pytest.raises(FileNotFoundError):
        parse_file(str(main_file))


def test_variable_ref_as_operand():
    """Test that variable_ref (qualified paths) can be used on the right side of 'is'"""
    dsl = """
    ?puzzle meaning "the puzzle" with
        ?target_person meaning "who we ask about"
        ?base_person meaning "base person"

    ?answer can be "Yes", "No"

    ?answer becomes "Yes"
        when ?target_person of ?puzzle is ?base_person of ?puzzle
    """
    ast = parse_string(dsl)
    rules = [node for node in ast if isinstance(node, Rule)]
    assert len(rules) == 1

    # Check that the condition has the right structure
    rule = rules[0]
    assert rule.conditions is not None
    # conditions can be a Condition or a list
    condition = rule.conditions if isinstance(rule.conditions, Condition) else rule.conditions[0]
    assert condition.operator == "IS"
    # Left side should be qualified path
    assert isinstance(condition.left, dict)
    assert "child" in condition.left
    # Right side should also be qualified path (variable_ref)
    assert isinstance(condition.right, (dict, tuple)), f"Expected dict or tuple, got {type(condition.right)}: {condition.right}"


def test_variable_ref_nested_as_operand():
    """Test nested variable_ref on right side of 'is'"""
    dsl = """
    ?outer meaning "outer" with
        ?middle meaning "middle" with
            ?inner meaning "inner"

    ?result can be "match", "no_match"

    ?result becomes "match"
        when ?inner of (?middle of ?outer) is ?inner of (?middle of ?outer)
    """
    ast = parse_string(dsl)
    rules = [node for node in ast if isinstance(node, Rule)]
    assert len(rules) == 1


def test_let_binding_any():
    """Test let binding with 'any' quantifier and compound conditions"""
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
    rules = [node for node in ast if isinstance(node, Rule)]
    assert len(rules) == 1

    rule = rules[0]
    assert rule.conditions is not None


def test_let_binding_all():
    """Test let binding with 'all' quantifier"""
    dsl = """
?items has a list of ?item meaning "items"
?item meaning "an item" with
    ?status can be "active", "inactive"

?all_active can be true, false

?all_active becomes true
    when let ?item be all in ?items where ?status of ?item is "active"
"""
    ast = parse_string(dsl)
    rules = [node for node in ast if isinstance(node, Rule)]
    assert len(rules) == 1


def test_let_binding_none():
    """Test let binding with 'none' quantifier"""
    dsl = """
?items has a list of ?item meaning "items"
?item meaning "an item" with
    ?is_error can be true, false

?no_errors can be true, false

?no_errors becomes true
    when let ?item be none in ?items where ?is_error of ?item is true
"""
    ast = parse_string(dsl)
    rules = [node for node in ast if isinstance(node, Rule)]
    assert len(rules) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
