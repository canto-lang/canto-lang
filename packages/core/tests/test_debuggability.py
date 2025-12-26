"""
Test case for DSL Debuggability (Conflict Detection & Gap Analysis)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from canto_core.parser.dsl_parser import parse_string
from canto_core.codegen import DeLPTranslator
from canto_core.delp.analyzer import DeLPReasoningAnalyzer

def test_conflict_detection():
    """
    Test that the analyzer correctly detects and reports conflicts.
    """
    print("\n=== Testing Conflict Detection ===")

    # DSL with a clear conflict:
    # 1. Strict rule says flag is True
    # 2. Strict rule says flag is False
    # 3. No superiority defined
    dsl_conflict = """
    ?flag can be true, false

    ?flag becomes true
    ?flag becomes false
    """

    ast = parse_string(dsl_conflict)
    translator = DeLPTranslator()
    program = translator.translate(ast)
    analyzer = DeLPReasoningAnalyzer(program)

    analysis = analyzer.analyze()

    # Check validation errors
    errors = analysis.get('validation_errors', [])
    print(f"Validation Errors found: {len(errors)}")
    for err in errors:
        print(f" - {err}")

    # We expect a "contradictory_strict_rules" error
    has_conflict_error = any(e['type'] == 'contradictory_strict_rules' for e in errors)
    if has_conflict_error:
        print("✓ Correctly detected contradictory strict rules")
    else:
        print("✗ Failed to detect contradictory strict rules")

def test_gap_analysis():
    """
    Test that the analyzer detects gaps (unreachable rules).
    """
    print("\n=== Testing Gap Analysis (Unreachable Rules) ===")

    # DSL with an unreachable rule:
    # 1. Strict rule says flag is True (always)
    # 2. Defeasible rule says flag is False (normally)
    # 3. Strict beats defeasible, so the second rule is unreachable
    dsl_gap = """
    ?flag can be true, false

    ?flag becomes true
    normally ?flag becomes false
    """

    ast = parse_string(dsl_gap)
    translator = DeLPTranslator()
    program = translator.translate(ast)
    analyzer = DeLPReasoningAnalyzer(program)

    analysis = analyzer.analyze()

    # Check validation errors
    errors = analysis.get('validation_errors', [])
    print(f"Validation Errors found: {len(errors)}")
    for err in errors:
        print(f" - {err}")

    # We expect an "unreachable_defeasible_rule" error
    has_gap_error = any(e['type'] == 'unreachable_defeasible_rule' for e in errors)
    if has_gap_error:
        print("✓ Correctly detected unreachable defeasible rule (gap)")
    else:
        print("✗ Failed to detect unreachable defeasible rule")

def test_var_equals_static_analysis():
    """
    Test that static analysis works correctly with var_equals predicate.
    This tests the case where we compare two qualified variables:
    ?target of ?puzzle is ?base of ?puzzle
    """
    print("\n=== Testing var_equals Static Analysis ===")

    dsl_var_equals = """
?puzzle meaning "the puzzle" with
    ?target_person meaning "who we ask about"
    ?base_person meaning "base person"
    ?base_is_truthful can be true, false meaning "whether base tells truth"

?answer can be "Yes", "No"

?answer becomes "Yes"
    when ?target_person of ?puzzle is ?base_person of ?puzzle
    and ?base_is_truthful of ?puzzle is true

?answer becomes "No"
    when ?target_person of ?puzzle is ?base_person of ?puzzle
    and ?base_is_truthful of ?puzzle is false
"""

    ast = parse_string(dsl_var_equals)
    translator = DeLPTranslator()
    program = translator.translate(ast)
    analyzer = DeLPReasoningAnalyzer(program)

    # This should not raise an exception
    analysis = analyzer.analyze()

    # Check that analysis completed without error
    print(f"Analysis completed: {len(analysis.get('variables', {}))} variables analyzed")

    # Check for the var_equals usage in generated prolog
    prolog_kb = program.to_prolog_string()
    has_var_equals = 'var_equals' in prolog_kb
    if has_var_equals:
        print("✓ var_equals predicate correctly generated in Prolog KB")
    else:
        print("✗ var_equals predicate not found in generated Prolog KB")

    # The analysis should have detected that ?answer has competing rules
    answer_analysis = analysis.get('variables', {}).get('answer', {})
    print(f"Answer analysis: {answer_analysis}")

    assert has_var_equals, "var_equals should be generated for variable-to-variable comparison"


if __name__ == "__main__":
    test_conflict_detection()
    test_gap_analysis()
    test_var_equals_static_analysis()
