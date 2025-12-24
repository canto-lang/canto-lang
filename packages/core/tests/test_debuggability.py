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

if __name__ == "__main__":
    test_conflict_detection()
    test_gap_analysis()
