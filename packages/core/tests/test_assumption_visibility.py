"""
Test Assumption Visibility feature.

Tests that the DeLP engine correctly identifies and warns about:
1. Unknown semantic categories (typos like vaccine_termz instead of vaccine_terms)
2. Unknown predicates
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from canto_core.delp.program import DeLPProgram
from canto_core.delp.models import DeLPRule, DeLPRuleSource, DeLPDeclaration


def make_decl(name, values_from=None):
    """Create a DeLPDeclaration."""
    return DeLPDeclaration(
        name=name,
        description=None,
        values_from=values_from
    )


def make_category(description, patterns):
    """Create a mock category object."""
    return type('Category', (), {
        'description': description,
        'patterns': patterns
    })()


def make_rule(rule_id, head, body, head_variable, head_value):
    """Create a DeLPRule with source."""
    source = DeLPRuleSource(
        head_variable=head_variable,
        head_value=head_value,
        conditions=[],
        exceptions=[],
        priority='normal',
        override_target=None
    )
    return DeLPRule(id=rule_id, head=head, body=body, source=source)


def create_test_program_with_typo():
    """Create a DeLP program with a typo in category name."""
    program = DeLPProgram()

    # Add a semantic category
    program.semantic_categories['vaccine_terms'] = make_category(
        'Vaccine-related terms',
        ['vaccine', 'immunization', 'shot']
    )

    # Add patterns
    program.patterns = [
        ('vaccine_terms', 'vaccine'),
        ('vaccine_terms', 'immunization'),
        ('vaccine_terms', 'shot'),
    ]

    # Add a declaration
    program.declarations['vaccine_flag'] = make_decl('vaccine_flag', [True, False])

    # Add a rule with TYPO: vaccine_termz instead of vaccine_terms
    program.defeasible_rules.append(make_rule(
        'vaccine_flag_1',
        'vaccine_flag(true)',
        ['matches(patient_query, vaccine_termz)'],  # TYPO HERE
        'vaccine_flag',
        True
    ))

    return program


def create_test_program_correct():
    """Create a DeLP program with correct category name."""
    program = DeLPProgram()

    # Add a semantic category
    program.semantic_categories['vaccine_terms'] = make_category(
        'Vaccine-related terms',
        ['vaccine', 'immunization', 'shot']
    )

    # Add patterns
    program.patterns = [
        ('vaccine_terms', 'vaccine'),
        ('vaccine_terms', 'immunization'),
        ('vaccine_terms', 'shot'),
    ]

    # Add a declaration
    program.declarations['vaccine_flag'] = make_decl('vaccine_flag', [True, False])

    # Add a rule with correct category name
    program.defeasible_rules.append(make_rule(
        'vaccine_flag_1',
        'vaccine_flag(true)',
        ['matches(patient_query, vaccine_terms)'],  # CORRECT
        'vaccine_flag',
        True
    ))

    return program


def test_typo_detection():
    """Test that typos in category names are detected."""
    print("\n" + "=" * 60)
    print("TEST: Typo Detection in Category Names")
    print("=" * 60)

    from canto_core.delp.engine import JanusDeLP

    program = create_test_program_with_typo()
    engine = JanusDeLP(program)
    engine.load()

    # Clear any previous warnings
    engine.clear_assumption_warnings()

    # Run a query - this should trigger assumption validation
    result = engine.delp_query("vaccine_flag(true)")
    print(f"\nQuery result: {result['status']}")

    # Get warnings
    warnings = engine.get_assumption_warnings()
    print(f"\nWarnings found: {len(warnings)}")

    engine.print_assumption_warnings()

    # Verify we got a warning about unknown category
    assert len(warnings) > 0, "Expected at least one warning"

    # Check that it's about vaccine_termz
    found_typo_warning = False
    for w in warnings:
        if 'vaccine_termz' in str(w.get('predicate', '')):
            found_typo_warning = True
            print(f"\n✓ Found typo warning for 'vaccine_termz'")
            if w.get('suggestion') and w['suggestion'] != 'none':
                print(f"✓ Suggestion provided: {w['suggestion']}")
            break

    assert found_typo_warning, "Expected warning about 'vaccine_termz' typo"
    print("\n✓ TEST PASSED: Typo detection works!")

    engine.cleanup()


def test_no_warnings_for_correct_program():
    """Test that correct programs don't generate warnings."""
    print("\n" + "=" * 60)
    print("TEST: No Warnings for Correct Program")
    print("=" * 60)

    from canto_core.delp.engine import JanusDeLP

    program = create_test_program_correct()
    engine = JanusDeLP(program)
    engine.load()

    # Clear any previous warnings
    engine.clear_assumption_warnings()

    # Run a query
    result = engine.delp_query("vaccine_flag(true)")
    print(f"\nQuery result: {result['status']}")

    # Get warnings - should be empty for known category
    warnings = engine.get_assumption_warnings()

    # Filter out warnings about vaccine_terms (which is a known category)
    unknown_category_warnings = [
        w for w in warnings
        if w.get('type') == 'unknown_category'
    ]

    print(f"\nUnknown category warnings: {len(unknown_category_warnings)}")
    engine.print_assumption_warnings()

    # matches(X, vaccine_terms) should NOT generate a warning
    # because vaccine_terms is defined in pattern/2
    for w in unknown_category_warnings:
        pred = str(w.get('predicate', ''))
        assert 'vaccine_terms' not in pred or 'vaccine_termz' in pred, \
            f"Unexpected warning for known category: {pred}"

    print("\n✓ TEST PASSED: No false positives for correct categories!")

    engine.cleanup()


if __name__ == '__main__':
    try:
        test_typo_detection()
        test_no_warnings_for_correct_program()
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
