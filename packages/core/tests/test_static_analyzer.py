#!/usr/bin/env python3
"""
Test script for DeLP static analyzer
Tests compile-time validation of DeLP programs
"""

from canto_core.parser.dsl_parser import parse_string
from canto_core.fol import translate_to_fol
from canto_core.delp import create_janus_engine

print("=" * 70)
print("DeLP STATIC ANALYZER TEST")
print("=" * 70)

# Test 1: Valid program
print("\n" + "=" * 70)
print("TEST 1: Valid Program (should pass)")
print("=" * 70)

valid_dsl = """
$vaccine_flag IS BOOLEAN
$query_intent IS TEXT
$patient_query IS TEXT

$vaccine_terms IS LIKE ["vaccine", "vaccination"]

# Strict rule with OVERRIDES
$vaccine_flag IS True
    WHEN $patient_query MATCHES $vaccine_terms
    OVERRIDES ALL

# Defeasible rule
NORMALLY $vaccine_flag IS True
    WHEN $query_intent IS "prevention"
"""

try:
    ast = parse_string(valid_dsl)
    delp = translate_to_fol(ast)

    engine = create_janus_engine(delp)
    engine.load()

    # Run validation
    result = engine.validate_program()

    if result['valid']:
        print("✓ Validation PASSED (as expected)")
    else:
        print("✗ Validation FAILED (unexpected!)")
        print("Errors:", result['errors'])

    engine.cleanup()
except Exception as e:
    print(f"✗ Test failed with exception: {e}")

# Test 2: Circular superiority
print("\n" + "=" * 70)
print("TEST 2: Circular Superiority (should fail)")
print("=" * 70)

circular_dsl = """
$flag IS BOOLEAN

# Create circular superiority: r1 > r2 and r2 > r1
$flag IS True
    WHEN $condition1 IS True

$flag IS False
    WHEN $condition2 IS True
"""

# Manually create circular superiority for testing
try:
    from canto_core.codegen.translator import DeLPProgram

    program = DeLPProgram()
    program.add_strict_rule('r1', 'flag(true)', ['condition1(true)'], None)
    program.add_strict_rule('r2', 'flag(false)', ['condition2(true)'], None)

    # Create circular superiority
    program.add_superiority('r1', 'r2')
    program.add_superiority('r2', 'r1')

    engine = create_janus_engine(program)
    engine.load()

    # Run validation
    result = engine.validate_program()

    if not result['valid']:
        print("✓ Validation correctly FAILED (detected circular superiority)")
        print("Errors found:")
        for error in result['errors']:
            print(f"  - [{error['type']}] {error['details']}")
    else:
        print("✗ Validation PASSED (should have detected circular superiority!)")

    engine.cleanup()
except Exception as e:
    print(f"✗ Test failed with exception: {e}")

# Test 3: Contradictory strict rules without superiority
print("\n" + "=" * 70)
print("TEST 3: Contradictory Strict Rules (should fail)")
print("=" * 70)

contradictory_dsl = """
$flag IS BOOLEAN

# Two strict rules, same variable, different values, NO superiority
$flag IS True
    WHEN $condition1 IS True

$flag IS False
    WHEN $condition2 IS True
"""

try:
    ast = parse_string(contradictory_dsl)
    delp = translate_to_fol(ast)

    engine = create_janus_engine(delp)
    engine.load()

    # Run validation
    result = engine.validate_program()

    if not result['valid']:
        print("✓ Validation correctly FAILED (detected contradictory strict rules)")
        print("Errors found:")
        for error in result['errors']:
            print(f"  - [{error['type']}] {error['details']}")
    else:
        print("✗ Validation PASSED (should have detected contradiction!)")

    engine.cleanup()
except Exception as e:
    print(f"✗ Test failed with exception: {e}")

# Test 4: Test analyzer integration
print("\n" + "=" * 70)
print("TEST 4: Analyzer Integration Test")
print("=" * 70)

try:
    from canto_core.delp.analyzer import DeLPReasoningAnalyzer

    ast = parse_string(valid_dsl)
    delp = translate_to_fol(ast)

    analyzer = DeLPReasoningAnalyzer(delp)
    analysis = analyzer.analyze()

    print(f"Valid: {analysis['valid']}")
    print(f"Validation errors: {len(analysis['validation_errors'])}")
    print(f"Variables analyzed: {len(analysis['variables'])}")
    print(f"Semantic categories: {len(analysis['semantic_categories'])}")
    print(f"Conflict resolutions: {len(analysis['conflict_summary'])}")

    if analysis['valid']:
        print("\n✓ Analyzer integration PASSED")
    else:
        print("\n✗ Analyzer found issues:")
        for error in analysis['validation_errors']:
            print(f"  - {error}")

    analyzer.engine.cleanup()
except Exception as e:
    print(f"✗ Test failed with exception: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("✓ All tests completed!")
print("=" * 70)
