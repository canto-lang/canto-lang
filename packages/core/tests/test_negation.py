"""
Test: Negation Semantics (NAF vs Warrant-based)

This test verifies the two negation operators:
1. NOT (NAF) - Fails if ANY argument exists for the goal
2. NOT WARRANTED - Fails only if the goal is WARRANTED (has undefeated argument)

Key difference:
- NOT: Conservative - fails even if arguments are defeated
- NOT WARRANTED: Checks actual warrant status
"""

from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).parent))

try:
    from janus_swi import consult, query
except ImportError:
    print("ERROR: janus_swi not available. Install with: pip install janus-swi")
    sys.exit(1)


def load_delp_meta():
    """Load the DeLP meta-interpreter"""
    meta_path = Path(__file__).parent / "delp" / "delp_meta.pl"
    if not meta_path.exists():
        raise FileNotFoundError(f"DeLP meta-interpreter not found at {meta_path}")
    consult(str(meta_path))
    print(f"✓ Loaded DeLP meta-interpreter from {meta_path}")


def load_prolog_string(prolog_code: str):
    """Load Prolog code from string via temp file"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pl', delete=False) as f:
        f.write(prolog_code)
        temp_path = f.name
    consult(temp_path)
    return temp_path


def query_once(query_string: str):
    """Execute a Prolog query and return first result"""
    for result in query(query_string):
        return result
    return None


def test_naf_vs_warrant():
    """Test the difference between NAF and warrant-based negation."""

    print("=" * 60)
    print("TEST: NAF vs Warrant-Based Negation")
    print("=" * 60)

    # Scenario:
    # - Rule 1: query_intent(prevention) :- matches(X, prevention_terms)
    # - Rule 2: query_intent(treatment) :- matches(X, treatment_terms)  [OVERRIDES Rule 1]
    # - If both match, treatment DEFEATS prevention
    #
    # Now test:
    # - Rule using NOT query_intent(prevention) - should FAIL (argument exists)
    # - Rule using NOT WARRANTED query_intent(prevention) - should SUCCEED (prevention is defeated)

    test_program = """
% Patterns for both intents
pattern(prevention_terms, "prevent").
pattern(treatment_terms, "treat").

% Both patterns will match (simulating text that contains both)
matches(input, prevention_terms).
matches(input, treatment_terms).

% Rule 1: Intent is prevention
rule_info(rule_prevention, query_intent(prevention), defeasible, [matches(input, prevention_terms)]).

% Rule 2: Intent is treatment (OVERRIDES prevention)
rule_info(rule_treatment, query_intent(treatment), defeasible, [matches(input, treatment_terms)]).

% Superiority: treatment beats prevention
sup(rule_treatment, rule_prevention).

% Test rules using different negation types:
% Rule A: Uses NAF - should FAIL to derive because prevention has an argument
rule_info(rule_naf_test, result(naf_test), defeasible, [query_intent(treatment), \\+(query_intent(prevention))]).

% Rule B: Uses warrant-based - should SUCCEED because prevention is DEFEATED
rule_info(rule_warrant_test, result(warrant_test), defeasible, [query_intent(treatment), not_warranted(query_intent(prevention))]).
"""

    print("\n1. Loading test program...")
    print("   Scenario: Both prevention and treatment patterns match")
    print("   Rule: treatment OVERRIDES prevention")
    print()

    load_prolog_string(test_program)

    print("2. Testing warrant status of query_intent(prevention)...")
    result = query_once("delp_query(query_intent(prevention), Status, _)")
    if result:
        status = result.get('Status', 'unknown')
        print(f"   query_intent(prevention) status: {status}")
        if status == 'defeated':
            print("   ✓ Prevention is correctly DEFEATED by treatment")
        else:
            print(f"   ✗ Expected 'defeated', got '{status}'")
    else:
        print("   ✗ Query failed")

    print("\n3. Testing warrant status of query_intent(treatment)...")
    result = query_once("delp_query(query_intent(treatment), Status, _)")
    if result:
        status = result.get('Status', 'unknown')
        print(f"   query_intent(treatment) status: {status}")
        if status == 'warranted':
            print("   ✓ Treatment is correctly WARRANTED")
        else:
            print(f"   ✗ Expected 'warranted', got '{status}'")
    else:
        print("   ✗ Query failed")

    print("\n4. Testing NAF rule (NOT query_intent(prevention))...")
    # Use a query that doesn't return the complex Arg term
    result = query_once("find_argument(result(naf_test), _, []) -> Found = yes ; Found = no")
    if result and result.get('Found') == 'yes':
        print("   ✗ NAF rule SUCCEEDED (should have failed)")
        naf_success = False
    else:
        print("   ✓ NAF rule correctly FAILED")
        print("      (Because an argument for prevention EXISTS, even though defeated)")
        naf_success = True

    print("\n5. Testing warrant-based rule (NOT WARRANTED query_intent(prevention))...")
    # Use a query that doesn't return the complex Arg term
    result = query_once("find_argument(result(warrant_test), _, []) -> Found = yes ; Found = no")
    if result and result.get('Found') == 'yes':
        print("   ✓ Warrant-based rule correctly SUCCEEDED")
        print("      (Because prevention is NOT warranted - it's defeated)")
        warrant_success = True
    else:
        print("   ✗ Warrant-based rule FAILED (should have succeeded)")
        warrant_success = False

    print("\n" + "=" * 60)
    if naf_success and warrant_success:
        print("TEST PASSED: NAF and warrant-based negation work correctly!")
        print()
        print("Summary:")
        print("  - NOT (NAF): Fails if ANY argument exists")
        print("  - NOT WARRANTED: Fails only if goal is WARRANTED")
        print("=" * 60)
        return True
    else:
        print("TEST FAILED")
        print(f"  NAF test: {'PASS' if naf_success else 'FAIL'}")
        print(f"  Warrant test: {'PASS' if warrant_success else 'FAIL'}")
        print("=" * 60)
        return False


def test_parser():
    """Test that the parser handles NOT WARRANTED syntax."""
    print("\n" + "=" * 60)
    print("TEST: Parser handles NOT WARRANTED syntax")
    print("=" * 60)

    try:
        from canto_core.parser.dsl_parser import parse_string

        dsl_code = '''
$intent IS TEXT
$result IS TEXT

$intent IS "prevention" WHEN $input MATCHES $prevention_terms

$result IS "test"
    WHEN NOT WARRANTED $intent IS "prevention"
'''
        print("\n1. Parsing DSL with NOT WARRANTED...")
        ast = parse_string(dsl_code)
        print(f"   ✓ Parsed {len(ast)} AST nodes")

        # Find the rule with NOT WARRANTED
        for node in ast:
            if hasattr(node, 'conditions') and node.conditions:
                for cond in node.conditions:
                    if hasattr(cond, 'operator') and cond.operator == 'NOT_WARRANTED':
                        print(f"   ✓ Found NOT_WARRANTED condition: {cond}")
                        print("=" * 60)
                        print("TEST PASSED: Parser correctly handles NOT WARRANTED")
                        print("=" * 60)
                        return True

        print("   ✗ NOT_WARRANTED condition not found in AST")
        return False

    except Exception as e:
        print(f"   ✗ Parser test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("NEGATION SEMANTICS TEST SUITE")
    print("=" * 60)

    try:
        # Load the DeLP meta-interpreter first
        load_delp_meta()

        success1 = test_parser()
        success2 = test_naf_vs_warrant()

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"  Parser test: {'PASS' if success1 else 'FAIL'}")
        print(f"  NAF vs Warrant test: {'PASS' if success2 else 'FAIL'}")

        if success1 and success2:
            print("\nAll tests passed!")
            sys.exit(0)
        else:
            print("\nSome tests failed!")
            sys.exit(1)

    except Exception as e:
        print(f"\nTest error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
