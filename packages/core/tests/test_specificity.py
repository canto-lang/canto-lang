"""
Test: Improved Specificity Calculation

This test verifies that the new information-theoretic specificity calculation
works correctly, ensuring that rules matching narrow semantic categories
are considered more specific than rules with multiple generic conditions.

The key improvement:
- OLD: Body length (more conditions = more specific)
- NEW: Semantic specificity (narrower categories = more specific)

Example scenario:
- Rule A: matches(X, specific_vaccines) → 5 patterns → score = 1000/5 = 200
- Rule B: has(X, a), has(X, b) → 2 conditions → score = 10 + 10 = 20
- Result: Rule A (200) > Rule B (20), so A is more specific
"""

from pathlib import Path
import sys
import tempfile

# Add project root to path
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


def query_all(query_string: str):
    """Execute a Prolog query and return all results"""
    results = []
    for result in query(query_string):
        results.append(result)
    return results


def test_specificity_calculation():
    """Test that specificity is based on semantic content, not just body length."""

    print("=" * 60)
    print("TEST: Information-Theoretic Specificity Calculation")
    print("=" * 60)

    # Create a test DeLP program with competing rules
    test_program = """
% Semantic categories with different sizes
% specific_vaccines: 5 patterns (more specific)
pattern(specific_vaccines, "Comirnaty").
pattern(specific_vaccines, "Moderna").
pattern(specific_vaccines, "Spikevax").
pattern(specific_vaccines, "Novavax").
pattern(specific_vaccines, "Pfizer-BioNTech").

% general_terms: 20 patterns (less specific)
pattern(general_terms, "medicine").
pattern(general_terms, "drug").
pattern(general_terms, "medication").
pattern(general_terms, "treatment").
pattern(general_terms, "therapy").
pattern(general_terms, "prescription").
pattern(general_terms, "pill").
pattern(general_terms, "tablet").
pattern(general_terms, "capsule").
pattern(general_terms, "injection").
pattern(general_terms, "shot").
pattern(general_terms, "dose").
pattern(general_terms, "remedy").
pattern(general_terms, "cure").
pattern(general_terms, "pharmaceutical").
pattern(general_terms, "medical").
pattern(general_terms, "health").
pattern(general_terms, "wellness").
pattern(general_terms, "care").
pattern(general_terms, "supplement").

% Test rules with different specificity profiles
% Rule 1: Single condition with NARROW category (5 patterns)
% Expected specificity: 1000/5 = 200
rule_info(rule_narrow, result(vaccine), defeasible, [matches(query, specific_vaccines)]).

% Rule 2: TWO conditions with BROAD categories (20 patterns each)
% Expected specificity: (1000/20) + (1000/20) = 50 + 50 = 100
rule_info(rule_broad_2cond, result(general), defeasible, [matches(query, general_terms), matches(query, general_terms)]).

% Rule 3: THREE conditions with has predicates
% Expected specificity: 10 + 10 + 10 = 30
rule_info(rule_has_3cond, result(other), defeasible, [has(query, a), has(query, b), has(query, c)]).

% Make them conflict on the same variable
% (In real usage, they'd have the same functor but different values)
"""

    print("\n1. Creating test DeLP program...")
    print("   - rule_narrow: 1 condition with 5-pattern category")
    print("   - rule_broad_2cond: 2 conditions with 20-pattern categories")
    print("   - rule_has_3cond: 3 'has' conditions")

    # Load the test program
    load_prolog_string(test_program)

    print("\n2. Calculating specificity scores...")

    # Query the specificity for each rule using Prolog
    # Create a helper to avoid binding Body (which contains complex terms)
    helper_code = """
get_rule_specificity(RuleIdStr, Score) :-
    rule_info(RuleId, _, _, Body),
    calculate_specificity(Body, Score),
    atom_string(RuleId, RuleIdStr).
"""
    load_prolog_string(helper_code)
    results = query_all("get_rule_specificity(RuleIdStr, Score)")

    print("\n   Results:")
    scores = {}
    for result in results:
        rule_id = result.get('RuleIdStr', 'unknown')
        score = result.get('Score', 0)
        scores[rule_id] = score
        print(f"   - {rule_id}: specificity = {score:.2f}")

    print("\n3. Verifying specificity ordering...")

    # Verify the ordering is correct
    narrow_score = scores.get('rule_narrow', 0)
    broad_score = scores.get('rule_broad_2cond', 0)
    has_score = scores.get('rule_has_3cond', 0)

    print(f"\n   Expected ordering: rule_narrow > rule_broad_2cond > rule_has_3cond")
    print(f"   Actual scores: {narrow_score:.2f} > {broad_score:.2f} > {has_score:.2f}")

    # Assertions
    errors = []

    if not (narrow_score > broad_score):
        errors.append(f"FAIL: rule_narrow ({narrow_score}) should beat rule_broad_2cond ({broad_score})")
    else:
        print(f"   ✓ rule_narrow ({narrow_score:.2f}) > rule_broad_2cond ({broad_score:.2f})")

    if not (broad_score > has_score):
        errors.append(f"FAIL: rule_broad_2cond ({broad_score}) should beat rule_has_3cond ({has_score})")
    else:
        print(f"   ✓ rule_broad_2cond ({broad_score:.2f}) > rule_has_3cond ({has_score:.2f})")

    # Under OLD system (body length), rule_has_3cond would win (3 > 2 > 1)
    # Under NEW system (semantic specificity), rule_narrow wins (200 > 100 > 30)

    print("\n4. Comparing with old body-length approach...")
    print("   OLD (body length): rule_has_3cond (3) > rule_broad_2cond (2) > rule_narrow (1)")
    print("   NEW (specificity): rule_narrow (200) > rule_broad_2cond (100) > rule_has_3cond (30)")

    if narrow_score > has_score:
        print("   ✓ NEW approach correctly prefers narrow category over more conditions")
    else:
        errors.append("FAIL: New approach should prefer narrow category over more conditions")

    print("\n" + "=" * 60)
    if errors:
        print("TEST FAILED:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("TEST PASSED: Specificity calculation working correctly!")
        print("=" * 60)
        return True


def test_explicit_weights():
    """Test that explicit category weights override automatic calculation."""

    print("\n" + "=" * 60)
    print("TEST: Explicit Category Weights")
    print("=" * 60)

    test_program = """
% Category with explicit weight
category_weight(high_priority_terms, 500).

% Category without explicit weight (will use pattern count)
pattern(normal_terms, "a").
pattern(normal_terms, "b").
pattern(normal_terms, "c").
pattern(normal_terms, "d").
pattern(normal_terms, "e").

% Rules using these categories
rule_info(rule_weighted, result(weighted), defeasible, [matches(query, high_priority_terms)]).
rule_info(rule_normal, result(normal), defeasible, [matches(query, normal_terms)]).
"""

    load_prolog_string(test_program)

    # Redefine helper for this test context
    helper_code = """
get_rule_specificity2(RuleIdStr, Score) :-
    rule_info(RuleId, _, _, Body),
    calculate_specificity(Body, Score),
    atom_string(RuleId, RuleIdStr).
"""
    load_prolog_string(helper_code)
    results = query_all("get_rule_specificity2(RuleIdStr, Score)")

    scores = {}
    for result in results:
        rule_id = result.get('RuleIdStr', 'unknown')
        score = result.get('Score', 0)
        scores[rule_id] = score
        print(f"   - {rule_id}: specificity = {score:.2f}")

    weighted_score = scores.get('rule_weighted', 0)
    normal_score = scores.get('rule_normal', 0)

    # Explicit weight of 500 should beat automatic 1000/5 = 200
    if weighted_score == 500:
        print(f"   ✓ Explicit weight (500) applied correctly")
    else:
        print(f"   ✗ Expected 500 for explicit weight, got {weighted_score}")
        return False

    if normal_score == 200:  # 1000/5 patterns
        print(f"   ✓ Automatic calculation (200) working for normal category")
    else:
        print(f"   ✗ Expected 200 for 5-pattern category, got {normal_score}")
        return False

    print("\n" + "=" * 60)
    print("TEST PASSED: Explicit weights override automatic calculation!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SPECIFICITY CALCULATION TEST SUITE")
    print("=" * 60)

    try:
        # Load the DeLP meta-interpreter first
        load_delp_meta()

        success1 = test_specificity_calculation()
        success2 = test_explicit_weights()

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"  Specificity calculation: {'PASS' if success1 else 'FAIL'}")
        print(f"  Explicit weights: {'PASS' if success2 else 'FAIL'}")

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
