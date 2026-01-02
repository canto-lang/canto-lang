"""
Test: OR Condition Translation using De Morgan's Law

This test verifies that rules with OR conditions are properly translated
using De Morgan's law: A or B ≡ ¬(¬A ∧ ¬B)

This approach:
- Keeps OR as a single rule (no expansion into multiple rules)
- Avoids cycles when combined with "overriding all"
- Preserves logical equivalence: conclusion holds if either branch succeeds
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from canto_core.parser.dsl_parser import parse_string
from canto_core.fol import translate_to_fol


def translate_to_delp(ast):
    """Wrapper that uses translate_to_fol"""
    return translate_to_fol(ast)


def test_simple_or_expansion():
    """Test that a simple OR is translated using De Morgan (single rule)."""

    print("=" * 60)
    print("TEST: Simple OR with De Morgan")
    print("=" * 60)

    dsl_code = '''
?input meaning "input text"
?result meaning "output result"

?pattern1 resembles "vaccine", "immunization"
?pattern2 resembles "treatment", "therapy"

?result becomes "matched"
    when ?input is like ?pattern1 or ?input is like ?pattern2
'''

    print("\n1. Parsing DSL with OR condition...")
    ast = parse_string(dsl_code)
    print(f"   Parsed {len(ast)} AST nodes")

    print("\n2. Translating to DeLP...")
    program = translate_to_delp(ast)

    # Count rules for result
    result_rules = [r for r in program.defeasible_rules + program.strict_rules
                    if r.head.startswith('result(')]

    print(f"   Generated {len(result_rules)} rule(s) for result:")
    for rule in result_rules:
        print(f"     - {rule.id}: {rule.head} :- {', '.join(rule.body)}")

    # Should have 1 rule with De Morgan translation
    assert len(result_rules) == 1, f"Expected 1 rule (De Morgan), got {len(result_rules)}"
    print("\n   OR correctly translated to single rule using De Morgan")

    # The body property shows or([...]) format, actual Prolog uses De Morgan
    rule = result_rules[0]
    body_str = ', '.join(rule.body)
    # Accept both or([...]) format in body and De Morgan \+ in actual Prolog
    assert 'or([' in body_str or '\\+' in body_str, "Rule should have OR structure"
    print(f"   Rule uses OR structure (or De Morgan in actual Prolog)")


def test_nested_or_expansion():
    """Test that nested ORs are handled correctly with De Morgan."""

    print("\n" + "=" * 60)
    print("TEST: Nested OR with De Morgan")
    print("=" * 60)

    dsl_code = '''
?input meaning "input text"
?result meaning "output result"

?p1 resembles "a"
?p2 resembles "b"
?p3 resembles "c"

?result becomes "matched"
    when ?input is like ?p1 or ?input is like ?p2 or ?input is like ?p3
'''

    print("\n1. Parsing DSL with nested OR conditions...")
    ast = parse_string(dsl_code)

    print("\n2. Translating to DeLP...")
    program = translate_to_delp(ast)

    result_rules = [r for r in program.defeasible_rules + program.strict_rules
                    if r.head.startswith('result(')]

    print(f"   Generated {len(result_rules)} rule(s) for result:")
    for rule in result_rules:
        print(f"     - {rule.id}: {', '.join(rule.body)}")

    # Should have 1 rule with nested De Morgan
    assert len(result_rules) == 1, f"Expected 1 rule, got {len(result_rules)}"
    print("\n   Nested OR correctly translated to single rule")


def test_or_with_and():
    """Test OR combined with AND conditions."""

    print("\n" + "=" * 60)
    print("TEST: OR with AND using De Morgan")
    print("=" * 60)

    dsl_code = '''
?input meaning "input text"
?flag can be true, false
?result meaning "output result"

?p1 resembles "a"
?p2 resembles "b"

?result becomes "matched"
    when (?input is like ?p1 or ?input is like ?p2) and ?flag is true
'''

    print("\n1. Parsing DSL with (A OR B) AND C pattern...")
    ast = parse_string(dsl_code)

    print("\n2. Translating to DeLP...")
    program = translate_to_delp(ast)

    result_rules = [r for r in program.defeasible_rules + program.strict_rules
                    if r.head.startswith('result(')]

    print(f"   Generated {len(result_rules)} rule(s) for result:")
    for rule in result_rules:
        print(f"     - {rule.id}: {', '.join(rule.body)}")

    # Should have 1 rule with both De Morgan OR and the AND condition
    assert len(result_rules) == 1, f"Expected 1 rule, got {len(result_rules)}"

    # Rule should have both the De Morgan OR and the flag condition
    rule = result_rules[0]
    body_str = ', '.join(rule.body)
    assert '\\+' in body_str, "Rule should have De Morgan pattern"
    assert 'flag(' in body_str, "Rule should have flag condition"
    print("\n   (A OR B) AND C correctly translated with De Morgan for OR part")


def test_cartesian_product():
    """Test multiple ORs - should still be single rule with De Morgan."""

    print("\n" + "=" * 60)
    print("TEST: Multiple ORs with De Morgan")
    print("=" * 60)

    dsl_code = '''
?x meaning "first input"
?y meaning "second input"
?result meaning "output result"

?a resembles "a"
?b resembles "b"
?c resembles "c"
?d resembles "d"

?result becomes "matched"
    when (?x is like ?a or ?x is like ?b) and (?y is like ?c or ?y is like ?d)
'''

    print("\n1. Parsing DSL with (A OR B) AND (C OR D) pattern...")
    ast = parse_string(dsl_code)

    print("\n2. Translating to DeLP...")
    program = translate_to_delp(ast)

    result_rules = [r for r in program.defeasible_rules + program.strict_rules
                    if r.head.startswith('result(')]

    print(f"   Generated {len(result_rules)} rule(s) for result:")
    for rule in result_rules:
        print(f"     - {rule.id}: {', '.join(rule.body)}")

    # Should have 1 rule (De Morgan handles both ORs in a single rule)
    assert len(result_rules) == 1, f"Expected 1 rule, got {len(result_rules)}"
    print("\n   Multiple ORs handled in single rule with De Morgan")


def test_or_with_overrides():
    """Test that OVERRIDES works correctly without creating cycles."""

    print("\n" + "=" * 60)
    print("TEST: OR with OVERRIDES (no cycles)")
    print("=" * 60)

    dsl_code = '''
?input meaning "input text"
?result meaning "output result"

?p1 resembles "a"
?p2 resembles "b"

?result becomes "default"

?result becomes "matched"
    when ?input is like ?p1 or ?input is like ?p2
    overriding all
'''

    print("\n1. Parsing DSL with OR + overriding...")
    ast = parse_string(dsl_code)

    print("\n2. Translating to DeLP...")
    program = translate_to_delp(ast)

    result_rules = [r for r in program.defeasible_rules + program.strict_rules
                    if r.head.startswith('result(')]

    print(f"   Generated {len(result_rules)} rules for result:")
    for rule in result_rules:
        print(f"     - {rule.id}: {rule.head} :- {', '.join(rule.body) if rule.body else 'true'}")

    print(f"\n   Superiority relations: {len(program.superiority)}")
    for sup in program.superiority:
        print(f"     - {sup['superior']} > {sup['inferior']}")

    # Should have 2 rules: 1 default + 1 from OR (with De Morgan)
    default_rule = [r for r in result_rules if not r.body or r.body == ['true']]
    or_rules = [r for r in result_rules if r.body and r.body != ['true']]

    assert len(default_rule) == 1, f"Expected 1 default rule, got {len(default_rule)}"
    assert len(or_rules) == 1, f"Expected 1 OR rule (De Morgan), got {len(or_rules)}"

    # Check superiority - single OR rule overrides default
    default_id = default_rule[0].id
    overriding_count = sum(1 for sup in program.superiority if sup['inferior'] == default_id)

    assert overriding_count == 1, f"Expected 1 superiority relation, found {overriding_count}"
    print("\n   Single OR rule (De Morgan) overrides the default - no sibling cycles!")


def test_no_disjunction_in_output():
    """Test that no Prolog disjunction (;) appears in output."""

    print("\n" + "=" * 60)
    print("TEST: No Prolog Disjunction in Output")
    print("=" * 60)

    dsl_code = '''
?input meaning "input text"
?result meaning "output result"

?p1 resembles "a"
?p2 resembles "b"

?result becomes "matched"
    when ?input is like ?p1 or ?input is like ?p2
'''

    print("\n1. Translating DSL with OR...")
    ast = parse_string(dsl_code)
    program = translate_to_delp(ast)

    print("\n2. Checking Prolog output for disjunctions...")
    prolog_output = program.to_prolog_string()

    # Check that there's no (a ; b) pattern in the output
    assert '; ' not in prolog_output, "Found Prolog disjunction (;) in output"
    print("   No Prolog disjunction (;) found in output")
    print("   OR conditions translated using De Morgan's law")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("OR CONDITION DE MORGAN TRANSLATION TEST SUITE")
    print("=" * 60)

    test_simple_or_expansion()
    test_nested_or_expansion()
    test_or_with_and()
    test_cartesian_product()
    test_or_with_overrides()
    test_no_disjunction_in_output()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
