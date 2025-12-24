"""
Test Trace Mode for Dialectical Trees.

Tests that the DeLP engine correctly traces the reasoning process:
1. Query start
2. Arguments found
3. Defeaters identified
4. Marking progression
5. Final status

Also tests the TracePresenter for user-friendly output.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from canto_core.delp.program import DeLPProgram
from canto_core.delp.models import DeLPRule, DeLPRuleSource, DeLPDeclaration
from canto_core.ast_nodes.rules import Rule, Condition, RulePriority, OverrideTarget


def make_decl(name, values_from=None):
    """Create a DeLPDeclaration."""
    return DeLPDeclaration(
        name=name,
        description=None,
        values_from=values_from
    )


def make_category(description, patterns):
    return type('Category', (), {
        'description': description,
        'patterns': patterns
    })()


def create_conflict_program():
    """Create a program with conflicting rules to test defeat tracing."""
    program = DeLPProgram()

    # Categories
    program.semantic_categories['vaccine_terms'] = make_category(
        'Vaccine terms', ['vaccine', 'immunization']
    )
    program.semantic_categories['treatment_terms'] = make_category(
        'Treatment terms', ['medication', 'treatment', 'drug']
    )

    program.patterns = [
        ('vaccine_terms', 'vaccine'),
        ('vaccine_terms', 'immunization'),
        ('treatment_terms', 'medication'),
        ('treatment_terms', 'treatment'),
        ('treatment_terms', 'drug'),
    ]

    # Declaration
    program.declarations['query_type'] = make_decl('query_type', ['vaccine', 'treatment'])

    # Rule 1: $query_type IS "vaccine" WHEN $input MATCHES $vaccine_terms
    rule1 = Rule(
        head_variable='query_type',
        head_value='vaccine',
        conditions=[Condition(operator='MATCHES', left='input', right='vaccine_terms')],
        priority=RulePriority.NORMAL,
        override_target=OverrideTarget.NONE
    )
    program.defeasible_rules.append(DeLPRule(
        id='query_type_1',
        head="query_type('vaccine')",
        body=['matches(input, vaccine_terms)'],
        source=DeLPRuleSource.from_ast(rule1)
    ))

    # Rule 2: $query_type IS "treatment" WHEN $input MATCHES $treatment_terms OVERRIDES ALL
    rule2 = Rule(
        head_variable='query_type',
        head_value='treatment',
        conditions=[Condition(operator='MATCHES', left='input', right='treatment_terms')],
        priority=RulePriority.NORMAL,
        override_target=OverrideTarget.ALL
    )
    program.defeasible_rules.append(DeLPRule(
        id='query_type_2',
        head="query_type('treatment')",
        body=['matches(input, treatment_terms)'],
        source=DeLPRuleSource.from_ast(rule2)
    ))

    # Rule 2 overrides Rule 1
    program.superiority.append({
        'superior': 'query_type_2',
        'inferior': 'query_type_1'
    })

    return program


def test_trace_basic():
    """Test that trace captures basic reasoning events."""
    print("\n" + "=" * 60)
    print("TEST: Basic Trace Capture")
    print("=" * 60)

    from canto_core.delp.engine import JanusDeLP

    program = create_conflict_program()
    engine = JanusDeLP(program)
    engine.load()

    # Run query with trace
    result = engine.delp_query_with_trace("query_type('vaccine')")

    print(f"\nQuery result: {result['status']}")
    print(f"Trace events: {len(result.get('trace', []))}")

    # Verify trace has events
    trace = result.get('trace', [])
    assert len(trace) > 0, "Expected trace events"

    # Check for key event types
    event_types = [e.get('type') for e in trace]
    print(f"Event types captured: {set(event_types)}")

    assert 'query_start' in event_types, "Expected query_start event"
    assert 'final_status' in event_types, "Expected final_status event"

    print("\n✓ TEST PASSED: Basic trace capture works!")
    engine.cleanup()


def test_trace_with_defeat():
    """Test that trace captures defeat relationships with accurate reasons."""
    print("\n" + "=" * 60)
    print("TEST: Trace with Defeat (using TracePresenter)")
    print("=" * 60)

    from canto_core.delp.engine import JanusDeLP
    from canto_core.devtools.trace_presenter import TracePresenter

    program = create_conflict_program()
    engine = JanusDeLP(program)
    engine.load()

    # Enable trace
    engine.enable_trace()

    # Query for vaccine - should show defeat by treatment rule
    result = engine.delp_query("query_type('vaccine')")
    print(f"\nQuery for vaccine: {result['status']}")

    # Get trace and present with TracePresenter
    trace = engine.get_trace()
    presenter = TracePresenter(program)

    print("\n--- User Mode ---")
    presenter.present(trace, mode='user')

    print("\n--- Debug Mode ---")
    presenter.present(trace, mode='debug')

    event_types = [e.get('type') for e in trace]

    # Should have defeater_found with reason details
    assert 'defeater_found' in event_types, "Expected defeater_found event"

    # Check that defeat reason is captured
    for event in trace:
        if event.get('type') == 'defeater_found':
            details = event.get('details', {})
            assert 'reason_type' in details, "Expected reason_type in defeater details"
            assert details['reason_type'] == 'explicit_superiority', \
                f"Expected explicit_superiority, got {details['reason_type']}"
            print(f"\n✓ Defeat reason captured: {details['reason_type']}")

    print("\n✓ TEST PASSED: Trace captures defeat with accurate reasons!")
    engine.disable_trace()
    engine.cleanup()


def test_trace_no_arguments():
    """Test trace when no arguments exist."""
    print("\n" + "=" * 60)
    print("TEST: Trace with No Arguments")
    print("=" * 60)

    from canto_core.delp.engine import JanusDeLP
    from canto_core.devtools.trace_presenter import TracePresenter

    program = create_conflict_program()
    engine = JanusDeLP(program)
    engine.load()

    engine.enable_trace()

    # Query for something that doesn't exist
    result = engine.delp_query("query_type('unknown')")
    print(f"\nQuery for unknown: {result['status']}")

    trace = engine.get_trace()
    presenter = TracePresenter(program)
    presenter.present(trace, mode='user')

    event_types = [e.get('type') for e in trace]

    # Should have no_arguments event
    assert 'no_arguments' in event_types, "Expected no_arguments event"

    print("\n✓ TEST PASSED: Trace captures no arguments case!")
    engine.disable_trace()
    engine.cleanup()


def test_presenter_formats_dsl():
    """Test that TracePresenter correctly formats DSL from metadata."""
    print("\n" + "=" * 60)
    print("TEST: TracePresenter DSL Formatting")
    print("=" * 60)

    from canto_core.devtools.trace_presenter import TracePresenter

    program = create_conflict_program()
    presenter = TracePresenter(program)

    # Test rule formatting
    rule_source = program.defeasible_rules[0].source
    formatted = presenter._format_rule(rule_source)
    print(f"\nFormatted rule 1: {formatted}")
    assert '$query_type IS "vaccine"' in formatted, "Expected variable format"
    assert '$input MATCHES $vaccine_terms' in formatted, "Expected condition format"

    rule_source2 = program.defeasible_rules[1].source
    formatted2 = presenter._format_rule(rule_source2)
    print(f"Formatted rule 2: {formatted2}")
    assert 'OVERRIDES ALL' in formatted2, "Expected OVERRIDES in output"

    # Test goal formatting
    goal_formatted = presenter._format_goal("query_type(vaccine)")
    print(f"Formatted goal: {goal_formatted}")
    assert goal_formatted == '$query_type IS "vaccine"', f"Unexpected format: {goal_formatted}"

    print("\n✓ TEST PASSED: TracePresenter formats DSL correctly!")


if __name__ == '__main__':
    try:
        test_trace_basic()
        test_trace_with_defeat()
        test_trace_no_arguments()
        test_presenter_formats_dsl()
        print("\n" + "=" * 60)
        print("ALL TRACE TESTS PASSED!")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
