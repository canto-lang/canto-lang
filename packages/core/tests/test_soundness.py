"""
Test case for DeLP Soundness (Meta-Interpreter Logic)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from canto_core.parser.dsl_parser import parse_string
from canto_core.codegen import DeLPTranslator
from canto_core.delp.engine import create_janus_engine

def test_specificity_principle():
    """
    Test that specificity is correctly applied.
    Rule 1: Birds fly (general)
    Rule 2: Penguins don't fly (specific)
    Penguin is a Bird.
    Result: Penguin should NOT fly.
    """
    print("\n=== Testing Specificity Principle ===")

    dsl = """
    ?fly can be true, false
    ?bird can be true, false
    ?penguin can be true, false

    ?bird becomes true when ?penguin is true

    normally ?fly becomes true when ?bird is true
    normally ?fly becomes false when ?penguin is true

    ?penguin becomes true
    """

    ast = parse_string(dsl)
    translator = DeLPTranslator()
    program = translator.translate(ast)
    engine = create_janus_engine(program)

    # Check fly(false) - should be warranted because penguin rule is more specific
    result = engine.delp_query("fly(false)")
    print(f"fly(false) status: {result['status']}")

    if result['status'] == 'warranted':
        print("✓ Specificity correctly applied (Penguin > Bird)")
    else:
        print("✗ Specificity failed")
        engine.pretty_print_tree("fly(false)")

def test_cycle_detection():
    """
    Test that the meta-interpreter handles cycles in reasoning without infinite loops.
    A -> B, B -> A
    """
    print("\n=== Testing Cycle Handling ===")

    dsl = """
    ?a can be true, false
    ?b can be true, false

    normally ?a becomes true when ?b is true
    normally ?b becomes true when ?a is true
    """

    ast = parse_string(dsl)
    translator = DeLPTranslator()
    program = translator.translate(ast)
    engine = create_janus_engine(program)

    # This should not hang
    try:
        result = engine.delp_query("a(true)")
        print(f"Cycle query status: {result['status']}")
        print("✓ Cycle handled without infinite loop")
    except Exception as e:
        print(f"✗ Cycle caused error: {e}")

def test_consistency():
    """
    Test that arguments are consistent (no contradictory literals in argument).
    """
    print("\n=== Testing Argument Consistency ===")

    # Define a rule that relies on a contradiction
    # If the meta-interpreter is sound, it should NOT form an argument from this
    dsl = """
    ?flag can be true, false
    ?a can be true, false

    ?a becomes true
    ?a becomes false

    normally ?flag becomes true when ?a is true and ?a is false
    """

    ast = parse_string(dsl)
    translator = DeLPTranslator()
    program = translator.translate(ast)
    engine = create_janus_engine(program)

    result = engine.delp_query("flag(true)")
    print(f"Contradictory premise status: {result['status']}")

    if result['status'] in ['undecided', 'error'] or result['tree'] is None:
        print("✓ Correctly rejected argument with contradictory premises")
    else:
        print("✗ Formed argument despite contradiction")

if __name__ == "__main__":
    test_specificity_principle()
    test_cycle_detection()
    test_consistency()
