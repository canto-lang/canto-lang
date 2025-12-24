"""
Integration test for LLM predicate injection into DeLP Analyzer
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from canto_core.parser.dsl_parser import parse_string
from canto_core.codegen import DeLPTranslator
from canto_core.delp.analyzer import DeLPReasoningAnalyzer

def test_llm_predicate_injection():
    """
    Test that extra Prolog code (simulating LLM output) is correctly
    loaded and used by the analyzer.
    """
    # 1. Define DSL with a semantic category
    dsl = """
    ?flag can be true, false
    ?text meaning "input text"
    ?category resembles "test"

    ?flag becomes true
        when ?text is like ?category
    """

    ast = parse_string(dsl)
    translator = DeLPTranslator()
    program = translator.translate(ast)

    # 2. Define "LLM-generated" predicates
    # This predicate says: is_like(Text, category) is true if Text contains 'magic_word'
    llm_predicates = """
    is_like(Text, category) :-
        sub_string(Text, _, _, _, 'magic_word').
    """

    # 3. Run analyzer with extra prolog
    analyzer = DeLPReasoningAnalyzer(program, extra_prolog=llm_predicates)

    # 4. Verify that the predicate works
    # We run analyze() which triggers loading of both program and extra prolog
    analyzer.analyze()

    # Test case 1: Text contains magic word -> should match
    result_match = analyzer.engine.check_fact("is_like('this has magic_word in it', category)")
    assert result_match is True, "Should match when magic word is present"

    # Test case 2: Text does NOT contain magic word -> should fail
    result_no_match = analyzer.engine.check_fact("is_like('this is boring', category)")
    assert result_no_match is False, "Should not match when magic word is missing"

    print("\n✓ LLM predicate injection verified successfully")

if __name__ == "__main__":
    test_llm_predicate_injection()
