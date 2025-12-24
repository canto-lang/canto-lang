"""
Tests for mutual exclusivity detection in conflict analysis.

When two rules have conditions that reference DIFFERENT categories,
they are mutually exclusive (can't both fire) and should NOT trigger
a COMPETES warning.
"""

import pytest
from pathlib import Path

from canto_core import parse_file, DeLPTranslator, DeLPReasoningAnalyzer
from canto_core.parser import CantoParser


FIXTURES_DIR = Path(__file__).parent / "fixtures"


def analyze_dsl(dsl_code: str) -> str:
    """Helper to parse DSL and return analysis report."""
    parser = CantoParser()
    result = parser.parse(dsl_code)
    translator = DeLPTranslator()
    program = translator.translate(result.ast)
    analyzer = DeLPReasoningAnalyzer(program)
    return analyzer.get_dsl_analysis_report()


class TestBasicMutualExclusivity:
    """Basic tests for mutual exclusivity with simple is like conditions."""

    def test_same_category_is_real_conflict(self):
        """Same category = CAN both fire = real conflict."""
        report = analyze_dsl('''
            ?food_terms resembles "pizza", "burger"
            ?input meaning "input"
            ?result can be "a", "b"
            ?result becomes "a" when ?input is like ?food_terms
            ?result becomes "b" when ?input is like ?food_terms
        ''')
        assert "CONFLICT" in report or "COMPETES" in report

    def test_different_categories_no_conflict(self):
        """Different categories = mutually exclusive = no conflict."""
        report = analyze_dsl('''
            ?fruit_terms resembles "apple", "orange"
            ?vehicle_terms resembles "car", "truck"
            ?input meaning "input"
            ?result can be "fruit", "vehicle"
            ?result becomes "fruit" when ?input is like ?fruit_terms
            ?result becomes "vehicle" when ?input is like ?vehicle_terms
        ''')
        assert "COMPETES" not in report, f"Got:\n{report}"

    def test_three_different_categories(self):
        """Three rules with different categories = all mutually exclusive."""
        report = analyze_dsl('''
            ?cat_a resembles "a1", "a2"
            ?cat_b resembles "b1", "b2"
            ?cat_c resembles "c1", "c2"
            ?input meaning "input"
            ?result can be "a", "b", "c"
            ?result becomes "a" when ?input is like ?cat_a
            ?result becomes "b" when ?input is like ?cat_b
            ?result becomes "c" when ?input is like ?cat_c
        ''')
        assert "COMPETES" not in report, f"Got:\n{report}"


class TestAndConditions:
    """Tests with AND conditions."""

    def test_and_with_different_categories_no_conflict(self):
        """
        AND conditions with different categories are still mutually exclusive.
        Rule 1: cat_a AND flag
        Rule 2: cat_b AND flag
        If cat_a != cat_b, only one is_like can match.
        """
        report = analyze_dsl('''
            ?cat_a resembles "apple"
            ?cat_b resembles "banana"
            ?input meaning "input"
            ?flag can be true, false
            ?result can be "a", "b"
            ?result becomes "a" when ?input is like ?cat_a and ?flag is true
            ?result becomes "b" when ?input is like ?cat_b and ?flag is true
        ''')
        assert "COMPETES" not in report, f"Got:\n{report}"

    def test_and_with_same_category_is_conflict(self):
        """
        AND conditions with SAME category can both fire.
        Both rules could match if input matches cat_a.
        """
        report = analyze_dsl('''
            ?cat_a resembles "apple"
            ?input meaning "input"
            ?flag can be true, false
            ?result can be "x", "y"
            ?result becomes "x" when ?input is like ?cat_a and ?flag is true
            ?result becomes "y" when ?input is like ?cat_a and ?flag is false
        ''')
        # Same category - could conflict (though flag makes them exclusive)
        # Conservative: should still warn since category overlaps
        assert "CONFLICT" in report or "COMPETES" in report

    def test_and_with_multiple_different_categories(self):
        """Multiple AND conditions, all with different categories."""
        report = analyze_dsl('''
            ?cat_a resembles "a"
            ?cat_b resembles "b"
            ?cat_c resembles "c"
            ?cat_d resembles "d"
            ?input meaning "input"
            ?result can be "x", "y"
            ?result becomes "x" when ?input is like ?cat_a and ?input is like ?cat_b
            ?result becomes "y" when ?input is like ?cat_c and ?input is like ?cat_d
        ''')
        # All categories different = mutually exclusive
        assert "COMPETES" not in report, f"Got:\n{report}"


class TestOrConditions:
    """Tests with OR conditions (De Morgan transformation: A OR B -> NOT(NOT A AND NOT B))."""

    def test_or_with_no_overlap(self):
        """
        OR conditions with completely different categories.
        Rule 1: cat_a OR cat_b
        Rule 2: cat_c OR cat_d
        No shared categories = mutually exclusive.
        """
        report = analyze_dsl('''
            ?cat_a resembles "a"
            ?cat_b resembles "b"
            ?cat_c resembles "c"
            ?cat_d resembles "d"
            ?input meaning "input"
            ?result can be "x", "y"
            ?result becomes "x" when ?input is like ?cat_a or ?input is like ?cat_b
            ?result becomes "y" when ?input is like ?cat_c or ?input is like ?cat_d
        ''')
        assert "COMPETES" not in report, f"Got:\n{report}"

    def test_or_with_overlap_is_conflict(self):
        """
        OR conditions with overlapping category.
        Rule 1: cat_a OR cat_b
        Rule 2: cat_b OR cat_c
        cat_b is shared = can both fire = conflict.
        """
        report = analyze_dsl('''
            ?cat_a resembles "a"
            ?cat_b resembles "b"
            ?cat_c resembles "c"
            ?input meaning "input"
            ?result can be "x", "y"
            ?result becomes "x" when ?input is like ?cat_a or ?input is like ?cat_b
            ?result becomes "y" when ?input is like ?cat_b or ?input is like ?cat_c
        ''')
        # cat_b is in both - they can both fire
        assert "CONFLICT" in report or "COMPETES" in report

    def test_triple_or_no_overlap(self):
        """
        Triple OR conditions with no overlap.
        Rule 1: cat_a OR cat_b OR cat_c
        Rule 2: cat_d OR cat_e OR cat_f
        No shared categories = mutually exclusive.
        """
        report = analyze_dsl('''
            ?cat_a resembles "a"
            ?cat_b resembles "b"
            ?cat_c resembles "c"
            ?cat_d resembles "d"
            ?cat_e resembles "e"
            ?cat_f resembles "f"
            ?input meaning "input"
            ?result can be "x", "y"
            ?result becomes "x" when ?input is like ?cat_a or ?input is like ?cat_b or ?input is like ?cat_c
            ?result becomes "y" when ?input is like ?cat_d or ?input is like ?cat_e or ?input is like ?cat_f
        ''')
        assert "COMPETES" not in report, f"Got:\n{report}"

    def test_triple_or_with_one_overlap(self):
        """
        Triple OR conditions with one shared category.
        Rule 1: cat_a OR cat_b OR cat_c
        Rule 2: cat_c OR cat_d OR cat_e
        cat_c is shared = can both fire = conflict.
        """
        report = analyze_dsl('''
            ?cat_a resembles "a"
            ?cat_b resembles "b"
            ?cat_c resembles "c"
            ?cat_d resembles "d"
            ?cat_e resembles "e"
            ?input meaning "input"
            ?result can be "x", "y"
            ?result becomes "x" when ?input is like ?cat_a or ?input is like ?cat_b or ?input is like ?cat_c
            ?result becomes "y" when ?input is like ?cat_c or ?input is like ?cat_d or ?input is like ?cat_e
        ''')
        # cat_c is in both - they can both fire
        assert "CONFLICT" in report or "COMPETES" in report

    def test_or_with_non_category_branch_is_conflict(self):
        """
        OR where one branch has no category condition.
        Rule 1: cat_a OR flag is true
        Rule 2: cat_b OR flag is true
        The flag branch can overlap even if categories don't = conflict.
        """
        report = analyze_dsl('''
            ?cat_a resembles "a"
            ?cat_b resembles "b"
            ?input meaning "input"
            ?flag can be true, false
            ?result can be "x", "y"
            ?result becomes "x" when ?input is like ?cat_a or ?flag is true
            ?result becomes "y" when ?input is like ?cat_b or ?flag is true
        ''')
        # Both rules can fire when flag is true
        assert "CONFLICT" in report or "COMPETES" in report

    def test_or_one_rule_vs_simple_no_overlap(self):
        """
        One rule with OR vs one simple rule, no overlap.
        Rule 1: cat_a OR cat_b
        Rule 2: cat_c (simple)
        No shared categories = mutually exclusive.
        """
        report = analyze_dsl('''
            ?cat_a resembles "a"
            ?cat_b resembles "b"
            ?cat_c resembles "c"
            ?input meaning "input"
            ?result can be "x", "y"
            ?result becomes "x" when ?input is like ?cat_a or ?input is like ?cat_b
            ?result becomes "y" when ?input is like ?cat_c
        ''')
        assert "COMPETES" not in report, f"Got:\n{report}"

    def test_or_one_rule_vs_simple_with_overlap(self):
        """
        One rule with OR vs one simple rule, with overlap.
        Rule 1: cat_a OR cat_b
        Rule 2: cat_b (simple) - matches one OR branch
        cat_b is shared = can both fire = conflict.
        """
        report = analyze_dsl('''
            ?cat_a resembles "a"
            ?cat_b resembles "b"
            ?input meaning "input"
            ?result can be "x", "y"
            ?result becomes "x" when ?input is like ?cat_a or ?input is like ?cat_b
            ?result becomes "y" when ?input is like ?cat_b
        ''')
        # cat_b is in both - they can both fire
        assert "CONFLICT" in report or "COMPETES" in report


class TestQuantifiedHasConditions:
    """Tests with has any/all/none that conditions."""

    def test_has_any_with_different_categories(self):
        """has any that is like with different categories = mutually exclusive."""
        report = analyze_dsl('''
            ?cat_a resembles "emergency"
            ?cat_b resembles "routine"
            ?patient meaning "the patient" with
                ?symptoms has a list of ?symptom meaning "symptoms"
            ?symptom meaning "a symptom" with
                ?description meaning "desc"
            ?urgency can be "high", "low"
            ?urgency becomes "high" when ?symptoms of ?patient has any that is like ?cat_a
            ?urgency becomes "low" when ?symptoms of ?patient has any that is like ?cat_b
        ''')
        assert "COMPETES" not in report, f"Got:\n{report}"

    def test_has_any_with_same_category_is_conflict(self):
        """has any that is like with same category = can both fire."""
        report = analyze_dsl('''
            ?cat_a resembles "symptom"
            ?patient meaning "the patient" with
                ?symptoms has a list of ?symptom meaning "symptoms"
            ?symptom meaning "a symptom" with
                ?description meaning "desc"
            ?result can be "x", "y"
            ?result becomes "x" when ?symptoms of ?patient has any that is like ?cat_a
            ?result becomes "y" when ?symptoms of ?patient has any that is like ?cat_a
        ''')
        assert "CONFLICT" in report or "COMPETES" in report

    def test_has_all_with_different_categories(self):
        """has all that is like with different categories."""
        report = analyze_dsl('''
            ?cat_a resembles "valid"
            ?cat_b resembles "invalid"
            ?container meaning "the container" with
                ?items has a list of ?item meaning "items"
            ?item meaning "an item" with
                ?status meaning "status"
            ?result can be "valid", "invalid"
            ?result becomes "valid" when ?items of ?container has all that is like ?cat_a
            ?result becomes "invalid" when ?items of ?container has all that is like ?cat_b
        ''')
        assert "COMPETES" not in report, f"Got:\n{report}"

    def test_has_none_with_different_categories(self):
        """has none that is like with different categories."""
        report = analyze_dsl('''
            ?cat_a resembles "blocked"
            ?cat_b resembles "allowed"
            ?user meaning "the user" with
                ?permissions has a list of ?permission meaning "perms"
            ?permission meaning "a permission" with
                ?type meaning "type"
            ?access can be "granted", "denied"
            ?access becomes "denied" when ?permissions of ?user has none that is like ?cat_a
            ?access becomes "granted" when ?permissions of ?user has none that is like ?cat_b
        ''')
        assert "COMPETES" not in report, f"Got:\n{report}"


class TestLengthConditions:
    """Tests with length of where conditions."""

    def test_length_where_different_categories(self):
        """length of where with different categories = mutually exclusive."""
        report = analyze_dsl('''
            ?cat_a resembles "critical"
            ?cat_b resembles "minor"
            ?patient meaning "the patient" with
                ?symptoms has a list of ?symptom meaning "symptoms"
            ?symptom meaning "a symptom" with
                ?severity meaning "severity"
            ?urgency can be "high", "low"
            ?urgency becomes "high" when (length of ?symptoms of ?patient where ?severity is like ?cat_a) > 0
            ?urgency becomes "low" when (length of ?symptoms of ?patient where ?severity is like ?cat_b) > 0
        ''')
        assert "COMPETES" not in report, f"Got:\n{report}"

    def test_length_where_same_category_is_conflict(self):
        """length of where with same category = can both fire."""
        report = analyze_dsl('''
            ?cat_a resembles "item"
            ?container meaning "the container" with
                ?items has a list of ?item meaning "items"
            ?item meaning "an item" with
                ?type meaning "type"
            ?result can be "few", "many"
            ?result becomes "few" when (length of ?items of ?container where ?type is like ?cat_a) > 0
            ?result becomes "many" when (length of ?items of ?container where ?type is like ?cat_a) > 5
        ''')
        # Same category, different thresholds - could both be true
        assert "CONFLICT" in report or "COMPETES" in report


class TestMixedConditions:
    """Tests with mixed condition types."""

    def test_is_like_and_has_any_different_categories(self):
        """Mix of is like and has any with different categories."""
        report = analyze_dsl('''
            ?cat_a resembles "urgent"
            ?cat_b resembles "routine"
            ?input meaning "input"
            ?patient meaning "the patient" with
                ?symptoms has a list of ?symptom meaning "symptoms"
            ?symptom meaning "a symptom" with
                ?desc meaning "desc"
            ?result can be "a", "b"
            ?result becomes "a" when ?input is like ?cat_a
            ?result becomes "b" when ?symptoms of ?patient has any that is like ?cat_b
        ''')
        # Different condition types, different categories
        # These are independent axes - could both be true!
        # This is actually NOT mutually exclusive
        assert "CONFLICT" in report or "COMPETES" in report

    def test_complex_and_or_mix(self):
        """Complex mix of AND and OR with different categories."""
        report = analyze_dsl('''
            ?cat_a resembles "a"
            ?cat_b resembles "b"
            ?cat_c resembles "c"
            ?cat_d resembles "d"
            ?input meaning "input"
            ?flag can be true, false
            ?result can be "x", "y"
            ?result becomes "x" when (?input is like ?cat_a or ?input is like ?cat_b) and ?flag is true
            ?result becomes "y" when (?input is like ?cat_c or ?input is like ?cat_d) and ?flag is true
        ''')
        # Different category sets in OR, same flag condition
        # Categories are mutually exclusive
        assert "COMPETES" not in report, f"Got:\n{report}"


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_single_rule_no_conflict(self):
        """Single rule should never show conflict."""
        report = analyze_dsl('''
            ?cat_a resembles "a"
            ?input meaning "input"
            ?result can be "x"
            ?result becomes "x" when ?input is like ?cat_a
        ''')
        assert "CONFLICT" not in report
        assert "COMPETES" not in report

    def test_same_value_different_conditions_no_conflict(self):
        """Multiple rules concluding same value = no conflict (they agree)."""
        report = analyze_dsl('''
            ?cat_a resembles "a"
            ?cat_b resembles "b"
            ?input meaning "input"
            ?result can be "same"
            ?result becomes "same" when ?input is like ?cat_a
            ?result becomes "same" when ?input is like ?cat_b
        ''')
        # Same conclusion - no conflict even if conditions differ
        assert "CONFLICT" not in report

    def test_defeasible_vs_strict_with_different_categories(self):
        """Defeasible and strict rules with different categories."""
        report = analyze_dsl('''
            ?cat_a resembles "a"
            ?cat_b resembles "b"
            ?input meaning "input"
            ?result can be "strict", "default"
            ?result becomes "strict" when ?input is like ?cat_a
            normally ?result becomes "default" when ?input is like ?cat_b
        ''')
        # Different categories = mutually exclusive
        assert "COMPETES" not in report, f"Got:\n{report}"
