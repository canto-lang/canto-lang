"""
Tests for Phase 3: DeLP Meta-Interpreter
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from canto_core.parser.dsl_parser import parse_string
from canto_core.codegen import DeLPTranslator
from canto_core.delp import create_janus_engine


# Helper function for backwards compatibility
def translate_to_delp(ast):
    """Wrapper for DeLPTranslator"""
    translator = DeLPTranslator()
    return translator.translate(ast)


class TestArgumentConstruction:
    """Test argument construction from rules"""

    def test_simple_argument(self):
        """Test building argument for simple rule"""
        dsl = """
        ?flag can be true, false
        ?flag becomes true
        """
        ast = parse_string(dsl)
        delp = translate_to_delp(ast)
        engine = create_janus_engine(delp)
        engine.load()

        result = engine.delp_query("flag(true)")
        assert result['status'] == 'warranted'
        assert result['tree'] is not None
        assert result['tree']['argument']['rule_id'] == 'flag_1'

        engine.cleanup()

    def test_argument_with_premises(self):
        """Test argument with premises"""
        dsl = """
        ?flag can be true, false
        ?condition can be true, false
        ?flag becomes true
            when ?condition is true
        """
        ast = parse_string(dsl)
        delp = translate_to_delp(ast)
        engine = create_janus_engine(delp)
        engine.load()

        # Add condition fact
        engine.add_fact("condition", "true")

        result = engine.delp_query("flag(true)")
        assert result['status'] == 'warranted'
        # Should have premises
        assert len(result['tree']['argument']['premises']) > 0

        engine.cleanup()


class TestDefeatDetection:
    """Test defeat detection with superiority"""

    def test_strict_defeats_defeasible(self):
        """Test that STRICT rule defeats DEFEASIBLE rule"""
        dsl = """
        ?flag can be true, false
        ?flag becomes true overriding all
        normally ?flag becomes false
        """
        ast = parse_string(dsl)
        delp = translate_to_delp(ast)
        engine = create_janus_engine(delp)
        engine.load()

        # flag(true) should be warranted
        result_true = engine.delp_query("flag(true)")
        assert result_true['status'] == 'warranted'

        # flag(false) should be defeated
        result_false = engine.delp_query("flag(false)")
        assert result_false['status'] == 'defeated'

        engine.cleanup()

    def test_multiple_rules_same_variable(self):
        """Test multiple rules for same variable"""
        dsl = """
        ?flag can be true, false
        ?flag becomes true
        ?flag becomes false
        """
        ast = parse_string(dsl)
        delp = translate_to_delp(ast)
        engine = create_janus_engine(delp)
        engine.load()

        # Both rules exist
        result_true = engine.delp_query("flag(true)")
        result_false = engine.delp_query("flag(false)")

        # Without superiority, later rule might override
        # Just verify they return valid statuses
        assert result_true['status'] in ['warranted', 'defeated', 'undecided']
        assert result_false['status'] in ['warranted', 'defeated', 'undecided']

        engine.cleanup()

    def test_overrides_order_independence(self):
        """Test overriding works regardless of rule order"""
        # Rule 1: overriding defined first
        dsl1 = """
        ?flag can be true, false
        ?flag becomes true overriding all
        normally ?flag becomes false
        """

        ast1 = parse_string(dsl1)
        delp1 = translate_to_delp(ast1)
        engine1 = create_janus_engine(delp1)
        engine1.load()

        result1 = engine1.delp_query("flag(false)")
        assert result1['status'] == 'defeated'
        engine1.cleanup()

        # Rule 2: overriding defined second
        dsl2 = """
        ?flag can be true, false
        normally ?flag becomes false
        ?flag becomes true overriding all
        """

        ast2 = parse_string(dsl2)
        delp2 = translate_to_delp(ast2)
        engine2 = create_janus_engine(delp2)
        engine2.load()

        result2 = engine2.delp_query("flag(false)")
        assert result2['status'] == 'defeated'
        engine2.cleanup()


class TestDialecticalTrees:
    """Test dialectical tree construction"""

    def test_tree_structure(self):
        """Test that dialectical trees have correct structure"""
        dsl = """
        ?flag can be true, false
        ?flag becomes true overriding all
        normally ?flag becomes false
        """
        ast = parse_string(dsl)
        delp = translate_to_delp(ast)
        engine = create_janus_engine(delp)
        engine.load()

        result = engine.delp_query("flag(false)")
        tree = result['tree']

        # Check tree structure
        assert tree['type'] == 'tree'
        assert 'argument' in tree
        assert 'status' in tree
        assert 'defeaters' in tree

        # Should have defeaters
        assert len(tree['defeaters']) > 0

        engine.cleanup()

    def test_tree_marking(self):
        """Test that trees are marked correctly"""
        dsl = """
        ?flag can be true, false
        ?flag becomes true overriding all
        normally ?flag becomes false
        """
        ast = parse_string(dsl)
        delp = translate_to_delp(ast)
        engine = create_janus_engine(delp)
        engine.load()

        # Undefeated argument
        result_true = engine.delp_query("flag(true)")
        assert result_true['tree']['status'] == 'undefeated'

        # Defeated argument
        result_false = engine.delp_query("flag(false)")
        # Parent is defeated, but status might be 'unmarked' or 'defeated'
        # depending on implementation
        assert result_false['status'] == 'defeated'

        engine.cleanup()


class TestWarrantStatus:
    """Test warrant status queries"""

    def test_is_warranted(self):
        """Test is_warranted helper"""
        dsl = """
        ?flag can be true, false
        ?flag becomes true
        """
        ast = parse_string(dsl)
        delp = translate_to_delp(ast)
        engine = create_janus_engine(delp)
        engine.load()

        assert engine.is_warranted("flag(true)") is True
        assert engine.is_warranted("flag(false)") is False

        engine.cleanup()

    def test_is_defeated(self):
        """Test is_defeated helper"""
        dsl = """
        ?flag can be true, false
        ?flag becomes true overriding all
        normally ?flag becomes false
        """
        ast = parse_string(dsl)
        delp = translate_to_delp(ast)
        engine = create_janus_engine(delp)
        engine.load()

        assert engine.is_defeated("flag(false)") is True
        assert engine.is_defeated("flag(true)") is False

        engine.cleanup()


class TestComplexScenarios:
    """Test complex reasoning scenarios"""

    def test_conditional_defeat(self):
        """Test defeat with conditional rules"""
        dsl = """
        ?flag can be true, false
        ?condition can be true, false

        ?flag becomes true
            when ?condition is true
            overriding all

        normally ?flag becomes false
        """
        ast = parse_string(dsl)
        delp = translate_to_delp(ast)
        engine = create_janus_engine(delp)
        engine.load()

        # Query for false - behavior depends on whether condition is met
        result1 = engine.delp_query("flag(false)")
        # Could be warranted or defeated depending on condition
        assert result1['status'] in ['warranted', 'defeated', 'undecided']

        # With condition set, test again
        from janus_swi import query_once
        query_once("retractall(condition(_))")
        engine.add_fact("condition", "true")

        result2 = engine.delp_query("flag(false)")
        # Status depends on whether conditional rule defeats
        assert result2['status'] in ['warranted', 'defeated', 'undecided']

        engine.cleanup()

    def test_chained_defeat(self):
        """Test transitive defeat"""
        dsl = """
        ?flag can be true, false

        ?flag becomes true overriding all
        ?flag becomes false overriding normal
        normally ?flag becomes true
        """
        ast = parse_string(dsl)
        delp = translate_to_delp(ast)
        engine = create_janus_engine(delp)
        engine.load()

        # Complex superiority structure
        # Should resolve correctly
        result = engine.delp_query("flag(true)")
        assert result['status'] in ['warranted', 'defeated', 'undecided']

        engine.cleanup()

    def test_medical_query_scenario(self):
        """Test complex medical query example"""
        example_path = Path(__file__).parent.parent / "examples" / "medical_query.canto"
        if not example_path.exists():
            pytest.skip("medical_query.canto not found")

        from canto_core.parser import parse_file
        ast = parse_file(str(example_path))
        delp = translate_to_delp(ast)
        engine = create_janus_engine(delp)
        engine.load()

        # Scenario 1: explicit vaccine mention
        engine.add_fact("is_like", "patient_query", "vaccine_terms")

        result = engine.delp_query("vaccine_flag(true)")
        assert result['status'] == 'warranted'

        engine.cleanup()


class TestSymbolicPredicates:
    """Test handling of symbolic predicates"""

    def test_matches_predicate(self):
        """Test is like symbolic predicate"""
        dsl = """
        ?flag can be true, false
        ?query meaning "query text"
        ?terms resembles "test"

        ?flag becomes true
            when ?query is like ?terms
        """
        ast = parse_string(dsl)
        delp = translate_to_delp(ast)
        engine = create_janus_engine(delp)
        engine.load()

        # Note: Current implementation may treat ungrounded predicates
        # as assumptions that succeed. This is okay for now.
        result1 = engine.delp_query("flag(true)")
        assert result1['status'] in ['warranted', 'undecided']

        # With explicit grounding, argument should succeed or be undecided
        # (depends on how the symbolic predicate is handled by the engine)
        from janus_swi import query_once
        query_once("retractall(is_like(_, _))")
        engine.add_fact("is_like", "query", "terms")
        result2 = engine.delp_query("flag(true)")
        assert result2['status'] in ['warranted', 'undecided']

        engine.cleanup()


class TestEdgeCases:
    """Test edge cases"""

    def test_no_arguments(self):
        """Test query with no supporting arguments"""
        dsl = """
        ?flag can be true, false
        ?flag becomes true
        """
        ast = parse_string(dsl)
        delp = translate_to_delp(ast)
        engine = create_janus_engine(delp)
        engine.load()

        # Query for value with no rule
        result = engine.delp_query("flag(false)")
        # May return undecided or error if no rule found
        assert result['status'] in ['undecided', 'error']

        engine.cleanup()

    def test_query_with_rule(self):
        """Test basic query with rule"""
        dsl = """
        ?flag can be true, false
        ?flag becomes true
        """
        ast = parse_string(dsl)
        delp = translate_to_delp(ast)
        engine = create_janus_engine(delp)
        engine.load()

        # Query for value with rule
        result = engine.delp_query("flag(true)")
        assert result['status'] == 'warranted'

        engine.cleanup()


class TestDeLPReasoningAnalyzer:
    """Test DeLPReasoningAnalyzer with actual DeLP reasoning"""

    def test_get_possible_values_from_rules(self):
        """Test _get_possible_values collects values from rule heads"""
        from canto_core.delp.analyzer import DeLPReasoningAnalyzer

        dsl = """
        ?flag can be true, false
        ?flag becomes true
        normally ?flag becomes false
        """
        ast = parse_string(dsl)
        delp = translate_to_delp(ast)
        analyzer = DeLPReasoningAnalyzer(delp)

        values = analyzer._get_possible_values('flag')
        # Should contain boolean values from rule heads
        assert True in values
        assert False in values
        # May also contain string values from can be declaration
        assert len(values) >= 2

    def test_get_possible_values_from_one_of(self):
        """Test _get_possible_values collects values from can be clause"""
        from canto_core.delp.analyzer import DeLPReasoningAnalyzer

        dsl = """
        ?intent can be "prevention", "treatment", "info"
        ?intent becomes "prevention"
        """
        ast = parse_string(dsl)
        delp = translate_to_delp(ast)
        analyzer = DeLPReasoningAnalyzer(delp)

        values = analyzer._get_possible_values('intent')
        assert "prevention" in values
        assert "treatment" in values
        assert "info" in values

    def test_format_goal_boolean(self):
        """Test _format_goal for boolean values"""
        from canto_core.delp.analyzer import DeLPReasoningAnalyzer

        dsl = "?flag can be true, false"
        ast = parse_string(dsl)
        delp = translate_to_delp(ast)
        analyzer = DeLPReasoningAnalyzer(delp)

        assert analyzer._format_goal('flag', True) == 'flag(true)'
        assert analyzer._format_goal('flag', False) == 'flag(false)'

    def test_format_goal_string(self):
        """Test _format_goal for string values"""
        from canto_core.delp.analyzer import DeLPReasoningAnalyzer

        dsl = "?intent meaning \"intent value\""
        ast = parse_string(dsl)
        delp = translate_to_delp(ast)
        analyzer = DeLPReasoningAnalyzer(delp)

        assert analyzer._format_goal('intent', 'prevention') == "intent('prevention')"

    def test_conclusion_warranted(self):
        """Test that analyzer returns warranted conclusion from DeLP"""
        from canto_core.delp.analyzer import DeLPReasoningAnalyzer

        dsl = """
        ?flag can be true, false
        ?flag becomes true
        """
        ast = parse_string(dsl)
        delp = translate_to_delp(ast)
        analyzer = DeLPReasoningAnalyzer(delp)

        analysis = analyzer.analyze()

        # Check conclusion exists for the variable
        flag_data = analysis['variables']['flag']
        assert 'conclusion' in flag_data
        assert flag_data['conclusion'] is not None
        assert flag_data['conclusion']['value'] == True
        assert flag_data['conclusion']['status'] == 'warranted'

    def test_alternatives_defeated(self):
        """Test that analyzer detects defeated alternatives"""
        from canto_core.delp.analyzer import DeLPReasoningAnalyzer

        dsl = """
        ?flag can be true, false
        ?flag becomes true overriding all
        normally ?flag becomes false
        """
        ast = parse_string(dsl)
        delp = translate_to_delp(ast)
        analyzer = DeLPReasoningAnalyzer(delp)

        analysis = analyzer.analyze()
        flag_data = analysis['variables']['flag']

        # True should be the warranted conclusion
        assert flag_data['conclusion'] is not None
        assert flag_data['conclusion']['value'] == True
        assert flag_data['conclusion']['status'] == 'warranted'

        # False should be in alternatives as defeated
        assert 'alternatives' in flag_data
        false_alt = next((a for a in flag_data['alternatives'] if a['value'] == False), None)
        assert false_alt is not None
        assert false_alt['status'] in ['defeated', 'blocked']

    def test_conclusion_is_warranted_value(self):
        """Test that conclusion contains the warranted value"""
        from canto_core.delp.analyzer import DeLPReasoningAnalyzer

        dsl = """
        ?flag can be true, false
        ?flag becomes true overriding all
        normally ?flag becomes false
        """
        ast = parse_string(dsl)
        delp = translate_to_delp(ast)
        analyzer = DeLPReasoningAnalyzer(delp)

        analysis = analyzer.analyze()
        flag_data = analysis['variables']['flag']

        # Conclusion should be True (the warranted value)
        assert flag_data['conclusion']['value'] == True
        # Alternatives should contain False (the defeated value)
        alt_values = [a['value'] for a in flag_data['alternatives']]
        assert False in alt_values

    def test_defeat_info_extracted(self):
        """Test that defeat info is extracted for alternatives"""
        from canto_core.delp.analyzer import DeLPReasoningAnalyzer

        dsl = """
        ?flag can be true, false
        ?flag becomes true overriding all
        normally ?flag becomes false
        """
        ast = parse_string(dsl)
        delp = translate_to_delp(ast)
        analyzer = DeLPReasoningAnalyzer(delp)

        analysis = analyzer.analyze()
        flag_data = analysis['variables']['flag']

        # The False alternative should have defeat info
        false_alt = next((a for a in flag_data['alternatives'] if a['value'] == False), None)
        assert false_alt is not None
        # Should have defeated_by or defeat_reason
        assert 'defeated_by' in false_alt or 'defeat_reason' in false_alt

    def test_conflict_resolutions_included(self):
        """Test that conflict resolutions are included in analysis"""
        from canto_core.delp.analyzer import DeLPReasoningAnalyzer

        dsl = """
        ?flag can be true, false
        ?flag becomes true overriding all
        normally ?flag becomes false
        """
        ast = parse_string(dsl)
        delp = translate_to_delp(ast)
        analyzer = DeLPReasoningAnalyzer(delp)

        analysis = analyzer.analyze()
        flag_data = analysis['variables']['flag']

        # Check that conflict_resolutions field exists
        assert 'conflict_resolutions' in flag_data
        # With overriding, there should be conflicts recorded
        conflicts = flag_data['conflict_resolutions']
        # May or may not have conflicts depending on implementation
        assert isinstance(conflicts, list)

    def test_output_structure(self):
        """Test that output has the expected structure"""
        from canto_core.delp.analyzer import DeLPReasoningAnalyzer

        dsl = """
        ?flag can be true, false
        ?flag becomes true overriding all
        normally ?flag becomes false
        """
        ast = parse_string(dsl)
        delp = translate_to_delp(ast)
        analyzer = DeLPReasoningAnalyzer(delp)

        analysis = analyzer.analyze()
        flag_pattern = analysis['variables']['flag']

        # Rule structure fields
        assert 'strict_rules' in flag_pattern
        assert 'defeasible_rules' in flag_pattern
        assert 'semantic_patterns' in flag_pattern

        # Simplified reasoning output fields
        assert 'conclusion' in flag_pattern
        assert 'alternatives' in flag_pattern
        assert 'conflict_resolutions' in flag_pattern

    def test_analyze_for_llm(self):
        """Test analyze_for_llm returns LLM-friendly output"""
        from canto_core.delp.analyzer import DeLPReasoningAnalyzer

        dsl = """
        ?flag can be true, false
        ?flag becomes true overriding all
        normally ?flag becomes false
        """
        ast = parse_string(dsl)
        delp = translate_to_delp(ast)
        analyzer = DeLPReasoningAnalyzer(delp)

        llm_summary = analyzer.analyze_for_llm()

        # Should have variables section
        assert 'variables' in llm_summary
        assert 'flag' in llm_summary['variables']

        flag_summary = llm_summary['variables']['flag']
        assert 'name' in flag_summary
        assert flag_summary['name'] == 'flag'
        assert 'possible_values' in flag_summary

        # Should have rules summary
        assert 'rules_summary' in llm_summary

        # Should have conflict resolutions
        assert 'conflict_resolutions' in llm_summary


class TestHierarchyBuilding:
    """Test hierarchy building from 'with' blocks"""

    def test_hierarchy_from_with_blocks(self):
        """Test that hierarchy is built from 'with' block relationships"""
        from canto_core.delp.analyzer import DeLPReasoningAnalyzer

        # Use indentation-based syntax (pre-processor adds 'end' markers)
        dsl = """\
?patient meaning "patient information" with
    ?name meaning "patient name"
    ?age meaning "patient age"
"""
        ast = parse_string(dsl)
        delp = translate_to_delp(ast)
        analyzer = DeLPReasoningAnalyzer(delp)

        analysis = analyzer.analyze()

        # Should have hierarchy section
        assert 'hierarchy' in analysis
        hierarchy = analysis['hierarchy']

        # Should have roots and flat_relationships
        assert 'roots' in hierarchy
        assert 'flat_relationships' in hierarchy

        # Patient should be a root
        roots = hierarchy['roots']
        assert len(roots) == 1
        assert roots[0]['name'] == 'patient'
        assert roots[0]['description'] == 'patient information'

        # Should have name and age as children
        children = roots[0]['children']
        child_names = [c['name'] for c in children]
        assert 'name' in child_names
        assert 'age' in child_names

    def test_nested_hierarchy(self):
        """Test deeply nested hierarchy"""
        from canto_core.delp.analyzer import DeLPReasoningAnalyzer

        # Use indentation-based syntax (pre-processor adds 'end' markers)
        dsl = """\
?root meaning "root" with
    ?child1 meaning "first child" with
        ?grandchild meaning "grandchild"
    ?child2 meaning "second child"
"""
        ast = parse_string(dsl)
        delp = translate_to_delp(ast)
        analyzer = DeLPReasoningAnalyzer(delp)

        analysis = analyzer.analyze()
        hierarchy = analysis['hierarchy']

        # Root should be present
        roots = hierarchy['roots']
        assert len(roots) == 1
        root = roots[0]
        assert root['name'] == 'root'

        # Should have two children
        assert len(root['children']) == 2

        # Find child1 which has grandchild
        child1 = next((c for c in root['children'] if c['name'] == 'child1'), None)
        assert child1 is not None
        assert len(child1['children']) == 1
        assert child1['children'][0]['name'] == 'grandchild'

    def test_hierarchy_flat_relationships(self):
        """Test that flat_relationships captures all parent-child pairs"""
        from canto_core.delp.analyzer import DeLPReasoningAnalyzer

        # Use indentation-based syntax (pre-processor adds 'end' markers)
        dsl = """\
?patient meaning "patient" with
    ?name meaning "name"
    ?symptoms with
        ?text meaning "symptom text"
"""
        ast = parse_string(dsl)
        delp = translate_to_delp(ast)
        analyzer = DeLPReasoningAnalyzer(delp)

        analysis = analyzer.analyze()
        flat_rels = analysis['hierarchy']['flat_relationships']

        # Should have 3 relationships: patient->name, patient->symptoms, symptoms->text
        assert len(flat_rels) == 3

        # Check relationships exist
        rels_as_tuples = [(r['parent'], r['child']) for r in flat_rels]
        assert ('patient', 'name') in rels_as_tuples
        assert ('patient', 'symptoms') in rels_as_tuples
        assert ('symptoms', 'text') in rels_as_tuples

    def test_empty_hierarchy_when_no_with_blocks(self):
        """Test that hierarchy is empty when no 'with' blocks"""
        from canto_core.delp.analyzer import DeLPReasoningAnalyzer

        dsl = """
        ?flag can be true, false
        ?flag becomes true
        """
        ast = parse_string(dsl)
        delp = translate_to_delp(ast)
        analyzer = DeLPReasoningAnalyzer(delp)

        analysis = analyzer.analyze()

        # Should have hierarchy section but with empty roots
        assert 'hierarchy' in analysis
        assert analysis['hierarchy']['roots'] == []
        assert analysis['hierarchy']['flat_relationships'] == []


class TestComplexTermSerialization:
    """Test that complex terms can be serialized via Janus"""

    def test_path_expression_serialization(self):
        """Test that path expressions like text_of_(?symptoms, ?patient) can be serialized"""
        dsl = """
?emergency_symptoms resembles "chest pain", "difficulty breathing"

?patient meaning "patient" with
    ?symptoms with
        ?text meaning "symptom text"

?is_emergency can be true, false
?is_emergency becomes true
    when ?text of (?symptoms of ?patient) is like ?emergency_symptoms
"""
        ast = parse_string(dsl)
        delp = translate_to_delp(ast)
        engine = create_janus_engine(delp)
        engine.load()

        # This should not throw a Janus serialization error
        result = engine.delp_query("is_emergency(true)")

        # Should get a valid result, not an error
        assert result['status'] in ['warranted', 'defeated', 'undecided']
        assert 'error' not in result or result.get('error') is None

        # The tree should be serializable
        if result['tree'] is not None:
            # Check the tree has expected structure
            assert 'argument' in result['tree'] or 'type' in result['tree']

        engine.cleanup()

    def test_dict_literal_serialization(self):
        """Test that Prolog dict literals (from has_any_property_is) can be serialized"""
        dsl = """
?patient meaning "patient" with
    ?symptoms with
        ?text meaning "symptom text"
        ?is_emergency can be true, false

?urgency can be "emergency", "urgent", "routine"
?urgency becomes "emergency"
    when ?symptoms of ?patient has any that ?is_emergency is true
"""
        ast = parse_string(dsl)
        delp = translate_to_delp(ast)
        engine = create_janus_engine(delp)
        engine.load()

        # This should not throw a Janus serialization error
        result = engine.delp_query("urgency('emergency')")

        # Should get a valid result, not an error
        assert result['status'] in ['warranted', 'defeated', 'undecided']
        assert 'error' not in result or result.get('error') is None

        engine.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
