"""
Janus-based DeLP engine using SWI-Prolog
"""

from typing import List, Dict, Any, Optional
from pathlib import Path

from .program import DeLPProgram
from .models import DeLPQueryResult
from .prolog_engine import PrologEngine
from .tracer import DeLPTracer


class JanusDeLP:
    """
    DeLP engine using Janus (SWI-Prolog bridge)
    """

    def __init__(self, delp_program: DeLPProgram):
        self.program = delp_program
        self._loaded = False
        self._prolog = PrologEngine()
        self._tracer = DeLPTracer(self._prolog)

        # Load DeLP meta-interpreter and support files
        delp_dir = Path(__file__).parent
        self._prolog.consult_file(delp_dir / "delp_meta.pl", "DeLP meta-interpreter")
        self._prolog.consult_file(delp_dir / "dev_tools.pl", "DeLP dev tools")
        self._prolog.consult_file(delp_dir / "static_analyzer.pl", "DeLP static analyzer")

    def load(self, force_reload=False):
        """Load the DeLP program into SWI-Prolog"""
        if self._loaded and not force_reload:
            return

        # Abolish existing predicates if reloading
        if self._loaded:
            for var_name in self.program.declarations.keys():
                self._prolog.abolish(var_name, 1)
            for pred, arity in [("rule_info", 4), ("sup", 2), ("pattern", 2),
                                ("matches", 2), ("has", 2), ("like", 2)]:
                self._prolog.abolish(pred, arity)

        # Load program
        prolog_code = self.program.to_prolog_string()
        self._prolog.consult_string(prolog_code, "DeLP program")
        self._loaded = True

    def load_prolog_string(self, prolog_code: str):
        """
        Load raw Prolog code into the engine.
        Useful for adding LLM-generated predicates or auxiliary rules.
        """
        self._prolog.consult_string(prolog_code, "extra Prolog code")

    def query_variable(self, variable: str, value: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Query a variable in the DeLP program

        Args:
            variable: Variable name (without $)
            value: Optional specific value to check

        Returns:
            List of solutions with variable bindings
        """
        if not self._loaded:
            self.load()

        if value is not None:
            query_str = f"{variable}({value})"
        else:
            query_str = f"{variable}(X)"

        try:
            return self._prolog.query_all(query_str)
        except Exception as e:
            print(f"Query failed: {e}")
            return []

    def check_fact(self, predicate: str) -> bool:
        """
        Check if a fact/predicate is true

        Args:
            predicate: Prolog predicate (e.g., "vaccine_flag(true)")

        Returns:
            True if provable, False otherwise
        """
        if not self._loaded:
            self.load()

        try:
            result = self._prolog.query_once(predicate)
            return result is not None and result.get('truth', False)
        except:
            return False

    def add_fact(self, predicate: str, *args):
        """
        Dynamically add a fact to the knowledge base

        Args:
            predicate: Predicate name (e.g., "matches")
            args: Arguments to the predicate
        """
        if not self._loaded:
            self.load()

        self._prolog.assert_fact(predicate, *args)

    def set_match_predicate(self, var1: str, var2: str, result: bool = True):
        """
        Set a matches/2 predicate result

        Args:
            var1: First variable
            var2: Second variable (pattern)
            result: True or False
        """
        if result:
            self.add_fact("matches", var1, var2)
        else:
            # For now, we don't add negative facts
            # In full DeLP, we'd need to handle this differently
            print(f"Note: Negative facts not yet implemented")

    def explain(self, goal: str) -> Dict[str, Any]:
        """
        Get explanation/proof tree for a goal

        Args:
            goal: Goal to prove (e.g., "vaccine_flag(true)")

        Returns:
            Dictionary with proof information
        """
        if not self._loaded:
            self.load()

        results = self._prolog.query_all(goal)
        return {
            'goal': goal,
            'provable': len(results) > 0,
            'solutions': results
        }

    def delp_query(self, goal: str) -> Dict[str, Any]:
        """
        Query with full DeLP semantics using meta-interpreter

        Args:
            goal: Goal to query (e.g., "vaccine_flag(true)")

        Returns:
            Dictionary with:
                - status: 'warranted' | 'defeated' | 'undecided'
                - tree: Dialectical tree structure (as dict)
                - goal: The queried goal
        """
        if not self._loaded:
            self.load()

        try:
            query_str = f"delp_query_dict({goal}, Status, TreeDict)"
            result = self._prolog.query_once(query_str, normalize=False)

            # Use Pydantic model for normalization (converts 'true'/'false' to booleans)
            query_result = DeLPQueryResult.from_prolog(goal, result)
            return query_result.model_dump()

        except Exception as e:
            print(f"DeLP query failed: {e}")
            return {
                'goal': goal,
                'status': 'error',
                'tree': None,
                'error': str(e)
            }

    def is_warranted(self, goal: str) -> bool:
        """
        Check if goal is warranted (has undefeated argument)

        Args:
            goal: Goal to check (e.g., "vaccine_flag(true)")

        Returns:
            True if warranted, False otherwise
        """
        if not self._loaded:
            self.load()

        try:
            result = self._prolog.query_once(f"warranted({goal})")
            return result is not None and result.get('truth', False)
        except:
            return False

    def is_defeated(self, goal: str) -> bool:
        """
        Check if goal is defeated (all arguments defeated)

        Args:
            goal: Goal to check (e.g., "vaccine_flag(true)")

        Returns:
            True if defeated, False otherwise
        """
        if not self._loaded:
            self.load()

        try:
            result = self._prolog.query_once(f"defeated({goal})")
            return result is not None and result.get('truth', False)
        except:
            return False

    def get_dialectical_tree(self, goal: str) -> Optional[Dict[str, Any]]:
        """
        Get full dialectical tree for a goal

        Args:
            goal: Goal to analyze

        Returns:
            Dialectical tree structure or None
        """
        result = self.delp_query(goal)
        return result.get('tree')

    def extract_assumptions(self, tree: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract all assumptions from a dialectical tree.

        Assumptions are predicates that couldn't be derived or proven as facts,
        so they were assumed to be true. This can indicate:
        - Typos in predicate names
        - Undefined semantic categories
        - Missing fact definitions

        Args:
            tree: Dialectical tree from delp_query

        Returns:
            List of assumption dicts with 'functor' and 'args' keys
        """
        assumptions = []
        self._collect_assumptions(tree, assumptions)
        # Remove duplicates while preserving order
        seen = set()
        unique = []
        for a in assumptions:
            key = (a.get('functor'), tuple(a.get('args', [])))
            if key not in seen:
                seen.add(key)
                unique.append(a)
        return unique

    def _collect_assumptions(self, node: Optional[Dict[str, Any]], assumptions: List[Dict[str, Any]]):
        """Recursively collect assumptions from tree nodes."""
        if not node or not isinstance(node, dict):
            return

        # Check if this node is a 'no_arguments' leaf
        if node.get('type') == 'no_arguments':
            return

        # Check the argument's premises for assumptions
        argument = node.get('argument', {})
        for premise in argument.get('premises', []):
            if isinstance(premise, dict):
                if premise.get('type') == 'assumption':
                    assumptions.append({
                        'functor': premise.get('functor'),
                        'args': premise.get('args', [])
                    })
                # Recursively check nested arguments
                elif premise.get('type') == 'tree' or 'premises' in premise:
                    self._collect_assumptions(premise, assumptions)

        # Check defeater subtrees
        for defeater in node.get('defeaters', []):
            self._collect_assumptions(defeater, assumptions)

    def pretty_print_tree(self, goal: str):
        """
        Pretty print dialectical tree

        Args:
            goal: Goal to visualize
        """
        if not self._loaded:
            self.load()

        try:
            result = self.delp_query(goal)
            tree = result.get('tree')

            if tree:
                self._print_tree_dict(tree, 0)
            else:
                print(f"No tree available for {goal}")
        except Exception as e:
            print(f"Failed to print tree: {e}")

    def _print_tree_dict(self, tree: Dict[str, Any], indent: int):
        """
        Print tree dict structure (Python-side implementation)

        Args:
            tree: Tree dict from delp_query
            indent: Current indentation level
        """
        prefix = "  " * indent

        if tree.get('type') == 'no_arguments':
            print(f"{prefix}No arguments for {tree.get('goal')}")
            return

        arg = tree.get('argument', {})
        status = tree.get('status', 'unknown')
        goal = arg.get('goal', '?')
        goal_args = arg.get('goal_args', [])
        rule_id = arg.get('rule_id', '?')
        specificity = arg.get('specificity', 0)

        goal_str = f"{goal}({', '.join(str(a) for a in goal_args)})" if goal_args else goal
        print(f"{prefix}{goal_str} [{status}] via {rule_id} (spec={specificity})")

        for defeater in tree.get('defeaters', []):
            self._print_tree_dict(defeater, indent + 1)

    def validate_program(self) -> Dict[str, Any]:
        """
        Run static analysis validation on the loaded DeLP program

        Returns:
            Dictionary with:
                - valid: bool - True if program is valid
                - errors: list - List of validation error dicts
        """
        if not self._loaded:
            self.load()

        try:
            result = self._prolog.query_once("validate_delp_program(Errors)")

            if result and 'Errors' in result:
                errors_raw = result['Errors']
                errors = self._parse_validation_errors(errors_raw)

                return {
                    'valid': len(errors) == 0,
                    'errors': errors
                }
            else:
                return {
                    'valid': True,
                    'errors': []
                }
        except Exception as e:
            return {
                'valid': False,
                'errors': [{'type': 'validation_error', 'details': str(e)}]
            }

    def get_dsl_analysis(self) -> Dict[str, Any]:
        """
        Get comprehensive DSL analysis from Prolog.

        Returns a structured report with:
        - conflicts: Variables with competing rules (different values)
        - no_conflicts: Variables with single rules or same-value rules
        - issues: Potential problems (always-defeated rules, etc.)

        Each conflict includes detailed rule status information
        (which rules win, which are defeated, and why).

        Conflict analysis uses mutual exclusivity detection from delp_meta.pl
        to avoid false positives when rules have mutually exclusive conditions
        (e.g., different semantic categories).
        """
        if not self._loaded:
            self.load()

        # First run validation to populate validation_error facts
        self.validate_program()

        try:
            # Get conflict analysis from delp_meta.pl (uses mutual exclusivity detection)
            result = self._prolog.query_once("get_variable_conflicts(Report)")
            if result and 'Report' in result:
                report = result['Report']
            else:
                report = {'conflicts': [], 'no_conflicts': []}

            # Get issues from static_analyzer.pl (validation errors)
            issues = self._get_static_analysis_issues()
            report['issues'] = issues

            return report
        except Exception as e:
            print(f"Failed to get DSL analysis: {e}")
            return {'conflicts': [], 'no_conflicts': [], 'issues': [], 'error': str(e)}

    def _get_static_analysis_issues(self) -> List[Dict[str, Any]]:
        """
        Get issues from static analyzer validation errors.

        Issues include:
        - Unreachable defeasible rules
        - Circular superiority
        - Contradictory strict rules without resolution
        """
        try:
            result = self._prolog.query_once("findall(Issue, format_issue_safe(Issue), Issues)")
            if result and 'Issues' in result:
                return result['Issues']
            return []
        except Exception as e:
            print(f"Failed to get static analysis issues: {e}")
            return []

    def validate_or_fail(self):
        """
        Validate program and raise exception if invalid

        Raises:
            ValueError: If validation fails with error details
        """
        result = self.validate_program()

        if not result['valid']:
            error_msg = "DeLP program validation failed:\n"
            for error in result['errors']:
                error_msg += f"  [{error['type']}] {error['details']}\n"
            raise ValueError(error_msg)

        print("✓ DeLP program validation passed")

    def clear_assumption_warnings(self):
        """
        Clear all assumption warnings.
        Call this before running queries to get fresh warnings.
        """
        try:
            self._prolog.query_once("clear_assumption_warnings")
        except Exception as e:
            print(f"Warning: Failed to clear assumption warnings: {e}")

    def get_assumption_warnings(self) -> List[Dict[str, Any]]:
        """
        Get all assumption warnings from the last analysis.

        Warnings indicate potential problems like:
        - Unknown semantic categories (typos in MATCHES clauses)
        - Unknown predicates

        Returns:
            List of warning dicts with keys:
                - type: 'unknown_category' or 'unknown_predicate'
                - predicate: The problematic predicate string
                - suggestion: Suggested correction or 'none'
        """
        try:
            result = self._prolog.query_once("get_assumption_warnings(Warnings)")
            if result and 'Warnings' in result:
                return result['Warnings']
            return []
        except Exception as e:
            print(f"Warning: Failed to get assumption warnings: {e}")
            return []

    def print_assumption_warnings(self):
        """
        Print assumption warnings in a human-readable format.
        """
        warnings = self.get_assumption_warnings()
        if not warnings:
            print("\n✓ No assumption warnings")
            return

        print("\n" + "=" * 50)
        print(" ASSUMPTION WARNINGS")
        print("=" * 50)
        for w in warnings:
            w_type = w.get('type', 'unknown')
            pred = w.get('predicate', '?')
            suggestion = w.get('suggestion', 'none')

            if w_type == 'unknown_category':
                # Extract category from matches(X, category)
                print(f"  ✗ Unknown category in: {pred}")
            else:
                print(f"  ✗ Unknown predicate: {pred}")

            if suggestion and suggestion != 'none':
                print(f"    → Did you mean: {suggestion}?")

    # =========================================================================
    # Trace Mode (delegated to DeLPTracer)
    # =========================================================================

    def enable_trace(self):
        """Enable trace mode. Trace events will be recorded during reasoning."""
        self._tracer.enable()

    def disable_trace(self):
        """Disable trace mode."""
        self._tracer.disable()

    def clear_trace(self):
        """Clear all recorded trace events."""
        self._tracer.clear()

    def get_trace(self) -> List[Dict[str, Any]]:
        """Get all trace events from the last query."""
        return self._tracer.get_events()

    def print_trace(self):
        """Print the reasoning trace in a human-readable format."""
        self._tracer.print_events()

    def delp_query_with_trace(self, goal: str) -> Dict[str, Any]:
        """
        Run a DeLP query with tracing enabled.

        Args:
            goal: Goal to query (e.g., "vaccine_flag(true)")

        Returns:
            Dictionary with query result plus trace events
        """
        self._tracer.enable()
        result = self.delp_query(goal)
        result['trace'] = self._tracer.get_events()
        self._tracer.disable()
        return result

    def _parse_validation_errors(self, errors_raw: List) -> List[Dict[str, Any]]:
        """
        Parse Prolog error dicts into Python dicts
        """
        errors = []

        for error_item in errors_raw:
            try:
                # Janus returns dicts directly now
                if isinstance(error_item, dict):
                    error_type = str(error_item.get('type', 'unknown'))
                    details = error_item.get('details')
                    
                    # Handle Prolog lists in details (converted to Python lists/tuples)
                    if isinstance(details, (list, tuple)):
                        details = [str(d) for d in details]
                    else:
                        details = str(details)

                    errors.append({
                        'type': error_type,
                        'details': self._format_error_details(error_type, details)
                    })
                else:
                    errors.append({
                        'type': 'unknown',
                        'details': str(error_item)
                    })
            except Exception as e:
                errors.append({
                    'type': 'parse_error',
                    'details': f"Failed to parse error: {error_item} ({e})"
                })

        return errors

    def _format_error_details(self, error_type: str, details) -> str:
        """Format error details for human readability"""
        if isinstance(details, list):
            if error_type == 'circular_superiority':
                return f"Circular superiority: {details[0]} <-> {details[1]}"
            elif error_type == 'contradictory_strict_rules':
                return f"Contradictory strict rules: {details[0]} vs {details[2]}"
            elif error_type == 'undefined_superior_rule':
                return f"Undefined rule in superiority: {details[0]}"
            elif error_type == 'unreachable_defeasible_rule':
                return f"Unreachable defeasible rule: {details[0]}"

        return str(details)

    def cleanup(self):
        """Clean up temporary files"""
        self._prolog.cleanup()

    def __del__(self):
        """Cleanup on deletion"""
        self.cleanup()


def create_janus_engine(delp_program: DeLPProgram) -> JanusDeLP:
    """
    Create a Janus-based DeLP engine

    Args:
        delp_program: Translated DeLP program

    Returns:
        JanusDeLP engine instance
    """
    return JanusDeLP(delp_program)
