"""
Trace Presenter for Canto DSL.

Formats DeLP reasoning traces in user-friendly (DSL) or debug (Prolog) modes.
Uses stored DSL metadata to reconstruct human-readable rule representations.
"""

from typing import List, Dict, Any, Optional
from ..delp.program import DeLPProgram


class TracePresenter:
    """
    Presents DeLP reasoning traces in different formats.

    Modes:
        - 'user': DSL-friendly output for rule authors
        - 'debug': Prolog-level details for debugging
    """

    def __init__(self, program: DeLPProgram):
        """
        Initialize presenter with program metadata.

        Args:
            program: The DeLPProgram containing rule sources and declarations
        """
        self.program = program
        self._rule_lookup = self._build_rule_lookup()

    def _build_rule_lookup(self) -> Dict[str, Any]:
        """Build a lookup from rule_id to source Rule object."""
        lookup = {}
        for rule in self.program.defeasible_rules:
            if rule.source:
                lookup[rule.id] = rule.source
        for rule in self.program.strict_rules:
            if rule.source:
                lookup[rule.id] = rule.source
        return lookup

    def present(self, events: List[Dict[str, Any]], mode: str = 'user'):
        """
        Present trace events.

        Args:
            events: List of trace event dicts from engine.get_trace()
            mode: 'user' for DSL-friendly, 'debug' for Prolog details
        """
        if not events:
            print("\n(No trace events)")
            return

        if mode == 'debug':
            self._present_debug(events)
        else:
            self._present_user(events)

    # =========================================================================
    # Debug Mode (Prolog details)
    # =========================================================================

    def _present_debug(self, events: List[Dict[str, Any]]):
        """Present trace in debug mode with Prolog details."""
        print("\n" + "=" * 60)
        print(" REASONING TRACE (debug)")
        print("=" * 60)

        for event in events:
            step = event.get('step', '?')
            event_type = event.get('type', 'unknown')
            details = event.get('details', {})

            if event_type == 'query_start':
                print(f"[{step}] QUERY: {details.get('goal', '?')}")

            elif event_type == 'argument_found':
                print(f"[{step}] ARGUMENT FOUND:")
                print(f"      Goal: {details.get('goal', '?')}")
                print(f"      Rule: {details.get('rule_id', '?')}")
                print(f"      Specificity: {details.get('specificity', '?')}")

            elif event_type == 'no_arguments':
                print(f"[{step}] NO ARGUMENTS for {details.get('goal', '?')}")

            elif event_type == 'tree_build_start':
                print(f"[{step}] BUILDING TREE for {details.get('goal', '?')} via {details.get('rule_id', '?')}")

            elif event_type == 'defeater_found':
                arg_spec = details.get('argument_specificity', '?')
                def_spec = details.get('defeater_specificity', '?')
                print(f"[{step}] DEFEATER FOUND:")
                print(f"      {details.get('argument_goal', '?')} ({details.get('argument_rule', '?')}, spec={arg_spec})")
                print(f"      defeated by: {details.get('defeater_goal', '?')} ({details.get('defeater_rule', '?')}, spec={def_spec})")
                print(f"      type: {details.get('defeat_type', '?')}")
                reason_type = details.get('reason_type', '')
                reason_details = details.get('reason_details', '')
                if reason_type:
                    print(f"      reason: {reason_type} - {reason_details}")

            elif event_type == 'marking':
                print(f"[{step}] MARKING: {details.get('goal', '?')} ({details.get('rule_id', '?')}) = {details.get('status', '?')}")
                reason = details.get('reason', 'none')
                if reason and reason != 'none':
                    print(f"      reason: {reason}")

            elif event_type == 'final_status':
                print(f"[{step}] FINAL STATUS: {details.get('goal', '?')} = {details.get('status', '?')}")
                winning = details.get('winning_rule', 'none')
                if winning and winning != 'none':
                    print(f"      winning rule: {winning}")

            else:
                print(f"[{step}] {event_type}: {details}")

        print("=" * 60)

    # =========================================================================
    # User Mode (DSL-friendly)
    # =========================================================================

    def _present_user(self, events: List[Dict[str, Any]]):
        """Present trace in user mode with DSL-friendly output."""
        # Extract query goal
        query_goal_dsl = "?"
        for e in events:
            if e.get('type') == 'query_start':
                query_goal_dsl = self._format_goal(e.get('details', {}).get('goal', '?'))
                break

        print("\n" + "=" * 60)
        print(f" REASONING TRACE: {query_goal_dsl}")
        print("=" * 60)

        for event in events:
            event_type = event.get('type', 'unknown')
            details = event.get('details', {})

            if event_type == 'query_start':
                goal_dsl = self._format_goal(details.get('goal', '?'))
                print(f"\n-> Checking: Can {goal_dsl}?")

            elif event_type == 'argument_found':
                rule_id = details.get('rule_id', '?')
                rule_source = self._rule_lookup.get(rule_id)
                rule_dsl = self._format_rule(rule_source)
                spec = details.get('specificity', 0)
                spec_display = round(spec) if isinstance(spec, (int, float)) else spec
                print(f"\n   Found rule ({rule_id}):")
                print(f"      {rule_dsl}")
                print(f"      strength: {spec_display}")

            elif event_type == 'no_arguments':
                goal_dsl = self._format_goal(details.get('goal', '?'))
                print(f"\n   No rules found for {goal_dsl}")

            elif event_type == 'tree_build_start':
                pass  # Skip - too technical for user mode

            elif event_type == 'defeater_found':
                self._present_conflict(details)

            elif event_type == 'marking':
                pass  # Skip - internal detail

            elif event_type == 'final_status':
                goal_dsl = self._format_goal(details.get('goal', '?'))
                status = details.get('status', '?')
                print(f"\n=> Result: {goal_dsl} is {status.upper()}")

        print("\n" + "=" * 60)

    def _present_conflict(self, details: Dict[str, Any]):
        """Present a conflict between two rules."""
        arg_rule_id = details.get('argument_rule', '?')
        def_rule_id = details.get('defeater_rule', '?')
        arg_source = self._rule_lookup.get(arg_rule_id)
        def_source = self._rule_lookup.get(def_rule_id)

        arg_dsl = self._format_rule(arg_source)
        def_dsl = self._format_rule(def_source)

        arg_spec = details.get('argument_specificity', 0)
        def_spec = details.get('defeater_specificity', 0)
        reason_type = details.get('reason_type', '')

        print(f"\n   Conflict:")
        print(f"      {arg_dsl}")
        print(f"         vs")
        print(f"      {def_dsl}")

        # Explain why using actual metadata
        if reason_type == 'explicit_superiority':
            if def_source and hasattr(def_source, 'override_target') and def_source.override_target:
                override_val = def_source.override_target
                # Handle both string and enum value
                if hasattr(override_val, 'value'):
                    override_val = override_val.value
                if override_val and override_val != 'none':
                    print(f"      Winner: {def_rule_id} (has OVERRIDES {override_val.upper()})")
                else:
                    print(f"      Winner: {def_rule_id} (explicit priority)")
            else:
                print(f"      Winner: {def_rule_id} (explicit priority)")
        elif reason_type == 'more_specific':
            print(f"      Winner: {def_rule_id} (more specific: {round(def_spec)} > {round(arg_spec)})")
        elif reason_type == 'blocking':
            print(f"      Result: Mutual block (equal strength, no OVERRIDES)")

    # =========================================================================
    # DSL Formatting (uses stored metadata)
    # =========================================================================

    def _format_goal(self, goal_str: str) -> str:
        """
        Format a Prolog goal as DSL.

        Uses declaration metadata when available.
        """
        if not goal_str or goal_str == '?':
            return '?'

        # Parse functor(arg) format
        if '(' in goal_str and goal_str.endswith(')'):
            paren_idx = goal_str.index('(')
            var_name = goal_str[:paren_idx]
            value = goal_str[paren_idx+1:-1].strip("'\"")

            # Check if this is a known variable
            decl = self.program.declarations.get(var_name)

            # Format value based on type
            if value in ('true', 'false'):
                return f"${var_name} IS {value}"
            else:
                return f'${var_name} IS "{value}"'

        return goal_str

    def _format_rule(self, rule_source) -> str:
        """
        Format a Rule object as DSL string.

        Uses the actual Rule object's attributes to reconstruct DSL.
        """
        if rule_source is None:
            return "?"

        # Head: $variable IS value
        var = f"${rule_source.head_variable}"
        val = rule_source.head_value
        if isinstance(val, str):
            val_str = f'"{val}"'
        elif isinstance(val, bool):
            val_str = str(val).lower()
        else:
            val_str = str(val)

        result = f"{var} IS {val_str}"

        # Conditions: WHEN ...
        if rule_source.conditions:
            conds = " AND ".join(self._format_condition(c) for c in rule_source.conditions)
            result += f" WHEN {conds}"

        # Overrides: OVERRIDES ALL/NORMAL
        if hasattr(rule_source, 'override_target') and rule_source.override_target:
            override_val = rule_source.override_target
            # Handle both string and enum value
            if hasattr(override_val, 'value'):
                override_val = override_val.value
            if override_val and override_val != 'none':
                result += f" OVERRIDES {override_val.upper()}"

        return result

    def _format_condition(self, condition) -> str:
        """
        Format a Condition object as DSL string.

        Recursively handles nested conditions (AND, OR, NOT).
        """
        op = condition.operator

        if op == "NOT":
            inner = self._format_condition(condition.right) if hasattr(condition.right, 'operator') else str(condition.right)
            return f"NOT {inner}"

        elif op == "NOT_WARRANTED":
            inner = self._format_condition(condition.right) if hasattr(condition.right, 'operator') else str(condition.right)
            return f"NOT WARRANTED {inner}"

        elif op in ("AND", "OR"):
            left = self._format_condition(condition.left) if hasattr(condition.left, 'operator') else str(condition.left)
            right = self._format_condition(condition.right) if hasattr(condition.right, 'operator') else str(condition.right)
            return f"({left} {op} {right})"

        elif op == "MATCHES":
            return f"${condition.left} MATCHES ${condition.right}"

        elif op == "HAS":
            return f"${condition.left} HAS ${condition.right}"

        elif op == "LIKE":
            patterns = condition.right
            if isinstance(patterns, list):
                patterns_str = ", ".join(f'"{p}"' for p in patterns)
                return f"${condition.left} LIKE [{patterns_str}]"
            return f"${condition.left} LIKE {condition.right}"

        elif op == "IS":
            val = condition.right
            if isinstance(val, str):
                val_str = f'"{val}"'
            elif isinstance(val, bool):
                val_str = str(val).lower()
            else:
                val_str = str(val)
            return f"${condition.left} IS {val_str}"

        else:
            # Fallback to condition's own repr
            return str(condition)
