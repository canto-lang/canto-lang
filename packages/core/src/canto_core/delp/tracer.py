"""
DeLP Tracer - Debugging and tracing for DeLP reasoning.

This module provides tracing functionality to understand
the reasoning process in DeLP queries.
"""

from typing import List, Dict, Any

from .prolog_engine import PrologEngine


class DeLPTracer:
    """
    Tracer for DeLP reasoning process.

    Provides methods to enable/disable tracing and inspect
    the reasoning steps during query execution.
    """

    def __init__(self, prolog: PrologEngine):
        self._prolog = prolog

    def enable(self):
        """
        Enable trace mode. Trace events will be recorded during reasoning.
        Call this before running queries to see the reasoning process.
        """
        try:
            self._prolog.query_once("enable_trace")
        except Exception as e:
            print(f"Warning: Failed to enable trace: {e}")

    def disable(self):
        """Disable trace mode."""
        try:
            self._prolog.query_once("disable_trace")
        except Exception as e:
            print(f"Warning: Failed to disable trace: {e}")

    def clear(self):
        """Clear all recorded trace events."""
        try:
            self._prolog.query_once("clear_trace")
        except Exception as e:
            print(f"Warning: Failed to clear trace: {e}")

    def get_events(self) -> List[Dict[str, Any]]:
        """
        Get all trace events from the last query.

        Returns:
            List of trace event dicts with keys:
                - step: Event sequence number
                - type: Event type (query_start, argument_found, etc.)
                - details: Event-specific details dict
        """
        try:
            result = self._prolog.query_once("get_trace(Events)")
            if result and 'Events' in result:
                return result['Events']
            return []
        except Exception as e:
            print(f"Warning: Failed to get trace: {e}")
            return []

    def print_events(self):
        """Print the reasoning trace in a human-readable format."""
        events = self.get_events()
        if not events:
            print("\n(No trace events. Did you call enable() before querying?)")
            return

        print("\n" + "=" * 60)
        print(" REASONING TRACE")
        print("=" * 60)

        for event in events:
            self._print_event(event)

        print("=" * 60)

    def _print_event(self, event: Dict[str, Any]):
        """Print a single trace event."""
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
            print(f"[{step}] DEFEATER FOUND:")
            arg_spec = details.get('argument_specificity', '?')
            def_spec = details.get('defeater_specificity', '?')
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
