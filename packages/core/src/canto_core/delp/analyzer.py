"""
DeLP Reasoning Analyzer

Analyzes DeLP knowledge base using the Prolog meta-interpreter and produces
simplified output suitable for LLM prompt generation.

The Prolog side does deep defeasible reasoning (dialectical trees, blocking,
specificity). This analyzer extracts actionable conclusions for the LLM.
"""

from typing import Dict, List, Any, Set, Optional, TypedDict, Tuple
from enum import Enum
from difflib import get_close_matches
from .engine import JanusDeLP
from .models import normalize_prolog_value
from .program import DeLPProgram


class DefeatReason(Enum):
    """Enumeration for reasons one argument defeats another."""
    EXPLICIT_OVERRIDE = "explicit_override"
    STRICT_BEATS_DEFEASIBLE = "strict_beats_defeasible"
    MORE_SPECIFIC = "more_specific"
    PREFERENCE = "preference"
    BLOCKED_BY_EQUAL = "blocked_by_equal_arguments"


# Typed Dictionaries for structured data
class RuleInfo(TypedDict):
    id: str
    value: Any
    conditions: str
    description: str

class Conclusion(TypedDict):
    value: Any
    status: str
    rule_id: Optional[str]
    reason: Optional[str]

class Alternative(TypedDict):
    value: Any
    status: str
    rule_id: Optional[str]
    defeated_by: Optional[str]
    defeat_reason: Optional[str]

class Conflict(TypedDict):
    winner: Dict[str, Any]
    loser: Dict[str, Any]
    reason: str

class DefeatInfo(TypedDict, total=False):
    defeater_rule: Optional[str]
    reason: Optional[str]

class ReasoningDict(TypedDict):
    variable: str
    strict_rules: List[RuleInfo]
    defeasible_rules: List[RuleInfo]
    conclusion: Optional[Conclusion]
    alternatives: List[Alternative]
    conflict_resolutions: List[Conflict]
    semantic_patterns: Dict[str, List[str]]


class AssumptionWarning(TypedDict, total=False):
    """Warning about an assumption made during reasoning."""
    predicate: str  # Full predicate string, e.g., "matches(query, emergency_termz)"
    functor: str    # Predicate name, e.g., "matches"
    args: List[Any] # Arguments, e.g., ["query", "emergency_termz"]
    severity: str   # "warning" or "error"
    message: str    # Human-readable message
    suggestion: Optional[str]  # Suggested correction (if typo detected)


class ReasoningPattern:
    """
    Simplified reasoning pattern for a variable.

    Contains actionable information for LLM prompt generation,
    not full dialectical trees.
    """

    def __init__(self, variable: str):
        self.variable = variable
        # Rule structure (for context)
        self.strict_rules: List[RuleInfo] = []
        self.defeasible_rules: List[RuleInfo] = []
        # Simplified reasoning results
        self.conclusion: Optional[Conclusion] = None  # The winning value
        self.alternatives: List[Alternative] = []  # Defeated/blocked alternatives
        self.conflicts: List[Conflict] = []  # How conflicts were resolved
        self.semantic_patterns: Dict[str, List[str]] = {}

    def to_dict(self) -> ReasoningDict:
        """Convert to dictionary for LLM consumption"""
        return {
            'variable': self.variable,
            'strict_rules': self.strict_rules,
            'defeasible_rules': self.defeasible_rules,
            'conclusion': self.conclusion,
            'alternatives': self.alternatives,
            'conflict_resolutions': self.conflicts,
            'semantic_patterns': self.semantic_patterns
        }


class DeLPReasoningAnalyzer:
    """
    Analyzes DeLP knowledge base and produces simplified output for LLMs.

    Uses the deep Prolog DeLP meta-interpreter for proper defeasible reasoning,
    then simplifies the results to actionable conclusions.
    """

    def __init__(self, program: DeLPProgram, extra_prolog: Optional[str] = None):
        self.program = program
        self.engine = JanusDeLP(program)

        # Load program and extra prolog once
        self.engine.load()
        if extra_prolog:
            self.engine.load_prolog_string(extra_prolog)

        # Build rule index for O(1) lookups
        self.rule_index = {}
        for rule in self.program.strict_rules:
            self.rule_index[rule.id] = rule
        for rule in self.program.defeasible_rules:
            self.rule_index[rule.id] = rule

        # Index rules by variable for efficient lookups
        self.strict_rules_by_var: Dict[str, List] = {}
        self.defeasible_rules_by_var: Dict[str, List] = {}
        for rule in self.program.strict_rules:
            if rule.source:
                var = rule.source.head_variable
                self.strict_rules_by_var.setdefault(var, []).append(rule)
        for rule in self.program.defeasible_rules:
            if rule.source:
                var = rule.source.head_variable
                self.defeasible_rules_by_var.setdefault(var, []).append(rule)

        # Build set of known predicates for assumption validation
        self._known_predicates: Set[str] = set()
        self._known_categories: Set[str] = set()
        self._build_known_predicates()

        # Track assumptions found during analysis
        self._collected_assumptions: List[Dict[str, Any]] = []

    def _build_known_predicates(self):
        """Build sets of known predicates and categories for assumption checking."""
        # Add semantic categories
        for cat_name in self.program.semantic_categories.keys():
            self._known_categories.add(cat_name)
            self._known_predicates.add(cat_name)

        # Add variable names (they become predicates like vaccine_flag/1)
        for var_name in self.program.declarations.keys():
            self._known_predicates.add(var_name)

        # Add standard predicate names
        self._known_predicates.update(['matches', 'has', 'like', 'pattern'])

    def analyze(self) -> Dict[str, Any]:
        """
        Analyze the KB and extract simplified reasoning results.

        Returns:
            Dictionary with:
            - valid: bool
            - validation_errors: List
            - variables: Map of variable -> simplified reasoning
            - semantic_categories: Map of category -> patterns
            - conflict_summary: Human-readable conflict resolutions
            - hierarchy: Nested structure of parent-child relationships
        """
        # Run static validation
        validation_result = self.engine.validate_program()

        # Analyze each variable
        variables: Dict[str, ReasoningDict] = {}
        for var_name in self.program.declarations.keys():
            pattern = self._analyze_variable(var_name)
            variables[var_name] = pattern.to_dict()

        # Extract semantic categories
        semantic_categories = {}
        for cat_name, category in self.program.semantic_categories.items():
            semantic_categories[cat_name] = {
                'description': category.description,
                'patterns': category.patterns
            }

        # Generate human-readable conflict summary for LLM
        conflict_summary = self._generate_conflict_summary()

        # Build hierarchy from has_relationships
        hierarchy = self._build_hierarchy()

        return {
            'valid': validation_result['valid'],
            'validation_errors': validation_result['errors'],
            'variables': variables,
            'semantic_categories': semantic_categories,
            'conflict_summary': conflict_summary,
            'hierarchy': hierarchy,
            'rule_count': len(self.program.strict_rules) + len(self.program.defeasible_rules)
        }

    def analyze_for_llm(self) -> Dict[str, Any]:
        """
        Generate a simplified summary specifically for LLM prompt generation.

        Returns plain-English descriptions of the reasoning logic.
        """
        full_analysis = self.analyze()

        llm_summary = {
            'variables': {},
            'rules_summary': [],
            'conflict_resolutions': []
        }

        for var_name, var_data in full_analysis['variables'].items():
            # Skip variables with no rules (just declarations)
            if not var_data['strict_rules'] and not var_data['defeasible_rules']:
                continue

            var_summary = {
                'name': var_name,
                'possible_values': self._get_possible_values_list(var_name),
            }

            if var_data['conclusion']:
                var_summary['default_value'] = var_data['conclusion']['value']
                var_summary['determined_by'] = var_data['conclusion']['rule_id']
                var_summary['reason'] = var_data['conclusion'].get('reason', 'No conflicts')

            if var_data['alternatives']:
                var_summary['overridden_values'] = [
                    {
                        'value': alt['value'],
                        'would_be_set_by': alt['rule_id'],
                        'defeated_by': alt.get('defeated_by'),
                        'defeat_reason': alt.get('defeat_reason')
                    }
                    for alt in var_data['alternatives']
                ]

            llm_summary['variables'][var_name] = var_summary

        # Add rule descriptions
        for rule in self.program.strict_rules + self.program.defeasible_rules:
            if rule.source:
                llm_summary['rules_summary'].append({
                    'id': rule.id,
                    'type': 'strict' if rule in self.program.strict_rules else 'defeasible',
                    'description': self._describe_rule(rule)
                })

        # Add conflict resolutions in plain English
        llm_summary['conflict_resolutions'] = full_analysis['conflict_summary']

        return llm_summary

    def _analyze_variable(self, var_name: str) -> ReasoningPattern:
        """
        Analyze a variable using DeLP and extract simplified results.
        """
        pattern = ReasoningPattern(var_name)

        # Get rule structure using pre-built index
        strict_rules = self.strict_rules_by_var.get(var_name, [])
        defeasible_rules = self.defeasible_rules_by_var.get(var_name, [])

        for rule in strict_rules:
            pattern.strict_rules.append({
                'id': rule.id,
                'value': rule.source.head_value,
                'conditions': self._describe_conditions(rule.source),
                'description': self._describe_rule(rule)
            })

        for rule in defeasible_rules:
            pattern.defeasible_rules.append({
                'id': rule.id,
                'value': rule.source.head_value,
                'conditions': self._describe_conditions(rule.source),
                'description': self._describe_rule(rule)
            })

        # Query DeLP for each possible value
        if strict_rules or defeasible_rules:
            reasoning_results = self._query_and_simplify(var_name)

            # Find the warranted value (conclusion)
            for result in reasoning_results:
                if result['status'] == 'warranted':
                    pattern.conclusion = result
                    break

            # Collect alternatives (defeated/blocked)
            pattern.alternatives = [
                r for r in reasoning_results
                if r['status'] in ['defeated', 'blocked']
            ]

            # Extract conflict resolutions
            pattern.conflicts = self._extract_conflicts(var_name, reasoning_results)

        # Get semantic patterns
        pattern.semantic_patterns = self._extract_semantic_patterns(var_name)

        return pattern

    def _query_and_simplify(self, var_name: str) -> List[Conclusion]:
        """
        Query DeLP and return simplified results (no full trees).
        """
        results: List[Conclusion] = []
        possible_values = self._get_possible_values(var_name)

        for value in possible_values:
            goal = self._format_goal(var_name, value)
            try:
                result = self.engine.delp_query(goal)

                simplified: Conclusion = {
                    'value': value,
                    'status': result.get('status', 'error'),
                    'rule_id': self._extract_winning_rule(result.get('tree')),
                    'reason': None, # To be filled in later if needed
                }

                # If defeated/blocked, explain why
                if result.get('status') in ['defeated', 'blocked']:
                    defeat_info = self._extract_defeat_reason(result.get('tree'))
                    simplified['defeated_by'] = defeat_info.get('defeater_rule')
                    simplified['defeat_reason'] = defeat_info.get('reason')

                results.append(simplified)

            except Exception as e:
                results.append({
                    'value': value,
                    'status': 'error',
                    'rule_id': None,
                    'error': str(e)
                })

        return results

    def _extract_winning_rule(self, tree: Optional[Dict]) -> Optional[str]:
        """Extract the rule ID that supports the argument."""
        if not tree or tree.get('type') == 'no_arguments':
            return None
        return tree.get('argument', {}).get('rule_id')

    def _extract_defeat_reason(self, tree: Optional[Dict]) -> DefeatInfo:
        """Extract why an argument was defeated."""
        if not tree or tree.get('type') == 'no_arguments':
            return {}

        defeaters = tree.get('defeaters', [])
        if not defeaters:
            return {}

        # Find the first undefeated defeater
        for defeater in defeaters:
            if defeater.get('status') == 'undefeated':
                defeater_rule = defeater.get('argument', {}).get('rule_id')
                # Determine reason
                my_rule = tree.get('argument', {}).get('rule_id')
                reason = self._determine_defeat_reason(defeater_rule, my_rule)
                return {
                    'defeater_rule': defeater_rule,
                    'reason': reason
                }

        return {'reason': DefeatReason.BLOCKED_BY_EQUAL.value}

    def _determine_defeat_reason(self, winner_rule: str, loser_rule: str) -> str:
        """Determine why one rule beats another."""
        # Check explicit superiority
        for sup in self.program.superiority:
            if sup['superior'] == winner_rule and sup['inferior'] == loser_rule:
                return DefeatReason.EXPLICIT_OVERRIDE.value

        # Check rule types
        winner = self.rule_index.get(winner_rule)
        loser = self.rule_index.get(loser_rule)

        if winner and loser:
            winner_type = 'strict' if winner in self.program.strict_rules else 'defeasible'
            loser_type = 'strict' if loser in self.program.strict_rules else 'defeasible'

            if winner_type == 'strict' and loser_type == 'defeasible':
                return DefeatReason.STRICT_BEATS_DEFEASIBLE.value

            # Check specificity (body length)
            winner_len = len(winner.body)
            loser_len = len(loser.body)
            if winner_len > loser_len:
                return DefeatReason.MORE_SPECIFIC.value

        return DefeatReason.PREFERENCE.value

    def _extract_conflicts(self, var_name: str, results: List[Conclusion]) -> List[Conflict]:
        """Extract conflict resolution information."""
        conflicts: List[Conflict] = []

        warranted = [r for r in results if r['status'] == 'warranted']
        defeated = [r for r in results if r['status'] in ['defeated', 'blocked']]

        for winner in warranted:
            for loser in defeated:
                conflicts.append({
                    'winner': {
                        'value': winner['value'],
                        'rule': winner.get('rule_id')
                    },
                    'loser': {
                        'value': loser['value'],
                        'rule': loser.get('rule_id')
                    },
                    'reason': loser.get('defeat_reason', 'unknown')
                })

        return conflicts

    def _generate_conflict_summary(self) -> List[str]:
        """Generate human-readable conflict resolution summary."""
        summaries = []

        for sup in self.program.superiority:
            superior = self.rule_index.get(sup['superior'])
            inferior = self.rule_index.get(sup['inferior'])

            if superior and inferior and superior.source and inferior.source:
                sup_id = sup['superior']
                inf_id = sup['inferior']
                sup_desc = f"{superior.source.head_variable}={superior.source.head_value}"
                inf_desc = f"{inferior.source.head_variable}={inferior.source.head_value}"
                summaries.append(
                    f"{sup_id} ({sup_desc}) > {inf_id} ({inf_desc})"
                )

        return summaries

    def _get_possible_values(self, var_name: str) -> Set[Any]:
        """Collect all possible values for a variable."""
        values = set()

        # From VALUES FROM clause
        decl = self.program.declarations.get(var_name)
        if decl and hasattr(decl, 'values_from') and decl.values_from:
            values.update(decl.values_from)

        # From rule head values
        for rule in self.program.strict_rules + self.program.defeasible_rules:
            if rule.source and rule.source.head_variable == var_name:
                values.add(rule.source.head_value)

        return values

    def _get_possible_values_list(self, var_name: str) -> List[Any]:
        """Get possible values as a list."""
        return list(self._get_possible_values(var_name))

    def _format_goal(self, var_name: str, value: Any) -> str:
        """Format a Prolog goal string."""
        if isinstance(value, bool):
            return f"{var_name}({str(value).lower()})"
        elif isinstance(value, str):
            if value.startswith("'") or value.startswith('"'):
                return f"{var_name}({value})"
            return f"{var_name}('{value}')"
        else:
            return f"{var_name}({value})"

    def _describe_rule(self, rule) -> str:
        """Generate human-readable rule description."""
        if not rule.source:
            return f"Rule {rule.id}"

        source = rule.source
        var = source.head_variable
        val = source.head_value

        if not source.conditions:
            if source.priority == 'normal':
                return f"By default, {var} is {val}"
            else:
                return f"{var} is always {val}"

        cond_desc = self._describe_conditions(source)
        if source.priority == 'normal':
            return f"Normally, {var} is {val} when {cond_desc}"
        else:
            return f"{var} is {val} when {cond_desc}"

    def _describe_conditions(self, source_rule) -> str:
        """Describe rule conditions in plain English."""
        if not source_rule.conditions:
            return "always"

        descriptions = []
        for cond in source_rule.conditions:
            if hasattr(cond, 'operator'):
                if cond.operator == 'MATCHES':
                    descriptions.append(f"{cond.left} matches {cond.right}")
                elif cond.operator == 'IS':
                    descriptions.append(f"{cond.left} is {cond.right}")
                elif cond.operator == 'HAS':
                    descriptions.append(f"{cond.left} has {cond.right}")
                else:
                    descriptions.append(str(cond))
            else:
                descriptions.append(str(cond))

        return " AND ".join(descriptions)

    def _extract_semantic_patterns(self, var_name: str) -> Dict[str, List[str]]:
        """Extract semantic categories used by this variable's rules."""
        patterns = {}

        all_rules = [r for r in self.program.strict_rules + self.program.defeasible_rules
                    if r.source and r.source.head_variable == var_name]

        for rule in all_rules:
            if not rule.source:
                continue

            for condition in rule.source.conditions:
                if hasattr(condition, 'operator') and condition.operator == 'MATCHES':
                    category_name = str(condition.right).strip('$')
                    if category_name in self.program.semantic_categories:
                        category = self.program.semantic_categories[category_name]
                        patterns[category_name] = category.patterns

        return patterns

    def _build_hierarchy(self) -> Dict[str, Any]:
        """
        Build a nested hierarchy structure from has_relationships.

        Returns a dictionary representing the tree structure of variables,
        where each node has:
        - name: variable name
        - description: optional description
        - is_list: whether this is a list relationship
        - children: list of child nodes
        """
        # Build parent -> children mapping
        parent_to_children: Dict[str, List[Dict[str, Any]]] = {}
        child_vars: Set[str] = set()

        for key, has_decl in self.program.has_relationships.items():
            parent = has_decl.parent
            child = has_decl.child

            child_vars.add(child)

            if parent not in parent_to_children:
                parent_to_children[parent] = []

            # Get child's declaration for description
            child_decl = self.program.declarations.get(child)
            child_info = {
                'name': child,
                'description': has_decl.description or (child_decl.description if child_decl else None),
                'is_list': has_decl.is_list,
                'children': []  # Will be populated recursively
            }
            parent_to_children[parent].append(child_info)

        # Find root nodes (parents that are not children of anything)
        root_nodes = []
        for parent in parent_to_children.keys():
            if parent not in child_vars:
                root_nodes.append(parent)

        # Build nested structure recursively
        def build_tree(var_name: str) -> Dict[str, Any]:
            decl = self.program.declarations.get(var_name)
            node = {
                'name': var_name,
                'description': decl.description if decl else None,
                'is_list': False,  # Root nodes are not lists by default
                'children': []
            }

            if var_name in parent_to_children:
                for child_info in parent_to_children[var_name]:
                    # Recursively build children
                    child_node = build_tree(child_info['name'])
                    child_node['is_list'] = child_info['is_list']
                    if child_info['description']:
                        child_node['description'] = child_info['description']
                    node['children'].append(child_node)

            return node

        # Build trees for all roots
        roots = [build_tree(root) for root in root_nodes]

        return {
            'roots': roots,
            'flat_relationships': [
                {
                    'parent': has_decl.parent,
                    'child': has_decl.child,
                    'is_list': has_decl.is_list,
                    'description': has_decl.description
                }
                for has_decl in self.program.has_relationships.values()
            ]
        }

    def get_reasoning_summary(self) -> str:
        """Generate human-readable summary for debugging."""
        analysis = self.analyze()

        lines = ["=" * 60, "REASONING ANALYSIS SUMMARY", "=" * 60, ""]

        for var_name, var_data in analysis['variables'].items():
            if not var_data['strict_rules'] and not var_data['defeasible_rules']:
                continue

            lines.append(f"\n{var_name}:")

            if var_data['conclusion']:
                c = var_data['conclusion']
                lines.append(f"  → Conclusion: {c['value']} (via {c['rule_id']})")

            if var_data['alternatives']:
                lines.append(f"  → Defeated alternatives:")
                for alt in var_data['alternatives']:
                    reason = alt.get('defeat_reason', 'unknown')
                    lines.append(f"     - {alt['value']} ({alt.get('rule_id')}) - {reason}")

        if analysis['conflict_summary']:
            lines.append("\n\nCONFLICT RESOLUTIONS:")
            for summary in analysis['conflict_summary']:
                lines.append(f"  • {summary}")

        return "\n".join(lines)

    def get_dsl_analysis_report(self) -> str:
        """
        Generate a visual tree-based DSL analysis report.

        Shows each variable with its rules as a tree structure,
        making it easy to see conflicts, winners, and issues at a glance.
        """
        # Get static analysis from Prolog
        static_analysis = self.engine.get_dsl_analysis()

        # Get runtime reasoning results
        runtime_analysis = self.analyze()

        lines = ["=" * 60, "DSL ANALYSIS REPORT", "=" * 60]

        conflicts = static_analysis.get('conflicts', [])
        no_conflicts = static_analysis.get('no_conflicts', [])
        issues = static_analysis.get('issues', [])

        # Build issue lookup by rule_id for quick access
        issue_rules = set()
        contradicting_vars = set()
        for issue in issues:
            msg = issue.get('message', '')
            if 'can never win' in msg or 'unreachable' in msg.lower():
                # Extract rule_id from message
                details = issue.get('details', '')
                if details:
                    issue_rules.add(details.split(',')[0].strip('['))
            if 'contradict' in msg.lower():
                # Extract variable name
                details = issue.get('details', '')
                contradicting_vars.add(details)

        # Process variables with conflicts
        for var_info in conflicts:
            var_name = var_info.get('variable', '?')
            rules = var_info.get('rules', [])

            # Get runtime conclusion
            var_data = runtime_analysis.get('variables', {}).get(var_name, {})
            conclusion = var_data.get('conclusion')
            winner_rule = conclusion.get('rule_id') if conclusion else None

            # Determine variable status
            has_explicit_winner = any(
                (r.get('has_override') == True or r.get('has_override') == 'true')
                and r.get('status') == 'wins'
                for r in rules
            )
            has_contradiction = any(
                var_name in str(issue.get('details', ''))
                and 'contradict' in issue.get('message', '').lower()
                for issue in issues
            )

            # Variable header with status
            if has_contradiction:
                lines.append("")
                lines.append(f"?{var_name} ❌ CONFLICT")
            elif has_explicit_winner:
                lines.append("")
                lines.append(f"?{var_name} ✅ RESOLVED")
            else:
                lines.append("")
                lines.append(f"?{var_name} ⚠️  COMPETES ({len(rules)} rules)")

            # Draw tree of rules
            for i, rule in enumerate(rules):
                rule_id = rule.get('rule_id', '?')
                value = rule.get('value', '?')
                rule_type = rule.get('type', 'unknown')
                status = rule.get('status', 'unknown')
                has_override = rule.get('has_override') == True or rule.get('has_override') == 'true'
                defeated_by = rule.get('defeated_by', [])

                is_last = (i == len(rules) - 1)
                prefix = "  └── " if is_last else "  ├── "

                # Determine rule status display
                if rule_id == winner_rule:
                    status_str = "✓ WINNER"
                elif status == 'defeated' or defeated_by:
                    status_str = "✗ defeated"
                elif has_override:
                    status_str = "⚡ overrides"
                else:
                    status_str = "○ competes"

                type_char = "S" if rule_type == 'strict' else "D"
                lines.append(f"{prefix}{rule_id} [{type_char}] \"{value}\" → {status_str}")

            # Show winner summary
            if conclusion:
                lines.append(f"  ─────→ Result: \"{conclusion['value']}\"")

        # Process variables without conflicts
        for var_info in no_conflicts:
            var_name = var_info.get('variable', '?')
            num_rules = var_info.get('num_rules', 0)
            rules = var_info.get('rules', [])

            if num_rules == 0:
                continue

            var_data = runtime_analysis.get('variables', {}).get(var_name, {})
            conclusion = var_data.get('conclusion')

            lines.append("")
            if num_rules == 1:
                rule = rules[0]
                value = rule.get('value', '?')
                type_char = "S" if rule.get('type') == 'strict' else "D"
                lines.append(f"?{var_name} ✅ SINGLE RULE")
                lines.append(f"  └── {rule.get('rule_id', '?')} [{type_char}] \"{value}\" → ✓")
            else:
                value = rules[0].get('value', '?') if rules else '?'
                lines.append(f"?{var_name} ✅ ALIGNED ({num_rules} rules, same value)")
                for i, rule in enumerate(rules):
                    is_last = (i == len(rules) - 1)
                    prefix = "  └── " if is_last else "  ├── "
                    type_char = "S" if rule.get('type') == 'strict' else "D"
                    lines.append(f"{prefix}{rule.get('rule_id', '?')} [{type_char}] \"{value}\"")

        # Issues section - only show actionable issues
        actionable_issues = [
            issue for issue in issues
            if 'contradict' in issue.get('message', '').lower()
            or 'can never win' in issue.get('message', '').lower()
        ]

        if actionable_issues:
            lines.append("")
            lines.append("─" * 60)
            lines.append("❌ ISSUES TO FIX:")
            for issue in actionable_issues:
                msg = issue.get('message', str(issue))
                lines.append(f"  • {msg}")

        lines.append("")
        lines.append("=" * 60)

        # Summary line
        error_count = len([i for i in issues if 'contradict' in i.get('message', '').lower()])
        warning_count = len([v for v in conflicts if not any(
            (r.get('has_override') == True or r.get('has_override') == 'true')
            for r in v.get('rules', [])
        )])
        ok_count = len(no_conflicts) + len([v for v in conflicts if any(
            (r.get('has_override') == True or r.get('has_override') == 'true')
            for r in v.get('rules', [])
        )])

        lines.insert(3, f"Summary: {error_count} errors, {warning_count} warnings, {ok_count} ok")
        lines.insert(4, "")

        return "\n".join(lines)
