"""
Format Canto reasoning patterns into PDO-compatible context
"""

from typing import Dict, Any, List


def format_reasoning_context(
    reasoning_structure: Dict[str, Any],
    dsl_instructions: str = ""
) -> Dict[str, Any]:
    """
    Convert Canto reasoning structure to PDO context format.

    Args:
        reasoning_structure: Output from DeLPReasoningAnalyzer
        dsl_instructions: Instructions from DSL file to include in generated prompts

    Returns:
        Context dict for PDO instruction generation
    """
    variables = reasoning_structure.get('variables', {})
    semantic_categories = reasoning_structure.get('semantic_categories', {})

    # Extract output variable and possible values
    output_vars = list(variables.keys())
    possible_values = {}

    for var_name, var_info in variables.items():
        values = set()
        for rule in var_info.get('strict_rules', []) + var_info.get('defeasible_rules', []):
            head = rule.get('head', '')
            if '(' in head and ')' in head:
                value = head.split('(')[1].rstrip(')')
                values.add(value)
        if values:
            possible_values[var_name] = list(values)

    # Format rules
    rules = []
    for var_name, var_info in variables.items():
        for rule in var_info.get('strict_rules', []):
            rules.append({
                'id': rule.get('id'),
                'conclusion': f"{var_name} = {_extract_value(rule.get('head', ''))}",
                'conditions': rule.get('conditions', []),
                'priority': 'strict',
                'description': rule.get('description', '')
            })
        for rule in var_info.get('defeasible_rules', []):
            rules.append({
                'id': rule.get('id'),
                'conclusion': f"{var_name} = {_extract_value(rule.get('head', ''))}",
                'conditions': rule.get('conditions', []),
                'priority': 'defeasible',
                'description': rule.get('description', '')
            })

    # Format conflicts
    conflicts = []
    for var_name, var_info in variables.items():
        for conflict in var_info.get('conflicts', []):
            conflicts.append({
                'winner': conflict.get('superior'),
                'loser': conflict.get('inferior'),
                'reason': conflict.get('type', 'OVERRIDES')
            })

    # Format semantic patterns
    patterns = {}
    for name, info in semantic_categories.items():
        patterns[name] = info.get('examples', [])

    return {
        'instructions': dsl_instructions,
        'output_variables': output_vars,
        'possible_values': possible_values,
        'rules': rules,
        'conflicts': conflicts,
        'semantic_patterns': patterns
    }


def _extract_value(head: str) -> str:
    """Extract value from Prolog head like 'var(value)'"""
    if '(' in head and ')' in head:
        return head.split('(')[1].rstrip(')')
    return head


def format_context_for_prompt(context: Dict[str, Any]) -> str:
    """
    Format context dict as string for inclusion in prompts.

    Args:
        context: Context from format_reasoning_context

    Returns:
        Formatted string for prompt inclusion
    """
    lines = []

    if context.get('instructions'):
        lines.append("Instructions:")
        lines.append(context['instructions'])
        lines.append("")

    # Output specification
    lines.append("Output Variables:")
    for var in context.get('output_variables', []):
        values = context.get('possible_values', {}).get(var, [])
        if values:
            lines.append(f"  - {var}: one of [{', '.join(values)}]")
        else:
            lines.append(f"  - {var}")
    lines.append("")

    # Rules
    if context.get('rules'):
        lines.append("Decision Rules:")
        for rule in context['rules']:
            priority_marker = "[STRICT]" if rule['priority'] == 'strict' else "[DEFAULT]"
            lines.append(f"  {priority_marker} {rule['conclusion']}")
            if rule.get('conditions'):
                conditions = rule['conditions']
                if isinstance(conditions, str):
                    lines.append(f"      when: {conditions}")
                else:
                    lines.append(f"      when: {', '.join(str(c) for c in conditions)}")
        lines.append("")

    # Conflicts
    if context.get('conflicts'):
        lines.append("Priority Overrides:")
        for conflict in context['conflicts']:
            lines.append(f"  - {conflict['winner']} overrides {conflict['loser']}")
        lines.append("")

    # Patterns
    if context.get('semantic_patterns'):
        lines.append("Semantic Patterns:")
        for name, examples in context['semantic_patterns'].items():
            if isinstance(examples, str):
                lines.append(f"  - {name}: {examples}")
            elif examples:
                lines.append(f"  - {name}: {', '.join(str(e) for e in examples[:5])}")
            else:
                lines.append(f"  - {name}: (no examples)")

    return "\n".join(lines)
