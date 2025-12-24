"""
DeLPProgram - Represents a complete DeLP (Defeasible Logic Programming) program.
"""

from typing import List, Dict

from .models import DeLPRule, DeLPRuleSource, DeLPDeclaration
from ..utils.prolog import normalize_prolog_atom


class DeLPProgram:
    """
    Represents a complete DeLP program.

    Uses Pydantic models for rules and declarations to ensure
    automatic normalization of values (e.g., 'true' -> True).
    """
    def __init__(self):
        # Import here to avoid circular dependency
        from ..ast_nodes import SemanticCategory, HasDeclaration

        self.facts: List[str] = []  # Strict facts
        self.strict_rules: List[DeLPRule] = []  # Strict rules (:-)
        self.defeasible_rules: List[DeLPRule] = []  # Defeasible rules (->)
        self.superiority: List[Dict] = []  # Superiority relations
        self.declarations: Dict[str, DeLPDeclaration] = {}
        self.semantic_categories: Dict[str, SemanticCategory] = {}
        self.has_relationships: Dict[str, HasDeclaration] = {}  # parent -> HasDeclaration
        self.rule_counter = 0

    def add_fact(self, fact: str):
        """Add a strict fact"""
        self.facts.append(fact)

    def add_strict_rule(self, rule_id: str, head: str, body: List[str], source_rule=None):
        """Add a strict rule with normalized source."""
        source = DeLPRuleSource.from_ast(source_rule) if source_rule else None
        self.strict_rules.append(DeLPRule(
            id=rule_id,
            head=head,
            body=body,
            source=source
        ))

    def add_defeasible_rule(self, rule_id: str, head: str, body: List[str], source_rule=None):
        """Add a defeasible rule with normalized source."""
        source = DeLPRuleSource.from_ast(source_rule) if source_rule else None
        self.defeasible_rules.append(DeLPRule(
            id=rule_id,
            head=head,
            body=body,
            source=source
        ))

    def add_superiority(self, superior_id: str, inferior_id: str):
        """Add superiority relation: superior_id > inferior_id"""
        self.superiority.append({
            'superior': superior_id,
            'inferior': inferior_id
        })

    def get_next_rule_id(self, prefix: str = "r") -> str:
        """Generate unique rule ID"""
        self.rule_counter += 1
        return f"{prefix}{self.rule_counter}"

    def to_prolog_string(self) -> str:
        """Convert to Prolog/DeLP string representation"""
        lines = []

        # Dynamic predicates declaration
        lines.append("% ============ DYNAMIC PREDICATES ============")
        lines.append(":- dynamic is_like/2.")
        lines.append(":- dynamic has/2.")
        lines.append(":- dynamic has_property/3.")  # has_property(Parent, Child, Cardinality)
        lines.append(":- dynamic rule_info/4.")
        lines.append(":- dynamic sup/2.")
        lines.append(":- dynamic pattern/2.")
        # Quantified has predicates
        lines.append(":- dynamic has_any_like/2.")
        lines.append(":- dynamic has_any_eq/2.")
        lines.append(":- dynamic has_any_neq/2.")
        lines.append(":- dynamic has_all_like/2.")
        lines.append(":- dynamic has_all_eq/2.")
        lines.append(":- dynamic has_all_neq/2.")
        lines.append(":- dynamic has_none_like/2.")
        lines.append(":- dynamic has_none_eq/2.")
        lines.append(":- dynamic has_none_neq/2.")
        # Length where predicates
        lines.append(":- dynamic length_where_is_eq/4.")
        lines.append(":- dynamic length_where_is_gt/4.")
        lines.append(":- dynamic length_where_is_lt/4.")
        lines.append(":- dynamic length_where_is_gte/4.")
        lines.append(":- dynamic length_where_is_lte/4.")
        lines.append(":- dynamic length_where_like_eq/4.")
        lines.append(":- dynamic length_where_like_gt/4.")
        lines.append(":- dynamic length_where_like_lt/4.")
        lines.append(":- dynamic length_where_like_gte/4.")
        lines.append(":- dynamic length_where_like_lte/4.")

        # Make all variable predicates dynamic
        for var_name in self.declarations.keys():
            lines.append(f":- dynamic {var_name}/1.")

        # Discontiguous declarations for variables with rules
        # (needed because strict and defeasible rules are in separate sections)
        variables_with_rules = set()
        for rule in self.strict_rules:
            if rule.source:
                variables_with_rules.add(rule.source.head_variable)
        for rule in self.defeasible_rules:
            if rule.source:
                variables_with_rules.add(rule.source.head_variable)

        if variables_with_rules:
            for var_name in sorted(variables_with_rules):
                lines.append(f":- discontiguous {var_name}/1.")

        lines.append("")

        lines.append("% ============ DECLARATIONS ============")
        for var_name, var_decl in self.declarations.items():
            lines.append(f"% Variable: {var_name}")
            if var_decl.description:
                lines.append(f"%   {var_decl.description}")

        lines.append("")
        lines.append("% ============ SEMANTIC CATEGORIES ============")
        for cat_name, cat in self.semantic_categories.items():
            lines.append(f"% Category: {cat_name}")
            if cat.description:
                lines.append(f"%   {cat.description}")
            for pattern in cat.patterns:
                normalized = normalize_prolog_atom(pattern)
                lines.append(f"pattern({cat_name}, {normalized}).")

        lines.append("")
        lines.append("% ============ HAS RELATIONSHIPS ============")
        for key, has_decl in self.has_relationships.items():
            cardinality = "list" if has_decl.is_list else "single"
            lines.append(f"% {has_decl.parent} has a {'list of ' if has_decl.is_list else ''}{has_decl.child}")
            if has_decl.description:
                lines.append(f"%   {has_decl.description}")
            lines.append(f"has_property({has_decl.parent}, {has_decl.child}, {cardinality}).")

        lines.append("")
        lines.append("% ============ FACTS ============")
        for fact in self.facts:
            lines.append(f"{fact}.")

        lines.append("")
        lines.append("% ============ STRICT RULES ============")
        for rule in self.strict_rules:
            body_str = ", ".join(rule.body) if rule.body else "true"
            lines.append(f"% Rule: {rule.id}")
            lines.append(f"{rule.head} :- {body_str}.")

        lines.append("")
        lines.append("% ============ DEFEASIBLE RULES ============")
        lines.append("% Note: Using standard Prolog (:-)  - defeasibility handled via superiority")
        for rule in self.defeasible_rules:
            body_str = ", ".join(rule.body) if rule.body else "true"
            lines.append(f"% Rule: {rule.id} (defeasible)")
            lines.append(f"{rule.head} :- {body_str}.")

        lines.append("")
        lines.append("% ============ RULE METADATA (for DeLP meta-interpreter) ============")
        lines.append("% rule_info(RuleId, Head, Type, BodyList)")

        for rule in self.strict_rules:
            body_list_str = self._format_body_list(rule.body)
            lines.append(f"rule_info({rule.id}, {rule.head}, strict, {body_list_str}).")

        for rule in self.defeasible_rules:
            body_list_str = self._format_body_list(rule.body)
            lines.append(f"rule_info({rule.id}, {rule.head}, defeasible, {body_list_str}).")

        if self.superiority:
            lines.append("")
            lines.append("% ============ SUPERIORITY RELATIONS ============")
            for sup in self.superiority:
                lines.append(f"sup({sup['superior']}, {sup['inferior']}).")

        return "\n".join(lines)

    def _format_body_list(self, body: List[str]) -> str:
        """Format body list for Prolog list syntax"""
        if not body:
            return "[]"
        # Escape and format each body element
        formatted = [b for b in body]
        return "[" + ", ".join(formatted) + "]"
