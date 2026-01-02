"""
Prolog Backend Generator.

Generates Prolog/DeLP code from the canonical FOL IR.
This ensures Prolog is generated from verified FOL, not directly from AST.

The generated Prolog uses atom-based representation for compatibility
with the existing DeLP meta-interpreter.
"""

from typing import List, Dict, Any, Optional, Set
from datetime import datetime

from ..types import (
    FOLFormula,
    FOLPredicate,
    FOLEquals,
    FOLImplies,
    FOLAnd,
    FOLOr,
    FOLNot,
    FOLForall,
    FOLExists,
    FOLVar,
    FOLConstant,
    FOLFunctionApp,
    FOLTerm,
    FOLSort,
)
from ..canto_fol import CantoFOL, CantoRule, RuleType


def normalize_prolog_atom(value: Any) -> str:
    """
    Normalize a value to a Prolog atom.

    - Strings with spaces or special chars get quoted
    - Simple identifiers stay unquoted
    - Booleans become true/false atoms
    - Numbers stay as-is
    """
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return str(value)
    if value is None:
        return "null"

    s = str(value)

    # Check if it's a simple atom (lowercase, alphanumeric, underscores)
    if s and s[0].islower() and all(c.isalnum() or c == '_' for c in s):
        return s

    # Otherwise quote it
    escaped = s.replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"


class PrologBackend:
    """
    Generates Prolog code from Canto FOL IR.

    This replaces the direct AST → Prolog translation,
    ensuring Prolog is generated from the verified FOL IR.

    The output format uses atom-based representation for compatibility
    with the existing DeLP meta-interpreter and runtime.
    """

    def __init__(self, fol: CantoFOL):
        self.fol = fol

    def generate(self) -> str:
        """
        Generate complete Prolog program.

        Returns:
            Prolog source code as string
        """
        lines = []

        # Dynamic predicates declaration
        lines.append("% ============ DYNAMIC PREDICATES ============")
        lines.extend(self._generate_dynamic_declarations())
        lines.append("")

        # Variable declarations (as comments for documentation)
        lines.append("% ============ DECLARATIONS ============")
        for name, var in self.fol.variables.items():
            lines.append(f"% Variable: {name}")
            if var.description:
                lines.append(f"%   {var.description}")
        lines.append("")

        # Semantic category patterns
        lines.append("% ============ SEMANTIC CATEGORIES ============")
        for name, cat in self.fol.categories.items():
            lines.append(f"% Category: {name}")
            if cat.description:
                lines.append(f"%   {cat.description}")
            for pattern in cat.patterns:
                normalized = normalize_prolog_atom(pattern)
                lines.append(f"pattern({name}, {normalized}).")
        lines.append("")

        # Has relationships (for structured data)
        lines.append("% ============ HAS RELATIONSHIPS ============")
        for key, rel in self.fol.has_relationships.items():
            cardinality = "list" if rel.is_list else "single"
            list_str = "list of " if rel.is_list else ""
            lines.append(f"% {rel.parent} has a {list_str}{rel.child}")
            if rel.description:
                lines.append(f"%   {rel.description}")
            lines.append(f"has_property({rel.parent}, {rel.child}, {cardinality}).")
        lines.append("")

        # Facts
        lines.append("% ============ FACTS ============")
        lines.append("")

        # Strict Rules
        lines.append("% ============ STRICT RULES ============")
        for rule in self.fol.strict_rules:
            lines.extend(self._generate_rule_clause(rule))
        lines.append("")

        # Defeasible Rules
        lines.append("% ============ DEFEASIBLE RULES ============")
        lines.append("% Note: Using standard Prolog (:-)  - defeasibility handled via superiority")
        for rule in self.fol.defeasible_rules:
            lines.extend(self._generate_rule_clause(rule))
        lines.append("")

        # Rule metadata for DeLP meta-interpreter
        lines.append("% ============ RULE METADATA (for DeLP meta-interpreter) ============")
        lines.append("% rule_info(RuleId, Head, Type, BodyList)")
        for rule in self.fol.strict_rules:
            lines.append(self._generate_rule_info(rule, "strict"))
        for rule in self.fol.defeasible_rules:
            lines.append(self._generate_rule_info(rule, "defeasible"))

        # Superiority relations
        if self.fol.superiority:
            lines.append("")
            lines.append("% ============ SUPERIORITY RELATIONS ============")
            for sup in self.fol.superiority:
                lines.append(f"sup({sup.superior}, {sup.inferior}).")

        return "\n".join(lines)

    def _generate_dynamic_declarations(self) -> List[str]:
        """Generate dynamic predicate declarations."""
        lines = []

        # Core predicates
        lines.append(":- dynamic is_like/2.")
        lines.append(":- dynamic has/2.")
        lines.append(":- dynamic has_property/3.")
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

        # Property-based quantified predicates
        lines.append(":- dynamic has_any_prop_eq/4.")
        lines.append(":- dynamic has_any_prop_like/4.")
        lines.append(":- dynamic has_all_prop_eq/4.")
        lines.append(":- dynamic has_all_prop_like/4.")

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

        # Let binding predicates
        lines.append(":- dynamic let_any/4.")
        lines.append(":- dynamic let_all/4.")
        lines.append(":- dynamic let_none/4.")

        # Comparison predicates
        lines.append(":- dynamic compare_gt/2.")
        lines.append(":- dynamic compare_lt/2.")
        lines.append(":- dynamic compare_gte/2.")
        lines.append(":- dynamic compare_lte/2.")

        # Make all variable predicates dynamic
        for var_name in self.fol.variables.keys():
            lines.append(f":- dynamic {var_name}/1.")

        # Discontiguous declarations for variables with rules
        variables_with_rules: Set[str] = set()
        for rule in self.fol.all_rules():
            variables_with_rules.add(rule.head_variable)

        for var_name in sorted(variables_with_rules):
            lines.append(f":- discontiguous {var_name}/1.")

        return lines

    def _generate_rule_clause(self, rule: CantoRule) -> List[str]:
        """Generate Prolog clause for a single rule."""
        lines = []

        # Comment with rule info
        rule_type = "defeasible" if rule.rule_type == RuleType.DEFEASIBLE else ""
        if rule_type:
            lines.append(f"% Rule: {rule.id} ({rule_type})")
        else:
            lines.append(f"% Rule: {rule.id}")

        # Build head: var(value)
        head = f"{rule.head_variable}({self._format_value(rule.head_value)})"

        # Build body
        if rule.conditions:
            body = self._translate_formula(rule.conditions)
            lines.append(f"{head} :- {body}.")
        else:
            lines.append(f"{head} :- true.")

        return lines

    def _generate_rule_info(self, rule: CantoRule, rule_type: str) -> str:
        """Generate rule_info metadata for DeLP meta-interpreter."""
        head = f"{rule.head_variable}({self._format_value(rule.head_value)})"
        body_list = self._get_body_predicates(rule.conditions) if rule.conditions else []
        body_str = "[" + ", ".join(body_list) + "]"
        return f"rule_info({rule.id}, {head}, {rule_type}, {body_str})."

    def _translate_formula(self, formula: FOLFormula) -> str:
        """Translate FOL formula to Prolog."""
        if formula is None:
            return "true"

        if isinstance(formula, FOLPredicate):
            return self._translate_predicate(formula)

        elif isinstance(formula, FOLEquals):
            return self._translate_equals(formula)

        elif isinstance(formula, FOLNot):
            inner = self._translate_formula(formula.formula)
            return f"\\+({inner})"

        elif isinstance(formula, FOLAnd):
            if not formula.conjuncts:
                return "true"
            parts = [self._translate_formula(c) for c in formula.conjuncts]
            return ", ".join(parts)

        elif isinstance(formula, FOLOr):
            if not formula.disjuncts:
                return "fail"
            # Translate OR using De Morgan: A ∨ B ≡ ¬(¬A ∧ ¬B)
            parts = [self._translate_formula(d) for d in formula.disjuncts]
            negated_parts = [f"\\+({p})" for p in parts]
            return f"\\+ (({', '.join(negated_parts)}))"

        elif isinstance(formula, FOLImplies):
            ant = self._translate_formula(formula.antecedent)
            con = self._translate_formula(formula.consequent)
            return f"({ant} -> {con} ; true)"

        elif isinstance(formula, FOLForall):
            # Universal quantification in Prolog using forall/2
            var = formula.variable.name.capitalize()
            body = self._translate_formula(formula.formula)
            return f"forall({var}, {body})"

        elif isinstance(formula, FOLExists):
            # Existential quantification - just use the body
            # (Prolog variables are implicitly existentially quantified)
            return self._translate_formula(formula.formula)

        else:
            return str(formula)

    def _translate_predicate(self, pred: FOLPredicate) -> str:
        """
        Translate predicate to Prolog.

        Uses atom-based representation for compatibility with DeLP meta-interpreter.
        FOL variables (like 'x') are replaced with actual variable names from metadata.
        """
        if pred.name == "true":
            return "true"
        elif pred.name == "false":
            return "fail"
        elif pred.name == "pattern":
            return "true"  # Pattern facts are separate
        elif pred.name == "implicit_sup":
            return "true"  # Implicit superiority is handled by meta-interpreter
        elif pred.name == "not_warranted":
            # Special handling for warrant-based negation
            if pred.args and isinstance(pred.args[0], FOLFormula):
                inner = self._translate_formula(pred.args[0])
                return f"not_warranted({inner})"
            return "not_warranted"

        # Get source variable from metadata (for atom-based output)
        source_var = pred.metadata.get("source_var") if pred.metadata else None

        # Translate arguments, replacing FOL input variable 'x' with source_var
        args = []
        for a in pred.args:
            if isinstance(a, FOLVar) and a.name == "x":
                # Replace abstract 'x' with actual variable name
                if source_var:
                    args.append(source_var)
                # If no source_var, skip the argument (backward compat)
                continue
            args.append(self._translate_term(a))

        if not args:
            return pred.name
        else:
            return f"{pred.name}({', '.join(args)})"

    def _translate_equals(self, formula: FOLEquals) -> str:
        """
        Translate equality to Prolog.

        For var(x) = value, generates var(value) as a predicate call.
        For var1(x) = var2(x), generates var_equals(var1, var2).
        """
        # Variable-to-variable comparison: var1(x) = var2(x) -> var_equals(var1, var2)
        if isinstance(formula.left, FOLFunctionApp) and isinstance(formula.right, FOLFunctionApp):
            left_name = formula.left.function
            right_name = formula.right.function
            return f"var_equals({left_name}, {right_name})"

        right = self._translate_term(formula.right)

        if isinstance(formula.left, FOLFunctionApp):
            # var(x) = value becomes var(value)
            func_name = formula.left.function
            return f"{func_name}({right})"
        else:
            left = self._translate_term(formula.left)
            return f"{left} = {right}"

    def _translate_term(self, term: FOLTerm) -> str:
        """
        Translate FOL term to Prolog.

        Uses atom-based representation - FOL variables become atoms, not Prolog variables.
        """
        if isinstance(term, FOLVar):
            # For atom-based Prolog, the variable name is used as an atom
            # (e.g., 'x' becomes 'x', not 'X')
            # But typically we filter these out in _translate_predicate
            return term.name

        elif isinstance(term, FOLConstant):
            return self._format_constant(term)

        elif isinstance(term, FOLFunctionApp):
            args = [self._translate_term(a) for a in term.args if not (isinstance(a, FOLVar) and a.name == "x")]
            if not args:
                return term.function
            return f"{term.function}({', '.join(args)})"

        else:
            return str(term)

    def _format_constant(self, const: FOLConstant) -> str:
        """Format FOL constant for Prolog based on its sort.

        String values are always quoted for consistency.
        """
        value = const.value

        # Booleans
        if isinstance(value, bool):
            return str(value).lower()

        # Numbers
        if isinstance(value, (int, float)):
            return str(value)

        # Categories are unquoted atoms
        if const.sort == FOLSort.CATEGORY:
            return str(value)

        # String values - always quote for consistency
        if isinstance(value, str):
            escaped = value.replace("\\", "\\\\").replace("'", "\\'")
            return f"'{escaped}'"

        # Other values - use normalize_prolog_atom
        return normalize_prolog_atom(value)

    def _format_value(self, value: Any) -> str:
        """Format value for Prolog (for head values).

        Always quotes string values for consistency.
        """
        if isinstance(value, bool):
            return str(value).lower()
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, list):
            formatted = [self._format_value(v) for v in value]
            return f"[{', '.join(formatted)}]"
        elif isinstance(value, str):
            # Always quote string values for consistency
            escaped = value.replace("\\", "\\\\").replace("'", "\\'")
            return f"'{escaped}'"
        else:
            return normalize_prolog_atom(value)

    def _escape_string(self, s: str) -> str:
        """Escape special characters for Prolog strings."""
        return s.replace("\\", "\\\\").replace("'", "\\'")

    def _get_body_predicates(self, formula: FOLFormula) -> List[str]:
        """Extract body predicates for rule_info metadata.

        Uses the same translation as actual Prolog output for consistency.
        """
        if formula is None:
            return []

        if isinstance(formula, FOLAnd):
            result = []
            for c in formula.conjuncts:
                result.extend(self._get_body_predicates(c))
            return result

        else:
            # Use _translate_formula for consistent output (including De Morgan for OR)
            translated = self._translate_formula(formula)
            return [translated]

    def generate_with_meta_interpreter(self) -> str:
        """
        Generate Prolog with meta-interpreter loading.

        This generates a complete file that loads the DeLP
        meta-interpreter and includes all program facts.
        """
        lines = []

        # Load meta-interpreter
        lines.append(":- use_module(library(janus)).")
        lines.append("")
        lines.append("% Load DeLP meta-interpreter")
        lines.append(":- ensure_loaded('delp_meta.pl').")
        lines.append("")

        # Add generated program
        lines.append(self.generate())

        return "\n".join(lines)
