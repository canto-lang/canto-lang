"""
Translator from Canto DSL AST to DeLP (Defeasible Logic Programming)
"""

from typing import List, Dict, Set, Union
from ..ast_nodes import VariableDeclaration, SemanticCategory, HasDeclaration, Rule, Condition
from ..ast_nodes.rules import RulePriority, OverrideTarget
from ..delp.program import DeLPProgram
from ..delp.models import DeLPRule, DeLPRuleSource, DeLPDeclaration


class DeLPTranslator:
    """
    Translates Canto DSL AST to DeLP program
    """

    def __init__(self):
        self.program = DeLPProgram()
        self.rule_map: Dict[str, List[str]] = {}  # Maps head variable to rule IDs
        self.rule_index: Dict[str, Dict] = {}  # Maps rule ID to rule dict

    def _collect_declarations(self, ast: List[Union[VariableDeclaration, SemanticCategory, HasDeclaration, Rule]], parent: str = None, inherited_source: str = None):
        """
        Recursively collect all declarations, including nested 'with' block children.
        Also generates has_property relationships for 'with' blocks.

        Args:
            ast: List of AST nodes to process
            parent: Parent variable name (for nested 'with' blocks)
            inherited_source: Source variable inherited from parent (for 'from ?source' clause)
        """
        for node in ast:
            if isinstance(node, VariableDeclaration):
                self.program.declarations[node.name] = DeLPDeclaration.from_ast(node)

                # If this is inside a 'with' block, create has_property relationship
                if parent:
                    key = f"{parent}_{node.name}"
                    self.program.has_relationships[key] = HasDeclaration(
                        parent=parent,
                        child=node.name,
                        is_list=False,  # 'with' blocks define single properties
                        description=node.description
                    )

                # Determine source to pass to children (node's own source or inherited)
                child_source = node.source if node.source else inherited_source

                # Recursively collect children from 'with' blocks, passing current node as parent
                if hasattr(node, 'children') and node.children:
                    self._collect_declarations(node.children, parent=node.name, inherited_source=child_source)

            elif isinstance(node, SemanticCategory):
                self.program.semantic_categories[node.name] = node
            elif isinstance(node, HasDeclaration):
                # If this HasDeclaration is inside a 'with' block, create relationship
                # from the outer parent to this declaration's parent variable
                if parent:
                    # Create has_property from outer parent to this declaration's parent
                    # e.g., entities -> companies (when ?companies has a list of ?company)
                    outer_key = f"{parent}_{node.parent}"
                    if outer_key not in self.program.has_relationships:
                        self.program.has_relationships[outer_key] = HasDeclaration(
                            parent=parent,
                            child=node.parent,
                            is_list=False,  # The container is a single property
                            description=None
                        )
                    # Also declare the inner parent as a variable
                    if node.parent not in self.program.declarations:
                        self.program.declarations[node.parent] = DeLPDeclaration(
                            name=node.parent,
                            description=None
                        )

                # If HasDeclaration doesn't have its own source, use inherited source
                if not node.source and inherited_source:
                    node.source = inherited_source

                # Store the actual has relationship (e.g., companies -> company)
                key = f"{node.parent}_{node.child}"
                self.program.has_relationships[key] = node
                # Recursively collect children from 'with' blocks
                if hasattr(node, 'children') and node.children:
                    self._collect_declarations(node.children, parent=node.child, inherited_source=inherited_source)

    def translate(self, ast: List[Union[VariableDeclaration, SemanticCategory, HasDeclaration, Rule]]) -> DeLPProgram:
        """
        Translate AST to DeLP program
        """
        # First pass: collect declarations (including nested 'with' block children)
        self._collect_declarations(ast)

        # Second pass: translate rules
        for node in ast:
            if isinstance(node, Rule):
                self._translate_rule(node)

        # Third pass: auto-detect variables from rule heads
        self._auto_declare_variables_from_rules()

        # Fourth pass: handle cross-rule OVERRIDES (for rules added after the OVERRIDES rule)
        self._finalize_superiority()

        # Note: Cycle detection is handled in Prolog reasoning, not here.
        # The Prolog meta-interpreter only checks superiority between
        # rules with contradicting conclusions (different values).
        # Cycles between same-value rules are harmless and never traversed.

        return self.program

    def _auto_declare_variables_from_rules(self):
        """
        Auto-declare variables that are assigned by rules but not explicitly declared.
        """
        for rule in self.program.strict_rules + self.program.defeasible_rules:
            if rule.source:
                var_name = rule.source.head_variable
                if var_name not in self.program.declarations:
                    self.program.declarations[var_name] = DeLPDeclaration(
                        name=var_name,
                        description=None,
                        values_from=None
                    )

    def _translate_rule(self, rule: Rule):
        """
        Translate a single rule to DeLP.

        OR conditions are handled inline using De Morgan's law (see _translate_condition),
        so no special expansion is needed here.
        """
        self._translate_single_rule(rule)

    def _translate_single_rule(self, rule: Rule):
        """Translate a single rule to DeLP predicates"""
        # Generate rule ID
        rule_id = self.program.get_next_rule_id(f"{rule.head_variable}_")

        # Build head predicate
        head = self._build_head(rule)

        # Build body (conditions)
        body = []
        for condition in rule.conditions:
            body.extend(self._translate_condition(condition))

        # Handle UNLESS (negated conditions)
        for exception in rule.exceptions:
            negated = self._translate_condition_negated(exception)
            body.extend(negated)

        # Handle OVERRIDES (must be done BEFORE adding rule to rule_map)
        self._handle_overrides(rule, rule_id)

        # Add rule based on priority
        if rule.is_strict():
            self.program.add_strict_rule(rule_id, head, body, rule)
        else:
            self.program.add_defeasible_rule(rule_id, head, body, rule)

        # Track rule for this variable (done AFTER handling overrides)
        if rule.head_variable not in self.rule_map:
            self.rule_map[rule.head_variable] = []
        self.rule_map[rule.head_variable].append(rule_id)

    def _build_head(self, rule: Rule) -> str:
        """Build head predicate from rule"""
        var_name = rule.head_variable
        value = self._format_value(rule.head_value)
        return f"{var_name}({value})"

    def _translate_condition(self, condition: Condition) -> List[str]:
        """Translate condition to DeLP predicates (kept symbolic)"""
        if condition.operator == "NOT":
            # NOT condition (Negation as Failure)
            # Semantics: Fails if ANY argument exists for the goal
            if isinstance(condition.right, Condition):
                inner = self._translate_condition(condition.right)
                return [f"\\+({pred})" for pred in inner]
            else:
                return [f"\\+({condition.right})"]

        elif condition.operator == "NOT_WARRANTED":
            # NOT WARRANTED condition (warrant-based negation)
            # Semantics: Fails only if the goal is WARRANTED (has undefeated argument)
            if isinstance(condition.right, Condition):
                inner = self._translate_condition(condition.right)
                return [f"not_warranted({pred})" for pred in inner]
            else:
                return [f"not_warranted({condition.right})"]

        elif condition.operator == "AND":
            # AND: combine predicates from both sides
            left_preds = self._translate_condition(condition.left) if isinstance(condition.left, Condition) else [str(condition.left)]
            right_preds = self._translate_condition(condition.right) if isinstance(condition.right, Condition) else [str(condition.right)]
            return left_preds + right_preds

        elif condition.operator == "OR":
            # Translate OR using De Morgan's law: A or B ≡ ¬(¬A ∧ ¬B)
            #
            # THE PROBLEM: OR expansion creates cycles with "overriding all"
            # ---------------------------------------------------------------
            # Canto DSL allows rules like:
            #   ?result becomes "matched" when ?input is like ?p1 or ?input is like ?p2 overriding all
            #
            # A naive approach would expand this into multiple rules:
            #   - Rule 1a: result=matched when input like p1 (overriding all)
            #   - Rule 1b: result=matched when input like p2 (overriding all)
            #
            # But both sibling rules inherit "overriding all", which means each tries
            # to override ALL other rules for the same variable - including its sibling:
            #   - 1a > 1b (1a overrides all rules for result)
            #   - 1b > 1a (1b overrides all rules for result)
            #
            # This creates a cycle in the superiority graph, which is invalid in DeLP.
            # The user's intent was "this conclusion should override other conclusions",
            # NOT "each OR branch should fight its siblings".
            #
            # THE SOLUTION: De Morgan's law
            # ------------------------------
            # Instead of expanding, we translate OR as: ¬(¬A ∧ ¬B)
            # In Prolog: \+ (\+ A, \+ B) means "it's not the case that both A and B fail"
            # which is logically equivalent to "A succeeds or B succeeds".
            #
            # This keeps OR as a single rule, avoiding sibling cycles while preserving
            # the exact same semantics: the conclusion holds if either branch succeeds.
            left_preds = self._translate_condition(condition.left) if isinstance(condition.left, Condition) else [str(condition.left)]
            right_preds = self._translate_condition(condition.right) if isinstance(condition.right, Condition) else [str(condition.right)]

            # Build: \+ (\+ left, \+ right)
            left_negated = ", ".join([f"\\+({p})" for p in left_preds])
            right_negated = ", ".join([f"\\+({p})" for p in right_preds])
            return [f"\\+ (({left_negated}), ({right_negated}))"]

        elif condition.operator == "MATCHES":
            # Symbolic predicate: matches(var, pattern)
            left = self._format_variable(condition.left)
            right = self._format_variable(condition.right)
            return [f"matches({left}, {right})"]

        elif condition.operator == "HAS":
            # Symbolic predicate: has(var, value)
            left = self._format_variable(condition.left)
            right = self._format_value(condition.right)
            return [f"has({left}, {right})"]

        # Quantified HAS conditions: has any/all/none that is/is like/is not
        elif condition.operator == "HAS_ANY_IS_LIKE":
            left = self._format_variable(condition.left)
            right = self._format_variable(condition.right)
            return [f"has_any_like({left}, {right})"]

        elif condition.operator == "HAS_ANY_IS":
            left = self._format_variable(condition.left)
            right = self._format_value(condition.right)
            return [f"has_any_eq({left}, {right})"]

        elif condition.operator == "HAS_ANY_IS_NOT":
            left = self._format_variable(condition.left)
            right = self._format_value(condition.right)
            return [f"has_any_neq({left}, {right})"]

        elif condition.operator == "HAS_ALL_IS_LIKE":
            left = self._format_variable(condition.left)
            right = self._format_variable(condition.right)
            return [f"has_all_like({left}, {right})"]

        elif condition.operator == "HAS_ALL_IS":
            left = self._format_variable(condition.left)
            right = self._format_value(condition.right)
            return [f"has_all_eq({left}, {right})"]

        elif condition.operator == "HAS_ALL_IS_NOT":
            left = self._format_variable(condition.left)
            right = self._format_value(condition.right)
            return [f"has_all_neq({left}, {right})"]

        elif condition.operator == "HAS_NONE_IS_LIKE":
            left = self._format_variable(condition.left)
            right = self._format_variable(condition.right)
            return [f"has_none_like({left}, {right})"]

        elif condition.operator == "HAS_NONE_IS":
            left = self._format_variable(condition.left)
            right = self._format_value(condition.right)
            return [f"has_none_eq({left}, {right})"]

        elif condition.operator == "HAS_NONE_IS_NOT":
            left = self._format_variable(condition.left)
            right = self._format_value(condition.right)
            return [f"has_none_neq({left}, {right})"]

        # Let binding quantified conditions: let ?var be any/all/none in ?collection where conditions
        elif condition.operator in ("LET_ANY_BOUND", "LET_ALL_BOUND", "LET_NONE_BOUND"):
            return self._translate_let_bound(condition)

        elif condition.operator == "IS_LIKE":
            # Semantic similarity: is_like(var, pattern) - for LLM-interpreted matching
            left = self._format_variable(condition.left)
            right = self._format_variable(condition.right)
            return [f"is_like({left}, {right})"]

        elif condition.operator == "IS":
            # Variable equality/assignment check
            left = self._format_variable(condition.left)
            right = self._format_value(condition.right)
            # If right side is also a variable reference, use equality check
            if self._is_variable_ref(condition.right):
                # Variable-to-variable comparison: check if both have same value
                return [f"var_equals({left}, {right})"]
            else:
                # Variable-to-value check: left(value)
                return [f"{left}({right})"]

        # LENGTH_WHERE conditions: (length of ?list where ?prop is/is_like value) comparison
        elif condition.operator.startswith("LENGTH_WHERE_"):
            return self._translate_length_where(condition)

        else:
            # Generic predicate
            left = self._format_variable(condition.left)
            right = self._format_value(condition.right)
            return [f"{condition.operator.lower()}({left}, {right})"]

    def _translate_condition_negated(self, condition: Condition) -> List[str]:
        """Translate UNLESS condition (negated)"""
        positive = self._translate_condition(condition)
        return [f"\\+({pred})" for pred in positive]

    def _translate_length_where(self, condition: Condition) -> List[str]:
        """
        Translate LENGTH_WHERE conditions.

        Condition structure:
        - operator: LENGTH_WHERE_IS_EQ, LENGTH_WHERE_IS_GT, LENGTH_WHERE_LIKE_EQ, etc.
        - left: {"list": list_ref, "property": prop_var, "condition": "IS"|"IS_LIKE", "value": operand}
        - right: comparison value (number)

        Generates: length_where_<type>_<cmp>(list, property, value, comparison_value)
        """
        # Parse operator: LENGTH_WHERE_IS_EQ -> (is, eq), LENGTH_WHERE_LIKE_GT -> (like, gt)
        parts = condition.operator.split("_")
        # parts = ["LENGTH", "WHERE", "IS"|"LIKE", "EQ"|"GT"|"LT"|"GTE"|"LTE"]
        cond_type = parts[2].lower()  # "is" or "like"
        cmp_type = parts[3].lower()   # "eq", "gt", "lt", "gte", "lte"

        # Extract components from left dict
        left_data = condition.left
        list_ref = left_data["list"]
        prop_var = left_data["property"]
        filter_value = left_data["value"]
        cmp_value = condition.right

        # Format the list reference
        list_str = self._format_variable(list_ref)

        # Format property variable
        prop_str = self._format_variable(prop_var)

        # Format filter value (the value being compared in the where clause)
        filter_str = self._format_value(filter_value)

        # Format comparison value (the number being compared to length)
        cmp_str = self._format_value(cmp_value)

        # Generate predicate: length_where_is_eq(list, prop, filter_val, cmp_val)
        predicate = f"length_where_{cond_type}_{cmp_type}({list_str}, {prop_str}, {filter_str}, {cmp_str})"
        return [predicate]

    def _translate_let_bound(self, condition: Condition) -> List[str]:
        """
        Translate let binding quantified conditions with explicit binding and compound conditions.

        Example DSL:
            let ?link be any in ?chain where
                ?speaker of ?link is ?target
                and ?speaker_is_truthful of ?link is true

        Condition structure:
        - operator: LET_ANY_BOUND, LET_ALL_BOUND, or LET_NONE_BOUND
        - left: collection variable (e.g., chain_of_puzzle)
        - right: {
            "binding": "?link",
            "conditions": [
                {"condition_type": "IS", "left": ("?speaker", "?link"), "right": "?target"},
                {"condition_type": "IS", "left": ("?speaker_is_truthful", "?link"), "right": true, "logic_op": "AND"}
            ]
          }

        Generates: let_any_bound(collection, binding, [cond1, cond2, ...])
        Each condition is translated to a Prolog term.
        """
        # Determine quantifier type: any, all, none
        if condition.operator == "LET_ANY_BOUND":
            quantifier = "any"
        elif condition.operator == "LET_ALL_BOUND":
            quantifier = "all"
        else:  # LET_NONE_BOUND
            quantifier = "none"

        # Format collection variable
        collection = self._format_variable(condition.left)

        # Get binding and conditions from right side
        right_data = condition.right
        binding = self._strip_prefix(right_data["binding"])
        bound_conditions = right_data["conditions"]

        # Translate each bound condition to a Prolog term
        prolog_conditions = []
        for cond in bound_conditions:
            cond_type = cond["condition_type"]
            left = self._format_variable(cond["left"])
            right = self._format_value(cond["right"])

            if cond_type == "IS":
                # Check if right is a variable reference
                if self._is_variable_ref(cond["right"]):
                    prolog_conditions.append(f"var_equals({left}, {right})")
                else:
                    prolog_conditions.append(f"{left}({right})")
            elif cond_type == "IS_LIKE":
                prolog_conditions.append(f"is_like({left}, {right})")

        # Generate the let binding predicate
        # Format: let_<quantifier>_bound(collection, binding, [cond1, cond2, ...])
        conditions_list = ", ".join(prolog_conditions)
        predicate = f"let_{quantifier}_bound({collection}, {binding}, [{conditions_list}])"
        return [predicate]

    def _format_variable(self, var) -> str:
        """Format variable name (remove ? prefix if present)

        Handles both simple variables (?var) and qualified variables (?child of ?parent).
        For qualified variables, returns the format: child_of_parent
        Also handles nested qualified variables: ?a of (?b of ?c) -> a_of_b_of_c
        """
        if isinstance(var, dict) and 'child' in var and 'parent' in var:
            # Qualified variable: ?child of ?parent
            child = self._strip_prefix(var['child'])
            # Parent might be nested (another dict or tuple)
            parent = self._format_variable(var['parent'])
            return f"{child}_of_{parent}"
        elif isinstance(var, tuple) and len(var) == 2:
            # Tuple form of qualified variable: (?child, ?parent)
            child = self._strip_prefix(var[0]) if isinstance(var[0], str) else self._format_variable(var[0])
            parent = self._format_variable(var[1]) if not isinstance(var[1], str) else self._strip_prefix(var[1])
            return f"{child}_of_{parent}"
        elif isinstance(var, str):
            return self._strip_prefix(var)
        return str(var)

    def _strip_prefix(self, var: str) -> str:
        """Strip ? prefix from variable name"""
        if isinstance(var, str) and var.startswith('?'):
            return var[1:]
        return str(var)

    def _format_value(self, value) -> str:
        """Format value for Prolog"""
        if isinstance(value, bool):
            return str(value).lower()
        elif isinstance(value, dict) and 'child' in value and 'parent' in value:
            # Qualified variable reference: ?child of ?parent
            return self._format_variable(value)
        elif isinstance(value, tuple) and len(value) == 2:
            # Tuple form of qualified variable (possibly nested)
            return self._format_variable(value)
        elif isinstance(value, str):
            # If it's a variable reference (? prefix)
            if value.startswith('?'):
                return self._format_variable(value)
            # If it's a string literal
            if value.startswith('"') or value.startswith("'"):
                return value
            # Otherwise, quote it
            return f"'{value}'"
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, list):
            # List of patterns
            formatted = [self._format_value(v) for v in value]
            return f"[{', '.join(formatted)}]"
        else:
            return str(value)

    def _is_variable_ref(self, value) -> bool:
        """Check if value is a variable reference (simple or qualified)"""
        if isinstance(value, dict) and 'child' in value and 'parent' in value:
            return True
        if isinstance(value, tuple) and len(value) == 2:
            return True
        if isinstance(value, str) and value.startswith('?'):
            return True
        return False

    def _handle_overrides(self, rule: Rule, rule_id: str):
        """Handle OVERRIDES clause for rules seen so far"""
        if rule.override_target == OverrideTarget.NONE:
            return

        var_name = rule.head_variable

        if rule.override_target == OverrideTarget.ALL:
            # This rule overrides ALL other rules for the same variable (so far)
            if var_name in self.rule_map:
                for other_rule_id in self.rule_map[var_name]:
                    if other_rule_id != rule_id:
                        self.program.add_superiority(rule_id, other_rule_id)

        elif rule.override_target == OverrideTarget.NORMAL:
            # This rule overrides only NORMAL (defeasible) rules
            # We need to check which rules are defeasible
            for defeasible in self.program.defeasible_rules:
                if defeasible.source and defeasible.source.head_variable == var_name:
                    if defeasible.id != rule_id:
                        self.program.add_superiority(rule_id, defeasible.id)

    def _finalize_superiority(self):
        """
        Finalize superiority relations after all rules are processed.
        Handles cases where OVERRIDES rule is defined before the rules it overrides.
        """
        # Check all strict rules with OVERRIDES ALL
        for strict_rule in self.program.strict_rules:
            source_rule = strict_rule.source
            if source_rule and source_rule.override_target == OverrideTarget.ALL.value:
                rule_id = strict_rule.id
                var_name = source_rule.head_variable

                # Override ALL rules for this variable
                if var_name in self.rule_map:
                    for other_rule_id in self.rule_map[var_name]:
                        if other_rule_id != rule_id:
                            # Check if not already added
                            already_exists = any(
                                sup['superior'] == rule_id and sup['inferior'] == other_rule_id
                                for sup in self.program.superiority
                            )
                            if not already_exists:
                                self.program.add_superiority(rule_id, other_rule_id)

        # Check all strict rules with OVERRIDES NORMAL
        for strict_rule in self.program.strict_rules:
            source_rule = strict_rule.source
            if source_rule and source_rule.override_target == OverrideTarget.NORMAL.value:
                rule_id = strict_rule.id
                var_name = source_rule.head_variable

                # Override only DEFEASIBLE rules for this variable
                for defeasible_rule in self.program.defeasible_rules:
                    def_source = defeasible_rule.source
                    if def_source and def_source.head_variable == var_name:
                        other_rule_id = defeasible_rule.id
                        if other_rule_id != rule_id:
                            already_exists = any(
                                sup['superior'] == rule_id and sup['inferior'] == other_rule_id
                                for sup in self.program.superiority
                            )
                            if not already_exists:
                                self.program.add_superiority(rule_id, other_rule_id)



def translate_to_delp(ast: List[Union[VariableDeclaration, SemanticCategory, Rule]]) -> DeLPProgram:
    """
    Convenience function to translate AST to DeLP
    """
    translator = DeLPTranslator()
    return translator.translate(ast)
