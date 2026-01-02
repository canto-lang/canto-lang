"""
Translates Canto DSL AST to FOL IR.

This is a deterministic translation - no LLM involved.
The resulting FOL IR is the canonical representation of the program.
"""

from typing import List, Dict, Any, Union, Optional

from .types import (
    FOLFormula,
    FOLPredicate,
    FOLEquals,
    FOLAnd,
    FOLOr,
    FOLNot,
    FOLVar,
    FOLConstant,
    FOLFunctionApp,
    FOLSort,
    FOLVariable,
    make_and,
    make_or,
)
from .canto_fol import (
    CantoFOL,
    CantoVariable,
    CantoCategory,
    CantoRule,
    CantoSuperiority,
    CantoHasRelationship,
    RuleType,
    ValueType,
)
from ..ast_nodes import (
    VariableDeclaration,
    SemanticCategory,
    HasDeclaration,
    Rule,
    Condition,
)
from ..ast_nodes.rules import RulePriority, OverrideTarget


class ASTToFOLTranslator:
    """
    Translates Canto AST to FOL IR.

    This is the primary entry point for creating the canonical FOL
    representation from parsed Canto DSL.
    """

    def __init__(self):
        self.fol = CantoFOL()
        self.rule_counter: Dict[str, int] = {}
        self.input_var = FOLVariable("x", FOLSort.INPUT)

    def translate(self, ast: List[Any], source_file: Optional[str] = None) -> CantoFOL:
        """
        Translate complete AST to FOL IR.

        Args:
            ast: List of AST nodes (declarations, categories, rules)
            source_file: Optional source file path for metadata

        Returns:
            CantoFOL: The canonical FOL representation
        """
        self.fol = CantoFOL(source_file=source_file)
        self.rule_counter = {}

        # First pass: collect declarations (including nested)
        self._collect_declarations(ast)

        # Second pass: translate rules
        self._translate_rules(ast)

        # Third pass: compute superiority relations
        self._compute_superiority()

        return self.fol

    def _collect_declarations(
        self,
        ast: List[Any],
        parent: Optional[str] = None,
        inherited_source: Optional[str] = None
    ):
        """
        Collect variable and category declarations, including nested ones.

        Args:
            ast: List of AST nodes
            parent: Parent variable name (for nested 'with' blocks)
            inherited_source: Source variable inherited from parent
        """
        for node in ast:
            if isinstance(node, VariableDeclaration):
                # Determine value type from values_from
                value_type = self._infer_value_type(node.values_from)
                # Normalize possible_values based on type
                possible_values = self._normalize_values(node.values_from, value_type)

                self.fol.variables[node.name] = CantoVariable(
                    name=node.name,
                    value_type=value_type,
                    description=node.description,
                    possible_values=possible_values,
                    source=node.source or inherited_source
                )

                # If inside a 'with' block, create has relationship
                if parent:
                    key = f"{parent}_{node.name}"
                    self.fol.has_relationships[key] = CantoHasRelationship(
                        parent=parent,
                        child=node.name,
                        is_list=False,
                        description=node.description
                    )

                # Determine source for children
                child_source = node.source if node.source else inherited_source

                # Recurse into children
                if hasattr(node, 'children') and node.children:
                    self._collect_declarations(
                        node.children,
                        parent=node.name,
                        inherited_source=child_source
                    )

            elif isinstance(node, SemanticCategory):
                self.fol.categories[node.name] = CantoCategory(
                    name=node.name,
                    patterns=node.patterns,
                    description=node.description
                )

            elif isinstance(node, HasDeclaration):
                # Store the has relationship
                key = f"{node.parent}_{node.child}"
                self.fol.has_relationships[key] = CantoHasRelationship(
                    parent=node.parent,
                    child=node.child,
                    is_list=node.is_list,
                    description=node.description
                )

                # Declare child variable if not exists
                if node.child not in self.fol.variables:
                    self.fol.variables[node.child] = CantoVariable(
                        name=node.child,
                        description=node.description
                    )

                # Recurse into children
                if hasattr(node, 'children') and node.children:
                    self._collect_declarations(
                        node.children,
                        parent=node.child,
                        inherited_source=node.source or inherited_source
                    )

    def _translate_rules(self, ast: List[Any]):
        """Translate all rules to FOL."""
        # Pre-pass: auto-declare all undeclared variables from rules
        # This ensures correct types are known before translating conditions
        for node in ast:
            if isinstance(node, Rule):
                var_name = node.head_variable.lstrip('?')
                if var_name not in self.fol.variables:
                    value_type = self._infer_value_type_from_value(node.head_value)
                    self.fol.variables[var_name] = CantoVariable(
                        name=var_name,
                        value_type=value_type,
                    )

        # Main pass: translate all rules
        for node in ast:
            if isinstance(node, Rule):
                fol_rule = self._translate_rule(node)
                if fol_rule.rule_type == RuleType.STRICT:
                    self.fol.strict_rules.append(fol_rule)
                else:
                    self.fol.defeasible_rules.append(fol_rule)

    def _translate_rule(self, rule: Rule) -> CantoRule:
        """Translate a single rule to FOL."""
        # Generate unique rule ID
        rule_id = self._generate_rule_id(rule.head_variable)

        # Translate 'when' conditions to FOL formula
        conditions_fol = self._translate_conditions(rule.conditions)

        # Handle 'unless' (exceptions) as negated conditions
        if rule.exceptions:
            exception_fol = self._translate_conditions(rule.exceptions)
            if exception_fol:
                if conditions_fol:
                    conditions_fol = FOLAnd([conditions_fol, FOLNot(exception_fol)])
                else:
                    conditions_fol = FOLNot(exception_fol)

        return CantoRule(
            id=rule_id,
            head_variable=rule.head_variable,
            head_value=rule.head_value,
            conditions=conditions_fol,
            rule_type=RuleType.DEFEASIBLE if rule.is_defeasible() else RuleType.STRICT,
            original_ast=rule
        )

    def _translate_conditions(
        self,
        conditions: Optional[List[Condition]]
    ) -> Optional[FOLFormula]:
        """Translate list of conditions to FOL formula (conjunction)."""
        if not conditions:
            return None

        fol_conditions = []
        for cond in conditions:
            fol_cond = self._translate_condition(cond)
            if fol_cond:
                fol_conditions.append(fol_cond)

        if not fol_conditions:
            return None
        return make_and(fol_conditions)

    def _translate_condition(self, cond: Condition) -> Optional[FOLFormula]:
        """Translate a single condition to FOL."""
        x = FOLVar(self.input_var.name)

        if cond.operator == "IS_LIKE":
            # is_like(x, category) with metadata for Prolog generation
            left = self._normalize_var(cond.left)
            category = self._normalize_var(cond.right)
            return FOLPredicate(
                "is_like",
                [x, FOLConstant(category, FOLSort.CATEGORY)],
                metadata={"source_var": left}
            )

        elif cond.operator == "MATCHES":
            # matches(x, category) - alias for is_like
            left = self._normalize_var(cond.left)
            category = self._normalize_var(cond.right)
            return FOLPredicate(
                "matches",
                [x, FOLConstant(category, FOLSort.CATEGORY)],
                metadata={"source_var": left}
            )

        elif cond.operator == "IS":
            # var(x) = value
            left = self._normalize_var(cond.left)
            right = self._normalize_value(cond.right)

            # Check if right is a variable reference
            if self._is_variable_ref(cond.right):
                # Variable comparison: var1(x) = var2(x)
                right_var = self._normalize_var(cond.right)
                left_app = FOLFunctionApp(left, [x])
                right_app = FOLFunctionApp(right_var, [x])
                return FOLEquals(left_app, right_app)
            else:
                # Value comparison: var(x) = value
                var_app = FOLFunctionApp(left, [x])
                # Check if the variable is boolean
                var_decl = self.fol.variables.get(left)
                if var_decl and var_decl.is_bool():
                    # Convert to boolean constant
                    if isinstance(right, bool):
                        return FOLEquals(var_app, FOLConstant(right, FOLSort.BOOL))
                    elif isinstance(right, str) and right.lower() in ("true", "false"):
                        return FOLEquals(var_app, FOLConstant(right.lower() == "true", FOLSort.BOOL))
                # Use _make_constant to preserve native types (int, float, etc.)
                return FOLEquals(var_app, self._make_constant(right))

        elif cond.operator == "HAS":
            # has(x, value) with metadata for Prolog generation
            left = self._normalize_var(cond.left)
            right = self._normalize_value(cond.right)
            return FOLPredicate(
                "has",
                [x, FOLConstant(str(right), FOLSort.VALUE)],
                metadata={"source_var": left}
            )

        elif cond.operator == "NOT":
            # Negation as failure
            inner = self._translate_condition(cond.right)
            if inner:
                return FOLNot(inner)
            return None

        elif cond.operator == "NOT_WARRANTED":
            # Warrant-based negation
            inner = self._translate_condition(cond.right)
            if inner:
                return FOLPredicate("not_warranted", [inner])
            return None

        elif cond.operator == "AND":
            left = self._translate_condition(cond.left)
            right = self._translate_condition(cond.right)
            parts = [p for p in [left, right] if p]
            return make_and(parts) if parts else None

        elif cond.operator == "OR":
            left = self._translate_condition(cond.left)
            right = self._translate_condition(cond.right)
            parts = [p for p in [left, right] if p]
            return make_or(parts) if parts else None

        # Quantified HAS conditions - use FOLVar(x) for Z3, metadata for Prolog
        elif cond.operator == "HAS_ANY_IS_LIKE":
            left = self._normalize_var(cond.left)
            category = self._normalize_var(cond.right)
            return FOLPredicate(
                "has_any_like",
                [x, FOLConstant(category, FOLSort.CATEGORY)],
                metadata={"source_var": left}
            )

        elif cond.operator == "HAS_ALL_IS_LIKE":
            left = self._normalize_var(cond.left)
            category = self._normalize_var(cond.right)
            return FOLPredicate(
                "has_all_like",
                [x, FOLConstant(category, FOLSort.CATEGORY)],
                metadata={"source_var": left}
            )

        elif cond.operator == "HAS_NONE_IS_LIKE":
            left = self._normalize_var(cond.left)
            category = self._normalize_var(cond.right)
            return FOLNot(
                FOLPredicate(
                    "has_any_like",
                    [x, FOLConstant(category, FOLSort.CATEGORY)],
                    metadata={"source_var": left}
                )
            )

        elif cond.operator == "HAS_ANY_IS":
            left = self._normalize_var(cond.left)
            value = self._normalize_value(cond.right)
            return FOLPredicate(
                "has_any_eq",
                [x, self._make_constant(value)],
                metadata={"source_var": left}
            )

        elif cond.operator == "HAS_ALL_IS":
            left = self._normalize_var(cond.left)
            value = self._normalize_value(cond.right)
            return FOLPredicate(
                "has_all_eq",
                [x, self._make_constant(value)],
                metadata={"source_var": left}
            )

        elif cond.operator == "HAS_NONE_IS":
            left = self._normalize_var(cond.left)
            value = self._normalize_value(cond.right)
            return FOLNot(
                FOLPredicate(
                    "has_any_eq",
                    [x, self._make_constant(value)],
                    metadata={"source_var": left}
                )
            )

        # HAS_*_IS_NOT conditions (negated equality)
        elif cond.operator == "HAS_ANY_IS_NOT":
            left = self._normalize_var(cond.left)
            value = self._normalize_value(cond.right)
            return FOLPredicate(
                "has_any_neq",
                [x, self._make_constant(value)],
                metadata={"source_var": left}
            )

        elif cond.operator == "HAS_ALL_IS_NOT":
            left = self._normalize_var(cond.left)
            value = self._normalize_value(cond.right)
            return FOLPredicate(
                "has_all_neq",
                [x, self._make_constant(value)],
                metadata={"source_var": left}
            )

        elif cond.operator == "HAS_NONE_IS_NOT":
            left = self._normalize_var(cond.left)
            value = self._normalize_value(cond.right)
            return FOLNot(
                FOLPredicate(
                    "has_any_neq",
                    [x, self._make_constant(value)],
                    metadata={"source_var": left}
                )
            )

        # Property-based quantified conditions: HAS_ANY_PROPERTY_IS, HAS_ANY_PROPERTY_IS_LIKE, etc.
        elif cond.operator.startswith("HAS_") and "PROPERTY_IS" in cond.operator:
            return self._translate_property_quantified(cond)

        # Comparison conditions: COMPARE_GT, COMPARE_LT, COMPARE_GTE, COMPARE_LTE
        elif cond.operator.startswith("COMPARE_"):
            return self._translate_comparison(cond)

        # Let binding conditions
        elif cond.operator in ("LET_ANY_BOUND", "LET_ALL_BOUND", "LET_NONE_BOUND"):
            return self._translate_let_bound(cond)

        # Length conditions
        elif cond.operator.startswith("LENGTH_WHERE_"):
            return self._translate_length_where(cond)

        else:
            # Generic predicate for unknown operators
            return FOLPredicate(
                cond.operator.lower(),
                [x, FOLConstant(str(cond.right), FOLSort.VALUE)]
            )

    def _translate_let_bound(self, cond: Condition) -> FOLFormula:
        """Translate let binding quantified conditions."""
        x = FOLVar(self.input_var.name)

        # Determine quantifier type
        if cond.operator == "LET_ANY_BOUND":
            quantifier = "let_any"
        elif cond.operator == "LET_ALL_BOUND":
            quantifier = "let_all"
        else:
            quantifier = "let_none"

        # Get collection and binding info
        collection = self._normalize_var(cond.left)
        right_data = cond.right
        binding = self._normalize_var(right_data.get("binding", ""))
        bound_conditions = right_data.get("conditions", [])

        # Build condition list
        cond_terms = []
        for bc in bound_conditions:
            cond_type = bc.get("condition_type", "")
            left = self._normalize_var(bc.get("left", ""))
            right = bc.get("right", "")

            cond_terms.append(
                FOLConstant(f"{cond_type}({left}, {right})", FOLSort.VALUE)
            )

        return FOLPredicate(
            quantifier,
            [
                x,
                FOLConstant(collection, FOLSort.VALUE),
                FOLConstant(binding, FOLSort.VALUE),
                FOLConstant(str(cond_terms), FOLSort.VALUE)
            ]
        )

    def _translate_length_where(self, cond: Condition) -> FOLFormula:
        """Translate LENGTH_WHERE conditions."""
        x = FOLVar(self.input_var.name)

        # Parse operator: LENGTH_WHERE_IS_EQ, LENGTH_WHERE_LIKE_GT, etc.
        parts = cond.operator.split("_")
        cond_type = parts[2].lower()  # "is" or "like"
        cmp_type = parts[3].lower()   # "eq", "gt", "lt", etc.

        left_data = cond.left
        list_ref = self._normalize_var(left_data.get("list", ""))
        prop_var = self._normalize_var(left_data.get("property", ""))
        filter_value = left_data.get("value", "")
        cmp_value = cond.right

        # For length_where_like, filter_value is a category reference
        if cond_type == "like":
            filter_const = FOLConstant(self._normalize_var(filter_value), FOLSort.CATEGORY)
        else:
            filter_const = self._make_constant(filter_value)

        # For length_where, list_ref and prop_var are atom names (unquoted)
        return FOLPredicate(
            f"length_where_{cond_type}_{cmp_type}",
            [
                x,
                FOLConstant(list_ref, FOLSort.CATEGORY),  # atom name
                FOLConstant(prop_var, FOLSort.CATEGORY),  # atom name
                filter_const,
                self._make_constant(cmp_value)
            ]
        )

    def _translate_property_quantified(self, cond: Condition) -> FOLFormula:
        """
        Translate property-based quantified conditions.

        Handles operators like:
        - HAS_ANY_PROPERTY_IS: ?list has any that ?prop is value
        - HAS_ANY_PROPERTY_IS_LIKE: ?list has any that ?prop is like @category
        - HAS_ALL_PROPERTY_IS, HAS_ALL_PROPERTY_IS_LIKE
        - HAS_NONE_PROPERTY_IS, HAS_NONE_PROPERTY_IS_LIKE
        """
        x = FOLVar(self.input_var.name)

        # Parse operator: HAS_ANY_PROPERTY_IS, HAS_ANY_PROPERTY_IS_LIKE, etc.
        parts = cond.operator.split("_")
        quantifier = parts[1].lower()  # "any", "all", "none"

        # Check if it's IS_LIKE or just IS
        is_like = cond.operator.endswith("_IS_LIKE")

        # Get the list variable
        list_var = self._normalize_var(cond.left)

        # Get property and value from right (dict with "property" and "value" keys)
        right_data = cond.right
        if isinstance(right_data, dict):
            prop_var = self._normalize_var(right_data.get("property", ""))
            value = right_data.get("value", "")
        else:
            # Fallback for unexpected format
            prop_var = ""
            value = right_data

        # Determine predicate name
        if is_like:
            pred_name = f"has_{quantifier}_prop_like"
            value_sort = FOLSort.CATEGORY
        else:
            pred_name = f"has_{quantifier}_prop_eq"
            value_sort = FOLSort.VALUE

        # NONE is negation of ANY
        if quantifier == "none":
            return FOLNot(
                FOLPredicate(
                    pred_name.replace("none", "any"),
                    [
                        x,
                        FOLConstant(list_var, FOLSort.CATEGORY),  # atom name
                        FOLConstant(prop_var, FOLSort.CATEGORY),  # atom name
                        self._make_constant(value) if value_sort == FOLSort.VALUE else FOLConstant(value, value_sort)
                    ]
                )
            )

        return FOLPredicate(
            pred_name,
            [
                x,
                FOLConstant(list_var, FOLSort.CATEGORY),  # atom name
                FOLConstant(prop_var, FOLSort.CATEGORY),  # atom name
                self._make_constant(value) if value_sort == FOLSort.VALUE else FOLConstant(value, value_sort)
            ]
        )

    def _translate_comparison(self, cond: Condition) -> FOLFormula:
        """
        Translate comparison conditions.

        Handles operators: COMPARE_GT, COMPARE_LT, COMPARE_GTE, COMPARE_LTE
        """
        x = FOLVar(self.input_var.name)

        # Parse operator: COMPARE_GT, COMPARE_LT, etc.
        cmp_type = cond.operator.split("_")[1].lower()  # "gt", "lt", "gte", "lte"

        # Get variable and value
        var_name = self._normalize_var(cond.left)
        value = cond.right

        return FOLPredicate(
            f"compare_{cmp_type}",
            [
                FOLFunctionApp(var_name, [x]),
                FOLConstant(value, FOLSort.VALUE)
            ]
        )

    def _compute_superiority(self):
        """Compute superiority relations from OVERRIDING clauses."""
        # Track rules by variable for override computation
        rules_by_var: Dict[str, List[str]] = {}

        for rule in self.fol.all_rules():
            var = rule.head_variable
            if var not in rules_by_var:
                rules_by_var[var] = []
            rules_by_var[var].append(rule.id)

        # Process override relationships from original AST
        for rule in self.fol.all_rules():
            if not rule.original_ast:
                continue

            ast_rule = rule.original_ast
            override = ast_rule.override_target

            if override == OverrideTarget.ALL:
                # This rule overrides all other rules for same variable
                var = rule.head_variable
                for other_id in rules_by_var.get(var, []):
                    if other_id != rule.id:
                        self.fol.superiority.append(
                            CantoSuperiority(superior=rule.id, inferior=other_id)
                        )

            elif override == OverrideTarget.NORMAL:
                # This rule overrides only defeasible rules
                var = rule.head_variable
                for def_rule in self.fol.defeasible_rules:
                    if def_rule.head_variable == var and def_rule.id != rule.id:
                        self.fol.superiority.append(
                            CantoSuperiority(superior=rule.id, inferior=def_rule.id)
                        )

    def _generate_rule_id(self, var_name: str) -> str:
        """Generate unique rule ID."""
        if var_name not in self.rule_counter:
            self.rule_counter[var_name] = 0
        self.rule_counter[var_name] += 1
        return f"{var_name}_{self.rule_counter[var_name]}"

    def _normalize_var(self, var: Any) -> str:
        """Normalize variable name (remove ? prefix, handle qualified vars)."""
        if isinstance(var, str):
            return var.lstrip('?')
        if isinstance(var, dict) and 'child' in var:
            child = self._normalize_var(var['child'])
            parent = self._normalize_var(var['parent'])
            return f"{child}_of_{parent}"
        if isinstance(var, tuple) and len(var) == 2:
            child = self._normalize_var(var[0])
            parent = self._normalize_var(var[1])
            return f"{child}_of_{parent}"
        return str(var)

    def _normalize_value(self, val: Any) -> Any:
        """Normalize value."""
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.strip('"\'').lstrip('?')
        return val

    def _make_constant(self, value: Any) -> FOLConstant:
        """Create FOL constant with proper sort based on value type.

        Preserves native Python types (bool, int, float) for proper Prolog output.
        """
        if isinstance(value, bool):
            return FOLConstant(value, FOLSort.BOOL)
        elif isinstance(value, (int, float)):
            return FOLConstant(value, FOLSort.VALUE)
        elif isinstance(value, str):
            # Check if it's a boolean string
            if value.lower() in ("true", "false"):
                return FOLConstant(value.lower() == "true", FOLSort.BOOL)
            return FOLConstant(value, FOLSort.VALUE)
        else:
            return FOLConstant(str(value), FOLSort.VALUE)

    def _is_variable_ref(self, val: Any) -> bool:
        """Check if value is a variable reference."""
        if isinstance(val, dict) and 'child' in val:
            return True
        if isinstance(val, tuple) and len(val) == 2:
            return True
        if isinstance(val, str) and val.startswith('?'):
            return True
        return False

    def _infer_value_type(self, values: Optional[List[Any]]) -> ValueType:
        """Infer value type from a list of possible values."""
        if not values:
            return ValueType.STRING

        bool_values = {True, False, "true", "false"}
        if all(v in bool_values for v in values):
            return ValueType.BOOL
        return ValueType.STRING

    def _infer_value_type_from_value(self, value: Any) -> ValueType:
        """Infer value type from a single value."""
        if isinstance(value, bool):
            return ValueType.BOOL
        if isinstance(value, str) and value.lower() in ("true", "false"):
            return ValueType.BOOL
        return ValueType.STRING

    def _normalize_values(
        self,
        values: Optional[List[Any]],
        value_type: ValueType
    ) -> List[Any]:
        """Normalize values to consistent types."""
        if not values:
            return []

        if value_type == ValueType.BOOL:
            # Normalize to Python bools
            result = []
            for v in values:
                if v in (True, "true"):
                    result.append(True)
                elif v in (False, "false"):
                    result.append(False)
            return result
        else:
            # Keep as strings
            return [str(v) for v in values]


def translate_to_fol(ast: List[Any], source_file: str = None) -> CantoFOL:
    """
    Convenience function to translate AST to FOL.

    Args:
        ast: List of AST nodes
        source_file: Optional source file path

    Returns:
        CantoFOL: The canonical FOL representation
    """
    translator = ASTToFOLTranslator()
    return translator.translate(ast, source_file)
