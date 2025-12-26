"""
Main parser implementation using Lark
"""

import os
import re
from typing import List, Union, Tuple, Optional
from dataclasses import dataclass
from lark import Lark, Transformer, Tree, Token
from pathlib import Path


@dataclass
class ParseResult:
    """
    Result of parsing a DSL file.

    Attributes:
        ast: List of AST nodes (declarations, rules, etc.)
        instructions: Optional instructions from triple-quoted string at top of file
    """
    ast: List
    instructions: Optional[str] = None

    def __iter__(self):
        """Allow iterating over AST nodes for backwards compatibility"""
        return iter(self.ast)

    def __len__(self):
        """Allow len() on ParseResult for backwards compatibility"""
        return len(self.ast)

    def __getitem__(self, index):
        """Allow indexing into AST for backwards compatibility"""
        return self.ast[index]


def preprocess_with_blocks(text: str) -> str:
    """
    Pre-process DSL text to convert indentation-based 'with' blocks into explicit 'end' markers.

    This allows users to write:
        ?patient meaning "info" with
            ?name meaning "name"
            ?symptoms with
                ?text meaning "description"

        ?rule becomes true when ...

    And converts it to:
        ?patient meaning "info" with
            ?name meaning "name"
            ?symptoms with
                ?text meaning "description"
            end
        end

        ?rule becomes true when ...

    The pre-processor:
    1. Detects lines ending with 'with'
    2. Tracks indentation levels using a stack
    3. Inserts 'end' when indentation decreases
    4. Leaves non-with content unchanged
    """
    lines = text.split('\n')
    result_lines = []

    # Stack of (indent_level, line_index) for open 'with' blocks
    with_stack: List[Tuple[int, int]] = []

    def get_indent(line: str) -> int:
        """Get the indentation level (number of leading spaces/tabs)"""
        stripped = line.lstrip()
        if not stripped:  # Empty line
            return -1  # Special marker for empty lines
        return len(line) - len(stripped)

    def ends_with_with(line: str) -> bool:
        """Check if line ends with 'with' keyword (not inside a string)"""
        stripped = line.rstrip()
        # Simple check: ends with 'with' and followed by word boundary
        # This handles: "?var meaning "desc" with" but not "?var meaning "something with quotes""
        if not stripped.endswith('with'):
            return False
        # Make sure 'with' is a separate word
        if len(stripped) == 4:
            return True
        char_before = stripped[-5]
        return char_before in ' \t'

    i = 0
    while i < len(lines):
        line = lines[i]
        current_indent = get_indent(line)

        # Skip empty lines - just pass them through
        if current_indent == -1:
            result_lines.append(line)
            i += 1
            continue

        # Check if we need to close any 'with' blocks
        # Close blocks where the current indentation is <= the block's indentation
        ends_to_insert = []
        while with_stack and current_indent <= with_stack[-1][0]:
            block_indent, _ = with_stack.pop()
            ends_to_insert.append(block_indent)

        # Insert 'end' markers with proper indentation
        for indent_level in ends_to_insert:
            result_lines.append(' ' * indent_level + 'end')

        # Add the current line
        result_lines.append(line)

        # Check if this line opens a new 'with' block
        if ends_with_with(line):
            with_stack.append((current_indent, i))

        i += 1

    # Close any remaining open 'with' blocks at the end
    while with_stack:
        block_indent, _ = with_stack.pop()
        result_lines.append(' ' * block_indent + 'end')

    return '\n'.join(result_lines)

from ..ast_nodes import (
    ImportDeclaration,
    VariableDeclaration,
    SemanticCategory,
    HasDeclaration,
    Rule,
    Condition,
    Predicate,
)
from ..ast_nodes.rules import RulePriority, OverrideTarget




class CantoTransformer(Transformer):
    """
    Transforms Lark parse tree into Canto AST nodes
    """

    def program(self, items):
        """Root node containing all statements, with optional instructions"""
        instructions = None
        statements = []

        for item in items:
            if item is None:
                continue
            elif isinstance(item, str) and item.startswith("__INSTRUCTIONS__:"):
                # Extract instructions content
                instructions = item[len("__INSTRUCTIONS__:"):]
            else:
                statements.append(item)

        return ParseResult(ast=statements, instructions=instructions)

    def import_statement(self, items):
        """Parse import statement: import module_name"""
        module_name = str(items[0])
        return ImportDeclaration(name=module_name)

    def IDENTIFIER(self, token):
        """Parse IDENTIFIER token"""
        return str(token)

    def instructions(self, items):
        """Parse instructions block (triple-quoted string)"""
        if not items:
            return None
        content = str(items[0])
        # Remove surrounding triple quotes
        if content.startswith('"""') and content.endswith('"""'):
            content = content[3:-3].strip()
        # Mark as instructions for program() to recognize
        return f"__INSTRUCTIONS__:{content}"

    def INSTRUCTIONS(self, token):
        """Parse INSTRUCTIONS token"""
        return str(token)

    def predicate(self, items):
        """Parse predicate - pass through to specific type"""
        return items[0]

    def freetext_variable(self, items):
        """Parse freetext variable: ?var meaning "description" [from ?source] [with ...]"""
        var_name = items[0]
        description = items[1]
        source = None
        children = []

        # Parse optional from and with_block
        for item in items[2:]:
            if item is None:
                continue
            elif isinstance(item, str) and item.startswith('?'):
                # This is the source variable
                source = item
            elif isinstance(item, list):
                # This is children from with_block
                children = item

        return VariableDeclaration(
            name=var_name,
            description=description,
            values_from=None,
            source=source,
            children=children
        )

    def enum_variable(self, items):
        """Parse enum variable: ?var can be "a", "b", "c" meaning "description" [with ...]"""
        var_name = items[0]
        values_from = items[1]  # enum_values
        description = None
        children = []

        for item in items[2:]:
            if isinstance(item, str):
                description = item
            elif isinstance(item, list):
                children = item

        return VariableDeclaration(
            name=var_name,
            description=description,
            values_from=values_from,
            children=children
        )

    def enum_values(self, items):
        """Parse enum values (comma-separated strings or booleans)"""
        return items

    def enum_value(self, items):
        """Parse single enum value (string or boolean keyword)"""
        val = items[0]
        # Convert boolean to string representation for consistency
        if isinstance(val, bool):
            return "true" if val else "false"
        return val

    def assignment_predicate(self, items):
        """Parse assignment predicate - delegates to existing rule logic"""
        priority, head_var, head_parent, head_value = items[0]

        conditions = []
        exceptions = []
        override_target = OverrideTarget.NONE

        if len(items) > 1 and isinstance(items[1], list):
            for item in items[1]:
                if isinstance(item, dict):
                    if "when" in item:
                        conditions = item["when"]
                    elif "unless" in item:
                        exceptions = item["unless"]
                    elif "override" in item:
                        override_target = item["override"]

        return Rule(
            head_variable=head_var,
            head_value=head_value,
            conditions=conditions,
            exceptions=exceptions,
            priority=priority,
            override_target=override_target,
            head_parent=head_parent
        )

    def pattern_declaration(self, items):
        """Parse pattern declaration (resembles)"""
        var_name = items[0]
        patterns = items[1]  # List of patterns, already unquoted by string_literal
        description = None

        if len(items) > 2:  # Description is optional
            description = items[2]  # Already unquoted by description rule

        return SemanticCategory(
            name=var_name,
            patterns=patterns,
            description=description
        )

    def pattern_items(self, items):
        """Parse pattern items (comma-separated strings)"""
        return items

    def has_list_declaration(self, items):
        """Parse has list declaration: ?parent has a list of ?child meaning "..." """
        parent = items[0]
        child = items[1]
        description = items[2] if len(items) > 2 else None

        return HasDeclaration(
            parent=parent,
            child=child,
            is_list=True,
            description=description
        )

    def has_single_declaration(self, items):
        """Parse has single declaration: ?parent has a ?child meaning "..." """
        parent = items[0]
        child = items[1]
        description = items[2] if len(items) > 2 else None

        return HasDeclaration(
            parent=parent,
            child=child,
            is_list=False,
            description=description
        )

    def has_list_from_declaration(self, items):
        """Parse has list from declaration: ?parent has a list of ?child from ?source [with ...]"""
        parent = items[0]
        child = items[1]
        source = items[2]
        children = items[3] if len(items) > 3 and items[3] is not None else []

        return HasDeclaration(
            parent=parent,
            child=child,
            is_list=True,
            source=source,
            children=children
        )

    # ============ WITH BLOCK HANDLERS ============

    def with_block(self, items):
        """Parse with block - returns list of children (with_item+)"""
        # items contains all the with_item results directly
        return [item for item in items if item is not None]

    def with_body(self, items):
        """Parse with body - list of with_items"""
        return [item for item in items if item is not None]

    def with_freetext(self, items):
        """Parse with item: ?var meaning "description" [from ?source] [with ...]"""
        var_name = items[0]
        description = items[1]
        source = None
        children = []

        # Parse optional from and with_block
        for item in items[2:]:
            if item is None:
                continue
            elif isinstance(item, str) and item.startswith('?'):
                # This is the source variable
                source = item
            elif isinstance(item, list):
                # This is children from with_block
                children = item

        return VariableDeclaration(
            name=var_name,
            description=description,
            source=source,
            children=children
        )

    def with_enum(self, items):
        """Parse with enum: ?var can be "a", "b" [meaning "desc"] [with ...]"""
        var_name = items[0]
        values_from = items[1]
        description = None
        children = []

        for item in items[2:]:
            if isinstance(item, str):
                description = item
            elif isinstance(item, list):
                children = item

        return VariableDeclaration(
            name=var_name,
            description=description,
            values_from=values_from,
            children=children
        )

    def with_nested(self, items):
        """Parse with nested: ?var with ..."""
        var_name = items[0]
        children = items[1] if len(items) > 1 and items[1] is not None else []

        return VariableDeclaration(
            name=var_name,
            children=children
        )

    def with_has_list_from(self, items):
        """Parse with has list from: ?var has a list of ?type from ?source"""
        var_name = items[0]
        type_var = items[1]
        source_var = items[2]

        return HasDeclaration(
            parent=var_name,
            child=type_var,
            is_list=True,
            source=source_var
        )

    def with_has_list(self, items):
        """Parse with has list: ?var has a list of ?type [meaning "desc"]"""
        var_name = items[0]
        type_var = items[1]
        description = items[2] if len(items) > 2 else None

        return HasDeclaration(
            parent=var_name,
            child=type_var,
            is_list=True,
            description=description
        )

    def with_has_single(self, items):
        """Parse with has single: ?var has a ?type [meaning "desc"]"""
        var_name = items[0]
        type_var = items[1]
        description = items[2] if len(items) > 2 else None

        return HasDeclaration(
            parent=var_name,
            child=type_var,
            is_list=False,
            description=description
        )

    def with_rule(self, items):
        """Parse with rule: ?var becomes value [when ...]"""
        head_var = items[0]
        head_value = items[1]
        conditions = []

        if len(items) > 2 and items[2] is not None:
            when_dict = items[2]
            conditions = when_dict.get("when", [])

        return Rule(
            head_variable=head_var,
            head_value=head_value,
            conditions=conditions,
            exceptions=[],
            priority=RulePriority.STRICT,
            override_target=OverrideTarget.NONE
        )

    def predicate_head(self, items):
        """Parse predicate head - returns (priority, var, parent, value)

        var can be a simple variable name or qualified (?child of ?parent).
        For qualified variables, parent is set; otherwise it's None.
        """
        priority = RulePriority.STRICT
        i = 0

        # Check for normally
        if isinstance(items[i], RulePriority):
            priority = items[i]
            i += 1

        # Get variable (can be simple string or tuple for qualified)
        var_ref = items[i]
        i += 1

        # Handle qualified vs simple variable
        if isinstance(var_ref, tuple):
            # Qualified: (child, parent)
            head_var = var_ref[0]
            head_parent = var_ref[1]
        else:
            # Simple variable
            head_var = var_ref
            head_parent = None

        # Skip BECOMES token
        if isinstance(items[i], Token) and items[i].type == 'BECOMES':
            i += 1

        # Get value
        head_value = items[i]

        return (priority, head_var, head_parent, head_value)

    def predicate_priority(self, items):
        """Parse predicate priority (normally)"""
        return RulePriority.NORMAL

    def predicate_value(self, items):
        """Parse predicate value"""
        return items[0]

    def predicate_body(self, items):
        """Parse predicate body - returns list of clause dicts"""
        return items if items else []

    def when_clause(self, items):
        """Parse when clause"""
        # items[0] is the conditions list
        return {"when": items[0] if items else []}

    def unless_clause(self, items):
        """Parse unless clause"""
        # items[0] is the conditions list
        return {"unless": items[0] if items else []}

    def conditions(self, items):
        """Parse conditions with and/or operators"""
        if len(items) == 1:
            return items[0] if isinstance(items[0], list) else [items[0]]

        # Helper to unwrap single-element lists from parenthesized conditions
        def unwrap(item):
            if isinstance(item, list) and len(item) == 1:
                return item[0]
            return item

        # Build tree of conditions with operators
        result = unwrap(items[0])
        for i in range(1, len(items), 2):
            op = items[i]  # and or or
            right = unwrap(items[i + 1])
            result = Condition(operator=op, left=result, right=right)

        return [result] if not isinstance(result, list) else result

    def qualified_variable(self, items):
        """Parse qualified variable: ?child of ?parent
        Returns a tuple (child, parent) to distinguish from simple variable
        """
        child = items[0]  # ?child
        parent = items[1]  # ?parent
        return (child, parent)

    def qualified_variable_nested(self, items):
        """Parse nested qualified variable: ?child of (?grandchild of ?grandparent)
        Returns a tuple (child, parent_ref) where parent_ref is itself a tuple
        Example: ?claims_truth of (?claims of ?puzzle) -> ('?claims_truth', ('?claims', '?puzzle'))
        """
        child = items[0]  # ?child
        parent_ref = items[1]  # The nested variable_ref (already parsed as tuple or string)
        return (child, parent_ref)

    def simple_variable(self, items):
        """Parse simple variable: ?var
        Returns just the variable name string
        """
        return items[0]

    def condition(self, items):
        """Parse single condition (regular conditions, not not/not warranted)"""
        if len(items) == 1:
            item = items[0]
            if isinstance(item, Condition):
                # Parenthesized single condition
                return item
            elif isinstance(item, list):
                # Parenthesized conditions - unwrap if single element
                if len(item) == 1 and isinstance(item[0], Condition):
                    return item[0]
                # Multiple conditions in parentheses - shouldn't happen with current grammar
                return item[0] if item else None

        # Regular condition: [variable_ref, operator_token, operand]
        var_ref = items[0]
        operator_token = str(items[1]) if len(items) > 1 else "IS"
        operand = items[2] if len(items) > 2 else None

        # Handle qualified vs simple variable reference
        if isinstance(var_ref, tuple):
            # Qualified: (child, parent) - e.g., ?symptoms of ?patient
            child, parent = var_ref
            # Store as dict to preserve the qualification info
            var = {"child": child, "parent": parent}
        else:
            # Simple variable
            var = var_ref

        return Condition(operator=operator_token, left=var, right=operand)

    def has_simple(self, items):
        """Parse simple has condition: variable_ref HAS_OP operand
        Example: ?symptoms of ?patient has "chest pain"

        Note: items = [variable_ref, HAS_OP token, operand]
        """
        var_ref = items[0]
        # items[1] is the HAS_OP token which we skip
        operand = items[2] if len(items) > 2 else items[1]

        # Handle qualified vs simple variable reference
        if isinstance(var_ref, tuple):
            child, parent = var_ref
            var = {"child": child, "parent": parent}
        else:
            var = var_ref

        return Condition(operator="HAS", left=var, right=operand)

    def has_quantified(self, items):
        """Parse quantified has condition: variable_ref HAS_OP quantified_check
        Example: ?symptoms of ?patient has any that is like ?emergency_symptoms

        For explicit binding with compound conditions, use "let ?x be any in ?coll where ..."

        Note: items = [variable_ref, HAS_OP token, quantified_check]
        """
        var_ref = items[0]
        # items[1] is the HAS_OP token which we skip
        quantified = items[2] if len(items) > 2 else items[1]

        # Handle qualified vs simple variable reference
        if isinstance(var_ref, tuple):
            child, parent = var_ref
            var = {"child": child, "parent": parent}
        else:
            var = var_ref

        # Build operator from quantifier + inline condition type
        # e.g., "HAS_ANY_LIKE", "HAS_ALL_IS", "HAS_NONE_IS_NOT"
        quantifier = quantified["quantifier"]  # "ANY", "ALL", "NONE"
        condition_type = quantified["condition_type"]  # "IS_LIKE", "IS", "IS_NOT", "PROPERTY_IS", "PROPERTY_IS_LIKE"
        condition_operand = quantified["operand"]

        operator = f"HAS_{quantifier}_{condition_type}"

        # For property-based conditions, include the property name
        if "property" in quantified:
            return Condition(
                operator=operator,
                left=var,
                right={"property": quantified["property"], "value": condition_operand}
            )

        return Condition(operator=operator, left=var, right=condition_operand)

    # ============ PROPERTY-BASED QUANTIFIED CONDITIONS ============

    def quantified_property(self, items):
        """Parse quantified property: quantifier property_condition
        Example: any that ?is_emergency is true
        """
        quantifier = items[0]  # "ANY", "ALL", "NONE"
        property_cond = items[1]  # dict with property, condition_type, operand

        return {
            "quantifier": quantifier,
            "condition_type": property_cond["condition_type"],
            "operand": property_cond["operand"],
            "property": property_cond["property"]
        }

    def property_is(self, items):
        """Parse property is: ?property is value"""
        property_name = items[0]
        # items[1] is IS_OP token
        operand = items[2] if len(items) > 2 else items[1]
        return {"condition_type": "PROPERTY_IS", "property": property_name, "operand": operand}

    def property_is_like(self, items):
        """Parse property is like: ?property is like ?category"""
        property_name = items[0]
        # items[1] is IS_LIKE_OP token
        operand = items[2] if len(items) > 2 else items[1]
        return {"condition_type": "PROPERTY_IS_LIKE", "property": property_name, "operand": operand}

    # ============ LENGTH OF CONDITIONS ============

    def length_where_is(self, items):
        """Parse: (length of ?list where ?prop is value) length_check

        list_var can be simple (?list) or qualified (?child of ?parent)
        """
        list_var = items[0]
        prop_var = items[1]
        # items[2] is IS_OP token
        operand = items[3] if len(items) > 3 else items[2]
        length_check = items[4] if len(items) > 4 else items[3]

        # Handle qualified vs simple variable for list
        if isinstance(list_var, tuple):
            child, parent = list_var
            list_ref = {"child": child, "parent": parent}
        else:
            list_ref = list_var

        return Condition(
            operator=f"LENGTH_WHERE_IS_{length_check['check_type']}",
            left={"list": list_ref, "property": prop_var, "condition": "IS", "value": operand},
            right=length_check.get("value")
        )

    def length_where_like(self, items):
        """Parse: (length of ?list where ?prop is like ?category) length_check

        list_var can be simple (?list) or qualified (?child of ?parent)
        """
        list_var = items[0]
        prop_var = items[1]
        # items[2] is IS_LIKE_OP token
        operand = items[3] if len(items) > 3 else items[2]
        length_check = items[4] if len(items) > 4 else items[3]

        # Handle qualified vs simple variable for list
        if isinstance(list_var, tuple):
            child, parent = list_var
            list_ref = {"child": child, "parent": parent}
        else:
            list_ref = list_var

        return Condition(
            operator=f"LENGTH_WHERE_LIKE_{length_check['check_type']}",
            left={"list": list_ref, "property": prop_var, "condition": "IS_LIKE", "value": operand},
            right=length_check.get("value")
        )

    def length_is_number(self, items):
        """Parse: is N"""
        # items[0] is IS_OP token ("IS"), items[1] is the number
        num = items[1] if len(items) > 1 else 0
        return {"check_type": "EQ", "value": num}

    def length_comparison(self, items):
        """Parse: > N, < N, etc."""
        op = str(items[0])  # comparison operator
        num = items[1]
        op_map = {">": "GT", "<": "LT", ">=": "GTE", "<=": "LTE"}
        return {"check_type": op_map.get(op, "GT"), "value": num}

    def COMPARE_OP(self, token):
        """Parse comparison operator"""
        return str(token)

    def comparison_condition(self, items):
        """Parse comparison condition: variable_ref > N"""
        var_ref = items[0]
        op = str(items[1])
        num = items[2]

        if isinstance(var_ref, tuple):
            child, parent = var_ref
            var = {"child": child, "parent": parent}
        else:
            var = var_ref

        op_map = {">": "GT", "<": "LT", ">=": "GTE", "<=": "LTE"}
        return Condition(operator=f"COMPARE_{op_map.get(op, 'GT')}", left=var, right=num)

    def quantified_check(self, items):
        """Parse quantified check: quantifier inline_condition
        Returns dict with quantifier info and inline condition
        """
        quantifier = items[0]  # "ANY", "ALL", "NONE"
        inline_cond = items[1]  # dict with condition_type and operand

        return {
            "quantifier": quantifier,
            "condition_type": inline_cond["condition_type"],
            "operand": inline_cond["operand"]
        }

    def any_quantifier(self, items):
        """Parse 'any that' quantifier"""
        return "ANY"

    def all_quantifier(self, items):
        """Parse 'all that' quantifier"""
        return "ALL"

    def none_quantifier(self, items):
        """Parse 'none that' quantifier"""
        return "NONE"

    # ============ BOUND CONDITIONS (used by let binding) ============

    def bound_conditions(self, items):
        """Parse compound bound conditions: bound_condition bound_continuation*
        Returns list of conditions with operators
        """
        conditions = []

        for item in items:
            if isinstance(item, dict):
                conditions.append(item)
            elif isinstance(item, list):
                # bound_continuation returns [logic_op, condition]
                conditions.extend(item)

        return conditions

    def bound_and_cont(self, items):
        """Parse 'and bound_condition' continuation"""
        # items[0] is 'and' token, items[1] is the bound_condition
        cond = items[1] if len(items) > 1 else items[0]
        if isinstance(cond, dict):
            cond["logic_op"] = "AND"
        return [cond]

    def bound_or_cont(self, items):
        """Parse 'or bound_condition' continuation"""
        # items[0] is 'or' token, items[1] is the bound_condition
        cond = items[1] if len(items) > 1 else items[0]
        if isinstance(cond, dict):
            cond["logic_op"] = "OR"
        return [cond]

    # ============ LET BINDING FOR QUANTIFIED CONDITIONS ============

    def let_condition(self, items):
        """Parse let binding condition: let ?var be any/all/none in collection where conditions
        Example: let ?link be any in ?chain where ?speaker of ?link is ?target

        items = [VARIABLE, let_quantifier, variable_ref, bound_conditions]
        """
        binding = items[0]  # The binding variable (e.g., ?link)
        quantifier = items[1]  # "ANY", "ALL", "NONE"
        collection = items[2]  # The collection variable_ref
        conditions = items[3]  # List of bound conditions

        # Handle qualified vs simple collection reference
        if isinstance(collection, tuple):
            child, parent = collection
            var = {"child": child, "parent": parent}
        else:
            var = collection

        operator = f"LET_{quantifier}_BOUND"

        return Condition(
            operator=operator,
            left=var,
            right={
                "binding": binding,
                "conditions": conditions
            }
        )

    def let_any(self, items):
        """Parse 'any' quantifier in let binding"""
        return "ANY"

    def let_all(self, items):
        """Parse 'all' quantifier in let binding"""
        return "ALL"

    def let_none(self, items):
        """Parse 'none' quantifier in let binding"""
        return "NONE"

    def bound_is(self, items):
        """Parse bound is condition: variable_ref IS_OP operand"""
        left = items[0]
        # items[1] is IS_OP token
        right = items[2] if len(items) > 2 else items[1]
        return {"condition_type": "IS", "left": left, "right": right}

    def bound_is_like(self, items):
        """Parse bound is like condition: variable_ref IS_LIKE_OP operand"""
        left = items[0]
        # items[1] is IS_LIKE_OP token
        right = items[2] if len(items) > 2 else items[1]
        return {"condition_type": "IS_LIKE", "left": left, "right": right}

    def bound_grouped(self, items):
        """Parse grouped bound conditions: ( bound_conditions )"""
        # items[0] is the bound_conditions result (list of conditions)
        return {"grouped": True, "conditions": items[0]}

    def inline_is_like(self, items):
        """Parse inline 'is like' condition
        items = [IS_LIKE_OP token, operand]
        """
        # items[0] is the IS_LIKE_OP token, items[1] is the operand
        operand = items[1] if len(items) > 1 else items[0]
        return {"condition_type": "IS_LIKE", "operand": operand}

    def inline_is(self, items):
        """Parse inline 'is' condition
        items = [IS_OP token, operand]
        """
        # items[0] is the IS_OP token, items[1] is the operand
        operand = items[1] if len(items) > 1 else items[0]
        return {"condition_type": "IS", "operand": operand}

    def inline_is_not(self, items):
        """Parse inline 'is not' condition
        items = [IS_OP token, 'not' literal, operand]
        """
        # items[0] is IS_OP, items[1] is 'not' literal, items[2] is operand
        operand = items[2] if len(items) > 2 else items[-1]
        return {"condition_type": "IS_NOT", "operand": operand}

    def not_condition(self, items):
        """Parse not condition (Negation as Failure)
        Syntax: not <condition>
        Semantics: Fails if ANY argument for the condition exists
        """
        # items: [NOT_OP token, condition]
        return Condition(operator="NOT", right=items[1])

    def not_warranted_condition(self, items):
        """Parse not warranted condition (warrant-based negation)
        Syntax: not warranted <condition>
        Semantics: Fails only if the condition is WARRANTED (has undefeated argument)
        """
        # items: [NOT_WARRANTED_OP token, condition]
        return Condition(operator="NOT_WARRANTED", right=items[1])

    def IS_LIKE_OP(self, token):
        """Parse is like operator (semantic similarity)"""
        return "IS_LIKE"

    def IS_OP(self, token):
        """Parse is operator (exact match)"""
        return "IS"

    def HAS_OP(self, token):
        """Parse has operator"""
        return "HAS"

    def NOT_OP(self, token):
        """Parse not operator"""
        return "NOT"

    def NOT_WARRANTED_OP(self, token):
        """Parse not warranted operator"""
        return "NOT_WARRANTED"

    def override_clause(self, items):
        """Parse override clause"""
        target = items[0]
        if target == "all":
            return {"override": OverrideTarget.ALL}
        elif target == "normal":
            return {"override": OverrideTarget.NORMAL}
        return {"override": OverrideTarget.NONE}

    def override_target(self, items):
        """Parse override target"""
        if len(items) == 0:
            return "none"
        return str(items[0])

    def OVERRIDE_ALL(self, token):
        """Parse OVERRIDE_ALL token"""
        return "all"

    def OVERRIDE_NORMAL(self, token):
        """Parse OVERRIDE_NORMAL token"""
        return "normal"


    def logic_op(self, items):
        """Parse logic operator (and or or)"""
        if len(items) == 0:
            return "AND"
        return str(items[0])

    def AND_OP(self, token):
        """Parse and operator"""
        return "AND"

    def OR_OP(self, token):
        """Parse or operator"""
        return "OR"

    def operand(self, items):
        """Parse operand"""
        if len(items) == 0:
            return None
        return items[0]

    def boolean(self, items):
        """Parse boolean - items[0] is BOOLEAN token"""
        if len(items) == 0:
            return False
        val = str(items[0])
        return val == "true"

    def BOOLEAN(self, token):
        """Parse BOOLEAN token"""
        return str(token)

    def number(self, items):
        """Parse number"""
        if len(items) == 0:
            return 0
        val = str(items[0])
        return float(val) if '.' in val else int(val)

    def string_literal(self, items):
        """Parse string literal"""
        if len(items) == 0:
            return ""
        val = str(items[0])
        # Remove surrounding quotes
        if val.startswith('"""') and val.endswith('"""'):
            return val[3:-3]
        elif (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
            return val[1:-1]
        return val

    def description(self, items):
        """Parse description"""
        if len(items) == 0:
            return ""
        return str(items[0])

    def VARIABLE(self, token):
        """Parse variable token"""
        return str(token)

    def comment(self, items):
        """Ignore comments"""
        return None


class CantoParser:
    """
    Main parser class for Canto DSL
    """

    def __init__(self, grammar_path: str = None):
        if grammar_path is None:
            # Default grammar path
            current_dir = Path(__file__).parent.parent
            grammar_path = current_dir / "grammar" / "dsl.lark"

        with open(grammar_path, 'r') as f:
            grammar = f.read()

        self.parser = Lark(
            grammar,
            parser='earley',
            start='program',
            ambiguity='resolve'
        )
        self.transformer = CantoTransformer()

    def parse(self, text: str) -> ParseResult:
        """
        Parse DSL text and return ParseResult with AST nodes and instructions.

        Automatically pre-processes 'with' blocks to convert indentation into explicit 'end' markers.

        Returns:
            ParseResult with:
                - ast: List of AST nodes (declarations, rules, etc.)
                - instructions: Optional instructions from triple-quoted string at top
        """
        try:
            # Pre-process to handle indentation-based 'with' blocks
            processed_text = preprocess_with_blocks(text)
            tree = self.parser.parse(processed_text)
            result = self.transformer.transform(tree)
            return result
        except Exception as e:
            raise SyntaxError(f"Failed to parse DSL: {e}")

    def parse_file(self, filepath: str, _imported: set = None) -> ParseResult:
        """
        Parse DSL file and return ParseResult with AST nodes and instructions.

        Resolves imports recursively, merging imported ASTs before the main file's AST.
        Tracks already-imported files to prevent cycles.
        """
        filepath = Path(filepath).resolve()

        # Track imported files to prevent cycles
        if _imported is None:
            _imported = set()

        if filepath in _imported:
            # Already imported, return empty result to avoid cycles
            return ParseResult(ast=[], instructions=None)

        _imported.add(filepath)

        with open(filepath, 'r') as f:
            text = f.read()

        result = self.parse(text)

        # Separate imports from other statements
        imports = [node for node in result.ast if isinstance(node, ImportDeclaration)]
        statements = [node for node in result.ast if not isinstance(node, ImportDeclaration)]

        # Resolve and merge imports
        merged_statements = []
        for imp in imports:
            # Resolve import path relative to the importing file
            import_path = filepath.parent / f"{imp.name}.canto"
            if import_path.exists():
                imported_result = self.parse_file(str(import_path), _imported)
                merged_statements.extend(imported_result.ast)
            else:
                raise FileNotFoundError(f"Cannot find import '{imp.name}' at {import_path}")

        # Add this file's statements after imports
        merged_statements.extend(statements)

        return ParseResult(ast=merged_statements, instructions=result.instructions)


def parse_string(text: str) -> ParseResult:
    """
    Convenience function to parse DSL string.

    Returns ParseResult with ast and instructions.
    """
    parser = CantoParser()
    return parser.parse(text)


def parse_file(filepath: str) -> ParseResult:
    """
    Convenience function to parse DSL file.

    Returns ParseResult with ast and instructions.
    """
    parser = CantoParser()
    return parser.parse_file(filepath)
