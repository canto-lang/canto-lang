"""
Encodes Canto FOL IR to Z3 formulas for verification.

This module provides the bridge between the FOL representation
and Z3's SMT solving capabilities.
"""

from typing import List, Dict, Any, Optional, Tuple

from z3 import (
    Solver,
    BoolRef,
    BoolVal,
    And as Z3And,
    Or as Z3Or,
    Not as Z3Not,
    Implies as Z3Implies,
    ForAll as Z3ForAll,
    Exists as Z3Exists,
    Function,
    DeclareSort,
    Const,
    StringSort,
    IntSort,
    BoolSort,
    StringVal,
    sat,
    unsat,
)

from .types import (
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
    FOLSort,
    FOLTerm,
)
from .canto_fol import CantoFOL, CantoRule, ValueType


class Z3Encoder:
    """
    Encodes Canto FOL IR to Z3 for SMT solving.

    This encoder creates Z3 formulas from the FOL representation,
    enabling formal verification of Canto programs.
    """

    def __init__(self):
        # Z3 sorts
        self.InputSort = DeclareSort('Input')
        self.ValueSort = StringSort()
        self.CategorySort = StringSort()

        # Built-in predicates as Z3 functions
        self.is_like = Function('is_like', self.InputSort, self.CategorySort, BoolSort())
        self.has = Function('has', self.InputSort, StringSort(), BoolSort())
        self.matches = Function('matches', self.InputSort, self.CategorySort, BoolSort())
        self.has_any_like = Function('has_any_like', self.InputSort, self.CategorySort, BoolSort())
        self.has_all_like = Function('has_all_like', self.InputSort, self.CategorySort, BoolSort())
        self.has_any_eq = Function('has_any_eq', self.InputSort, StringSort(), BoolSort())
        self.has_all_eq = Function('has_all_eq', self.InputSort, StringSort(), BoolSort())
        self.fires = Function('fires', StringSort(), BoolSort())
        self.wins = Function('wins', StringSort(), BoolSort())
        self.conflicts = Function('conflicts', StringSort(), StringSort(), BoolSort())
        self.not_warranted_pred = Function('not_warranted', BoolSort(), BoolSort())

        # Variable functions (created on demand)
        self.var_functions: Dict[str, Any] = {}

        # Variable type info from CantoFOL (set by encode())
        self._fol: Optional[CantoFOL] = None

        # Cache for created predicates
        self.predicate_cache: Dict[str, Any] = {}

        # Default input variable
        self.input_const = Const('x', self.InputSort)

    def encode(self, fol: CantoFOL) -> BoolRef:
        """
        Encode complete CantoFOL program to Z3 formula.

        Returns: Conjunction of all program constraints
        """
        # Store FOL for variable type lookups
        self._fol = fol

        constraints = []

        # Encode all FOL formulas from the program
        for formula in fol.to_fol_formulas():
            z3_formula = self.encode_formula(formula)
            if z3_formula is not None:
                constraints.append(z3_formula)

        if not constraints:
            return BoolVal(True)
        return Z3And(*constraints)

    def encode_formula(self, formula: FOLFormula) -> Optional[BoolRef]:
        """Encode a single FOL formula to Z3."""
        if formula is None:
            return None

        if isinstance(formula, FOLPredicate):
            return self._encode_predicate(formula)

        elif isinstance(formula, FOLEquals):
            left = self._encode_term(formula.left)
            right = self._encode_term(formula.right)
            return left == right

        elif isinstance(formula, FOLNot):
            inner = self.encode_formula(formula.formula)
            if inner is None:
                return None
            return Z3Not(inner)

        elif isinstance(formula, FOLAnd):
            encoded = [self.encode_formula(c) for c in formula.conjuncts]
            encoded = [e for e in encoded if e is not None]
            if not encoded:
                return BoolVal(True)
            return Z3And(*encoded)

        elif isinstance(formula, FOLOr):
            encoded = [self.encode_formula(d) for d in formula.disjuncts]
            encoded = [e for e in encoded if e is not None]
            if not encoded:
                return BoolVal(False)
            return Z3Or(*encoded)

        elif isinstance(formula, FOLImplies):
            antecedent = self.encode_formula(formula.antecedent)
            consequent = self.encode_formula(formula.consequent)
            if antecedent is None:
                return consequent
            if consequent is None:
                return BoolVal(True)
            return Z3Implies(antecedent, consequent)

        elif isinstance(formula, FOLForall):
            var = Const(formula.variable.name, self._sort_to_z3(formula.variable.sort))
            body = self.encode_formula(formula.formula)
            if body is None:
                return BoolVal(True)
            return Z3ForAll([var], body)

        elif isinstance(formula, FOLExists):
            var = Const(formula.variable.name, self._sort_to_z3(formula.variable.sort))
            body = self.encode_formula(formula.formula)
            if body is None:
                return BoolVal(False)
            return Z3Exists([var], body)

        else:
            raise ValueError(f"Unknown formula type: {type(formula)}")

    def _encode_predicate(self, pred: FOLPredicate) -> BoolRef:
        """Encode predicate to Z3."""
        args = [self._encode_term(a) for a in pred.args]

        # Built-in predicates
        if pred.name == "is_like" and len(args) >= 2:
            return self.is_like(args[0], args[1])

        elif pred.name == "has" and len(args) >= 2:
            return self.has(args[0], args[1])

        elif pred.name == "matches" and len(args) >= 2:
            return self.matches(args[0], args[1])

        elif pred.name == "has_any_like" and len(args) >= 2:
            return self.has_any_like(args[0], args[1])

        elif pred.name == "has_all_like" and len(args) >= 2:
            return self.has_all_like(args[0], args[1])

        elif pred.name == "has_any_eq" and len(args) >= 2:
            return self.has_any_eq(args[0], args[1])

        elif pred.name == "has_all_eq" and len(args) >= 2:
            return self.has_all_eq(args[0], args[1])

        elif pred.name == "fires" and len(args) >= 1:
            return self.fires(args[0])

        elif pred.name == "wins" and len(args) >= 1:
            return self.wins(args[0])

        elif pred.name == "conflicts" and len(args) >= 2:
            return self.conflicts(args[0], args[1])

        elif pred.name == "not_warranted" and len(args) >= 1:
            # Handle nested formula
            if isinstance(pred.args[0], FOLFormula):
                inner = self.encode_formula(pred.args[0])
                return self.not_warranted_pred(inner)
            return self.not_warranted_pred(args[0])

        elif pred.name == "true":
            return BoolVal(True)

        elif pred.name == "false":
            return BoolVal(False)

        elif pred.name == "pattern":
            # Pattern predicates are definitional, always true
            return BoolVal(True)

        elif pred.name == "implicit_sup":
            # Implicit superiority axiom
            return BoolVal(True)

        else:
            # Unknown predicate - create uninterpreted function
            return self._get_or_create_predicate(pred.name, args)

    def _get_or_create_predicate(self, name: str, args: List) -> BoolRef:
        """Get or create an uninterpreted predicate function."""
        key = f"{name}/{len(args)}"

        if key not in self.predicate_cache:
            # Determine argument sorts
            arg_sorts = []
            for arg in args:
                if hasattr(arg, 'sort'):
                    arg_sorts.append(arg.sort())
                else:
                    arg_sorts.append(self.InputSort)

            self.predicate_cache[key] = Function(
                name,
                *arg_sorts,
                BoolSort()
            )

        return self.predicate_cache[key](*args)

    def _encode_term(self, term: FOLTerm) -> Any:
        """Encode FOL term to Z3."""
        if isinstance(term, FOLVar):
            return Const(term.name, self.InputSort)

        elif isinstance(term, FOLConstant):
            if term.sort == FOLSort.VALUE:
                return StringVal(str(term.value))
            elif term.sort == FOLSort.CATEGORY:
                return StringVal(str(term.value))
            elif term.sort == FOLSort.BOOL:
                return BoolVal(term.value)
            else:
                return StringVal(str(term.value))

        elif isinstance(term, FOLFunctionApp):
            func = self._get_var_function(term.function)
            args = [self._encode_term(a) for a in term.args]
            return func(*args)

        elif isinstance(term, FOLFormula):
            # Handle nested formulas (e.g., in not_warranted)
            return self.encode_formula(term)

        else:
            # Fallback: convert to string
            return StringVal(str(term))

    def _get_var_function(self, name: str) -> Any:
        """Get or create Z3 function for a Canto variable."""
        if name not in self.var_functions:
            # Determine return sort based on variable type
            return_sort = self.ValueSort  # Default to string

            if self._fol:
                var = self._fol.get_variable_for_function(name)
                if var and var.is_bool():
                    return_sort = BoolSort()

            self.var_functions[name] = Function(
                name,
                self.InputSort,
                return_sort
            )
        return self.var_functions[name]

    def _sort_to_z3(self, sort: FOLSort) -> Any:
        """Convert FOL sort to Z3 sort."""
        if sort == FOLSort.INPUT:
            return self.InputSort
        elif sort == FOLSort.VALUE:
            return self.ValueSort
        elif sort == FOLSort.CATEGORY:
            return self.CategorySort
        elif sort == FOLSort.BOOL:
            return BoolSort()
        else:
            return self.InputSort


class Z3Verifier:
    """
    Verifies properties of Canto FOL programs using Z3.

    Provides static verification checks that can catch errors
    at compile time before prompt generation.
    """

    def __init__(self, fol: CantoFOL):
        self.fol = fol
        self.encoder = Z3Encoder()
        # Set FOL reference for variable type lookups
        self.encoder._fol = fol
        self._z3_formula = None

    @property
    def z3_formula(self) -> BoolRef:
        """Lazily compute Z3 formula."""
        if self._z3_formula is None:
            self._z3_formula = self.encoder.encode(self.fol)
        return self._z3_formula

    def check_satisfiability(self) -> Tuple[bool, Optional[Any]]:
        """
        Check if the program constraints are satisfiable.

        Returns:
            (True, model) if satisfiable
            (False, None) if unsatisfiable
        """
        s = Solver()
        s.add(self.z3_formula)

        if s.check() == sat:
            return (True, s.model())
        else:
            return (False, None)

    def verify_no_contradictions(self) -> Tuple[bool, Optional[Dict]]:
        """
        Verify no two strict rules can fire with contradicting conclusions.

        This catches cases where two strict rules could both fire
        but would assign different values to the same variable.

        Returns:
            (True, None) if no contradictions
            (False, details) if contradiction found
        """
        # Group strict rules by variable
        by_var: Dict[str, List[CantoRule]] = {}
        for rule in self.fol.strict_rules:
            var = rule.head_variable
            if var not in by_var:
                by_var[var] = []
            by_var[var].append(rule)

        # Check each pair of rules for same variable
        for var, rules in by_var.items():
            for i, r1 in enumerate(rules):
                for r2 in rules[i + 1:]:
                    # Skip if same value (no conflict)
                    if r1.head_value == r2.head_value:
                        continue

                    # Check if both conditions can be satisfied
                    s = Solver()

                    if r1.conditions:
                        s.add(self.encoder.encode_formula(r1.conditions))
                    if r2.conditions:
                        s.add(self.encoder.encode_formula(r2.conditions))

                    if s.check() == sat:
                        return (False, {
                            'type': 'contradiction',
                            'rule1': r1.id,
                            'rule2': r2.id,
                            'variable': var,
                            'value1': r1.head_value,
                            'value2': r2.head_value,
                            'message': f"Strict rules {r1.id} and {r2.id} can both fire with different values",
                            'model': str(s.model())
                        })

        return (True, None)

    def verify_acyclicity(self) -> Tuple[bool, Optional[List[str]]]:
        """
        Verify superiority relation has no cycles.

        A cycle in superiority would make the program unsound.

        Returns:
            (True, None) if acyclic
            (False, cycle) if cycle found (list of rule IDs in cycle)
        """
        s = Solver()

        # Create ordering variable for each rule
        order = {}
        for rule in self.fol.all_rules():
            order[rule.id] = Const(f"order_{rule.id}", IntSort())

        # sup(A, B) implies order(A) > order(B)
        for sup in self.fol.superiority:
            if sup.superior in order and sup.inferior in order:
                s.add(order[sup.superior] > order[sup.inferior])

        if s.check() == sat:
            return (True, None)
        else:
            # Find the actual cycle using DFS
            cycle = self._find_cycle()
            return (False, cycle)

    def _find_cycle(self) -> List[str]:
        """Find a cycle in the superiority graph using DFS."""
        # Build adjacency list
        adj: Dict[str, List[str]] = {}
        for sup in self.fol.superiority:
            if sup.superior not in adj:
                adj[sup.superior] = []
            adj[sup.superior].append(sup.inferior)

        visited = set()
        path = []
        path_set = set()

        def dfs(node: str) -> Optional[List[str]]:
            if node in path_set:
                # Found cycle
                cycle_start = path.index(node)
                return path[cycle_start:] + [node]

            if node in visited:
                return None

            visited.add(node)
            path.append(node)
            path_set.add(node)

            for neighbor in adj.get(node, []):
                cycle = dfs(neighbor)
                if cycle:
                    return cycle

            path.pop()
            path_set.remove(node)
            return None

        for node in adj:
            cycle = dfs(node)
            if cycle:
                return cycle

        return []

    def verify_determinism(self) -> Tuple[bool, Optional[Dict]]:
        """
        Verify that for any input, each variable has at most one value.

        This checks that conflicts are properly resolved by superiority
        or rule type (strict beats defeasible).

        Returns:
            (True, None) if deterministic
            (False, details) if non-determinism detected
        """
        for var_name in self.fol.variables:
            values = list(self.fol.get_values_for_variable(var_name))
            if len(values) <= 1:
                continue

            # Get rules for each value
            rules_by_value: Dict[Any, List[CantoRule]] = {}
            for rule in self.fol.all_rules():
                if rule.head_variable == var_name:
                    val = rule.head_value
                    if val not in rules_by_value:
                        rules_by_value[val] = []
                    rules_by_value[val].append(rule)

            # Check if conflicts are resolved
            for v1, rules1 in rules_by_value.items():
                for v2, rules2 in rules_by_value.items():
                    if v1 >= v2:  # Avoid duplicate checks
                        continue

                    # Check for resolution mechanisms
                    has_resolution = False

                    # Check explicit superiority
                    for r1 in rules1:
                        for r2 in rules2:
                            if any(s.superior == r1.id and s.inferior == r2.id
                                   for s in self.fol.superiority):
                                has_resolution = True
                            if any(s.superior == r2.id and s.inferior == r1.id
                                   for s in self.fol.superiority):
                                has_resolution = True

                    # Check strict vs defeasible
                    if not has_resolution:
                        r1_has_strict = any(r.rule_type.value == 'strict' for r in rules1)
                        r2_has_strict = any(r.rule_type.value == 'strict' for r in rules2)
                        r1_has_defeasible = any(r.rule_type.value == 'defeasible' for r in rules1)
                        r2_has_defeasible = any(r.rule_type.value == 'defeasible' for r in rules2)

                        # Strict beats defeasible
                        if r1_has_strict and r2_has_defeasible and not r2_has_strict:
                            has_resolution = True
                        if r2_has_strict and r1_has_defeasible and not r1_has_strict:
                            has_resolution = True

                    if not has_resolution:
                        return (False, {
                            'type': 'non_determinism',
                            'variable': var_name,
                            'value1': v1,
                            'value2': v2,
                            'message': f"No clear resolution between {v1} and {v2} for {var_name}"
                        })

        return (True, None)

    def verify_all(self) -> Dict[str, Tuple[bool, Optional[Any]]]:
        """
        Run all verification checks.

        Returns:
            Dict mapping check name to (passed, details)
        """
        return {
            'satisfiability': self.check_satisfiability(),
            'no_contradictions': self.verify_no_contradictions(),
            'acyclicity': self.verify_acyclicity(),
            'determinism': self.verify_determinism(),
        }

    def check_implication(
        self,
        antecedent: FOLFormula,
        consequent: FOLFormula
    ) -> Tuple[bool, Optional[Any]]:
        """
        Check if antecedent → consequent is valid.

        This is used for equivalence checking between DSL and prompt.

        Valid iff there's no counterexample where antecedent holds
        but consequent doesn't.

        Returns:
            (True, None) if valid
            (False, counterexample) if invalid
        """
        s = Solver()

        ant_z3 = self.encoder.encode_formula(antecedent)
        con_z3 = self.encoder.encode_formula(consequent)

        if ant_z3 is not None:
            s.add(ant_z3)
        if con_z3 is not None:
            s.add(Z3Not(con_z3))

        if s.check() == unsat:
            return (True, None)
        else:
            return (False, s.model())
