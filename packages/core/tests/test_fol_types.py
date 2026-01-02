"""
Unit tests for FOL types and basic operations.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from canto_core.fol.types import (
    FOLSort,
    FOLVariable,
    FOLConstant,
    FOLVar,
    FOLFunctionApp,
    FOLPredicate,
    FOLEquals,
    FOLNot,
    FOLAnd,
    FOLOr,
    FOLImplies,
    FOLForall,
    FOLExists,
    make_and,
    make_or,
    make_implies,
    make_forall,
    make_exists,
)


class TestFOLSorts:
    """Test FOL sort definitions."""

    def test_sort_values(self):
        """Test that all expected sorts exist."""
        assert FOLSort.INPUT.value == "input"
        assert FOLSort.VALUE.value == "value"
        assert FOLSort.CATEGORY.value == "category"
        assert FOLSort.BOOL.value == "bool"


class TestFOLTerms:
    """Test FOL term construction."""

    def test_fol_variable(self):
        """Test FOLVariable creation."""
        var = FOLVariable("x", FOLSort.INPUT)
        assert var.name == "x"
        assert var.sort == FOLSort.INPUT

    def test_fol_constant(self):
        """Test FOLConstant creation."""
        const = FOLConstant("high", FOLSort.VALUE)
        assert const.value == "high"
        assert const.sort == FOLSort.VALUE

    def test_fol_var(self):
        """Test FOLVar (untyped variable) creation."""
        var = FOLVar("x")
        assert var.name == "x"

    def test_fol_function_app(self):
        """Test FOLFunctionApp creation."""
        x = FOLVar("x")
        app = FOLFunctionApp("priority", [x])
        assert app.function == "priority"
        assert len(app.args) == 1
        assert app.args[0] == x


class TestFOLFormulas:
    """Test FOL formula construction."""

    def test_predicate(self):
        """Test FOLPredicate creation."""
        x = FOLVar("x")
        cat = FOLConstant("urgent", FOLSort.CATEGORY)
        pred = FOLPredicate("is_like", [x, cat])
        assert pred.name == "is_like"
        assert len(pred.args) == 2

    def test_equals(self):
        """Test FOLEquals creation."""
        x = FOLVar("x")
        val = FOLConstant("high", FOLSort.VALUE)
        eq = FOLEquals(x, val)
        assert eq.left == x
        assert eq.right == val

    def test_not(self):
        """Test FOLNot creation."""
        pred = FOLPredicate("true", [])
        neg = FOLNot(pred)
        assert neg.formula == pred

    def test_and(self):
        """Test FOLAnd creation."""
        p1 = FOLPredicate("p1", [])
        p2 = FOLPredicate("p2", [])
        conj = FOLAnd([p1, p2])
        assert len(conj.conjuncts) == 2

    def test_or(self):
        """Test FOLOr creation."""
        p1 = FOLPredicate("p1", [])
        p2 = FOLPredicate("p2", [])
        disj = FOLOr([p1, p2])
        assert len(disj.disjuncts) == 2

    def test_implies(self):
        """Test FOLImplies creation."""
        p1 = FOLPredicate("p1", [])
        p2 = FOLPredicate("p2", [])
        impl = FOLImplies(p1, p2)
        assert impl.antecedent == p1
        assert impl.consequent == p2

    def test_forall(self):
        """Test FOLForall creation."""
        x = FOLVariable("x", FOLSort.INPUT)
        pred = FOLPredicate("p", [FOLVar("x")])
        forall = FOLForall(x, pred)
        assert forall.variable == x
        assert forall.formula == pred

    def test_exists(self):
        """Test FOLExists creation."""
        x = FOLVariable("x", FOLSort.INPUT)
        pred = FOLPredicate("p", [FOLVar("x")])
        exists = FOLExists(x, pred)
        assert exists.variable == x
        assert exists.formula == pred


class TestHelperFunctions:
    """Test helper functions for building formulas."""

    def test_make_and_empty(self):
        """Test make_and with empty list."""
        result = make_and([])
        assert isinstance(result, FOLPredicate)
        assert result.name == "true"

    def test_make_and_single(self):
        """Test make_and with single formula."""
        p = FOLPredicate("p", [])
        result = make_and([p])
        assert result == p

    def test_make_and_multiple(self):
        """Test make_and with multiple formulas."""
        p1 = FOLPredicate("p1", [])
        p2 = FOLPredicate("p2", [])
        result = make_and([p1, p2])
        assert isinstance(result, FOLAnd)
        assert len(result.conjuncts) == 2

    def test_make_or_empty(self):
        """Test make_or with empty list."""
        result = make_or([])
        assert isinstance(result, FOLPredicate)
        assert result.name == "false"

    def test_make_or_single(self):
        """Test make_or with single formula."""
        p = FOLPredicate("p", [])
        result = make_or([p])
        assert result == p

    def test_make_or_multiple(self):
        """Test make_or with multiple formulas."""
        p1 = FOLPredicate("p1", [])
        p2 = FOLPredicate("p2", [])
        result = make_or([p1, p2])
        assert isinstance(result, FOLOr)
        assert len(result.disjuncts) == 2

    def test_make_implies(self):
        """Test make_implies."""
        p1 = FOLPredicate("p1", [])
        p2 = FOLPredicate("p2", [])
        result = make_implies(p1, p2)
        assert isinstance(result, FOLImplies)

    def test_make_forall(self):
        """Test make_forall."""
        p = FOLPredicate("p", [FOLVar("x")])
        result = make_forall("x", FOLSort.INPUT, p)
        assert isinstance(result, FOLForall)
        assert result.variable.name == "x"

    def test_make_exists(self):
        """Test make_exists."""
        p = FOLPredicate("p", [FOLVar("x")])
        result = make_exists("x", FOLSort.INPUT, p)
        assert isinstance(result, FOLExists)
        assert result.variable.name == "x"


class TestFormulaEquality:
    """Test formula equality and hashing."""

    def test_predicate_equality(self):
        """Test that identical predicates are equal."""
        p1 = FOLPredicate("p", [FOLVar("x")])
        p2 = FOLPredicate("p", [FOLVar("x")])
        assert p1 == p2

    def test_constant_equality(self):
        """Test that identical constants are equal."""
        c1 = FOLConstant("value", FOLSort.VALUE)
        c2 = FOLConstant("value", FOLSort.VALUE)
        assert c1 == c2

    def test_different_predicates(self):
        """Test that different predicates are not equal."""
        p1 = FOLPredicate("p1", [])
        p2 = FOLPredicate("p2", [])
        assert p1 != p2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
