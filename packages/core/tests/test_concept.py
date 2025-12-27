"""
Tests for Concept class - Python-Canto interop for semantic structures.
"""

import pytest
from canto_core.concept import Concept
from canto_core.ast_nodes import VariableDeclaration, SemanticCategory


class TestConceptBasic:
    """Basic Concept creation and properties."""

    def test_create_concept_with_name(self):
        """Test creating a concept with just a name."""
        c = Concept("vehicle")
        assert c.name == "vehicle"

    def test_concept_meaning(self):
        """Test setting meaning on a concept."""
        c = Concept("vehicle").meaning("a machine used for transportation")
        assert c._meaning == "a machine used for transportation"

    def test_concept_resembles(self):
        """Test setting resembles terms on a concept."""
        c = Concept("vehicle").resembles("car", "truck", "automobile")
        assert c._resembles == ["car", "truck", "automobile"]

    def test_concept_resembles_multiple_calls(self):
        """Test that multiple resembles calls accumulate."""
        c = (
            Concept("vehicle")
            .resembles("car", "truck")
            .resembles("automobile", "transport")
        )
        assert c._resembles == ["car", "truck", "automobile", "transport"]

    def test_concept_can_be_strings(self):
        """Test setting can_be with string values."""
        c = Concept("vehicle").can_be("land", "water", "air")
        assert c._can_be == ["land", "water", "air"]

    def test_concept_can_be_booleans(self):
        """Test setting can_be with boolean values."""
        c = Concept("is_valid").can_be(True, False)
        assert c._can_be == [True, False]

    def test_concept_can_be_mixed(self):
        """Test can_be with mixed string and boolean values."""
        c = Concept("status").can_be("active", "inactive", True, False)
        assert c._can_be == ["active", "inactive", True, False]


class TestConceptFluent:
    """Test fluent/builder pattern for Concept."""

    def test_fluent_chain(self):
        """Test chaining all methods."""
        c = (
            Concept("vehicle")
            .meaning("a machine used for transportation")
            .resembles("car", "truck", "automobile")
            .can_be("land", "water", "air")
        )

        assert c.name == "vehicle"
        assert c._meaning == "a machine used for transportation"
        assert c._resembles == ["car", "truck", "automobile"]
        assert c._can_be == ["land", "water", "air"]

    def test_fluent_returns_self(self):
        """Test that all methods return self for chaining."""
        c = Concept("test")
        assert c.meaning("desc") is c
        assert c.resembles("a", "b") is c
        assert c.can_be("x", "y") is c
        assert c.has(Concept("child")) is c


class TestConceptNested:
    """Test nested concepts with has()."""

    def test_has_single_child(self):
        """Test adding a single child concept."""
        parent = Concept("patient").has(
            Concept("name").meaning("patient's full name")
        )

        assert len(parent._children) == 1
        assert parent._children[0].name == "name"
        assert parent._children[0]._meaning == "patient's full name"

    def test_has_multiple_children(self):
        """Test adding multiple child concepts."""
        parent = Concept("patient").has(
            Concept("name").meaning("patient's full name"),
            Concept("age").meaning("patient's age in years"),
            Concept("dob").meaning("date of birth"),
        )

        assert len(parent._children) == 3
        assert [c.name for c in parent._children] == ["name", "age", "dob"]

    def test_has_nested_children(self):
        """Test deeply nested concepts."""
        patient = (
            Concept("patient")
            .meaning("extracted patient information")
            .has(
                Concept("name").meaning("patient's full name"),
                Concept("symptoms").has(
                    Concept("text").meaning("symptom description"),
                    Concept("is_emergency").can_be(True, False),
                ),
            )
        )

        assert len(patient._children) == 2
        symptoms = patient._children[1]
        assert symptoms.name == "symptoms"
        assert len(symptoms._children) == 2
        assert symptoms._children[0].name == "text"
        assert symptoms._children[1]._can_be == [True, False]

    def test_has_chained_calls(self):
        """Test multiple has() calls accumulate children."""
        patient = (
            Concept("patient")
            .has(Concept("name"))
            .has(Concept("age"))
        )

        assert len(patient._children) == 2


class TestConceptToAst:
    """Test converting Concept to AST nodes."""

    def test_to_ast_simple_meaning(self):
        """Test generating VariableDeclaration from meaning."""
        c = Concept("patient").meaning("extracted patient information")
        ast = c.to_ast()

        assert isinstance(ast, VariableDeclaration)
        assert ast.name == "patient"
        assert ast.description == "extracted patient information"

    def test_to_ast_resembles(self):
        """Test generating SemanticCategory from resembles."""
        c = Concept("vehicle_terms").resembles("car", "truck", "automobile")
        ast = c.to_ast()

        assert isinstance(ast, SemanticCategory)
        assert ast.name == "vehicle_terms"
        assert ast.patterns == ["car", "truck", "automobile"]

    def test_to_ast_resembles_with_meaning(self):
        """Test SemanticCategory includes meaning as description."""
        c = (
            Concept("vehicle_terms")
            .resembles("car", "truck")
            .meaning("vehicle-related terms")
        )
        ast = c.to_ast()

        assert isinstance(ast, SemanticCategory)
        assert ast.patterns == ["car", "truck"]
        assert ast.description == "vehicle-related terms"

    def test_to_ast_can_be_strings(self):
        """Test VariableDeclaration with string values_from."""
        c = Concept("status").can_be("active", "inactive", "pending")
        ast = c.to_ast()

        assert isinstance(ast, VariableDeclaration)
        assert ast.name == "status"
        assert ast.values_from == ["active", "inactive", "pending"]

    def test_to_ast_can_be_booleans(self):
        """Test VariableDeclaration with boolean values_from."""
        c = Concept("is_valid").can_be(True, False)
        ast = c.to_ast()

        assert isinstance(ast, VariableDeclaration)
        assert ast.values_from == ["true", "false"]

    def test_to_ast_can_be_with_meaning(self):
        """Test VariableDeclaration with values_from and description."""
        c = (
            Concept("status")
            .can_be("active", "inactive")
            .meaning("the current status")
        )
        ast = c.to_ast()

        assert isinstance(ast, VariableDeclaration)
        assert ast.values_from == ["active", "inactive"]
        assert ast.description == "the current status"

    def test_to_ast_nested_children(self):
        """Test VariableDeclaration with nested children."""
        patient = (
            Concept("patient")
            .meaning("extracted patient information")
            .has(
                Concept("name").meaning("patient's full name"),
                Concept("age").meaning("patient's age in years"),
            )
        )
        ast = patient.to_ast()

        assert isinstance(ast, VariableDeclaration)
        assert ast.name == "patient"
        assert ast.description == "extracted patient information"
        assert len(ast.children) == 2
        assert ast.children[0].name == "name"
        assert ast.children[1].name == "age"

    def test_to_ast_deeply_nested(self):
        """Test deeply nested AST structure."""
        patient = (
            Concept("patient")
            .meaning("patient info")
            .has(
                Concept("symptoms").has(
                    Concept("text").meaning("symptom description"),
                    Concept("is_emergency").can_be(True, False),
                )
            )
        )
        ast = patient.to_ast()

        assert isinstance(ast, VariableDeclaration)
        assert len(ast.children) == 1
        symptoms = ast.children[0]
        assert symptoms.name == "symptoms"
        assert len(symptoms.children) == 2
        assert symptoms.children[0].name == "text"
        assert symptoms.children[1].values_from == ["true", "false"]


class TestConceptToDict:
    """Test serialization to dictionary."""

    def test_to_dict_simple(self):
        """Test converting simple concept to dict."""
        c = Concept("vehicle").meaning("a transport machine")
        d = c.to_dict()

        assert d["name"] == "vehicle"
        assert d["meaning"] == "a transport machine"

    def test_to_dict_full(self):
        """Test converting concept with all properties to dict."""
        c = (
            Concept("vehicle")
            .meaning("a transport machine")
            .resembles("car", "truck")
            .can_be("land", "water")
        )
        d = c.to_dict()

        assert d["name"] == "vehicle"
        assert d["meaning"] == "a transport machine"
        assert d["resembles"] == ["car", "truck"]
        assert d["can_be"] == ["land", "water"]

    def test_to_dict_nested(self):
        """Test converting nested concept to dict."""
        patient = (
            Concept("patient")
            .meaning("patient info")
            .has(
                Concept("name").meaning("full name"),
                Concept("age").meaning("age in years"),
            )
        )
        d = patient.to_dict()

        assert d["name"] == "patient"
        assert len(d["has"]) == 2
        assert d["has"][0]["name"] == "name"
        assert d["has"][1]["name"] == "age"


class TestConceptFromDict:
    """Test creating Concept from dictionary."""

    def test_from_dict_simple(self):
        """Test creating concept from simple dict."""
        d = {"name": "vehicle", "meaning": "a transport machine"}
        c = Concept.from_dict(d)

        assert c.name == "vehicle"
        assert c._meaning == "a transport machine"

    def test_from_dict_full(self):
        """Test creating concept from dict with all properties."""
        d = {
            "name": "vehicle",
            "meaning": "a transport machine",
            "resembles": ["car", "truck"],
            "can_be": ["land", "water"],
        }
        c = Concept.from_dict(d)

        assert c.name == "vehicle"
        assert c._meaning == "a transport machine"
        assert c._resembles == ["car", "truck"]
        assert c._can_be == ["land", "water"]

    def test_from_dict_nested(self):
        """Test creating nested concept from dict."""
        d = {
            "name": "patient",
            "meaning": "patient info",
            "has": [
                {"name": "name", "meaning": "full name"},
                {"name": "age", "meaning": "age in years"},
            ],
        }
        c = Concept.from_dict(d)

        assert c.name == "patient"
        assert len(c._children) == 2
        assert c._children[0].name == "name"
        assert c._children[1].name == "age"

    def test_roundtrip_dict(self):
        """Test that to_dict -> from_dict preserves the concept."""
        original = (
            Concept("patient")
            .meaning("patient info")
            .resembles("client", "subject")
            .can_be("adult", "child")
            .has(
                Concept("name").meaning("full name"),
                Concept("status").can_be("active", "inactive"),
            )
        )

        d = original.to_dict()
        restored = Concept.from_dict(d)

        assert restored.name == original.name
        assert restored._meaning == original._meaning
        assert restored._resembles == original._resembles
        assert restored._can_be == original._can_be
        assert len(restored._children) == len(original._children)


class TestConceptValidation:
    """Test validation of concept definitions."""

    def test_empty_name_raises(self):
        """Test that empty name raises ValueError."""
        with pytest.raises(ValueError, match="name"):
            Concept("")

    def test_none_name_raises(self):
        """Test that None name raises TypeError."""
        with pytest.raises(TypeError):
            Concept(None)

    def test_invalid_name_characters(self):
        """Test that invalid characters in name raise ValueError."""
        with pytest.raises(ValueError, match="name"):
            Concept("invalid-name")  # hyphens not allowed

        with pytest.raises(ValueError, match="name"):
            Concept("123start")  # can't start with number


class TestConceptEquality:
    """Test concept equality and hashing."""

    def test_equality_same_name(self):
        """Test that concepts with same name are equal."""
        c1 = Concept("vehicle")
        c2 = Concept("vehicle")
        assert c1 == c2

    def test_equality_with_properties(self):
        """Test that concepts with same properties are equal."""
        c1 = Concept("vehicle").meaning("transport").resembles("car")
        c2 = Concept("vehicle").meaning("transport").resembles("car")
        assert c1 == c2

    def test_inequality_different_name(self):
        """Test that concepts with different names are not equal."""
        c1 = Concept("vehicle")
        c2 = Concept("transport")
        assert c1 != c2

    def test_hash_same_for_equal(self):
        """Test that equal concepts have same hash."""
        c1 = Concept("vehicle").meaning("transport")
        c2 = Concept("vehicle").meaning("transport")
        assert hash(c1) == hash(c2)


class TestCantoBuilder:
    """Test CantoBuilder for build-time concept injection."""

    def test_register_concept(self):
        """Test registering a concept."""
        from canto_core import CantoBuilder

        builder = CantoBuilder()
        concept = Concept("test").meaning("a test concept")

        result = builder.register_concept(concept)

        assert result is builder  # returns self for chaining
        assert "test" in builder._concepts

    def test_register_multiple_concepts(self):
        """Test registering multiple concepts."""
        from canto_core import CantoBuilder

        builder = CantoBuilder()
        c1 = Concept("patient").meaning("patient info")
        c2 = Concept("diagnosis").meaning("diagnosis info")

        builder.register_concept(c1).register_concept(c2)

        assert len(builder._concepts) == 2
        assert "patient" in builder._concepts
        assert "diagnosis" in builder._concepts

    def test_build_string_returns_build_result(self):
        """Test that build_string returns BuildResult with symbol table."""
        from canto_core import CantoBuilder, BuildResult

        builder = CantoBuilder()
        builder.register_concept(
            Concept("emergency_symptoms")
            .resembles("chest pain", "difficulty breathing")
            .meaning("emergency indicators")
        )

        result = builder.build_string("""
            ?is_emergency can be true, false
        """)

        # Should return BuildResult
        assert isinstance(result, BuildResult)

        # AST should only have the parsed node (not injected concept)
        assert len(result.ast) == 1
        assert result.ast[0].name == "is_emergency"

        # Symbol table should have both
        assert "emergency_symptoms" in result.symbols
        assert "is_emergency" in result.symbols

        # Concepts should be stored separately
        assert "emergency_symptoms" in result.concepts
        assert isinstance(result.concepts["emergency_symptoms"], Concept)

    def test_build_string_preserves_instructions(self):
        """Test that build_string preserves instructions."""
        from canto_core import CantoBuilder

        builder = CantoBuilder()
        builder.register_concept(Concept("test").meaning("test"))

        result = builder.build_string('''
            """
            These are instructions.
            """

            ?var meaning "a variable"
        ''')

        assert result.instructions == "These are instructions."

    def test_build_string_nested_concept(self):
        """Test injecting nested concepts."""
        from canto_core import CantoBuilder

        patient = (
            Concept("patient")
            .meaning("patient info")
            .has(
                Concept("name").meaning("full name"),
                Concept("age").meaning("age in years"),
            )
        )

        builder = CantoBuilder()
        builder.register_concept(patient)

        result = builder.build_string("?status can be true, false")

        # AST should only have the parsed declaration
        assert len(result.ast) == 1

        # Concepts stored separately with nested structure
        assert "patient" in result.concepts
        patient_concept = result.concepts["patient"]
        assert len(patient_concept._children) == 2
        assert patient_concept._children[0].name == "name"
        assert patient_concept._children[1].name == "age"

        # Symbol table should have the patient concept
        assert "patient" in result.symbols


class TestSymbolTable:
    """Test SymbolTable for name resolution."""

    def test_declare_and_resolve(self):
        """Test basic declare and resolve."""
        from canto_core import SymbolTable, SymbolKind

        symbols = SymbolTable()
        symbols.declare("patient", SymbolKind.VARIABLE, object())

        symbol = symbols.resolve("patient")
        assert symbol.name == "patient"
        assert symbol.kind == SymbolKind.VARIABLE

    def test_resolve_with_question_mark(self):
        """Test that resolve handles ? prefix."""
        from canto_core import SymbolTable, SymbolKind

        symbols = SymbolTable()
        symbols.declare("patient", SymbolKind.VARIABLE, object())

        # Should work with or without ?
        assert symbols.resolve("patient").name == "patient"
        assert symbols.resolve("?patient").name == "patient"

    def test_unresolved_reference_error(self):
        """Test that resolving unknown name raises error."""
        from canto_core import SymbolTable, UnresolvedReferenceError

        symbols = SymbolTable()

        with pytest.raises(UnresolvedReferenceError) as exc_info:
            symbols.resolve("unknown")

        assert "unknown" in str(exc_info.value)

    def test_duplicate_declaration_error(self):
        """Test that duplicate declarations raise error."""
        from canto_core import SymbolTable, SymbolKind, DuplicateDeclarationError

        symbols = SymbolTable()
        symbols.declare("patient", SymbolKind.VARIABLE, object())

        with pytest.raises(DuplicateDeclarationError) as exc_info:
            symbols.declare("patient", SymbolKind.CONCEPT, object())

        assert "patient" in str(exc_info.value)

    def test_validate_references(self):
        """Test reference validation."""
        from canto_core import SymbolTable, SymbolKind

        symbols = SymbolTable()
        symbols.declare("patient", SymbolKind.VARIABLE, object())

        symbols.add_reference("patient")
        symbols.add_reference("unknown")

        errors = symbols.validate_references()

        assert len(errors) == 1
        assert errors[0].name == "unknown"

    def test_get_unused_symbols(self):
        """Test finding unused symbols."""
        from canto_core import SymbolTable, SymbolKind

        symbols = SymbolTable()
        symbols.declare("used", SymbolKind.VARIABLE, object())
        symbols.declare("unused", SymbolKind.VARIABLE, object())

        symbols.add_reference("used")

        unused = symbols.get_unused_symbols()

        assert len(unused) == 1
        assert unused[0].name == "unused"


class TestBuildResolution:
    """Test resolution during build."""

    def test_build_with_unresolved_reference_strict(self):
        """Test that strict mode raises on unresolved references."""
        from canto_core import CantoBuilder, ResolutionErrors

        builder = CantoBuilder(strict=True)

        # This references ?unknown which is not declared
        with pytest.raises(ResolutionErrors) as exc_info:
            builder.build_string("""
                ?result becomes true when ?unknown is true
            """)

        assert "unknown" in str(exc_info.value)

    def test_build_with_unresolved_reference_non_strict(self):
        """Test that non-strict mode collects errors without raising."""
        from canto_core import CantoBuilder

        builder = CantoBuilder(strict=False)

        # This should not raise
        result = builder.build_string("""
            ?result becomes true when ?unknown is true
        """)

        # But we should still get a valid result
        assert len(result.ast) == 1

    def test_build_with_concept_resolves_reference(self):
        """Test that registered concepts satisfy references."""
        from canto_core import CantoBuilder

        builder = CantoBuilder(strict=True)
        builder.register_concept(
            Concept("emergency_symptoms")
            .resembles("chest pain")
        )

        # This references ?emergency_symptoms which is provided by concept
        # Also declare ?complaint so all references are satisfied
        result = builder.build_string("""
            ?complaint meaning "patient complaint"
            ?is_emergency can be true, false
            ?is_emergency becomes true when ?complaint is like ?emergency_symptoms
        """)

        # Should resolve without error
        assert "emergency_symptoms" in result.symbols
        assert result.symbols.resolve("emergency_symptoms").name == "emergency_symptoms"

    def test_symbol_table_has_concepts_and_declarations(self):
        """Test that symbol table contains both concepts and declarations."""
        from canto_core import CantoBuilder, SymbolKind

        builder = CantoBuilder()
        builder.register_concept(Concept("injected").meaning("from Python"))

        result = builder.build_string("""
            ?declared meaning "from Canto"
        """)

        # Both should be in symbol table
        assert "injected" in result.symbols
        assert "declared" in result.symbols

        # With correct kinds
        assert result.symbols.resolve("injected").kind == SymbolKind.CONCEPT
        assert result.symbols.resolve("declared").kind == SymbolKind.VARIABLE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
