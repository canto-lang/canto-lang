"""
Food Ontology Concepts

Semantic structures for food product classification and extraction.
"""

from canto_core.concept import Concept


# Food item extraction structure
food_item = (
    Concept("food_item")
    .meaning("extracted food ontology from input name")
    .has(
        Concept("canonical_name").meaning("base food name in English, singular"),
        Concept("variety").meaning("cultivar or size descriptor"),
        Concept("form")
        .meaning("processing states applied to the food")
        .has(
            Concept("form_name")
            .meaning("canonical form name")
            .can_be("dried", "frozen", "canned", "fresh", "ground", "sliced")
        ),
        Concept("presentation").meaning("packaging type"),
        Concept("is_organic").can_be(True, False).meaning("whether product is organic"),
        Concept("is_mix").can_be(True, False).meaning("whether product is a mixture"),
    )
)

# Processing method patterns
drying_terms = (
    Concept("drying_terms")
    .resembles("dried", "dehydrated", "sun-dried", "secchi", "essiccato")
    .meaning("drying process indicators")
)

freezing_terms = (
    Concept("freezing_terms")
    .resembles("frozen", "flash-frozen", "surgelato", "congelato")
    .meaning("freezing process indicators")
)

canning_terms = (
    Concept("canning_terms")
    .resembles("canned", "tinned", "preserved", "in scatola")
    .meaning("canning process indicators")
)

# Variety patterns
cherry_variety = (
    Concept("cherry_variety")
    .resembles("cherry", "ciliegino", "ciliegini", "pachino")
    .meaning("cherry tomato variety")
)

# Attribute patterns
organic_terms = (
    Concept("organic_terms")
    .resembles("organic", "bio", "biologico")
    .meaning("organic certification")
)

dop_terms = (
    Concept("dop_terms")
    .resembles("DOP", "denominazione di origine protetta")
    .meaning("DOP certification")
)
