"""
LLM-based Prolog Predicate Generation

Uses DSPy and ILP-inspired approach to generate Prolog predicates from LIKE/AS clauses.
"""

import dspy
from typing import List, Optional
from ..ast_nodes import SemanticCategory
from .prolog_verifier import PrologSyntaxVerifier
from ..utils.prolog import normalize_prolog_atom


class GenerateSymbolicPredicates(dspy.Signature):
    """Generate Prolog pattern facts for semantic category matching.

    Your task is to GENERALIZE from the given examples and generate additional
    pattern/2 facts. Think about synonyms, related terms, variations, and
    terms in other languages that belong to this category.

    ONLY generate pattern/2 facts in this format:
        pattern(category_name, term).

    Do NOT generate:
    - matches/2 or any other predicates
    - Helper predicates or rules (no :- clauses)
    - module/2 directives
    - Comments (just facts)

    TERM FORMAT RULES (CRITICAL):
    - All terms MUST be valid unquoted Prolog atoms
    - In Prolog, unquoted atoms MUST start with a lowercase letter
    - In Prolog, unquoted atoms contain only lowercase letters, digits, and underscores
    - NEVER start an atom with a digit - if you need to, add a prefix like 'n_'
      - BAD: pattern(cat, 2021_report). -- syntax error!
      - BAD: pattern(cat, 1st_place).   -- syntax error!
      - BAD: pattern(cat, 10k).         -- syntax error!
      - GOOD: pattern(cat, year_2021).  -- valid atom, starts with letter
      - GOOD: pattern(cat, first_place). -- valid atom
      - GOOD: pattern(cat, n_10k).      -- valid atom, prefixed with n_
    - NEVER use uppercase letters - in Prolog, uppercase = variable!
      - BAD: pattern(cat, Sale).        -- Sale is a variable!
      - GOOD: pattern(cat, sale).       -- sale is an atom
    - Replace spaces and special characters with underscores
    - Remove apostrophes and quotes: "don't" -> dont, "sott'olio" -> sottolio
    - Convert unicode/accents to ASCII: "café" -> cafe, "naïve" -> naive
    - Examples:
      - pattern(my_cat, sun_dried).     -- not 'sun-dried'
      - pattern(my_cat, dont).          -- not 'don''t'
      - pattern(my_cat, sottolio).      -- not 'sott''olio'
      - pattern(my_cat, cafe).          -- not 'café'
      - pattern(my_cat, year_2010).     -- not '2010'
    """

    category_name = dspy.InputField(desc="Name of the semantic category")
    description = dspy.InputField(desc="Natural language description of the category")
    examples = dspy.InputField(desc="List of example terms that belong to this category")

    reasoning = dspy.OutputField(desc="Your reasoning about patterns and generalizations")
    prolog_code = dspy.OutputField(desc="Complete Prolog predicates. Output ONLY valid Prolog code. Do NOT generate matches/2 predicates.")


# Few-shot examples for DSPy
# NOTE: Only pattern/2 facts - no helper predicates
# NOTE: All terms use unquoted lowercase atoms with underscores (no quotes, no unicode)
EXAMPLES = [
    dspy.Example(
        category_name="vaccine_terms",
        description="explicit vaccine-related medical terminology",
        examples=["vaccine", "vaccination", "immunization"],
        reasoning="""The examples show vaccine-related medical terms. I can generalize to:
1. Direct term matches (vaccine, vaccination, immunization)
2. Related terms (inoculation, immunize, vax, jab)
3. Variations and informal terms
All terms normalized to unquoted lowercase atoms.""",
        prolog_code="""pattern(vaccine_terms, vaccine).
pattern(vaccine_terms, vaccination).
pattern(vaccine_terms, immunization).
pattern(vaccine_terms, inoculation).
pattern(vaccine_terms, immunize).
pattern(vaccine_terms, vax).
pattern(vaccine_terms, jab).
pattern(vaccine_terms, shot)."""
    ),
    dspy.Example(
        category_name="treatment_intent",
        description="language indicating treatment of existing conditions",
        examples=["treat", "treating", "medication for", "drugs for"],
        reasoning="""The examples show treatment-related intent. Generalizing:
1. Direct verb forms (treat, treating, manage, cure)
2. Medication-seeking phrases normalized with underscores
3. Related therapeutic terms
All terms normalized to unquoted lowercase atoms with underscores.""",
        prolog_code="""pattern(treatment_intent, treat).
pattern(treatment_intent, treating).
pattern(treatment_intent, manage).
pattern(treatment_intent, cure).
pattern(treatment_intent, therapy).
pattern(treatment_intent, medication_for).
pattern(treatment_intent, drugs_for).
pattern(treatment_intent, medicine_for).
pattern(treatment_intent, remedy)."""
    ),
    dspy.Example(
        category_name="information_request_terms",
        description="phrases indicating request for information",
        examples=["what", "which", "list", "tell me about"],
        reasoning="""Information-seeking patterns. Generalizing:
1. Question words (what, which, who, how)
2. Request phrases normalized with underscores
3. Imperative information requests
All terms normalized to unquoted lowercase atoms.""",
        prolog_code="""pattern(information_request_terms, what).
pattern(information_request_terms, which).
pattern(information_request_terms, who).
pattern(information_request_terms, how).
pattern(information_request_terms, tell_me).
pattern(information_request_terms, list).
pattern(information_request_terms, show_me).
pattern(information_request_terms, explain).
pattern(information_request_terms, describe)."""
    )
]


class LLMPredicateGenerator:
    """
    Generates Prolog predicates from semantic categories using LLM as ILP engine
    """

    def __init__(self, max_retries: int = 3):
        """
        Initialize the predicate generator

        Args:
            max_retries: Maximum number of retry attempts for LLM generation
        """
        self.generator = dspy.ChainOfThought(GenerateSymbolicPredicates)
        self.verifier = PrologSyntaxVerifier()
        self.max_retries = max_retries

        # Compile with few-shot examples
        self._setup_examples()

    def _setup_examples(self):
        """Setup few-shot examples for the generator"""
        # Add examples to DSPy for in-context learning
        for example in EXAMPLES:
            self.generator.demos = EXAMPLES

    def generate_from_category(self, category: SemanticCategory) -> str:
        """
        Generate Prolog predicates from a semantic category

        Args:
            category: SemanticCategory with patterns (LIKE) and description (AS)

        Returns:
            Prolog code defining predicates for this category
        """
        # Build the prompt
        category_name = category.name
        description = category.description or f"Terms matching {category_name}"
        examples = category.patterns  # List of example terms from LIKE clause

        # Try generating with retries
        last_error = None
        for attempt in range(self.max_retries):
            try:
                print(f"Attempt {attempt + 1}/{self.max_retries} to generate predicates for {category_name}...")

                # Generate via DSPy
                result = self.generator(
                    category_name=category_name,
                    description=description,
                    examples=examples
                )

                prolog_code = result.prolog_code

                # Verify syntax
                is_valid, error_msg = self.verifier.verify_syntax(prolog_code)

                if is_valid:
                    print(f"✓ Generated valid Prolog predicates for {category_name}")
                    return self._format_output(category, prolog_code)
                else:
                    last_error = error_msg
                    print(f"✗ Syntax error on attempt {attempt + 1}: {error_msg}")

            except Exception as e:
                last_error = str(e)
                print(f"✗ Generation failed on attempt {attempt + 1}: {e}")

        # All retries failed - use fallback
        print(f"⚠ All retries failed for {category_name}. Using fallback predicates.")
        print(f"  Last error: {last_error}")
        return self._generate_fallback(category)

    def _format_output(self, category: SemanticCategory, prolog_code: str) -> str:
        """Format the LLM-generated output with header comments"""
        header = f"""% Generated predicates for: {category.name}
% Description: {category.description or 'N/A'}
% Generated via LLM (ILP-inspired)

"""
        return header + prolog_code.strip()

    def _generate_fallback(self, category: SemanticCategory) -> str:
        """
        Generate simple fallback predicates if LLM fails

        Uses pattern/2 facts (same as translator, used by meta-interpreter)
        """
        rules = []

        # Header
        rules.append(f"% Generated predicates for: {category.name} (FALLBACK)")
        rules.append(f"% Description: {category.description or 'N/A'}")
        rules.append("")

        # Generate pattern facts for each example
        for p in category.patterns:
            normalized = normalize_prolog_atom(p)
            rules.append(
                f"pattern({category.name}, {normalized})."
            )

        return "\n".join(rules)


def configure_dspy(model: str = "gpt-4o-mini", **kwargs):
    """
    Configure DSPy with LLM backend

    Args:
        model: Model identifier (e.g., "gpt-4o-mini", "gpt-4o", "claude-3-5-sonnet-20241022")
        **kwargs: Additional configuration for the model (like api_key)

    Example:
        configure_dspy("gpt-4o-mini", api_key="...")
        configure_dspy("claude-3-5-sonnet-20241022", api_key="...")
    """
    # DSPy 3.x uses dspy.LM
    lm = dspy.LM(model, **kwargs)
    dspy.configure(lm=lm)
