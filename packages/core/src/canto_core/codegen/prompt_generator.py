"""
Prompt Template Generator - BUILD Step 3

Converts DeLP reasoning structure into a reusable LLM prompt template.
Integrates FOL verification to ensure generated prompts match DSL semantics.
"""

import dspy
from typing import Dict, Any, List, Optional
from pathlib import Path

from ..fol import (
    ASTToFOLTranslator,
    EquivalenceVerifier,
    CantoFOL,
)


class GeneratePromptTemplate(dspy.Signature):
    """Generate LLM prompt template from symbolic reasoning structure.

    Your task is to convert formal reasoning rules (strict vs defeasible, with conflicts)
    into clear, natural language instructions for an LLM.

    Key requirements:
    1. CONSOLIDATE multiple strict rules into unified guidance (not mechanical listing)
    2. Convert OVERRIDES relationships into "Important:" or "Do NOT..." clarifications
    3. Embed semantic pattern examples INLINE with instructions (not separate sections)
    4. Provide POSITIVE and NEGATIVE examples to disambiguate edge cases
    5. Structure as natural instructions, not formal logic
    6. When hierarchy information is provided, structure the output to reflect nested relationships

    Example of good consolidation:
    - Bad: "Rule 1: if X then Y. Rule 2: if Z then Y"
    - Good: "Set Y when: explicit mentions of X, or specific instances like Z"

    Example of handling conflicts:
    - Bad: "Rule A overrides Rule B (priority 1 > priority 2)"
    - Good: "Important: Even if condition B is present, explicit X always takes precedence"

    Example of nested output structure (when hierarchy is provided):
    - If patient has symptoms, and symptoms has text and is_emergency, output:
      patient:
        symptoms:
          - text: "..."
            is_emergency: true/false

    The output should read like instructions written by a human expert, not a rules engine.
    """

    dsl_instructions = dspy.InputField(
        desc="""Instructions from the DSL file that MUST be included in the generated prompt.
        These are user-provided guidelines for how the task should be performed.
        Include these instructions verbatim or paraphrased near the top of your generated prompt."""
    )
    reasoning_structure = dspy.InputField(
        desc="""Complete reasoning structure from DeLP analysis containing:
        - variables: each with strict_rules, defeasible_rules, conclusion, alternatives, conflict_resolutions
        - semantic_patterns: pattern names with example values
        - conflict_summary: human-readable OVERRIDES relationship descriptions
        - hierarchy: nested structure of parent-child relationships (if any)
        Include rule descriptions and conditions."""
    )
    output_variables = dspy.InputField(
        desc="List of variable names that should appear in the final output"
    )

    reasoning = dspy.OutputField(
        desc="""Your step-by-step reasoning about:
        1. How to consolidate rules into natural instructions
        2. Which conflicts need clarification as 'Important:' notes
        3. Where to embed examples for disambiguation
        4. How to structure the prompt flow"""
    )

    prompt_template = dspy.OutputField(
        desc="""Natural language prompt template that encodes all reasoning logic.

        CRITICAL: Your template MUST include the text "INPUT: {user_input}" near the end.
        This is a required placeholder that will be replaced with actual user input at runtime.

        Requirements:
        - MUST include "INPUT: {user_input}" placeholder (required for template)
        - Use ONLY {user_input} as placeholder (no other curly braces)
        - Write as human-expert instructions, not mechanical rule translation
        - Consolidate related rules into unified guidance
        - Convert overrides into clarifying statements
        - Include inline examples from semantic patterns
        - Provide positive/negative examples for edge cases
        - End with clear output format specification after the INPUT line

        The prompt should feel like expert guidance, not a rules dump.

        Remember: The template MUST contain "{user_input}" exactly once."""
    )


# Few-shot examples demonstrating the consolidation approach
EXAMPLES = [
    dspy.Example(
        dsl_instructions="Categorize user-submitted content posts to determine if they require moderation",
        reasoning_structure="""{
  "variables": {
    "requires_moderation": {
      "strict_rules": [
        {
          "id": "mod_1",
          "head": "requires_moderation(true)",
          "conditions": ["matches(content, explicit_violations)"],
          "description": "When content matches explicit_violations"
        },
        {
          "id": "mod_2",
          "head": "requires_moderation(false)",
          "conditions": ["matches(content, safe_patterns)"],
          "description": "When content matches safe_patterns"
        }
      ],
      "defeasible_rules": [
        {
          "id": "mod_3",
          "head": "requires_moderation(true)",
          "conditions": ["has_high_report_count(content)"],
          "description": "When content has high report count"
        }
      ],
      "conflicts": [
        {"superior": "mod_1", "inferior": "mod_3", "type": "OVERRIDES"}
      ],
      "semantic_patterns": {
        "explicit_violations": ["spam", "harassment", "hate speech"],
        "safe_patterns": ["question", "discussion", "feedback"]
      }
    },
    "moderation_priority": {
      "strict_rules": [
        {
          "id": "priority_1",
          "head": "moderation_priority(urgent)",
          "conditions": ["requires_moderation(true)", "matches(content, explicit_violations)"],
          "description": "When requires moderation and explicit violations"
        }
      ]
    }
  },
  "semantic_categories": {
    "explicit_violations": {
      "description": "content that clearly violates policies",
      "examples": ["spam", "harassment", "hate speech", "illegal content"]
    },
    "safe_patterns": {
      "description": "benign content types",
      "examples": ["question", "discussion", "feedback", "tutorial"]
    }
  }
}""",
        output_variables=["requires_moderation", "moderation_priority", "reason"],
        reasoning="""I need to consolidate the moderation logic into clear guidance:

1. CONSOLIDATION: Two strict rules (explicit violations → true, safe patterns → false)
   should become one clear statement about when to flag content.

2. CONFLICTS: mod_1 overrides mod_3 means explicit violations always win even if
   there are few reports. This becomes: "Important clarification about edge cases"

3. EXAMPLES: Embed the semantic patterns (spam, harassment, etc.) inline where
   they're relevant, not in a separate section.

4. STRUCTURE: Start with what to check, then clarifications, then output format.""",

        prompt_template="""Analyze user-generated content to determine if it requires moderation.

MODERATION DETERMINATION:

Flag content for moderation (requires_moderation = true) when it contains:
- Explicit policy violations: spam, harassment, hate speech, illegal content, threats
- Promotional links with commercial intent
- Personal information or doxxing attempts

Do NOT flag (requires_moderation = false) when content is clearly benign:
- Questions seeking help or information
- Constructive discussions or feedback
- Educational tutorials or how-to guides
- General community conversations

Important Clarification:
Even if content has been reported multiple times by users, if it contains explicit
policy violations (spam, harassment, etc.), it ALWAYS requires moderation.
However, if content appears benign but has many reports, use your judgment -
the explicit violations take precedence over report counts.

PRIORITY DETERMINATION:

If requires_moderation = true:
- Set moderation_priority = "urgent" when: explicit violations are present
- Set moderation_priority = "normal" when: flagged only due to reports

Examples:

Requires moderation = TRUE:
- "Buy cheap meds here! [suspicious link]" → spam violation
- "You're an idiot, go away" → harassment
- Post with 50 user reports containing hate speech → explicit violation (urgent)

Requires moderation = FALSE:
- "How do I reset my password?" → question, benign
- "Great tutorial, thanks for sharing!" → feedback, benign
- "What's the best approach for X?" → discussion, benign

INPUT: {user_input}

OUTPUT: Return JSON with requires_moderation (true/false), moderation_priority (if true), and reason for decision."""
    ),

    dspy.Example(
        dsl_instructions="Route incoming job applications to appropriate hiring team based on role requirements",
        reasoning_structure="""{
  "variables": {
    "routing_team": {
      "strict_rules": [
        {
          "id": "route_1",
          "head": "routing_team(engineering)",
          "conditions": ["matches(application, technical_roles)"],
          "description": "When application matches technical_roles"
        },
        {
          "id": "route_2",
          "head": "routing_team(sales)",
          "conditions": ["matches(application, sales_roles)"],
          "description": "When application matches sales_roles"
        }
      ],
      "defeasible_rules": [
        {
          "id": "route_3",
          "head": "routing_team(engineering)",
          "conditions": ["has_coding_portfolio(application)"],
          "description": "When application has coding portfolio"
        }
      ],
      "conflicts": [
        {"superior": "route_1", "inferior": "route_3", "type": "OVERRIDES"},
        {"superior": "route_2", "inferior": "route_3", "type": "OVERRIDES"}
      ],
      "semantic_patterns": {
        "technical_roles": ["Software Engineer", "DevOps", "Data Scientist"],
        "sales_roles": ["Account Executive", "Sales Representative", "Business Development"]
      }
    }
  },
  "semantic_categories": {
    "technical_roles": {
      "description": "engineering and technical positions",
      "examples": ["Software Engineer", "DevOps Engineer", "Data Scientist", "ML Engineer"]
    },
    "sales_roles": {
      "description": "sales and business development positions",
      "examples": ["Account Executive", "Sales Representative", "Business Development Manager"]
    }
  }
}""",
        output_variables=["routing_team"],
        reasoning="""The routing logic is straightforward but needs one key clarification:

1. CONSOLIDATION: The strict rules are already simple - route based on role title.
   Just list the role categories clearly.

2. CONFLICTS: The defeasible rule (coding portfolio → engineering) gets overridden
   by explicit role mentions. This means: don't infer role from portfolio if the
   actual job title indicates something else.

3. CLARIFICATION: Need to address edge case where someone applies to "Sales Engineer"
   with a coding portfolio - the explicit role title wins.

4. KEEP SIMPLE: This is simpler than the moderation example, so don't over-complicate.""",

        prompt_template="""Route job application to the appropriate hiring team based on the position applied for.

ROUTING LOGIC:

Route to ENGINEERING team when applying for:
- Software Engineer, DevOps Engineer, Data Scientist, ML Engineer
- Backend Developer, Frontend Developer, Full Stack Developer
- Technical roles involving coding or infrastructure

Route to SALES team when applying for:
- Account Executive, Sales Representative, Business Development Manager
- Customer Success Manager (when sales-focused)
- Revenue-generating or client-facing roles

Important: Always route based on the actual job title in the application, not
inferred from candidate background. For example, if someone with a strong coding
portfolio applies for "Sales Engineer", route to SALES because that's the explicit
role, even though they have technical skills.

If the role title is ambiguous or doesn't clearly fit engineering or sales, indicate
routing_team = "general" for manual review.

INPUT: {user_input}

OUTPUT: Return JSON with routing_team (engineering/sales/general)."""
    ),

    dspy.Example(
        dsl_instructions="Classify customer product returns to determine refund eligibility",
        reasoning_structure="""{
  "variables": {
    "refund_eligible": {
      "strict_rules": [
        {
          "id": "refund_1",
          "head": "refund_eligible(true)",
          "conditions": ["matches(reason, defective_product)"],
          "description": "When reason matches defective_product"
        },
        {
          "id": "refund_2",
          "head": "refund_eligible(false)",
          "conditions": ["matches(reason, buyer_remorse)"],
          "description": "When reason matches buyer_remorse"
        },
        {
          "id": "refund_3",
          "head": "refund_eligible(false)",
          "conditions": ["return_window_exceeded(true)"],
          "description": "When return window exceeded"
        }
      ],
      "defeasible_rules": [
        {
          "id": "refund_4",
          "head": "refund_eligible(true)",
          "conditions": ["customer_loyalty_tier(premium)"],
          "description": "When customer is premium loyalty tier"
        }
      ],
      "conflicts": [
        {"superior": "refund_3", "inferior": "refund_4", "type": "OVERRIDES"}
      ],
      "semantic_patterns": {
        "defective_product": ["broken", "defective", "not working", "damaged"],
        "buyer_remorse": ["changed mind", "don't want", "ordered wrong"]
      }
    }
  },
  "semantic_categories": {
    "defective_product": {
      "description": "product quality issues",
      "examples": ["broken on arrival", "defective", "stops working", "damaged in shipping"]
    },
    "buyer_remorse": {
      "description": "customer change of mind",
      "examples": ["changed my mind", "don't need anymore", "ordered by mistake"]
    }
  }
}""",
        output_variables=["refund_eligible", "reason"],
        reasoning="""This has a critical policy override that needs emphasis:

1. STRICT RULES: Defective = refund yes, buyer remorse = no, expired window = no.
   These should be presented as clear policy.

2. THE KEY CONFLICT: Even premium customers don't get refunds past the window.
   This is the main thing to clarify - loyalty doesn't override policy deadlines.

3. POSITIVE/NEGATIVE EXAMPLES: Show when defective vs remorse, and importantly,
   show the premium customer edge case.

4. TONE: This is policy enforcement, so be clear and authoritative.""",

        prompt_template="""Determine if a product return qualifies for a refund based on company policy.

REFUND ELIGIBILITY:

Approve refund (refund_eligible = true) when:
- Product is defective or damaged: "broken on arrival", "stops working", "damaged in shipping"
- Product quality issues that aren't user-caused
- Return is within the 30-day return window

Deny refund (refund_eligible = false) when:
- Buyer's remorse: "changed my mind", "don't need anymore", "ordered by mistake"
- Return window (30 days from purchase) has expired
- Product damage is user-caused

Critical Policy Note:
The 30-day return window is absolute and overrides all other considerations.
Even premium loyalty customers cannot receive refunds for returns submitted after
30 days, regardless of the reason. The return window policy takes precedence.

Examples:

ELIGIBLE:
- "The item arrived broken" (within 30 days) → defective product
- "Stopped working after 2 weeks" → quality issue, within window

NOT ELIGIBLE:
- "I changed my mind" → buyer's remorse
- "It's defective" (but returned on day 35) → past window, even if defective
- "I'm a premium member and need a refund" (day 40) → window expired, premium status doesn't override

INPUT: {user_input}

OUTPUT: Return JSON with refund_eligible (true/false) and reason for decision."""
    )
]


class PromptTemplateGenerator:
    """
    Generates LLM prompt templates from DeLP reasoning structure

    Takes the output of DeLPReasoningAnalyzer and converts it into
    a natural language prompt template suitable for LLM execution.

    Integrates FOL verification to ensure generated prompts match DSL semantics.
    """

    def __init__(self, max_retries: int = 2):
        """
        Initialize the prompt template generator

        Args:
            max_retries: Maximum number of retry attempts for generation
        """
        self.generator = dspy.ChainOfThought(GeneratePromptTemplate)
        self.max_retries = max_retries

        # Setup few-shot examples
        self.generator.demos = EXAMPLES

        # FOL verification (set via set_fol or from AST)
        self._fol: Optional[CantoFOL] = None
        self._verifier: Optional[EquivalenceVerifier] = None

    def set_fol(self, fol: CantoFOL) -> None:
        """
        Set the FOL IR for verification.

        Args:
            fol: The CantoFOL IR to verify prompts against
        """
        self._fol = fol
        self._verifier = EquivalenceVerifier(fol)

    def set_fol_from_ast(self, ast: List, source_file: str = "<string>") -> None:
        """
        Set the FOL IR from AST nodes.

        Args:
            ast: List of AST nodes from parsed DSL
            source_file: Source file path for error messages
        """
        translator = ASTToFOLTranslator()
        self._fol = translator.translate(ast, source_file=source_file)
        self._verifier = EquivalenceVerifier(self._fol)

    def verify_prompt(self, prompt: str) -> List[str]:
        """
        Verify a prompt against the DSL using FOL.

        Args:
            prompt: The prompt template to verify

        Returns:
            List of constraint violations (empty if valid)
        """
        if self._verifier is None:
            return []  # No verification configured
        return self._verifier.verify_constraints(prompt)

    def generate_from_structure(
        self,
        reasoning_structure: Dict[str, Any],
        dsl_instructions: str = "",
        output_variables: List[str] = None,
        program = None,
        ast: List = None,
    ) -> str:
        """
        Generate prompt template from reasoning structure

        Args:
            reasoning_structure: Output from DeLPReasoningAnalyzer.analyze()
            dsl_instructions: Instructions from DSL file to include in the generated prompt
            output_variables: List of variables to include in output (if None, auto-detects OUTPUT variables)
            program: Optional DeLPProgram to detect OUTPUT variables
            ast: Optional AST nodes for FOL verification

        Returns:
            Prompt template string with {user_input} placeholder
        """
        # Setup FOL verification if AST provided
        if ast is not None and self._verifier is None:
            self.set_fol_from_ast(ast)

        # Extract output variables from structure if not provided
        if output_variables is None:
            # Use all variables from the reasoning structure
            output_variables = list(reasoning_structure.get('variables', {}).keys())

        # Convert structure to string for LLM
        import json
        structure_str = json.dumps(reasoning_structure, indent=2)

        # Try generating with retries, using FOL violations as feedback
        last_error = None
        fol_feedback = ""

        for attempt in range(self.max_retries):
            try:
                print(f"Generating prompt template (attempt {attempt + 1}/{self.max_retries})...")

                # Include FOL feedback in instructions if we have violations from previous attempt
                enhanced_instructions = dsl_instructions
                if fol_feedback:
                    enhanced_instructions = f"{dsl_instructions}\n\nIMPORTANT - Fix these issues from the previous attempt:\n{fol_feedback}"

                result = self.generator(
                    dsl_instructions=enhanced_instructions,
                    reasoning_structure=structure_str,
                    output_variables=", ".join(output_variables)
                )

                prompt_template = result.prompt_template

                # Validate that template has placeholder
                if "{user_input}" not in prompt_template:
                    last_error = "Template missing {user_input} placeholder"
                    print(f"✗ Validation failed: {last_error}")
                    continue

                # FOL verification
                violations = self.verify_prompt(prompt_template)
                if violations:
                    print(f"✗ FOL verification found {len(violations)} issue(s):")
                    for v in violations:
                        print(f"    - {v}")
                    # Build feedback for next attempt
                    fol_feedback = "\n".join(f"- {v}" for v in violations)
                    last_error = f"FOL verification failed: {len(violations)} violations"
                    continue

                print(f"✓ Generated prompt template successfully")
                if self._verifier is not None:
                    print(f"✓ FOL verification passed")
                return prompt_template

            except Exception as e:
                last_error = str(e)
                print(f"✗ Generation failed: {e}")

        # All retries failed - return fallback template
        print(f"⚠ All retries failed. Using fallback template.")
        return self._generate_fallback(reasoning_structure, output_variables)

    def _generate_fallback(
        self,
        reasoning_structure: Dict[str, Any],
        output_variables: List[str]
    ) -> str:
        """
        Generate a simple fallback template if LLM generation fails

        Args:
            reasoning_structure: The reasoning structure
            output_variables: Variables to include in output

        Returns:
            Basic prompt template
        """
        # Check for hierarchy - if present, use nested output structure
        hierarchy = reasoning_structure.get('hierarchy', {})
        roots = hierarchy.get('roots', [])

        if roots:
            # Generate nested output structure from hierarchy
            output_structure = self._generate_nested_output_structure(roots)
            variables_section = f"The output should follow this nested structure:\n\n{output_structure}"
        else:
            # Flat list of variables
            variables_section = "OUTPUT VARIABLES:\n" + "\n".join([
                f"- {var}" for var in output_variables
            ])

        # Extract semantic patterns
        patterns_section = ""
        patterns = reasoning_structure.get('semantic_categories', {})
        if patterns:
            patterns_list = []
            for name, info in patterns.items():
                examples = info.get('patterns', info.get('examples', []))[:5]  # First 5 examples
                patterns_list.append(f"  - {name}: {', '.join(examples)}")
            patterns_section = "\n".join(patterns_list)

        return f"""Analyze the input text and extract the following information:

{variables_section}

SEMANTIC PATTERNS TO RECOGNIZE:
{patterns_section}

INPUT TEXT: {{user_input}}

OUTPUT: Return a JSON object with the extracted variables following the structure above.
"""

    def _generate_nested_output_structure(self, roots: List[Dict], indent: int = 0) -> str:
        """
        Generate a nested output structure representation from hierarchy roots.

        Args:
            roots: List of root nodes from hierarchy
            indent: Current indentation level

        Returns:
            String representation of nested structure
        """
        lines = []
        indent_str = "  " * indent

        for node in roots:
            name = node.get('name', '')
            description = node.get('description', '')
            is_list = node.get('is_list', False)
            children = node.get('children', [])

            # Format the node
            desc_str = f" // {description}" if description else ""
            if is_list:
                lines.append(f"{indent_str}{name}: [{desc_str}")
                if children:
                    lines.append(f"{indent_str}  {{")
                    lines.append(self._generate_nested_output_structure(children, indent + 2))
                    lines.append(f"{indent_str}  }}")
                else:
                    lines.append(f"{indent_str}  // list of values")
                lines.append(f"{indent_str}]")
            else:
                if children:
                    lines.append(f"{indent_str}{name}: {{{desc_str}")
                    lines.append(self._generate_nested_output_structure(children, indent + 1))
                    lines.append(f"{indent_str}}}")
                else:
                    lines.append(f"{indent_str}{name}: <value>{desc_str}")

        return "\n".join(lines)

    def save_template(self, template: str, output_path: Path) -> None:
        """
        Save prompt template to file

        Args:
            template: The prompt template string
            output_path: Path to save the template
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            f.write(template)

        print(f"✓ Saved prompt template to {output_path}")

    def load_template(self, template_path: Path) -> str:
        """
        Load prompt template from file

        Args:
            template_path: Path to the template file

        Returns:
            Prompt template string
        """
        with open(template_path, 'r') as f:
            return f.read()
