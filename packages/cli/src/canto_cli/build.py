"""
BUILD Pipeline for Canto DSL

This module implements the build pipeline:
- Step 1: LLM predicate generation
- Step 2: DeLP reasoning analysis
- Step 3: Prompt template generation
- Step 4: PDO optimization (optional, with --optimize flag)

Usage:
    canto build <canto_file>
    canto build <canto_file> --optimize --eval-data examples.json
"""

import sys
import os
import json
import argparse
from pathlib import Path

from canto_core import parse_file
from canto_core.parser.semantic_analyzer import analyze as semantic_analyze
from canto_core import DeLPTranslator, LLMPredicateGenerator, configure_dspy
from canto_core import DeLPReasoningAnalyzer

from dotenv import load_dotenv

load_dotenv()


def build_pipeline(canto_file: str, verbose: bool = False, optimize: bool = False, eval_data: str = None) -> int:
    """Run the full BUILD pipeline (Steps 1, 2, 3, and optionally 4)"""

    print("=" * 60)
    print("CANTO BUILD PIPELINE")
    print("=" * 60)
    print()

    # Configure DSPy
    model = os.environ.get("LLM_MODEL", "gpt-4.1")
    print("Configuring DSPy...")
    configure_dspy(model)
    print(f"  DSPy configured (using {model})\n")

    # Parse the input .canto file
    dsl_file = Path(canto_file)
    print(f"Parsing {dsl_file.name}...")

    if not dsl_file.exists():
        print(f"  File not found: {dsl_file}")
        return 1

    result = parse_file(str(dsl_file))
    ast = result.ast
    dsl_instructions = result.instructions
    print(f"  Parsed {len(ast)} AST nodes")
    if dsl_instructions:
        print(f"  Found instructions block\n")
    else:
        print()

    # Semantic analysis
    print("Running semantic analysis...")
    semantic_result = semantic_analyze(ast)
    if not semantic_result.is_valid:
        print(f"  Semantic analysis failed:")
        print(semantic_result)
        return 1
    if semantic_result.warnings:
        print(f"  Warnings:")
        for warning in semantic_result.warnings:
            print(f"  - {warning}")
    print(f"  Semantic analysis passed\n")

    # Step 1: Compile DSL to Prolog KB
    print("=" * 60)
    print("STEP 1: COMPILATION (DSL -> Prolog KB)")
    print("=" * 60)
    print()

    print("Running deterministic translator...")
    translator = DeLPTranslator()
    program = translator.translate(ast)
    print(f"  Translated to DeLP program")
    print(f"  - {len(program.declarations)} variable declarations")
    print(f"  - {len(program.semantic_categories)} semantic categories")
    print(f"  - {len(program.strict_rules)} strict rules")
    print(f"  - {len(program.defeasible_rules)} defeasible rules")
    print(f"  - {len(program.superiority)} superiority relations")

    print()

    # Generate LLM predicates for semantic categories
    print("Generating LLM predicates for semantic categories...")
    generator = LLMPredicateGenerator(max_retries=3)

    llm_predicates = []
    for cat_name, category in program.semantic_categories.items():
        print(f"\n  Generating predicates for: {cat_name}")
        print(f"    Description: {category.description}")
        print(f"    Examples: {category.patterns[:3]}...")

        predicates = generator.generate_from_category(category)
        llm_predicates.append(predicates)
        print(f"    Generated {len(predicates.split(chr(10)))} lines of Prolog")

    print(f"\n  Generated predicates for {len(llm_predicates)} categories")
    print()

    # Join all generated predicates
    all_llm_predicates = "\n\n".join(llm_predicates)

    # Combine base prolog + LLM predicates
    base_prolog = program.to_prolog_string()
    full_prolog = base_prolog + "\n\n" + all_llm_predicates if all_llm_predicates else base_prolog

    if verbose:
        print("FULL PROLOG CODE:")
        print("=" * 60)
        print(full_prolog)
        print("=" * 60)
        print()
    elif llm_predicates:
        # Show sample of generated predicates
        print("Sample of generated predicates:")
        print("-" * 60)
        print(llm_predicates[0][:500] + "..." if len(llm_predicates[0]) > 500 else llm_predicates[0])
        print("-" * 60)
        print()

    # Save Prolog output to file
    prolog_output_path = dsl_file.parent / f"{dsl_file.stem}_prolog.pl"
    with open(prolog_output_path, 'w') as f:
        f.write(full_prolog)
    print(f"  Saved Prolog KB to {prolog_output_path}")
    print()

    # Step 2: Analyze KB structure
    print("=" * 60)
    print("STEP 2: REASONING ANALYSIS")
    print("=" * 60)
    print()

    print("Analyzing KB structure...")
    # Join all generated predicates into one string
    all_llm_predicates = "\n\n".join(llm_predicates)
    analyzer = DeLPReasoningAnalyzer(program, extra_prolog=all_llm_predicates)

    # Get reasoning structure
    reasoning_structure = analyzer.analyze()

    print(f"  Analysis complete")
    print(f"  - Analyzed {len(reasoning_structure['variables'])} variables")
    print(f"  - Found {len(reasoning_structure['semantic_categories'])} semantic categories")
    print(f"  - Detected {len(reasoning_structure['conflict_summary'])} conflict resolutions")
    print()

    # Show DSL analysis report
    print(analyzer.get_dsl_analysis_report())
    print()

    # Show detailed structure for first variable with conclusions
    variables = reasoning_structure.get('variables', {})
    var_with_conclusions = next(
        (name for name, data in variables.items() if data.get('conclusions')),
        None
    )
    if var_with_conclusions:
        print(f"DETAILED STRUCTURE ({var_with_conclusions}):")
        print("-" * 60)
        print(json.dumps(variables[var_with_conclusions], indent=2))
        print("-" * 60)
    print()

    # Step 3: Generate prompt template
    print("=" * 60)
    print("STEP 3: PROMPT TEMPLATE GENERATION")
    print("=" * 60)
    print()

    print("Generating prompt template from reasoning structure...")
    from canto_core import PromptTemplateGenerator

    prompt_generator = PromptTemplateGenerator(max_retries=3)

    # Use instructions from DSL file, or fallback to default
    instructions = dsl_instructions or "Process the input according to the defined rules and patterns."

    # Generate prompt template
    # Pass program to auto-detect OUTPUT variables
    prompt_template = prompt_generator.generate_from_structure(
        reasoning_structure=reasoning_structure,
        dsl_instructions=instructions,
        program=program  # Auto-detects variables marked with OUTPUT
    )

    print(f"  Prompt template generated")
    print()

    # Save to file
    output_path = dsl_file.parent / f"{dsl_file.stem}_prompt.txt"
    prompt_generator.save_template(prompt_template, output_path)
    print()

    # Show sample
    print("SAMPLE OF GENERATED PROMPT:")
    print("-" * 60)
    print(prompt_template[:800] + "..." if len(prompt_template) > 800 else prompt_template)
    print("-" * 60)
    print()

    # Step 4: PDO Optimization (optional)
    if optimize:
        print("=" * 60)
        print("STEP 4: PDO OPTIMIZATION")
        print("=" * 60)
        print()

        from canto_core import CantoPDO, PDOConfig

        # Load evaluation examples - required for PDO
        if not eval_data:
            print("  Error: --eval-data is required for PDO optimization")
            print("  Please provide a JSON file with evaluation examples")
            return 1

        eval_path = Path(eval_data)
        if not eval_path.exists():
            print(f"  Error: Eval data file not found: {eval_data}")
            return 1

        with open(eval_path) as f:
            eval_examples = json.load(f)
        print(f"  Loaded {len(eval_examples)} evaluation examples from {eval_data}")
        print()

        # Configure PDO
        config = PDOConfig(
            num_initial_instructions=8,
            num_rounds=30,
            duels_per_round=3,
            pool_update_frequency=10,
            verbose=True
        )

        # Run optimization
        optimizer = CantoPDO(config=config)
        optimized_prompt = optimizer.optimize(
            reasoning_context=reasoning_structure,
            eval_examples=eval_examples,
            dsl_instructions=instructions
        )

        print()
        print("  PDO optimization complete")
        print()

        # Save optimized prompt
        optimized_path = dsl_file.parent / f"{dsl_file.stem}_prompt_optimized.txt"
        with open(optimized_path, 'w') as f:
            f.write(optimized_prompt)
        print(f"  Saved optimized prompt to {optimized_path}")
        print()

        # Show sample
        print("SAMPLE OF OPTIMIZED PROMPT:")
        print("-" * 60)
        print(optimized_prompt[:800] + "..." if len(optimized_prompt) > 800 else optimized_prompt)
        print("-" * 60)
        print()

        # Update output path for final summary
        output_path = optimized_path

    print("=" * 60)
    if optimize:
        print("BUILD STEPS 1, 2, 3 & 4 COMPLETE!")
    else:
        print("BUILD STEPS 1, 2 & 3 COMPLETE!")
    print("=" * 60)
    print()
    print("  DSL compiled to Prolog KB")
    print("  LLM predicates generated and verified")
    print("  Reasoning structure analyzed")
    print("  Prompt template generated and saved")
    if optimize:
        print("  Prompt optimized via PDO")
    print()
    print(f"Outputs:")
    print(f"  - Prolog KB: {prolog_output_path}")
    print(f"  - Prompt template: {output_path}")

    return 0


def main():
    """Main entry point for canto CLI"""
    parser = argparse.ArgumentParser(
        prog="canto",
        description="Canto DSL - Build pipeline for .canto files"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Build command
    build_parser = subparsers.add_parser("build", help="Build a .canto file")
    build_parser.add_argument("canto_file", help="Path to .canto file")
    build_parser.add_argument("-v", "--verbose", action="store_true", help="Show full Prolog output")
    build_parser.add_argument("--optimize", action="store_true", help="Run PDO optimization (Step 4)")
    build_parser.add_argument("--eval-data", type=str, help="Path to JSON file with evaluation examples (required for --optimize)")

    args = parser.parse_args()

    if args.command == "build":
        sys.exit(build_pipeline(
            args.canto_file,
            verbose=args.verbose,
            optimize=args.optimize,
            eval_data=args.eval_data
        ))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
