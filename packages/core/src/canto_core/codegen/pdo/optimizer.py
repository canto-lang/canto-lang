"""
Canto PDO Optimizer - Prompt Duel Optimizer for instruction tuning
"""

import json
import logging
from typing import Dict, Any, List, Optional

import numpy as np
import dspy
from tqdm import tqdm

from .config import PDOConfig
from .context_formatter import format_reasoning_context, format_context_for_prompt

logger = logging.getLogger(__name__)


INITIAL_INSTRUCTION_TIPS = {
    "framing": "Set the context for the task by framing it as a concrete scenario.",
    "simple": "Keep the instruction clear and concise.",
    "description": "Make sure your instruction is very informative and descriptive.",
    "persona": "Provide the LM with a persona relevant to the task.",
    "relevance": "Focus on creating instructions that promote relevant, focused answers.",
    "completeness": "Instruct the model to provide complete answers.",
    "clarity": "Provide guidance on creating clear, well-articulated answers.",
    "evidence": "Emphasize using evidence from the context to support answers.",
}

MUTATION_TIPS = {
    "expansion": "Expand by adding helpful guidance while keeping the original.",
    "minimal": "Make minimal changes - modify only a few words.",
    "few_shot": "Add 1-3 concrete examples to demonstrate the expected output.",
    "emphasis": "Adjust the tone or emphasis to create different reasoning patterns.",
}


class CantoPDO:
    """
    Prompt Duel Optimizer for Canto.

    Generates optimized prompts from Canto reasoning patterns using
    dueling bandits with Thompson sampling.
    """

    def __init__(self, config: Optional[PDOConfig] = None):
        self.config = config or PDOConfig()
        self.instruction_pool: List[str] = []
        self.win_matrix: Optional[np.ndarray] = None

        # Use globally configured DSPy LM
        self.lm = dspy.settings.lm

    def optimize(
        self,
        reasoning_context: Dict[str, Any],
        eval_examples: List[Dict],
        dsl_instructions: str = ""
    ) -> str:
        """
        Run PDO optimization to find best prompt.

        Args:
            reasoning_context: Output from DeLPReasoningAnalyzer or format_reasoning_context
            eval_examples: List of evaluation examples (label-free)
            dsl_instructions: Instructions from DSL file to include in generated prompts

        Returns:
            Optimized prompt string
        """
        # Format context if needed
        if 'rules' not in reasoning_context:
            context = format_reasoning_context(reasoning_context, dsl_instructions)
        else:
            context = reasoning_context

        # Generate dataset summary
        dataset_summary = self._summarize_dataset(eval_examples)

        # Initialize instruction pool
        if self.config.verbose:
            print(f"Generating {self.config.num_initial_instructions} initial instructions...")

        self.instruction_pool = self._initialize_pool(context, eval_examples, dataset_summary)
        self.win_matrix = np.zeros((len(self.instruction_pool), len(self.instruction_pool)))

        if self.config.verbose:
            print(f"Starting optimization: {self.config.num_rounds} rounds, {self.config.duels_per_round} duels/round")

        # Run duel rounds with progress bar
        rounds_iter = tqdm(range(self.config.num_rounds), desc="Rounds", disable=not self.config.verbose)
        for round_num in rounds_iter:
            self._run_duel_round(eval_examples, context)

            # Periodic pool update
            if (round_num + 1) % self.config.pool_update_frequency == 0:
                self._update_pool(context, dataset_summary)
                top_win_rate = self._get_top_win_rate()
                rounds_iter.set_postfix(win_rate=f"{top_win_rate:.2f}", pool=len(self.instruction_pool))

        # Select best prompt
        rankings = self._compute_rankings()
        best_idx = rankings[0]

        if self.config.verbose:
            print(f"\nOptimization complete. Best prompt selected.")

        return self.instruction_pool[best_idx]

    def _summarize_dataset(self, examples: List[Dict]) -> str:
        """Generate dataset summary using LLM"""
        sample_inputs = [ex.get('input', ex.get('question', '')) for ex in examples[:10]]

        prompt = f"""You are a data analyst. Summarize this dataset in 3-5 sentences.
Focus on: input structure, common patterns, domain terminology.

Sample inputs:
{json.dumps(sample_inputs, indent=2)}

Summary:"""

        response = self.lm(prompt)
        return str(response)

    def _initialize_pool(
        self,
        context: Dict[str, Any],
        examples: List[Dict],
        dataset_summary: str
    ) -> List[str]:
        """Generate diverse initial instructions using different tips"""
        instructions = []
        context_str = format_context_for_prompt(context)
        sample_inputs = [ex.get('input', ex.get('question', '')) for ex in examples[:3]]

        tips = list(INITIAL_INSTRUCTION_TIPS.values())

        for i, tip in enumerate(tips[:self.config.num_initial_instructions]):
            prompt = f"""You are an expert prompt-engineer.
Generate exactly 1 high-quality system-level instruction for a classification task.

# Dataset Summary
{dataset_summary}

# Classification Context
{context_str}

# Sample Inputs (do NOT answer them)
{json.dumps(sample_inputs, indent=2)}

# Prompt-Engineering Tip
{tip}

# Requirements
- Include {{user_input}} placeholder for the input text
- Incorporate the decision rules naturally
- Specify output format as JSON

Return exactly 1 instruction as a JSON array:
["Your instruction here with {{user_input}} placeholder"]"""

            try:
                response = self.lm(prompt)
                response_str = str(response)
                if self.config.verbose:
                    print(f"  [DEBUG] Raw response {i+1} (first 200 chars): {response_str[:200]}")

                instruction = self._parse_instruction_response(response_str)
                if self.config.verbose:
                    if instruction:
                        print(f"  [DEBUG] Parsed instruction (first 100 chars): {instruction[:100]}")
                        print(f"  [DEBUG] Contains {{user_input}}: {'{{user_input}}' in instruction or '{user_input}' in instruction}")
                    else:
                        print(f"  [DEBUG] Failed to parse instruction from response")

                if instruction and "{user_input}" in instruction:
                    instructions.append(instruction)
                    if self.config.verbose:
                        print(f"  Generated instruction {i+1}/{self.config.num_initial_instructions}")
                elif instruction:
                    if self.config.verbose:
                        print(f"  Instruction {i+1} missing {{user_input}} placeholder")
            except Exception as e:
                if self.config.verbose:
                    print(f"  Error generating instruction {i+1}: {e}")

        # Ensure we have at least one instruction
        if not instructions:
            instructions.append(self._fallback_instruction(context))

        return instructions

    def _parse_instruction_response(self, response: str) -> Optional[str]:
        """Parse LLM response to extract instruction"""
        # Handle DSPy response format: ['["actual content"]']
        # First, try to extract from Python list representation
        if response.startswith("['") or response.startswith('["'):
            try:
                # It's a Python list representation, eval it carefully
                import ast
                parsed = ast.literal_eval(response)
                if isinstance(parsed, list) and len(parsed) > 0:
                    response = parsed[0]  # Get the inner content
            except:
                pass

        try:
            # Try JSON array format
            if '[' in response and ']' in response:
                start = response.index('[')
                end = response.rindex(']') + 1
                arr = json.loads(response[start:end])
                if arr and isinstance(arr, list):
                    return arr[0]
        except:
            pass

        # Try JSON object format
        try:
            if '{' in response:
                start = response.index('{')
                end = response.rindex('}') + 1
                obj = json.loads(response[start:end])
                return obj.get('instruction') or obj.get('mutated_prompt')
        except:
            pass

        return None

    def _fallback_instruction(self, context: Dict[str, Any]) -> str:
        """Generate fallback instruction if all else fails"""
        vars_str = ", ".join(context.get('output_variables', ['result']))
        return f"""Analyze the following input and classify it.

{format_context_for_prompt(context)}

INPUT: {{user_input}}

OUTPUT: Return JSON with {vars_str}."""

    def _run_duel_round(self, examples: List[Dict], context: Dict[str, Any]):
        """Execute one round of duels"""
        n = len(self.instruction_pool)
        if n < 2:
            return

        for _ in range(self.config.duels_per_round):
            # Sample duel pair using Thompson sampling
            idx_a, idx_b = self._sample_pair(n)

            # Select random example
            example = examples[np.random.randint(len(examples))]
            input_text = example.get('input', example.get('question', ''))

            # Execute both prompts
            prompt_a = self.instruction_pool[idx_a].replace('{user_input}', input_text)
            prompt_b = self.instruction_pool[idx_b].replace('{user_input}', input_text)

            try:
                response_a = str(self.lm(prompt_a))
                response_b = str(self.lm(prompt_b))

                # Judge
                winner = self._judge_responses(input_text, response_a, response_b, context)

                # Update win matrix
                if winner == 'A':
                    self.win_matrix[idx_a, idx_b] += 1
                elif winner == 'B':
                    self.win_matrix[idx_b, idx_a] += 1
                else:
                    self.win_matrix[idx_a, idx_b] += 0.5
                    self.win_matrix[idx_b, idx_a] += 0.5

            except Exception as e:
                logger.warning(f"Duel failed: {e}")

    def _sample_pair(self, n: int) -> tuple:
        """Sample pair using Thompson sampling with exploration"""
        indices = list(range(n))

        # Thompson sampling: sample from Beta distributions based on wins/losses
        scores = []
        for i in range(n):
            wins = np.sum(self.win_matrix[i, :]) + 1  # +1 for prior
            losses = np.sum(self.win_matrix[:, i]) + 1
            score = np.random.beta(wins, losses)
            scores.append(score)

        # Add exploration noise
        scores = np.array(scores) + self.config.exploration_tau * np.random.random(n)

        # Select top 2
        top_indices = np.argsort(scores)[-2:]
        return int(top_indices[0]), int(top_indices[1])

    def _judge_responses(
        self,
        input_text: str,
        response_a: str,
        response_b: str,
        context: Dict[str, Any]
    ) -> str:
        """Use LLM judge to compare responses"""
        # Randomize order to avoid position bias
        if np.random.random() < 0.5:
            first, second = ('A', response_a), ('B', response_b)
        else:
            first, second = ('B', response_b), ('A', response_a)

        judge_prompt = f"""You are an impartial referee evaluating two competing responses.

## Task Input
{input_text}

## Response from Prompt X
{first[1]}

## Response from Prompt Y
{second[1]}

## Evaluation Criteria
1. Correctness (50%) - Does the answer match task requirements?
2. Reasoning Quality (50%) - Is the logic coherent and complete?

## Output Format
Return JSON with "reasoning" and "winner" (X or Y):
{{"reasoning": "...", "winner": "X or Y"}}"""

        try:
            response = str(self.lm(judge_prompt))
            if '"winner"' in response.lower():
                if '"x"' in response.lower() or "'x'" in response.lower():
                    return first[0]
                elif '"y"' in response.lower() or "'y'" in response.lower():
                    return second[0]
        except:
            pass

        return 'TIE'

    def _update_pool(self, context: Dict[str, Any], dataset_summary: str):
        """Prune weak prompts and mutate strong ones"""
        rankings = self._compute_rankings()
        n = len(self.instruction_pool)

        # Keep top performers
        keep_count = max(2, int(n * (1 - self.config.prune_ratio)))
        keep_indices = rankings[:keep_count]
        new_pool = [self.instruction_pool[i] for i in keep_indices]

        # Mutate top performers
        mutate_count = max(1, int(len(rankings) * self.config.mutation_ratio))
        top_indices = rankings[:mutate_count]

        mutation_tips = list(MUTATION_TIPS.values())

        for i, idx in enumerate(top_indices):
            tip = mutation_tips[i % len(mutation_tips)]
            mutated = self._mutate_instruction(self.instruction_pool[idx], tip, context)
            if mutated and "{user_input}" in mutated:
                new_pool.append(mutated)

        self.instruction_pool = new_pool
        self._resize_win_matrix(len(new_pool), keep_count)

    def _mutate_instruction(self, instruction: str, tip: str, context: Dict[str, Any]) -> Optional[str]:
        """Generate mutation of an instruction"""
        prompt = f"""You are an expert prompt-engineer specializing in prompt optimization.
Generate 1 diverse, high-quality mutation of the BEST PERFORMING instruction.

# BEST PERFORMING Instruction
{instruction}

# Follow This Tip
{tip}

# Requirements
- Keep the {{user_input}} placeholder
- Maintain the core classification logic

Return exactly 1 mutated instruction as JSON:
{{"mutated_prompt": "Your mutated instruction here"}}"""

        try:
            response = str(self.lm(prompt))
            return self._parse_instruction_response(response)
        except:
            return None

    def _compute_rankings(self) -> List[int]:
        """Compute rankings using Copeland method"""
        n = len(self.instruction_pool)
        if n == 0:
            return []

        # Copeland ranking: count pairwise wins
        scores = []
        for i in range(n):
            wins = 0
            for j in range(n):
                if i != j:
                    if self.win_matrix[i, j] > self.win_matrix[j, i]:
                        wins += 1
                    elif self.win_matrix[i, j] == self.win_matrix[j, i]:
                        wins += 0.5
            scores.append((i, wins))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in scores]

    def _resize_win_matrix(self, new_size: int, keep_count: int):
        """Resize win matrix for updated pool"""
        new_matrix = np.zeros((new_size, new_size))
        copy_size = min(keep_count, new_size)
        new_matrix[:copy_size, :copy_size] = self.win_matrix[:copy_size, :copy_size]
        self.win_matrix = new_matrix

    def _get_top_win_rate(self) -> float:
        """Get win rate of top performer"""
        if self.win_matrix is None or len(self.instruction_pool) == 0:
            return 0.0

        total_wins = np.sum(self.win_matrix, axis=1)
        total_games = total_wins + np.sum(self.win_matrix, axis=0)
        win_rates = np.divide(total_wins, total_games, where=total_games > 0, out=np.zeros_like(total_wins))
        return float(np.max(win_rates)) if len(win_rates) > 0 else 0.0
