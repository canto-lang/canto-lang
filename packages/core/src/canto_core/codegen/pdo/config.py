"""
PDO Configuration
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PDOConfig:
    """Configuration for Prompt Duel Optimizer"""

    # Pool settings
    num_initial_instructions: int = 8

    # Duel settings
    num_rounds: int = 50
    duels_per_round: int = 5

    # Evolution settings
    pool_update_frequency: int = 10
    prune_ratio: float = 0.3
    mutation_ratio: float = 0.5

    # Ranking
    ranking_method: str = "aggregate"  # copeland, borda, elo, trueskill, aggregate

    # Models
    judge_model: str = "gpt-5.1"
    task_model: str = "gpt-5.1"

    # Sampling
    temperature: float = 1.0
    exploration_tau: float = 0.2

    # Task type
    task_type: str = "close_ended"  # close_ended or open_ended

    # Output format
    verbose: bool = True

    # Z3 verification level for DSL semantic compliance
    # "full" - Full Z3 verification via LLM logic extraction (most accurate, slower)
    # "quick" - Quick string matching for variables/categories (faster, less accurate)
    # "final_only" - Quick during optimization, full verification on final result
    verification_level: str = "full"
