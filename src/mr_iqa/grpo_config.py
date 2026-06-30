from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from transformers import TrainingArguments


@dataclass
class MRGRPOConfig(TrainingArguments):
    """Minimal GRPO config for the MR-IQA trainer on older TRL releases."""

    model_init_kwargs: dict[str, Any] = field(default_factory=dict)
    max_prompt_length: Optional[int] = field(default=None)
    max_completion_length: int = field(default=512)
    num_generations: int = field(default=2)
    temperature: float = field(default=1.0)
    beta: float = field(default=0.0)
    epsilon: float = field(default=0.2)
    epsilon_low: float = field(default=0.2)
    num_iterations: int = field(default=1)
    remove_unused_columns: bool = field(default=False)
