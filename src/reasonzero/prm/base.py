"""
Base interfaces and abstractions for Process Reward Models (PRMs).
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

from reasonzero.core.types import State, Action


@dataclass
class StepReward:
    """
    Step-level evaluation produced by a Process Reward Model.

    Attributes:
        score: Process reward scalar r in [-1.0, 1.0], reflecting step correctness/progress.
        prior: Policy prior probability P(a|s) in (0.0, 1.0].
        confidence: Model confidence in [0.0, 1.0].
        reasoning_tag: Diagnostic category (e.g. 'SOUND_INFERENCE', 'PROGRESS_TOWARD_GOAL').
        metadata: Detailed feature breakdown.
    """
    score: float
    prior: float = 1.0
    confidence: float = 1.0
    reasoning_tag: str = "DEFAULT"
    metadata: Dict[str, Any] = field(default_factory=dict)


class BasePRM(ABC):
    """
    Abstract Process Reward Model providing fine-grained step supervision
    and intermediate value estimation during test-time tree search.
    """

    @abstractmethod
    def evaluate_step(
        self,
        state: State,
        action: Action,
        next_state: State,
    ) -> StepReward:
        """Evaluate a single reasoning step transition (s_t, a_t, s_{t+1})."""
        pass

    @abstractmethod
    def estimate_value(self, state: State) -> float:
        """
        Heuristic or proxy value estimate V(s) in [-1.0, 1.0] representing
        the expected probability of successful task completion from state s.
        """
        pass

    def batch_evaluate(
        self,
        transitions: List[Tuple[State, Action, State]],
    ) -> List[StepReward]:
        """Batch evaluate multiple candidate reasoning steps."""
        return [self.evaluate_step(s, a, ns) for s, a, ns in transitions]
