"""
Core data types and interfaces for ReasonZero.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Protocol, runtime_checkable
import numpy as np


@dataclass(frozen=True)
class Action:
    """Atomic reasoning step / action taken in the reasoning tree."""
    name: str
    data: Tuple[Any, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict, hash=False, compare=False)

    def __str__(self) -> str:
        if self.data:
            return f"{self.name}({', '.join(str(x) for x in self.data)})"
        return self.name


@dataclass(frozen=True)
class State:
    """
    Immutable state representing the cumulative reasoning context.
    """
    step_index: int
    context: Tuple[Any, ...]
    is_terminal: bool = False
    is_successful: bool = False
    key: str = ""

    def __str__(self) -> str:
        return self.key or f"State(step={self.step_index}, ctx={len(self.context)})"


@dataclass
class StepResult:
    """Result of taking an action from a state."""
    next_state: State
    process_reward: float
    is_terminal: bool
    is_successful: bool
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """Comprehensive summary of test-time MCTS computation."""
    best_action: Optional[Action]
    best_path: List[Action]
    best_trajectory: List[State]
    root_q: float
    total_simulations: int
    expanded_nodes: int
    max_depth: int
    success: bool
    step_rewards: List[float] = field(default_factory=list)
    tree_stats: Dict[str, Any] = field(default_factory=dict)
    search_time_seconds: float = 0.0


@runtime_checkable
class Environment(Protocol):
    """Protocol for formal reasoning environments."""
    def initial_state(self) -> State:
        ...

    def valid_actions(self, state: State) -> List[Action]:
        ...

    def step(self, state: State, action: Action) -> StepResult:
        ...

    def verify(self, state: State) -> bool:
        ...
