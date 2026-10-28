"""
ReasonZero: Frontier Test-Time Compute & MCTS Reasoning Engine
with Step-Level Process Reward Models (PRM).
"""

__version__ = "0.1.0"

from reasonzero.core.node import MCTSNode
from reasonzero.core.tree import MCTSTree, MCTSConfig
from reasonzero.core.types import State, Action, StepResult, SearchResult
from reasonzero.prm.base import BasePRM, StepReward
from reasonzero.prm.heuristic import ReasoningPRM
from reasonzero.reasoning.engine import ReasoningEngine, TestTimeComputeConfig
from reasonzero.reasoning.tasks import (
    CountdownMathEnv,
    PropositionalLogicEnv,
    ReasoningEnvironment,
)
from reasonzero.visualizer.ascii_tree import render_tree_ascii

__all__ = [
    "MCTSNode",
    "MCTSTree",
    "MCTSConfig",
    "State",
    "Action",
    "StepResult",
    "SearchResult",
    "BasePRM",
    "StepReward",
    "ReasoningPRM",
    "ReasoningEngine",
    "TestTimeComputeConfig",
    "CountdownMathEnv",
    "PropositionalLogicEnv",
    "ReasoningEnvironment",
    "render_tree_ascii",
]
