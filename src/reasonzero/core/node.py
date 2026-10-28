"""
Monte Carlo Tree Search (MCTS) Node Representation with PUCT Selection
and Process Reward Model (PRM) integration.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import math
import numpy as np

from reasonzero.core.types import State, Action


class MCTSNode:
    """
    A node in the Monte Carlo Search Tree representing a deliberative reasoning state.

    Attributes:
        state: The formal State object at this node.
        parent: Reference to the parent MCTSNode (None for root).
        action_from_parent: The Action that led from parent to this node.
        step_reward: Process reward r(s_{parent}, a) assigned by PRM for this transition.
        children: Dictionary mapping Action -> MCTSNode.
        unexpanded_actions: Actions known valid from this state that haven't been expanded yet.
        visit_count: N(s), number of times this node has been visited.
        total_value: W(s), sum of backpropagated returns through this node.
        prior: P(s|parent), prior probability given by policy/heuristics.
        depth: Tree depth (root = 0).
    """

    def __init__(
        self,
        state: State,
        parent: Optional[MCTSNode] = None,
        action_from_parent: Optional[Action] = None,
        prior: float = 1.0,
        step_reward: float = 0.0,
        depth: int = 0,
    ):
        self.state = state
        self.parent = parent
        self.action_from_parent = action_from_parent
        self.prior = float(prior)
        self.step_reward = float(step_reward)
        self.depth = depth

        self.children: Dict[Action, MCTSNode] = {}
        self.unexpanded_actions: List[Action] = []
        self.visit_count: int = 0
        self.total_value: float = 0.0
        self.is_terminal: bool = state.is_terminal
        self.is_successful: bool = state.is_successful

    @property
    def q_value(self) -> float:
        """Expected return Q(s) = W(s) / N(s). Defaults to 0.0 if unvisited."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    @property
    def is_leaf(self) -> bool:
        """True if the node has no expanded children."""
        return len(self.children) == 0

    @property
    def is_fully_expanded(self) -> bool:
        """True if all valid actions from this state have been expanded."""
        return len(self.unexpanded_actions) == 0 and len(self.children) > 0

    def puct_score(self, parent_visits: int, c_puct: float = 1.414, temperature: float = 1.0) -> float:
        """
        Compute Predictor Upper Confidence bound for Trees (PUCT) score:
            PUCT(s, a) = Q(s, a) + c_puct * P(s, a) * (sqrt(N(parent)) / (1 + N(s)))
        """
        q = self.q_value
        # Exploration term
        exploration = (
            c_puct
            * self.prior
            * (math.sqrt(max(1, parent_visits)) / (1.0 + self.visit_count))
        )
        return q + exploration

    def select_best_child(
        self,
        c_puct: float = 1.414,
        temperature: float = 1.0,
    ) -> Tuple[Action, MCTSNode]:
        """
        Select child maximizing PUCT score.
        """
        if not self.children:
            raise ValueError("Cannot select child from a leaf node with no children.")

        best_score = -float("inf")
        best_action = None
        best_child = None

        for action, child in self.children.items():
            score = child.puct_score(self.visit_count, c_puct=c_puct, temperature=temperature)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        assert best_action is not None and best_child is not None
        return best_action, best_child

    def best_action_by_visits(self) -> Tuple[Action, MCTSNode]:
        """
        Select most visited child (robust policy for final decision).
        """
        if not self.children:
            raise ValueError("No children available to select best action.")
        return max(self.children.items(), key=lambda item: (item[1].visit_count, item[1].q_value))

    def best_action_by_q(self) -> Tuple[Action, MCTSNode]:
        """
        Select highest Q-value child.
        """
        if not self.children:
            raise ValueError("No children available to select best action.")
        return max(self.children.items(), key=lambda item: item[1].q_value)

    def add_child(
        self,
        action: Action,
        child_state: State,
        prior: float = 1.0,
        step_reward: float = 0.0,
    ) -> MCTSNode:
        """
        Instantiate and attach a new child node to this parent.
        """
        child = MCTSNode(
            state=child_state,
            parent=self,
            action_from_parent=action,
            prior=prior,
            step_reward=step_reward,
            depth=self.depth + 1,
        )
        self.children[action] = child
        if action in self.unexpanded_actions:
            self.unexpanded_actions.remove(action)
        return child

    def update(self, value: float) -> None:
        """Update visit count and accumulated return."""
        self.visit_count += 1
        self.total_value += float(value)

    def __repr__(self) -> str:
        return (
            f"MCTSNode(state={self.state}, N={self.visit_count}, "
            f"Q={self.q_value:.4f}, children={len(self.children)}, depth={self.depth})"
        )
