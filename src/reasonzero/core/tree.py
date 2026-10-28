"""
Monte Carlo Tree Search (MCTS) Engine for Test-Time Deliberative Reasoning.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
import math
import random
import time

from reasonzero.core.types import State, Action, StepResult, SearchResult, Environment
from reasonzero.core.node import MCTSNode
from reasonzero.prm.base import BasePRM


@dataclass
class MCTSConfig:
    """Hyperparameters governing test-time compute & MCTS exploration."""
    simulation_budget: int = 100
    c_puct: float = 1.414
    gamma: float = 0.95
    prm_weight: float = 0.5
    terminal_weight: float = 1.0
    max_depth: int = 25
    temperature: float = 1.0
    seed: Optional[int] = 42
    early_stop_on_success: bool = True


class MCTSTree:
    """
    Deliberative MCTS search tree integrating step-level Process Reward Models (PRMs)
    and Outcome Reward Models (ORMs).
    """

    def __init__(
        self,
        env: Environment,
        prm: BasePRM,
        config: Optional[MCTSConfig] = None,
    ):
        self.env = env
        self.prm = prm
        self.config = config or MCTSConfig()
        if self.config.seed is not None:
            random.seed(self.config.seed)

        self.root: Optional[MCTSNode] = None
        self.total_simulations: int = 0
        self.expanded_nodes: int = 0
        self.successful_leaves: List[MCTSNode] = []

    def initialize(self, initial_state: Optional[State] = None) -> MCTSNode:
        """Initialize the search tree with a given or environment root state."""
        state = initial_state if initial_state is not None else self.env.initial_state()
        self.root = MCTSNode(state=state, depth=0)
        self.root.unexpanded_actions = list(self.env.valid_actions(state))
        self.total_simulations = 0
        self.expanded_nodes = 1
        self.successful_leaves.clear()
        return self.root

    def search(
        self,
        initial_state: Optional[State] = None,
        budget: Optional[int] = None,
        on_step_callback: Optional[Callable[[int, MCTSNode], None]] = None,
    ) -> SearchResult:
        """
        Execute deliberative MCTS reasoning across the allocated compute budget.
        """
        start_time = time.perf_counter()
        if self.root is None or initial_state is not None:
            self.initialize(initial_state)

        assert self.root is not None
        sim_budget = budget if budget is not None else self.config.simulation_budget

        for sim_idx in range(sim_budget):
            leaf, path_actions, path_nodes = self._select(self.root)

            # Check terminal state
            if leaf.is_terminal:
                value = 1.0 if leaf.is_successful else -0.5
                if leaf.is_successful and leaf not in self.successful_leaves:
                    self.successful_leaves.append(leaf)
                self._backpropagate(path_nodes, path_actions, leaf_value=value)
                self.total_simulations += 1
                if on_step_callback:
                    on_step_callback(sim_idx, self.root)
                if self.config.early_stop_on_success and leaf.is_successful:
                    break
                continue

            # Expand
            child, action = self._expand(leaf)
            if child is not None and action is not None:
                path_nodes.append(child)
                path_actions.append(action)
                eval_node = child
            else:
                eval_node = leaf

            # Rollout / Value Evaluation
            value = self._evaluate(eval_node)
            if eval_node.is_successful and eval_node not in self.successful_leaves:
                self.successful_leaves.append(eval_node)

            # Backpropagate discounted returns and PRM rewards
            self._backpropagate(path_nodes, path_actions, leaf_value=value)
            self.total_simulations += 1

            if on_step_callback:
                on_step_callback(sim_idx, self.root)

            if self.config.early_stop_on_success and eval_node.is_successful:
                break

        elapsed = time.perf_counter() - start_time
        return self._build_result(elapsed)

    def _select(self, node: MCTSNode) -> Tuple[MCTSNode, List[Action], List[MCTSNode]]:
        """
        Selection phase: descend the tree using PUCT until reaching a node
        that is terminal or has unexpanded actions.
        """
        current = node
        path_actions: List[Action] = []
        path_nodes: List[MCTSNode] = [current]

        while not current.is_terminal and current.depth < self.config.max_depth:
            # If current node has unexpanded actions, stop selection here for expansion
            if len(current.unexpanded_actions) > 0:
                break
            if current.is_leaf:
                break
            action, next_node = current.select_best_child(
                c_puct=self.config.c_puct,
                temperature=self.config.temperature,
            )
            path_actions.append(action)
            path_nodes.append(next_node)
            current = next_node

        return current, path_actions, path_nodes

    def _expand(self, node: MCTSNode) -> Tuple[Optional[MCTSNode], Optional[Action]]:
        """
        Expansion phase: instantiate one unexpanded action from the node.
        """
        if node.is_terminal or len(node.unexpanded_actions) == 0:
            return None, None

        action = node.unexpanded_actions.pop(0)
        step_res: StepResult = self.env.step(node.state, action)

        # Step-level Process Reward Model evaluation
        prm_reward = self.prm.evaluate_step(node.state, action, step_res.next_state)

        child = node.add_child(
            action=action,
            child_state=step_res.next_state,
            prior=prm_reward.prior,
            step_reward=prm_reward.score,
        )
        child.unexpanded_actions = list(self.env.valid_actions(step_res.next_state))
        self.expanded_nodes += 1
        return child, action

    def _evaluate(self, node: MCTSNode) -> float:
        """
        Evaluation phase: combines PRM terminal verification with step rewards.
        """
        if node.is_terminal:
            return 1.0 if node.is_successful else -0.5
        # Process reward model state estimate
        return self.prm.estimate_value(node.state)

    def _backpropagate(
        self,
        path_nodes: List[MCTSNode],
        path_actions: List[Action],
        leaf_value: float,
    ) -> None:
        """
        Backpropagation phase: updates N(s) and backpropagates discounted cumulative
        returns blended with step-level process rewards:
            G_t = prm_weight * r(s_t, a_t) + gamma * G_{t+1}
        """
        discounted_return = float(leaf_value) * self.config.terminal_weight
        gamma = self.config.gamma
        prm_w = self.config.prm_weight

        # Backpropagate from leaf to root
        for i in reversed(range(len(path_nodes))):
            node = path_nodes[i]
            node.update(discounted_return)
            # Blend step reward of the node into return for preceding nodes
            discounted_return = prm_w * node.step_reward + gamma * discounted_return

    def _build_result(self, elapsed_seconds: float) -> SearchResult:
        """Construct comprehensive SearchResult summarizing search execution."""
        assert self.root is not None
        if self.root.is_leaf:
            return SearchResult(
                best_action=None,
                best_path=[],
                best_trajectory=[self.root.state],
                root_q=self.root.q_value,
                total_simulations=self.total_simulations,
                expanded_nodes=self.expanded_nodes,
                max_depth=0,
                success=self.root.is_successful,
                search_time_seconds=elapsed_seconds,
            )

        # Extract best trajectory
        best_path: List[Action] = []
        best_trajectory: List[State] = [self.root.state]
        step_rewards: List[float] = []
        curr = self.root
        max_depth = 0

        # If any successful leaves were found, trace the highest-Q trajectory to success
        while not curr.is_leaf and not curr.is_terminal:
            # Select most visited or highest Q child
            action, next_node = curr.best_action_by_visits()
            best_path.append(action)
            best_trajectory.append(next_node.state)
            step_rewards.append(next_node.step_reward)
            curr = next_node
            max_depth = max(max_depth, curr.depth)

        best_action = best_path[0] if best_path else None
        is_success = curr.is_successful or (len(self.successful_leaves) > 0)

        return SearchResult(
            best_action=best_action,
            best_path=best_path,
            best_trajectory=best_trajectory,
            root_q=self.root.q_value,
            total_simulations=self.total_simulations,
            expanded_nodes=self.expanded_nodes,
            max_depth=max_depth,
            success=is_success,
            step_rewards=step_rewards,
            tree_stats={
                "root_children": len(self.root.children),
                "root_visits": self.root.visit_count,
                "successful_leaves_found": len(self.successful_leaves),
            },
            search_time_seconds=elapsed_seconds,
        )
