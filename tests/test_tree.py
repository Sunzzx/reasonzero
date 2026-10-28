"""
Unit and integration tests for MCTSTree search algorithm.
"""

import pytest
from reasonzero.core.types import State, Action, StepResult
from reasonzero.core.tree import MCTSTree, MCTSConfig
from reasonzero.prm.base import BasePRM, StepReward
from reasonzero.reasoning.tasks import CountdownMathEnv


class MockPRM(BasePRM):
    def evaluate_step(self, state: State, action: Action, next_state: State) -> StepReward:
        return StepReward(score=0.5, prior=1.0, confidence=1.0)

    def estimate_value(self, state: State) -> float:
        return 0.2


def test_tree_initialization():
    env = CountdownMathEnv(numbers=(4, 4, 10, 10), target=24)
    prm = MockPRM()
    tree = MCTSTree(env=env, prm=prm)
    root = tree.initialize()

    assert root.depth == 0
    assert len(root.unexpanded_actions) > 0
    assert tree.expanded_nodes == 1
    assert tree.total_simulations == 0


def test_tree_search_finds_solution():
    # Easy 24 problem: 20 + 4 = 24
    env = CountdownMathEnv(numbers=(20, 4), target=24)
    prm = MockPRM()
    config = MCTSConfig(simulation_budget=20, early_stop_on_success=True)
    tree = MCTSTree(env=env, prm=prm, config=config)

    result = tree.search()
    assert result.success is True
    assert len(result.best_path) == 1
    assert result.best_path[0].name == "add"
    assert result.best_path[0].data == (4, 20, 24) or result.best_path[0].data == (20, 4, 24)


def test_backpropagation_reward_discounting():
    env = CountdownMathEnv(numbers=(6, 4), target=24)
    prm = MockPRM()
    config = MCTSConfig(simulation_budget=5, gamma=0.9, prm_weight=0.5)
    tree = MCTSTree(env=env, prm=prm, config=config)

    result = tree.search()
    assert tree.total_simulations >= 1
    assert tree.root is not None
    assert tree.root.visit_count >= 1
    assert tree.root.total_value != 0.0
