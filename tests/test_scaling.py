"""
Tests for Test-Time Compute Scaling laws and budget evaluations.
"""

import pytest
from reasonzero.reasoning.tasks import CountdownMathEnv
from reasonzero.reasoning.engine import ReasoningEngine
from reasonzero.visualizer.ascii_tree import render_tree_ascii
from reasonzero.core.node import MCTSNode
from reasonzero.core.types import State, Action


def test_compute_scaling_monotonicity():
    env = CountdownMathEnv(numbers=(2, 3, 4), target=24)
    engine = ReasoningEngine()
    scaling = engine.evaluate_scaling(env, budgets=[5, 20, 50])

    assert "scaling_data" in scaling
    assert len(scaling["scaling_data"]) == 3
    # Node expansion count should increase with budget
    expansions = [row["expanded_nodes"] for row in scaling["scaling_data"]]
    assert expansions[0] <= expansions[-1]


def test_ascii_tree_rendering():
    root_state = State(step_index=0, context=(6, 4))
    root = MCTSNode(state=root_state)
    root.visit_count = 5
    act = Action(name="mul", data=(6, 4, 24))
    child_state = State(step_index=1, context=(24,), is_successful=True, is_terminal=True)
    child = root.add_child(act, child_state, prior=1.0, step_reward=1.0)
    child.visit_count = 5
    child.total_value = 5.0

    rendered = render_tree_ascii(root, max_depth=2, highlight_path=[act], use_color=False)
    assert "ROOT" in rendered
    assert "mul(6, 4, 24)" in rendered
    assert "[GOAL SUCCESS]" in rendered
