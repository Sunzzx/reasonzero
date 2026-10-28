"""
Unit tests for MCTSNode data structure and PUCT computation.
"""

import math
import pytest
from reasonzero.core.types import State, Action
from reasonzero.core.node import MCTSNode


def test_node_initialization():
    state = State(step_index=0, context=(1, 2, 3), key="root")
    node = MCTSNode(state=state, depth=0)

    assert node.state == state
    assert node.visit_count == 0
    assert node.total_value == 0.0
    assert node.q_value == 0.0
    assert node.depth == 0
    assert node.is_leaf
    assert not node.is_terminal


def test_puct_exploration_formula():
    state = State(step_index=0, context=(), key="root")
    root = MCTSNode(state=state)
    root.visit_count = 10

    child_state = State(step_index=1, context=(), key="child")
    child = root.add_child(Action(name="act1"), child_state, prior=0.8, step_reward=0.2)
    child.visit_count = 2
    child.total_value = 1.0  # Q = 0.5

    c_puct = 1.414
    expected_q = 0.5
    expected_u = c_puct * 0.8 * (math.sqrt(10) / (1 + 2))
    expected_puct = expected_q + expected_u

    score = child.puct_score(parent_visits=10, c_puct=c_puct)
    assert pytest.approx(score, rel=1e-5) == expected_puct


def test_select_best_child():
    state = State(step_index=0, context=(), key="root")
    root = MCTSNode(state=state)
    root.visit_count = 5

    act1 = Action(name="act1")
    act2 = Action(name="act2")

    c1 = root.add_child(act1, State(step_index=1, context=(), key="c1"), prior=0.1)
    c2 = root.add_child(act2, State(step_index=1, context=(), key="c2"), prior=0.9)

    best_act, best_node = root.select_best_child(c_puct=1.414)
    assert best_act == act2
    assert best_node == c2


def test_best_action_by_visits_and_q():
    state = State(step_index=0, context=(), key="root")
    root = MCTSNode(state=state)

    act1 = Action(name="act1")
    act2 = Action(name="act2")

    c1 = root.add_child(act1, State(step_index=1, context=(), key="c1"))
    c2 = root.add_child(act2, State(step_index=1, context=(), key="c2"))

    c1.visit_count = 10
    c1.total_value = 5.0  # Q = 0.5

    c2.visit_count = 2
    c2.total_value = 1.8  # Q = 0.9

    assert root.best_action_by_visits()[0] == act1
    assert root.best_action_by_q()[0] == act2
