"""
Unit tests for Process Reward Models (PRMs).
"""

import pytest
from reasonzero.core.types import State, Action
from reasonzero.prm.heuristic import ReasoningPRM


def test_prm_terminal_success():
    prm = ReasoningPRM()
    s = State(step_index=0, context=(6, 4))
    act = Action(name="mul", data=(6, 4, 24))
    next_s = State(step_index=1, context=(24,), is_terminal=True, is_successful=True)

    reward = prm.evaluate_step(s, act, next_s)
    assert reward.score == 1.0
    assert reward.reasoning_tag == "TERMINAL_SUCCESS"


def test_prm_terminal_failure():
    prm = ReasoningPRM()
    s = State(step_index=0, context=(5, 5))
    act = Action(name="div", data=(5, 5, 1))
    next_s = State(step_index=1, context=(1,), is_terminal=True, is_successful=False)

    reward = prm.evaluate_step(s, act, next_s)
    assert reward.score == -1.0
    assert reward.reasoning_tag == "TERMINAL_FAILURE"


def test_prm_logic_deduction_step():
    prm = ReasoningPRM()
    s = State(step_index=0, context=("P", "P -> Q"))
    act = Action(
        name="modus_ponens",
        data=("P", "P -> Q", "Q"),
        metadata={"target_conclusion": "Q"},
    )
    next_s = State(step_index=1, context=("P", "P -> Q", "Q"), is_terminal=True, is_successful=True)

    reward = prm.evaluate_step(s, act, next_s)
    assert reward.score == 1.0


def test_prm_batch_evaluate():
    prm = ReasoningPRM()
    s = State(step_index=0, context=(2, 3))
    act1 = Action(name="add", data=(2, 3, 5))
    ns1 = State(step_index=1, context=(5,))
    act2 = Action(name="mul", data=(2, 3, 6))
    ns2 = State(step_index=1, context=(6,))

    evals = prm.batch_evaluate([(s, act1, ns1), (s, act2, ns2)])
    assert len(evals) == 2
    assert all(isinstance(e.score, float) for e in evals)
