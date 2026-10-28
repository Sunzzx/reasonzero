"""
Unit and integration tests for formal reasoning problem environments.
"""

import pytest
from reasonzero.core.types import Action, State
from reasonzero.reasoning.tasks import CountdownMathEnv, PropositionalLogicEnv
from reasonzero.reasoning.engine import ReasoningEngine


def test_countdown_env_step_logic():
    env = CountdownMathEnv(numbers=(3, 8), target=24)
    init_state = env.initial_state()
    assert not init_state.is_successful

    actions = env.valid_actions(init_state)
    mul_act = next(a for a in actions if a.name == "mul")

    step_res = env.step(init_state, mul_act)
    assert step_res.is_successful
    assert step_res.is_terminal
    assert step_res.next_state.context == (24,)
    assert env.verify(step_res.next_state)


def test_countdown_env_4_numbers_solve():
    # (4, 4, 10, 10) -> (10 - 4/4) * ... wait: (10 * 10 - 4) / 4 = 96 / 4 = 24!
    # Or (6, 6, 6, 6) -> 6 + 6 + 6 + 6 = 24
    env = CountdownMathEnv(numbers=(6, 6, 6, 6), target=24)
    engine = ReasoningEngine()
    result = engine.solve(env, budget=100)

    assert result.success is True
    assert len(result.best_path) == 3
    assert env.verify(result.best_trajectory[-1])


def test_propositional_logic_modus_ponens_chain():
    # Premises: A, A -> B, B -> C, C -> D. Query: D
    env = PropositionalLogicEnv(
        premises=("A", "A -> B", "B -> C", "C -> D"),
        query="D",
    )
    engine = ReasoningEngine()
    result = engine.solve(env, budget=60)

    assert result.success is True
    assert "D" in result.best_trajectory[-1].context
    assert env.verify(result.best_trajectory[-1])


def test_propositional_logic_disjunctive_syllogism():
    # Premises: ~P, P | Q. Query: Q
    env = PropositionalLogicEnv(
        premises=("~P", "P | Q"),
        query="Q",
    )
    engine = ReasoningEngine()
    result = engine.solve(env, budget=30)

    assert result.success is True
    assert "Q" in result.best_trajectory[-1].context
    assert env.verify(result.best_trajectory[-1])
