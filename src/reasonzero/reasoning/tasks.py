"""
Formal Reasoning Problem Environments for Test-Time Deliberation.
Includes Countdown / Math24 Arithmetic and Deductive Propositional Logic.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple

from reasonzero.core.types import Action, State, StepResult


class ReasoningEnvironment(ABC):
    """Abstract base class for formal reasoning environments."""

    @abstractmethod
    def initial_state(self) -> State:
        pass

    @abstractmethod
    def valid_actions(self, state: State) -> List[Action]:
        pass

    @abstractmethod
    def step(self, state: State, action: Action) -> StepResult:
        pass

    @abstractmethod
    def verify(self, state: State) -> bool:
        pass


class CountdownMathEnv(ReasoningEnvironment):
    """
    Countdown / Math24 Arithmetic Reasoning Environment.
    Given an initial list of numbers, reach a target value (default 24)
    by sequentially applying elementary operations: +, -, *, /.
    """

    def __init__(self, numbers: Tuple[int, ...] = (4, 4, 10, 10), target: int = 24):
        self.initial_numbers = tuple(sorted(numbers))
        self.target = target

    def initial_state(self) -> State:
        return State(
            step_index=0,
            context=self.initial_numbers,
            is_terminal=False,
            is_successful=any(n == self.target for n in self.initial_numbers),
            key=f"Nums: {self.initial_numbers} -> Target: {self.target}",
        )

    def valid_actions(self, state: State) -> List[Action]:
        if state.is_terminal or state.is_successful:
            return []

        nums = list(state.context)
        if len(nums) < 2:
            return []

        actions: List[Action] = []
        n_len = len(nums)
        seen_pairs = set()

        for i in range(n_len):
            for j in range(i + 1, n_len):
                a, b = nums[i], nums[j]
                pair_key = (a, b)
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                # Addition: a + b
                actions.append(
                    Action(
                        name="add",
                        data=(a, b, a + b),
                        metadata={"target": self.target, "i": i, "j": j},
                    )
                )

                # Multiplication: a * b
                actions.append(
                    Action(
                        name="mul",
                        data=(a, b, a * b),
                        metadata={"target": self.target, "i": i, "j": j},
                    )
                )

                # Subtraction: |a - b| (larger minus smaller, or both if different)
                if a >= b:
                    actions.append(
                        Action(
                            name="sub",
                            data=(a, b, a - b),
                            metadata={"target": self.target, "i": i, "j": j},
                        )
                    )
                if b > a:
                    actions.append(
                        Action(
                            name="sub",
                            data=(b, a, b - a),
                            metadata={"target": self.target, "i": j, "j": i},
                        )
                    )

                # Division: exact integer division only
                if b != 0 and a % b == 0:
                    actions.append(
                        Action(
                            name="div",
                            data=(a, b, a // b),
                            metadata={"target": self.target, "i": i, "j": j},
                        )
                    )
                if a != 0 and b % a == 0 and a != b:
                    actions.append(
                        Action(
                            name="div",
                            data=(b, a, b // a),
                            metadata={"target": self.target, "i": j, "j": i},
                        )
                    )

        return actions

    def step(self, state: State, action: Action) -> StepResult:
        nums = list(state.context)
        a, b, res = action.data[0], action.data[1], action.data[2]

        # Remove one occurrence of a and one of b
        if a in nums:
            nums.remove(a)
        if b in nums:
            nums.remove(b)
        nums.append(res)
        next_nums = tuple(sorted(nums))

        is_success = any(n == self.target for n in next_nums)
        is_terminal = is_success or len(next_nums) <= 1

        reward = 1.0 if is_success else (0.1 if res > 0 else -0.1)
        next_state = State(
            step_index=state.step_index + 1,
            context=next_nums,
            is_terminal=is_terminal,
            is_successful=is_success,
            key=f"{action} -> {next_nums}",
        )
        return StepResult(
            next_state=next_state,
            process_reward=reward,
            is_terminal=is_terminal,
            is_successful=is_success,
        )

    def verify(self, state: State) -> bool:
        return any(n == self.target for n in state.context)


class PropositionalLogicEnv(ReasoningEnvironment):
    """
    Deductive Logic Reasoning Environment.
    Given knowledge base axioms and a goal query proposition,
    derive the goal using formal deductive inference rules
    (Modus Ponens, Modus Tollens, Disjunctive Syllogism).
    """

    def __init__(
        self,
        premises: Tuple[str, ...] = ("P", "P -> Q", "Q -> R", "R -> S"),
        query: str = "S",
        max_steps: int = 10,
    ):
        self.premises = tuple(sorted(premises))
        self.query = query
        self.max_steps = max_steps

    def initial_state(self) -> State:
        is_success = self.query in self.premises
        return State(
            step_index=0,
            context=self.premises,
            is_terminal=is_success,
            is_successful=is_success,
            key=f"Premises: {self.premises} |= Query: {self.query}",
        )

    def valid_actions(self, state: State) -> List[Action]:
        if state.is_terminal or state.is_successful:
            return []
        if state.step_index >= self.max_steps:
            return []

        known = set(state.context)
        actions: List[Action] = []

        # 1. Modus Ponens: A and A -> B => B
        for prop in known:
            if " -> " in prop:
                antecedent, consequent = prop.split(" -> ", 1)
                if antecedent in known and consequent not in known:
                    actions.append(
                        Action(
                            name="modus_ponens",
                            data=(antecedent, prop, consequent),
                            metadata={"target_conclusion": self.query},
                        )
                    )

        # 2. Modus Tollens: ~B and A -> B => ~A
        for prop in known:
            if " -> " in prop:
                antecedent, consequent = prop.split(" -> ", 1)
                neg_consequent = f"~{consequent}" if not consequent.startswith("~") else consequent[1:]
                neg_antecedent = f"~{antecedent}" if not antecedent.startswith("~") else antecedent[1:]
                if neg_consequent in known and neg_antecedent not in known:
                    actions.append(
                        Action(
                            name="modus_tollens",
                            data=(neg_consequent, prop, neg_antecedent),
                            metadata={"target_conclusion": self.query},
                        )
                    )

        # 3. Disjunctive Syllogism: A | B and ~A => B
        for prop in known:
            if " | " in prop:
                parts = prop.split(" | ", 1)
                p1, p2 = parts[0], parts[1]
                neg_p1 = f"~{p1}" if not p1.startswith("~") else p1[1:]
                neg_p2 = f"~{p2}" if not p2.startswith("~") else p2[1:]
                if neg_p1 in known and p2 not in known:
                    actions.append(
                        Action(
                            name="disjunctive_syllogism",
                            data=(neg_p1, prop, p2),
                            metadata={"target_conclusion": self.query},
                        )
                    )
                if neg_p2 in known and p1 not in known:
                    actions.append(
                        Action(
                            name="disjunctive_syllogism",
                            data=(neg_p2, prop, p1),
                            metadata={"target_conclusion": self.query},
                        )
                    )

        return actions

    def step(self, state: State, action: Action) -> StepResult:
        new_fact = action.data[2]
        updated_context = tuple(sorted(set(state.context) | {new_fact}))
        is_success = self.query in updated_context
        is_terminal = is_success or (state.step_index + 1 >= self.max_steps)

        reward = 1.0 if is_success else 0.2
        next_state = State(
            step_index=state.step_index + 1,
            context=updated_context,
            is_terminal=is_terminal,
            is_successful=is_success,
            key=f"{action} => {new_fact}",
        )
        return StepResult(
            next_state=next_state,
            process_reward=reward,
            is_terminal=is_terminal,
            is_successful=is_success,
        )

    def verify(self, state: State) -> bool:
        return self.query in state.context
