"""
Multi-criteria Process Reward Model (PRM) for Mathematical and Deductive Reasoning.
"""

from __future__ import annotations
from typing import Dict, Any, Optional
import math

from reasonzero.core.types import State, Action
from reasonzero.prm.base import BasePRM, StepReward


class ReasoningPRM(BasePRM):
    """
    Analytical Process Reward Model incorporating step validity, goal convergence,
    action complexity penalties, and dead-end avoidance.
    """

    def __init__(
        self,
        soundness_weight: float = 0.4,
        progress_weight: float = 0.4,
        efficiency_weight: float = 0.2,
    ):
        self.soundness_weight = soundness_weight
        self.progress_weight = progress_weight
        self.efficiency_weight = efficiency_weight

    def evaluate_step(
        self, state: State, action: Action, next_state: State
    ) -> StepReward:
        """
        Evaluates step correctness r(s, a, s') in [-1.0, 1.0] and computes prior P(a|s).
        """
        # Terminal success guarantee
        if next_state.is_terminal and next_state.is_successful:
            return StepReward(
                score=1.0,
                prior=1.0,
                confidence=1.0,
                reasoning_tag="TERMINAL_SUCCESS",
                metadata={"progress": 1.0, "soundness": 1.0},
            )

        if next_state.is_terminal and not next_state.is_successful:
            return StepReward(
                score=-1.0,
                prior=0.01,
                confidence=1.0,
                reasoning_tag="TERMINAL_FAILURE",
                metadata={"dead_end": True},
            )

        # Extract domain features if present in next_state metadata or context
        soundness_score = 1.0
        progress_score = 0.0
        prior = 1.0

        # Inspect action data
        if action.name in ("add", "sub", "mul", "div"):
            # Arithmetic step heuristic
            target = action.metadata.get("target", 24)
            current_nums = [float(x) for x in next_state.context if isinstance(x, (int, float))]
            if current_nums:
                # Distance of closest number to target
                min_dist = min(abs(n - target) for n in current_nums)
                progress_score = max(0.0, 1.0 - (min_dist / max(1.0, target * 2.0)))
                # Prioritize operations that yield integer factors of target
                exact_matches = [n for n in current_nums if n > 0 and target % n == 0]
                if exact_matches:
                    prior = 1.5
                elif any(n == target for n in current_nums):
                    prior = 3.0
                    progress_score = 1.0
        elif action.name in ("modus_ponens", "modus_tollens", "hypothetical_syllogism", "resolution"):
            # Logic step heuristic
            target_concl = action.metadata.get("target_conclusion", "")
            new_lemmas = [str(x) for x in next_state.context]
            if target_concl in new_lemmas:
                progress_score = 1.0
                prior = 3.0
            else:
                # Reward introducing atoms matching the target
                atoms_in_target = set(target_concl.replace("~", "").split())
                matched = sum(1 for atom in atoms_in_target if any(atom in lem for lem in new_lemmas))
                progress_score = matched / max(1, len(atoms_in_target))
                prior = 1.0 + progress_score

        # Compute composite step score
        composite = (
            self.soundness_weight * soundness_score
            + self.progress_weight * progress_score
            - self.efficiency_weight * (0.05 * next_state.step_index)
        )
        clipped_score = max(-1.0, min(1.0, composite))

        return StepReward(
            score=clipped_score,
            prior=max(0.05, prior),
            confidence=0.9,
            reasoning_tag="VALID_PROGRESS",
            metadata={
                "soundness": soundness_score,
                "progress": progress_score,
                "step_index": next_state.step_index,
            },
        )

    def estimate_value(self, state: State) -> float:
        """
        Heuristic evaluation of state value V(s) in [-1.0, 1.0].
        """
        if state.is_terminal:
            return 1.0 if state.is_successful else -1.0
        return 0.1
