"""
Test-Time Compute Engine: Orchestrates MCTS Deliberation and Scaling Experiments.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import math
import time

from reasonzero.core.types import State, Action, SearchResult, Environment
from reasonzero.core.tree import MCTSTree, MCTSConfig
from reasonzero.prm.base import BasePRM
from reasonzero.prm.heuristic import ReasoningPRM


@dataclass
class TestTimeComputeConfig:
    """Configuration for test-time computation and scaling budgets."""
    simulation_budget: int = 100
    c_puct: float = 1.414
    gamma: float = 0.95
    prm_weight: float = 0.5
    early_stop_on_success: bool = True
    seed: Optional[int] = 42


class ReasoningEngine:
    """
    Frontier deliberative test-time reasoning engine combining
    Monte Carlo Tree Search with Step-Level Process Reward Models.
    """

    def __init__(
        self,
        prm: Optional[BasePRM] = None,
        config: Optional[TestTimeComputeConfig] = None,
    ):
        self.prm = prm or ReasoningPRM()
        self.config = config or TestTimeComputeConfig()

    def solve(
        self,
        env: Environment,
        budget: Optional[int] = None,
        initial_state: Optional[State] = None,
    ) -> SearchResult:
        """
        Perform test-time deliberative search to solve a reasoning problem.
        """
        mcts_config = MCTSConfig(
            simulation_budget=budget or self.config.simulation_budget,
            c_puct=self.config.c_puct,
            gamma=self.config.gamma,
            prm_weight=self.config.prm_weight,
            early_stop_on_success=self.config.early_stop_on_success,
            seed=self.config.seed,
        )
        tree = MCTSTree(env=env, prm=self.prm, config=mcts_config)
        return tree.search(initial_state=initial_state)

    def evaluate_scaling(
        self,
        env: Environment,
        budgets: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Empirically measure test-time compute scaling laws across simulation budgets.
        Returns metrics across budgets: success rates, Q-values, expanded nodes, latency.
        """
        test_budgets = budgets or [5, 15, 30, 60, 120]
        results: Dict[int, SearchResult] = {}
        scaling_table: List[Dict[str, Any]] = []

        for b in test_budgets:
            res = self.solve(env, budget=b)
            results[b] = res
            scaling_table.append({
                "budget": b,
                "success": res.success,
                "root_q": round(res.root_q, 4),
                "expanded_nodes": res.expanded_nodes,
                "time_ms": round(res.search_time_seconds * 1000, 2),
                "path_length": len(res.best_path),
            })

        return {
            "scaling_data": scaling_table,
            "results": results,
            "all_succeeded": any(r["success"] for r in scaling_table),
        }
