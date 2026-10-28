"""
ReasonZero CLI: Interactive Test-Time Compute & Deliberative Reasoning Engine.
"""

from __future__ import annotations
import argparse
import sys
import time

from reasonzero.reasoning.engine import ReasoningEngine, TestTimeComputeConfig
from reasonzero.reasoning.tasks import CountdownMathEnv, PropositionalLogicEnv
from reasonzero.visualizer.ascii_tree import render_tree_ascii
from reasonzero.prm.heuristic import ReasoningPRM
from reasonzero.core.tree import MCTSTree, MCTSConfig


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="reasonzero",
        description="ReasonZero: Test-Time Compute MCTS Reasoning Engine with Process Reward Models",
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-command to execute")

    # Math solver command
    math_parser = subparsers.add_parser("solve-math", help="Solve Countdown / Math24 puzzle via MCTS")
    math_parser.add_argument(
        "--numbers",
        type=int,
        nargs="+",
        default=[4, 4, 10, 10],
        help="List of numbers to combine (default: 4 4 10 10)",
    )
    math_parser.add_argument(
        "--target",
        type=int,
        default=24,
        help="Target value to reach (default: 24)",
    )
    math_parser.add_argument(
        "--budget",
        type=int,
        default=100,
        help="Test-time simulation budget (default: 100)",
    )
    math_parser.add_argument(
        "--render-tree",
        action="store_true",
        help="Print ASCII search tree after search",
    )

    # Logic solver command
    logic_parser = subparsers.add_parser("solve-logic", help="Solve Deductive Propositional Logic Theorem")
    logic_parser.add_argument(
        "--query",
        type=str,
        default="S",
        help="Goal proposition to prove (default: S)",
    )
    logic_parser.add_argument(
        "--budget",
        type=int,
        default=50,
        help="Simulation budget (default: 50)",
    )
    logic_parser.add_argument(
        "--render-tree",
        action="store_true",
        help="Print ASCII search tree after search",
    )

    # Compute scaling experiment
    scale_parser = subparsers.add_parser(
        "scaling-bench",
        help="Empirically evaluate Test-Time Compute scaling laws across simulation budgets",
    )
    scale_parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=[5, 15, 30, 60, 120],
        help="List of compute budgets to evaluate",
    )

    args = parser.parse_args()

    if args.command is None or args.command == "solve-math":
        numbers = tuple(getattr(args, "numbers", [4, 4, 10, 10]))
        target = getattr(args, "target", 24)
        budget = getattr(args, "budget", 100)
        render = getattr(args, "render_tree", False)

        print(f"\033[1;36m=== ReasonZero: Countdown Math Solver ===\033[0m")
        print(f"Initial numbers: {numbers} | Target: {target} | Budget: {budget}\n")

        env = CountdownMathEnv(numbers=numbers, target=target)
        prm = ReasoningPRM()
        mcts_config = MCTSConfig(simulation_budget=budget, early_stop_on_success=True)
        tree = MCTSTree(env=env, prm=prm, config=mcts_config)

        result = tree.search()
        print(f"Status: {'SUCCESS' if result.success else 'FAILED'}")
        print(f"Total Simulations: {result.total_simulations} | Expanded Nodes: {result.expanded_nodes}")
        print(f"Search Time: {result.search_time_seconds * 1000:.2f} ms\n")

        print(f"\033[1;32mOptimal Reasoning Trajectory:\033[0m")
        for step_i, act in enumerate(result.best_path):
            print(f"  Step {step_i + 1}: {act} => {result.best_trajectory[step_i + 1].context}")

        if render and tree.root:
            print(f"\n\033[1;33m=== MCTS Search Tree ===\033[0m")
            print(render_tree_ascii(tree.root, max_depth=3, highlight_path=result.best_path))
        return 0 if result.success else 1

    elif args.command == "solve-logic":
        query = args.query
        budget = args.budget
        render = args.render_tree

        print(f"\033[1;36m=== ReasonZero: Deductive Logic Theorem Prover ===\033[0m")
        env = PropositionalLogicEnv(query=query)
        prm = ReasoningPRM()
        tree = MCTSTree(env=env, prm=prm, config=MCTSConfig(simulation_budget=budget))

        result = tree.search()
        print(f"Goal Query: {query} | Success: {result.success}")
        print(f"Path Steps: {len(result.best_path)} | Time: {result.search_time_seconds * 1000:.2f} ms\n")

        for step_i, act in enumerate(result.best_path):
            print(f"  Lemma {step_i + 1}: {act} (reward: {result.step_rewards[step_i]:+.2f})")

        if render and tree.root:
            print(f"\n\033[1;33m=== Logic Derivation Tree ===\033[0m")
            print(render_tree_ascii(tree.root, max_depth=4, highlight_path=result.best_path))
        return 0 if result.success else 1

    elif args.command == "scaling-bench":
        budgets = args.budgets
        print(f"\033[1;36m=== ReasonZero: Test-Time Compute Scaling Benchmark ===\033[0m")
        print(f"Evaluating performance across simulation budgets: {budgets}\n")

        env = CountdownMathEnv(numbers=(4, 4, 10, 10), target=24)
        engine = ReasoningEngine()
        scaling = engine.evaluate_scaling(env, budgets=budgets)

        print(f"| Budget (Sims) | Success | Root Q   | Expanded | Time (ms) | Steps |")
        print(f"|--------------|---------|----------|----------|-----------|-------|")
        for row in scaling["scaling_data"]:
            succ_str = " PASS " if row['success'] else " FAIL "
            print(
                f"| {row['budget']:<12} | {succ_str} | {row['root_q']:<8} | "
                f"{row['expanded_nodes']:<8} | {row['time_ms']:<9} | {row['path_length']:<5} |"
            )
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
