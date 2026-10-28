"""
ANSI / ASCII Tree Visualizer for MCTS Search Trees and Deliberative Paths.
"""

from __future__ import annotations
from typing import List, Optional, Set

from reasonzero.core.node import MCTSNode
from reasonzero.core.types import Action

# ANSI color codes
COLOR_RESET = "\033[0m"
COLOR_BOLD = "\033[1m"
COLOR_GREEN = "\033[32m"
COLOR_CYAN = "\033[36m"
COLOR_YELLOW = "\033[33m"
COLOR_RED = "\033[31m"
COLOR_MAGENTA = "\033[35m"


def render_tree_ascii(
    root: MCTSNode,
    max_depth: int = 4,
    highlight_path: Optional[List[Action]] = None,
    use_color: bool = True,
) -> str:
    """
    Render the MCTS search tree into a structured, human-readable ASCII hierarchy.
    """
    lines: List[str] = []
    path_set = set(highlight_path) if highlight_path is not None else set()

    def _format_node(node: MCTSNode, action: Optional[Action], is_on_path: bool) -> str:
        prefix = ""
        status_tag = ""
        if node.is_successful:
            status_tag = " [GOAL SUCCESS]"
        elif node.is_terminal:
            status_tag = " [TERMINAL]"

        action_str = str(action) if action else "ROOT"
        stats = f"(N={node.visit_count}, Q={node.q_value:+.3f}, r={node.step_reward:+.2f})"

        if not use_color:
            star = "* " if is_on_path else ""
            return f"{star}{action_str} {stats} => {node.state.context}{status_tag}"

        color = COLOR_GREEN if is_on_path else COLOR_CYAN
        if node.is_successful:
            color = COLOR_BOLD + COLOR_GREEN
        return f"{color}{action_str}{COLOR_RESET} {COLOR_YELLOW}{stats}{COLOR_RESET} => {node.state.context}{COLOR_MAGENTA}{status_tag}{COLOR_RESET}"

    def _traverse(node: MCTSNode, action: Optional[Action], prefix: str, is_last: bool, depth: int):
        if depth > max_depth:
            return

        is_on_path = action in path_set if action else True
        branch = "└── " if is_last else "├── "
        indent = prefix + ("    " if is_last else "│   ")

        line = (prefix + branch if depth > 0 else "") + _format_node(node, action, is_on_path)
        lines.append(line)

        children = list(node.children.items())
        # Sort children: on-path and highest visits first
        children.sort(key=lambda item: (item[0] in path_set, item[1].visit_count), reverse=True)

        for idx, (child_act, child_node) in enumerate(children):
            _traverse(
                child_node,
                child_act,
                prefix="" if depth == 0 else indent,
                is_last=(idx == len(children) - 1),
                depth=depth + 1,
            )

    _traverse(root, None, prefix="", is_last=True, depth=0)
    return "\n".join(lines)
