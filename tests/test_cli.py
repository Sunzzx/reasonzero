"""
Tests for ReasonZero CLI entry points.
"""

import sys
import pytest
from unittest.mock import patch
from reasonzero.cli import main


def test_cli_solve_math(capsys):
    test_args = ["reasonzero", "solve-math", "--numbers", "6", "4", "--target", "24", "--budget", "20"]
    with patch.object(sys, "argv", test_args):
        code = main()
        captured = capsys.readouterr()
        assert code == 0
        assert "ReasonZero: Countdown Math Solver" in captured.out
        assert "SUCCESS" in captured.out


def test_cli_solve_logic(capsys):
    test_args = ["reasonzero", "solve-logic", "--query", "S", "--budget", "40"]
    with patch.object(sys, "argv", test_args):
        code = main()
        captured = capsys.readouterr()
        assert code == 0
        assert "ReasonZero: Deductive Logic Theorem Prover" in captured.out
        assert "Success: True" in captured.out


def test_cli_scaling_bench(capsys):
    test_args = ["reasonzero", "scaling-bench", "--budgets", "5", "10"]
    with patch.object(sys, "argv", test_args):
        code = main()
        captured = capsys.readouterr()
        assert code == 0
        assert "Budget (Sims)" in captured.out
