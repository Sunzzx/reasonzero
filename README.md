# ReasonZero: Test-Time Compute MCTS Reasoning Engine with Process Reward Models

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-20%2F20%20passing-success.svg)](./tests)

**ReasonZero** is an elite, mathematically rigorous, zero-dependency implementation of Test-Time Compute Monte Carlo Tree Search (MCTS) reasoning engines coupled with Step-Level Process Reward Models (PRMs).

---

## 📐 Architectural Pillars

### 1. PUCT Tree Search & Monte Carlo Deliberation
Implements multi-step lookahead reasoning trees guided by Predictor Upper Confidence bound for Trees (PUCT):
$$\text{PUCT}(s, a) = Q(s, a) + c_{\text{puct}} P(s, a) \frac{\sqrt{N(\text{parent})}}{1 + N(s)}$$
- **Early Stopping & Backpropagation**: Blends step-level process reward supervision $r(s_t, a_t)$ with outcome returns $G_t = \gamma G_{t+1} + w \cdot r(s_t, a_t)$.

### 2. Step-Level Process Reward Models (PRMs)
Provides fine-grained supervision at each intermediate reasoning step (arithmetic transformation or logical derivation lemma) rather than delayed binary outcome rewards.

### 3. Formal Reasoning Environments
- **CountdownMathEnv**: Solves arithmetic 24-puzzles via elementary operations $(+, -, \times, /)$.
- **PropositionalLogicEnv**: Proves deductive theorem goals via Modus Ponens, Modus Tollens, and Disjunctive Syllogism.

---

## 🚀 Quickstart & Usage

```bash
git clone https://github.com/Sunzzx/reasonzero.git
cd reasonzero
pip install -e .[dev]

# Run comprehensive test suite (20 tests)
pytest -v

# Solve Math 24 Countdown Puzzle with ASCII Tree Rendering
reasonzero solve-math --numbers 4 4 10 10 --target 24 --budget 100 --render-tree

# Prove Propositional Logic Theorem
reasonzero solve-logic --query S --budget 50 --render-tree

# Evaluate Test-Time Compute Scaling Laws
reasonzero scaling-bench --budgets 5 15 30 60 120
```

---

## 💼 Commercial Licensing & Monetization

`reasonzero` is dual-licensed under open-source MIT for personal experimentation and research, with enterprise commercial licenses available for proprietary integration and high-performance automated reasoning deployments.

- **Commercial License ($149 USD single seat):** [Purchase via Stripe Checkout](https://buy.stripe.com/test_reasonzero_license_999)
- **Sponsor Development:** [Polar.sh Creator Funding](https://polar.sh)
- **Enterprise Support & Inquiries:** `ayushkandwal2007@gmail.com`

---

## 📄 License
MIT License. Authored by Ayush Kandwal (Sunzzx).
