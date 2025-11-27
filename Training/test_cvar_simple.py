#!/usr/bin/env python3
"""
Simple standalone test for CVaR reward computation.

This test doesn't require torch or d3rlpy - just validates the reward logic.
"""

import numpy as np


def compute_cvar_reward(
    raw_pnl: float,
    q05: float,
    q95: float,
    prob_profit: float,
    portfolio_state: dict = None,
    risk_lambda: float = 0.5,
) -> float:
    """
    Advanced reward shaping using CVaR (Conditional Value at Risk).

    This replaces naive linear downside penalty with:
    1. CVaR-adjusted returns (tail risk beyond VaR)
    2. Opportunity cost (regret for missing better trades)
    3. Path-dependent drawdown penalty (Kelly-inspired)

    Args:
        raw_pnl: Raw profit/loss from the trade
        q05: 5th percentile of return distribution (CVaR estimate)
        q95: 95th percentile of return distribution
        prob_profit: Probability of profit
        portfolio_state: Optional dict with portfolio metrics
        risk_lambda: CVaR penalty weight

    Returns:
        Shaped reward value
    """
    # 1. CVaR penalty (expected loss beyond VaR)
    cvar_95 = q05  # CQF already gives us 5th percentile
    downside_penalty = risk_lambda * abs(min(0.0, cvar_95))

    # 2. Opportunity cost (if available)
    opportunity_cost = 0.0
    if portfolio_state is not None:
        max_expected = portfolio_state.get('max_expected_return_in_group', 0.0)
        opportunity_cost = 0.1 * max(0.0, max_expected - raw_pnl)

    # 3. Path-dependent drawdown multiplier
    dd_multiplier = 1.0
    if portfolio_state is not None:
        current_dd = portfolio_state.get('current_drawdown', 0.0)
        dd_multiplier = 1.0 / (1.0 + abs(current_dd))

    # 4. Sharpe-adjusted return (risk-adjusted performance)
    uncertainty = q95 - q05
    sharpe_proxy = raw_pnl / (uncertainty + 1e-6)

    # Final reward
    reward = (
        sharpe_proxy * dd_multiplier  # Risk-adjusted returns with drawdown scaling
        - downside_penalty             # Tail risk penalty
        - opportunity_cost             # Regret for missing better trades
    )

    return reward


def test_cvar_reward():
    """Test CVaR reward computation with various scenarios."""
    print("=" * 70)
    print("CVaR REWARD COMPUTATION TESTS")
    print("=" * 70)

    # Test case 1: Safe profitable trade (high win rate, positive tail)
    print("\n1. SAFE PROFITABLE TRADE")
    print("-" * 70)
    reward1 = compute_cvar_reward(
        raw_pnl=100.0,
        q05=20.0,     # Even worst case is profitable
        q95=200.0,
        prob_profit=0.85,
        risk_lambda=0.5
    )
    print(f"PnL: $100, CVaR(q05): $20, q95: $200, P(profit): 85%")
    print(f"Reward: {reward1:.4f}")
    print("Analysis: Very safe trade - positive tail risk")

    # Test case 2: Risky profitable trade (positive expected, negative tail)
    print("\n2. RISKY PROFITABLE TRADE")
    print("-" * 70)
    reward2 = compute_cvar_reward(
        raw_pnl=100.0,
        q05=-150.0,   # Bad tail risk
        q95=200.0,
        prob_profit=0.65,
        risk_lambda=0.5
    )
    print(f"PnL: $100, CVaR(q05): -$150, q95: $200, P(profit): 65%")
    print(f"Reward: {reward2:.4f}")
    print(f"CVaR Penalty: {0.5 * 150:.2f}")
    print("Analysis: Same expected return but dangerous tail - lower reward")

    # Test case 3: Losing trade with high tail risk
    print("\n3. LOSING TRADE")
    print("-" * 70)
    reward3 = compute_cvar_reward(
        raw_pnl=-50.0,
        q05=-200.0,
        q95=50.0,
        prob_profit=0.35,
        risk_lambda=0.5
    )
    print(f"PnL: -$50, CVaR(q05): -$200, q95: $50, P(profit): 35%")
    print(f"Reward: {reward3:.4f}")
    print("Analysis: Negative expectation + terrible tail = strongly negative reward")

    # Test case 4: With opportunity cost
    print("\n4. MISSED BETTER OPPORTUNITY")
    print("-" * 70)
    portfolio_state = {
        'max_expected_return_in_group': 200.0,  # There was a better trade
        'current_drawdown': 0.0
    }
    reward4 = compute_cvar_reward(
        raw_pnl=100.0,
        q05=20.0,
        q95=200.0,
        prob_profit=0.85,
        portfolio_state=portfolio_state,
        risk_lambda=0.5
    )
    opportunity_cost = 0.1 * (200.0 - 100.0)
    print(f"PnL: $100, Best alternative: $200")
    print(f"Reward: {reward4:.4f}")
    print(f"Opportunity cost: {opportunity_cost:.2f}")
    print("Analysis: Penalized for missing better trade in same group")

    # Test case 5: During drawdown (reduce position sizing)
    print("\n5. TRADE DURING DRAWDOWN")
    print("-" * 70)
    portfolio_state = {
        'max_expected_return_in_group': 0.0,
        'current_drawdown': 0.10  # 10% drawdown
    }
    reward5 = compute_cvar_reward(
        raw_pnl=100.0,
        q05=20.0,
        q95=200.0,
        prob_profit=0.85,
        portfolio_state=portfolio_state,
        risk_lambda=0.5
    )
    dd_multiplier = 1.0 / (1.0 + 0.10)
    print(f"PnL: $100, Current DD: 10%")
    print(f"Reward: {reward5:.4f}")
    print(f"DD Multiplier: {dd_multiplier:.3f}")
    print("Analysis: Same trade, but scaled down during drawdown (Kelly-inspired)")

    # Verification
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)

    assert reward1 > reward2, f"Safe trade ({reward1:.3f}) should beat risky ({reward2:.3f})"
    print(f"✓ Safe trade ({reward1:.3f}) > Risky trade ({reward2:.3f})")

    assert reward1 > reward3, f"Profitable ({reward1:.3f}) should beat losing ({reward3:.3f})"
    print(f"✓ Profitable ({reward1:.3f}) > Losing trade ({reward3:.3f})")

    assert reward1 > reward4, f"No regret ({reward1:.3f}) should beat with regret ({reward4:.3f})"
    print(f"✓ No opportunity cost ({reward1:.3f}) > With regret ({reward4:.3f})")

    assert reward1 > reward5, f"No DD ({reward1:.3f}) should beat during DD ({reward5:.3f})"
    print(f"✓ No drawdown ({reward1:.3f}) > During drawdown ({reward5:.3f})")

    print("\n✅ ALL TESTS PASSED - CVaR reward logic working correctly\n")

    # Summary
    print("=" * 70)
    print("REWARD RANKING (Best to Worst)")
    print("=" * 70)
    results = [
        ("Safe profitable (no regret, no DD)", reward1),
        ("During drawdown", reward5),
        ("Missed better opportunity", reward4),
        ("Risky profitable (bad tail)", reward2),
        ("Losing trade", reward3),
    ]
    for i, (desc, r) in enumerate(sorted(results, key=lambda x: x[1], reverse=True), 1):
        print(f"{i}. {desc:.<45} {r:>10.4f}")


if __name__ == '__main__':
    test_cvar_reward()
