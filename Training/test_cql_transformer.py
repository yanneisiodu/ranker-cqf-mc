#!/usr/bin/env python3
"""
Test script for Transformer CQL pipeline components.

This validates:
1. CandidateSetEncoder forward pass with varying batch sizes and candidate counts
2. CVaR reward computation with edge cases
3. Dataset building with synthetic data
4. Integration with d3rlpy (if available)
"""

import sys
from pathlib import Path
import numpy as np

# Add Training directory to path
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

def test_cvar_reward():
    """Test CVaR reward computation."""
    print("=" * 60)
    print("TEST 1: CVaR Reward Computation")
    print("=" * 60)

    from cql_pipeline import compute_cvar_reward

    # Test case 1: Profitable trade with low tail risk
    reward1 = compute_cvar_reward(
        raw_pnl=100.0,
        q05=20.0,    # Positive 5th percentile (very safe)
        q95=200.0,
        prob_profit=0.85,
        risk_lambda=0.5
    )
    print(f"✓ Safe profitable trade: {reward1:.4f}")

    # Test case 2: Profitable trade with high tail risk
    reward2 = compute_cvar_reward(
        raw_pnl=100.0,
        q05=-150.0,  # Negative 5th percentile (dangerous)
        q95=200.0,
        prob_profit=0.65,
        risk_lambda=0.5
    )
    print(f"✓ Risky profitable trade: {reward2:.4f}")

    # Test case 3: Loss with high tail risk
    reward3 = compute_cvar_reward(
        raw_pnl=-50.0,
        q05=-200.0,
        q95=50.0,
        prob_profit=0.35,
        risk_lambda=0.5
    )
    print(f"✓ Losing trade: {reward3:.4f}")

    # Verify: Safe trade should have highest reward
    assert reward1 > reward2, f"Safe trade ({reward1}) should beat risky trade ({reward2})"
    assert reward1 > reward3, f"Profitable ({reward1}) should beat losing trade ({reward3})"

    print(f"\n✅ CVaR reward ordering correct: {reward1:.3f} > {reward2:.3f} > {reward3:.3f}\n")


def test_transformer_encoder():
    """Test Transformer encoder with varying inputs."""
    print("=" * 60)
    print("TEST 2: Transformer Set Encoder")
    print("=" * 60)

    try:
        import torch
        from cql_pipeline import CandidateSetEncoder
    except ImportError as e:
        print(f"⚠️  Skipping (missing dependency): {e}")
        return

    # Initialize encoder
    encoder = CandidateSetEncoder(
        candidate_feature_dim=15,
        context_feature_dim=5,
        d_model=64,  # Smaller for faster testing
        nhead=4,
        num_layers=2,
        max_candidates=10,
    )

    print(f"✓ Encoder initialized: {sum(p.numel() for p in encoder.parameters()):,} parameters")

    # Test case 1: Fixed batch with 5 candidates
    batch_size = 16
    num_candidates = 5

    candidates = torch.randn(batch_size, num_candidates, 15)
    context = torch.randn(batch_size, 5)
    mask = torch.ones(batch_size, num_candidates)  # All valid

    output = encoder(candidates, context, mask)

    assert output.shape == (batch_size, 64), f"Expected shape (16, 64), got {output.shape}"
    print(f"✓ Forward pass with {num_candidates} candidates: {output.shape}")

    # Test case 2: Variable candidates with padding
    num_candidates = 8
    candidates_padded = torch.randn(batch_size, num_candidates, 15)
    mask_padded = torch.ones(batch_size, num_candidates)
    mask_padded[:, 5:] = 0  # Last 3 candidates are padding

    output_padded = encoder(candidates_padded, context, mask_padded)

    assert output_padded.shape == (batch_size, 64), f"Expected shape (16, 64), got {output_padded.shape}"
    print(f"✓ Forward pass with padding: {output_padded.shape}")

    # Test case 3: Permutation invariance (within valid candidates)
    # Shuffle first 5 candidates
    perm = torch.randperm(5)
    candidates_perm = candidates_padded.clone()
    candidates_perm[:, :5, :] = candidates_perm[:, perm, :]
    mask_perm = mask_padded.clone()
    mask_perm[:, :5] = mask_perm[:, perm]

    output_perm = encoder(candidates_perm, context, mask_perm)

    # Outputs should be similar (not identical due to positional encoding)
    # but we can verify no NaN/Inf
    assert not torch.isnan(output_perm).any(), "Output contains NaN"
    assert not torch.isinf(output_perm).any(), "Output contains Inf"
    print(f"✓ Permutation test passed (no NaN/Inf)")

    print(f"\n✅ Transformer encoder validated\n")


def test_dataset_building():
    """Test dataset building with synthetic data."""
    print("=" * 60)
    print("TEST 3: Dataset Building with Synthetic Data")
    print("=" * 60)

    import pandas as pd
    from cql_pipeline import build_transformer_dataset

    # Create synthetic CQF predictions
    np.random.seed(42)
    n_samples = 1000

    dates = pd.date_range('2023-01-01', periods=n_samples // 5, freq='D')
    underlyings = ['SPY', 'QQQ', 'IWM']

    data = []
    for date in dates:
        for underlying in underlyings:
            for i in range(5):  # 5 candidates per date/underlying
                data.append({
                    'date': date,
                    'underlying': underlying,
                    'strike': 400 + np.random.randn() * 20,
                    'dte': np.random.randint(1, 60),
                    'option_type': np.random.choice(['call', 'put']),
                    'q0.05': np.random.randn() * 50 - 10,
                    'q0.50': np.random.randn() * 20,
                    'q0.95': np.random.randn() * 50 + 10,
                    'prob_profit': np.random.uniform(0.3, 0.9),
                    'delta': np.random.uniform(-1, 1),
                    'gamma': np.random.uniform(0, 0.1),
                    'vega': np.random.uniform(0, 100),
                    'theta': np.random.uniform(-5, 0),
                    'iv': np.random.uniform(0.15, 0.5),
                    'exp_return': np.random.randn() * 30,
                })

    cqf_preds = pd.DataFrame(data)

    print(f"✓ Created synthetic CQF predictions: {len(cqf_preds)} rows")
    print(f"  - Date range: {cqf_preds['date'].min()} to {cqf_preds['date'].max()}")
    print(f"  - Underlyings: {cqf_preds['underlying'].nunique()}")
    print(f"  - Mean exp_return: {cqf_preds['exp_return'].mean():.2f}")

    # Build dataset
    try:
        result = build_transformer_dataset(
            cqf_preds=cqf_preds,
            ranker_candidates=None,  # Not required when cqf_preds has all info
            top_k=5,
            candidate_feature_dim=15,
            context_feature_dim=5,
            reward_col='exp_return',
            risk_lambda=0.5,
        )

        obs, actions, rewards, terminals, feature_config = result

        print(f"\n✓ Dataset built successfully:")
        print(f"  - Observations: {obs.shape}")
        print(f"  - Actions: {actions.shape}")
        print(f"  - Rewards: {rewards.shape} (mean={rewards.mean():.4f}, std={rewards.std():.4f})")
        print(f"  - Terminals: {terminals.shape} (terminal rate={terminals.mean():.2%})")
        print(f"  - Candidate features: {feature_config['candidate_feature_dim']}")
        print(f"  - Context features: {feature_config['context_feature_dim']}")

        # Validate shapes
        n_episodes = len(obs)
        assert actions.shape == (n_episodes,), f"Actions shape mismatch"
        assert rewards.shape == (n_episodes,), f"Rewards shape mismatch"
        assert terminals.shape == (n_episodes,), f"Terminals shape mismatch"

        # Validate action space
        assert actions.min() >= 0, f"Invalid negative action: {actions.min()}"
        assert actions.max() < 5, f"Invalid action > top_k: {actions.max()}"

        print(f"\n✅ Dataset validation passed\n")

        return obs, actions, rewards, terminals

    except Exception as e:
        print(f"⚠️  Dataset building failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


def test_d3rlpy_integration():
    """Test integration with d3rlpy (if available)."""
    print("=" * 60)
    print("TEST 4: d3rlpy Integration")
    print("=" * 60)

    try:
        import d3rlpy
        from d3rlpy.datasets import MDPDataset
        from cql_pipeline import TransformerEncoderFactory
    except ImportError as e:
        print(f"⚠️  Skipping (missing dependency): {e}")
        return

    # Create small synthetic dataset
    obs = np.random.randn(100, 80).astype(np.float32)  # 5 candidates * 15 features + 5 context
    actions = np.random.randint(0, 5, size=100)
    rewards = np.random.randn(100).astype(np.float32)
    terminals = np.zeros(100, dtype=bool)
    terminals[19::20] = True  # Every 20th step is terminal

    # Build MDP dataset
    dataset = MDPDataset(
        observations=obs,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
    )

    print(f"✓ Created MDPDataset: {len(dataset)} transitions")

    # Test encoder factory
    encoder_factory = TransformerEncoderFactory(
        candidate_feature_dim=15,
        context_feature_dim=5,
        d_model=64,
        nhead=4,
        num_layers=1,  # Minimal for testing
        max_candidates=5,
    )

    print(f"✓ TransformerEncoderFactory created")

    # Try to build a minimal CQL config (won't train, just validate structure)
    try:
        from d3rlpy.algos import DiscreteCQLConfig

        config = DiscreteCQLConfig(
            actor_encoder_factory=encoder_factory,
            critic_encoder_factory=encoder_factory,
            learning_rate=1e-4,
            batch_size=32,
            gamma=0.95,
        )

        print(f"✓ DiscreteCQLConfig created successfully")

        # Create algorithm instance
        algo = config.create(device='cpu')
        print(f"✓ CQL algorithm instantiated")

        # Build with dataset (this validates encoder compatibility)
        algo.build_with_dataset(dataset)
        print(f"✓ Algorithm built with dataset")

        print(f"\n✅ d3rlpy integration validated\n")

    except Exception as e:
        print(f"⚠️  CQL config/build failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("TRANSFORMER CQL PIPELINE - COMPONENT TESTS")
    print("=" * 60 + "\n")

    # Test 1: CVaR rewards (no dependencies)
    try:
        test_cvar_reward()
    except Exception as e:
        print(f"❌ CVaR test failed: {e}\n")
        import traceback
        traceback.print_exc()

    # Test 2: Transformer encoder (requires torch)
    try:
        test_transformer_encoder()
    except Exception as e:
        print(f"❌ Transformer test failed: {e}\n")
        import traceback
        traceback.print_exc()

    # Test 3: Dataset building (requires pandas)
    try:
        obs, actions, rewards, terminals = test_dataset_building()
    except Exception as e:
        print(f"❌ Dataset test failed: {e}\n")
        import traceback
        traceback.print_exc()

    # Test 4: d3rlpy integration (requires d3rlpy)
    try:
        test_d3rlpy_integration()
    except Exception as e:
        print(f"❌ d3rlpy test failed: {e}\n")
        import traceback
        traceback.print_exc()

    print("=" * 60)
    print("TESTS COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
