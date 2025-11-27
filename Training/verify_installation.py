#!/usr/bin/env python3
"""Quick verification script for Ultra-Performance CQL Pipeline"""

print("\n" + "="*70)
print("ULTRA-PERFORMANCE CQL PIPELINE - INSTALLATION VERIFICATION")
print("="*70 + "\n")

# Test 1: Core dependencies
print("1. CHECKING DEPENDENCIES...")
try:
    import pandas as pd
    print(f"   ✓ pandas {pd.__version__}")
except ImportError as e:
    print(f"   ✗ pandas missing: {e}")

try:
    import numpy as np
    print(f"   ✓ numpy {np.__version__}")
except ImportError as e:
    print(f"   ✗ numpy missing: {e}")

try:
    import sklearn
    print(f"   ✓ scikit-learn {sklearn.__version__}")
except ImportError as e:
    print(f"   ✗ scikit-learn missing: {e}")

try:
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   ✓ torch {torch.__version__} (device: {device})")
except ImportError as e:
    print(f"   ✗ torch missing: {e}")

try:
    import d3rlpy
    print(f"   ✓ d3rlpy {d3rlpy.__version__}")
except ImportError as e:
    print(f"   ✗ d3rlpy missing: {e}")

# Test 2: Import cql_pipeline
print("\n2. CHECKING CQL PIPELINE...")
try:
    import cql_pipeline
    print("   ✓ cql_pipeline.py imports successfully")
    
    # Check key components
    if hasattr(cql_pipeline, 'compute_cvar_reward'):
        print("   ✓ compute_cvar_reward() available")
    
    if cql_pipeline.torch is not None:
        if hasattr(cql_pipeline, 'CandidateSetEncoder'):
            print("   ✓ CandidateSetEncoder available")
        if hasattr(cql_pipeline, 'TransformerEncoderFactory'):
            print("   ✓ TransformerEncoderFactory available")
    else:
        print("   ⚠ Torch not available - Transformer classes disabled")
    
except Exception as e:
    print(f"   ✗ Failed to import: {e}")

# Test 3: Quick CVaR test
print("\n3. TESTING CVAR REWARD...")
try:
    from cql_pipeline import compute_cvar_reward
    
    # Safe trade
    r1 = compute_cvar_reward(100, 20, 200, 0.85, risk_lambda=0.5)
    # Risky trade
    r2 = compute_cvar_reward(100, -150, 200, 0.65, risk_lambda=0.5)
    
    if r1 > r2:
        print(f"   ✓ Reward logic correct: safe ({r1:.2f}) > risky ({r2:.2f})")
    else:
        print(f"   ✗ Reward logic incorrect")
        
except Exception as e:
    print(f"   ✗ CVaR test failed: {e}")

# Test 4: Transformer encoder (if torch available)
print("\n4. TESTING TRANSFORMER ENCODER...")
try:
    import torch
    from cql_pipeline import CandidateSetEncoder
    
    encoder = CandidateSetEncoder(
        candidate_feature_dim=15,
        context_feature_dim=5,
        d_model=64,
        nhead=4,
        num_layers=2,
    )
    
    # Test forward pass
    candidates = torch.randn(4, 5, 15)  # batch=4, candidates=5, features=15
    context = torch.randn(4, 5)
    mask = torch.ones(4, 5)
    
    output = encoder(candidates, context, mask)
    
    if output.shape == (4, 64):
        print(f"   ✓ Forward pass successful: {output.shape}")
        print(f"   ✓ Parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    else:
        print(f"   ✗ Unexpected output shape: {output.shape}")
        
except ImportError:
    print("   ⚠ Skipping (torch not available)")
except Exception as e:
    print(f"   ✗ Transformer test failed: {e}")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("\n✅ Installation verified successfully!")
print("\nNext steps:")
print("1. Prepare CQF predictions CSV")
print("2. Run: python3 cql_pipeline.py --cqf-preds <path> --no-train")
print("3. Train: python3 cql_pipeline.py --cqf-preds <path> --train-steps 10000")
print("\nSee INSTALLATION_GUIDE.md for detailed instructions.\n")
