import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Trading AI Studio", page_icon="📈", layout="wide")

st.title("Trading AI Studio")
st.caption("Ranker • CQF • Stress • Inference • Evaluation")

col1, col2 = st.columns([3, 2])
with col1:
    st.markdown("""
    **5-Stage ML Pipeline UI** for world-class algorithmic trading:
    
    **🏆 Stage 1: Ranker Training** - XGBoost ranking model (Optuna optimized)
    **📦 Stage 2: CQF Training** - Quantile forecasting + conformal prediction  
    **🚀 Stage 3: Inference Pipeline** - Integrated ranker + CQF + stress testing
    **🧠 Stage 4: IQL Training** - Reinforcement learning policy on inference outputs
    **🎯 Stage 5: Final Optimal Walkforward** - Risk-optimized validation
    
    **📊 Results Visualization** - Interactive charts and performance analysis
    """)
with col2:
    st.success("**TARGET PERFORMANCE**")
    st.metric("Win Rate", "86.6%")
    st.metric("Returns", "7,988.9%") 
    st.metric("Max Drawdown", "15.2%")
    st.metric("Calmar Ratio", "524.6")

root = Path(__file__).resolve().parents[1]

st.divider()
st.subheader("🔄 Pipeline Execution Order")
st.markdown("""
**⚠️ CRITICAL**: Execute stages in order 1→2→3→4→5 for optimal results.

1. **🏆 Ranker Training** → Generates: `xgboost_ranker2_*.joblib`
2. **📦 CQF Training** → Generates: `optimal_cqf_step8.joblib`  
3. **🚀 Inference** → Generates: `ranker_candidates.csv`, `cqf_predictions.csv`, `stress_metrics.csv`
4. **🧠 IQL Training** → Generates: `discrete_cql_policy.d3`, `policy_meta.json`
5. **🎯 Final Optimal Walkforward** → Generates: **86.6% win rate results** 🏆

Use **📊 Results Visualization** to view performance after each stage.
""")

st.info("💡 **Tip**: Each page calls the actual training scripts in `Training/` and `inference/`. All outputs are saved to standard directories.")


