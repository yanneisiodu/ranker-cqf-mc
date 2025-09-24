import streamlit as st
from pathlib import Path
import subprocess
import sys

st.set_page_config(page_title="IQL Training", page_icon="🧠", layout="wide")
st.title("🧠 IQL Policy Training (Stage 4)")

base = Path(__file__).resolve().parents[2]
training_dir = base / "Training"

with st.sidebar:
    st.header("IQL Training Configuration")
    
    # Training mode selection
    training_mode = st.selectbox(
        "Training Mode",
        ["Precomputed Inference Artifacts", "End-to-End (Raw Data)"],
        index=0
    )
    
    if training_mode == "Precomputed Inference Artifacts":
        st.subheader("Inference Artifacts")
        cqf_preds = st.text_input("CQF Predictions", str(base / "inference_outputs/cqf_predictions.csv"))
        ranker_candidates = st.text_input("Ranker Candidates", str(base / "inference_outputs/ranker_candidates.csv"))
        stress_metrics = st.text_input("Stress Metrics", str(base / "inference_outputs/stress_metrics.csv"))
    else:
        st.subheader("Raw Data + Models")
        raw_data = st.text_input("Raw Data CSV", str(base / "data/year_2024_data.csv"))
        config = st.text_input("Config YAML", str(base / "config.yaml"))
        ranker_model = st.text_input("Ranker Model", str(base / "model_output/xgboost_ranker_2022_2024_optuna_tuned_20250829_182605.joblib"))
        ranker_features = st.text_input("Ranker Features", str(base / "model_output/xgb_feature_names_2022_2024_20250829_182605.pkl"))
        sharpe_edges = st.text_input("Sharpe Edges", str(base / "model_output/sharpe_qcut_edges_2022_2024_20250829_182605.pkl"))
        cqf_model = st.text_input("CQF Model", str(base / "model_output/optimal_cqf_step8.joblib"))
    
    st.subheader("IQL Parameters")
    outdir = st.text_input("Output Directory", str(base / "iql_training_2024"))
    top_k = st.number_input("Top-K Candidates", min_value=3, max_value=10, value=5)
    size_bins = st.text_input("Size Bins", "0.5,1.0")
    train_steps = st.number_input("Training Steps", min_value=10000, max_value=500000, value=200000, step=10000)
    batch_size = st.number_input("Batch Size", min_value=256, max_value=4096, value=1024, step=256)
    expectile = st.number_input("Expectile", min_value=0.5, max_value=0.9, value=0.7, step=0.05)
    gamma = st.number_input("Gamma (Discount)", min_value=0.9, max_value=0.999, value=0.99, step=0.001)
    seed = st.number_input("Random Seed", min_value=1, max_value=999, value=42)
    
    # Advanced options
    with st.expander("Advanced Options"):
        no_train = st.checkbox("Build Dataset Only (Skip Training)", value=False)
        group_top_n = st.number_input("Group Top-N", min_value=10, max_value=1000, value=50)
        include_future_targets = st.checkbox("Include Future Targets", value=True)
        
    run_btn = st.button("Train IQL Policy", type="primary")

st.markdown("Uses `Training/iql_pipeline.py` to train DiscreteCQL policy on inference outputs.")

# Show usage examples
st.subheader("📋 Usage Examples")

if training_mode == "Precomputed Inference Artifacts":
    st.code(f"""
# Standard IQL training with inference artifacts:
python Training/iql_pipeline.py \\
    --cqf-preds {cqf_preds} \\
    --ranker-candidates {ranker_candidates} \\
    --stress-metrics {stress_metrics} \\
    --outdir {outdir} \\
    --train-steps {train_steps} \\
    --train-batch-size {batch_size}
    """, language="bash")
else:
    st.code(f"""
# End-to-end training (raw data → inference → IQL):
python Training/iql_pipeline.py \\
    --raw-data {raw_data} \\
    --config {config} \\
    --ranker-model {ranker_model} \\
    --cqf-model {cqf_model} \\
    --outdir {outdir} \\
    --train-steps {train_steps}
    """, language="bash")

if run_btn:
    if training_mode == "Precomputed Inference Artifacts":
        cmd = [
            sys.executable, str(training_dir / "iql_pipeline.py"),
            "--cqf-preds", str(Path(cqf_preds).resolve()),
            "--ranker-candidates", str(Path(ranker_candidates).resolve()),
            "--stress-metrics", str(Path(stress_metrics).resolve()),
            "--outdir", str(Path(outdir).resolve()),
            "--top-k", str(int(top_k)),
            "--size-bins", size_bins,
            "--train-steps", str(int(train_steps)),
            "--train-batch-size", str(int(batch_size)),
            "--expectile", str(float(expectile)),
            "--gamma", str(float(gamma)),
            "--seed", str(int(seed)),
        ]
        if no_train:
            cmd.append("--no-train")
        if group_top_n > 0:
            cmd.extend(["--group-top-n", str(int(group_top_n))])
    else:
        cmd = [
            sys.executable, str(training_dir / "iql_pipeline.py"),
            "--raw-data", str(Path(raw_data).resolve()),
            "--config", str(Path(config).resolve()),
            "--ranker-model", str(Path(ranker_model).resolve()),
            "--ranker-features", str(Path(ranker_features).resolve()),
            "--sharpe-edges", str(Path(sharpe_edges).resolve()),
            "--cqf-model", str(Path(cqf_model).resolve()),
            "--outdir", str(Path(outdir).resolve()),
            "--train-steps", str(int(train_steps)),
            "--train-batch-size", str(int(batch_size)),
            "--stress-mode", "mc",
        ]
        if include_future_targets:
            cmd.append("--include-future-targets")
    
    st.code(" ".join(cmd), language="bash")
    with st.status("Training IQL policy…", expanded=True) as status:
        try:
            proc = subprocess.run(cmd, cwd=str(base), check=True, capture_output=True, text=True)
            st.write(proc.stdout)
            if proc.stderr:
                st.write(proc.stderr)
            status.update(label="IQL training complete", state="complete")
            st.success("✅ IQL policy trained successfully!")
            st.info(f"Check {outdir} for: discrete_cql_policy.d3, policy_meta.json, decision_table.csv")
        except subprocess.CalledProcessError as e:
            st.error("IQL training failed")
            st.write(e.stdout)
            st.write(e.stderr)
