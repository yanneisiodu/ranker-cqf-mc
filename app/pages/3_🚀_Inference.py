import streamlit as st
from pathlib import Path
import subprocess
import sys

st.set_page_config(page_title="Inference", page_icon="🚀", layout="wide")
st.title("🚀 Inference Pipeline")

base = Path(__file__).resolve().parents[2]

with st.sidebar:
    st.header("Inputs")
    raw_data = st.text_input("Raw Data CSV", str(base / "data/year_2024_data.csv"))
    config_yaml = st.text_input("Config YAML", str(base / "config.yaml"))
    ranker_model = st.text_input("Ranker Model", str(base / "model_output/xgboost_ranker_2022_2024_optuna_tuned_20250829_182605.joblib"))
    ranker_features = st.text_input("Ranker Feature List", str(base / "model_output/xgb_feature_names_2022_2024_20250829_182605.pkl"))
    sharpe_edges = st.text_input("Sharpe Edges", str(base / "model_output/sharpe_qcut_edges_2022_2024_20250829_182605.pkl"))
    cqf_model = st.text_input("CQF Artifact", str(base / "model_output/optimal_cqf_step8.joblib"))
    outdir = st.text_input("Output Dir", str(base / "inference_output"))
    stress_mode = st.selectbox("Stress Mode", ["mc", "shadow", "llm"], index=0)
    llm_engine = st.selectbox("LLM Engine", ["basic", "agent"], index=0)
    llm_log = st.checkbox("Log LLM scenarios", value=False)
    min_prob_profit = st.number_input("Min Prob Profit", min_value=0.0, max_value=1.0, value=0.45)
    max_downside_var = st.number_input("Max Downside VaR", min_value=0.0, max_value=1.0, value=0.15)
    top_n = st.number_input("Top N", min_value=10, max_value=10000, value=1000)
    run_btn = st.button("Run Inference", type="primary")

st.markdown("Runs `inference/run_inference.py` end-to-end.")

if run_btn:
    cmd = [
        sys.executable, "-m", "inference.run_inference",
        "--raw-data", str(Path(raw_data).resolve()),
        "--config", str(Path(config_yaml).resolve()),
        "--ranker-model", str(Path(ranker_model).resolve()),
        "--ranker-features", str(Path(ranker_features).resolve()),
        "--sharpe-edges", str(Path(sharpe_edges).resolve()),
        "--cqf-model", str(Path(cqf_model).resolve()),
        "--top-n", str(int(top_n)),
        "--output-dir", str(Path(outdir).resolve()),
        "--stress-mode", stress_mode,
        "--llm-engine", llm_engine,
        "--min-prob-profit", str(float(min_prob_profit)),
        "--max-downside-var", str(float(max_downside_var)),
    ]
    if llm_log:
        cmd.append("--llm-log-scenarios")
    st.code(" ".join(cmd), language="bash")
    with st.status("Running inference…", expanded=True) as status:
        try:
            proc = subprocess.run(cmd, cwd=str(base), check=True, capture_output=True, text=True)
            st.write(proc.stdout)
            if proc.stderr:
                st.write(proc.stderr)
            status.update(label="Inference complete", state="complete")
        except subprocess.CalledProcessError as e:
            st.error("Inference failed")
            st.write(e.stdout)
            st.write(e.stderr)


