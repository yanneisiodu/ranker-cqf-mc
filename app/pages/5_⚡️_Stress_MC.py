import streamlit as st
from pathlib import Path
import subprocess
import sys

st.set_page_config(page_title="Stress MC", page_icon="⚡️", layout="wide")
st.title("⚡️ Stress Monte Carlo")

base = Path(__file__).resolve().parents[2]
training_dir = base / "Training"

with st.sidebar:
    st.header("Inputs")
    contracts_csv = st.text_input("Contracts (CQF outputs)", str(base / "model_output/optimal_cqf_step8_predictions.csv"))
    spy_history = st.text_input("SPY history (optional)", "")
    source_data_dir = st.text_input("Source Data Dir", "data_combined")
    n_paths = st.number_input("Paths", min_value=1000, max_value=100000, value=5000, step=1000)
    risk_aversion = st.number_input("Risk Aversion", min_value=0.0, max_value=2.0, value=0.5)
    min_prob_profit = st.number_input("Min Prob Profit", min_value=0.0, max_value=1.0, value=0.45)
    max_downside_var = st.number_input("Max Downside VaR", min_value=0.0, max_value=1.0, value=0.15)
    n_jobs = st.number_input("n_jobs", min_value=1, max_value=16, value=1)
    out = st.text_input("Output CSV", str(base / "results/stress_ranked.csv"))
    top_k = st.number_input("Top K save", min_value=1, max_value=10000, value=50)
    run_btn = st.button("Run Stress MC", type="primary")

st.markdown("Runs `Training/prod_stress_mc.py` standalone.")

if run_btn:
    cmd = [
        sys.executable, str(training_dir / "prod_stress_mc.py"),
        "--contracts", str(Path(contracts_csv).resolve()),
        "--source-data-dir", source_data_dir,
        "--n-paths", str(int(n_paths)),
        "--risk-aversion", str(float(risk_aversion)),
        "--min-prob-profit", str(float(min_prob_profit)),
        "--max-downside-var", str(float(max_downside_var)),
        "--n-jobs", str(int(n_jobs)),
        "--out", str(Path(out).resolve()),
        "--top-k", str(int(top_k)),
    ]
    if spy_history.strip():
        cmd += ["--spy-history", str(Path(spy_history).resolve())]
    st.code(" ".join(cmd), language="bash")
    with st.status("Running Stress MC…", expanded=True) as status:
        try:
            proc = subprocess.run(cmd, cwd=str(base), check=True, capture_output=True, text=True)
            st.write(proc.stdout)
            if proc.stderr:
                st.write(proc.stderr)
            status.update(label="Stress MC complete", state="complete")
        except subprocess.CalledProcessError as e:
            st.error("Stress MC failed")
            st.write(e.stdout)
            st.write(e.stderr)


