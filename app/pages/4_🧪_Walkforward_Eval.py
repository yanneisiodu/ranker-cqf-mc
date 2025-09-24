import streamlit as st
from pathlib import Path
import subprocess
import sys

st.set_page_config(page_title="Walkforward Eval", page_icon="🧪", layout="wide")
st.title("🧪 Basic Walkforward Evaluation (CQL policy)")
st.warning("⚠️ This is the BASIC walkforward. For optimal performance, use '🎯 Final Optimal Walkforward' page.")

base = Path(__file__).resolve().parents[2]
training_dir = base / "Training"

with st.sidebar:
    st.header("Inputs")
    decision_table = st.text_input("Decision Table CSV", str(base / "final_iql_training_2023/decision_table.csv"))
    policy = st.text_input("Policy .d3", str(base / "final_iql_training_2023/discrete_cql_policy.d3"))
    meta = st.text_input("policy_meta.json", str(base / "final_iql_training_2023/policy_meta.json"))
    outdir = st.text_input("Output Dir", str(base / "walkforward_results"))
    initial_capital = st.number_input("Initial Capital", min_value=1000.0, max_value=1e7, value=10000.0, step=1000.0)
    risk_pct = st.number_input("Risk % per trade", min_value=0.0, max_value=0.1, value=0.005)
    run_btn = st.button("Run Walkforward", type="primary")

st.markdown("Uses `Training/walkforward_simulation.py`. Note: test version relaxes constraints for experimentation.")

if run_btn:
    cmd = [
        sys.executable, str(training_dir / "walkforward_simulation.py"),
        "--decision-table", str(Path(decision_table).resolve()),
        "--policy", str(Path(policy).resolve()),
        "--meta", str(Path(meta).resolve()),
        "--outdir", str(Path(outdir).resolve()),
        "--initial-capital", str(float(initial_capital)),
        "--risk-pct", str(float(risk_pct)),
    ]
    st.code(" ".join(cmd), language="bash")
    with st.status("Running walkforward…", expanded=True) as status:
        try:
            proc = subprocess.run(cmd, cwd=str(base), check=True, capture_output=True, text=True)
            st.write(proc.stdout)
            if proc.stderr:
                st.write(proc.stderr)
            status.update(label="Walkforward complete", state="complete")
        except subprocess.CalledProcessError as e:
            st.error("Walkforward failed")
            st.write(e.stdout)
            st.write(e.stderr)


