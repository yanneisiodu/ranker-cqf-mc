import streamlit as st
from pathlib import Path
import subprocess
import sys

st.set_page_config(page_title="Walkforward Eval", page_icon="🧪", layout="wide")
st.title("🧪 Walkforward Evaluation")
st.info("💡 Uses unified walkforward_simulator.py with backtest or leak-free modes")

base = Path(__file__).resolve().parents[2]
training_dir = base / "Training"

with st.sidebar:
    st.header("Inputs")
    decision_table = st.text_input("Decision Table CSV", str(base / "iql_out/2023_with_targets/decision_table.csv"))
    policy = st.text_input("Policy .d3", str(base / "iql_out/2023_training_2022models/discrete_cql_policy.d3"))
    meta = st.text_input("policy_meta.json", str(base / "iql_out/2023_training_2022models/policy_meta.json"))
    outdir = st.text_input("Output Dir", str(base / "results/walkforward_eval"))
    
    st.subheader("Configuration")
    mode = st.selectbox("Mode", ["backtest", "leakfree"], index=0, 
                       help="backtest: uses actual targets | leakfree: simulated PnL")
    config = st.selectbox("Config", ["trial74", "trial62"], index=0,
                         help="trial74: 86.6% WR, 2.5× leverage | trial62: 83.9% WR, dynamic sizing")
    initial_capital = st.number_input("Initial Capital", min_value=1000.0, max_value=1e7, value=10000.0, step=1000.0)
    
    run_btn = st.button("Run Walkforward", type="primary")

st.markdown("Uses `Training/walkforward_simulator.py` with unified backtest/leak-free modes.")

if run_btn:
    cmd = [
        sys.executable, str(training_dir / "walkforward_simulator.py"),
        "--decision-table", str(Path(decision_table).resolve()),
        "--policy", str(Path(policy).resolve()),
        "--meta", str(Path(meta).resolve()),
        "--outdir", str(Path(outdir).resolve()),
        "--mode", mode,
        "--config", config,
        "--initial-capital", str(float(initial_capital)),
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


