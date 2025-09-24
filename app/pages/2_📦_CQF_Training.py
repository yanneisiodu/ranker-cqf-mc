import streamlit as st
from pathlib import Path
import subprocess
import sys

st.set_page_config(page_title="CQF Training", page_icon="📦", layout="wide")
st.title("📦 CQF Training (Step 8)")

base = Path(__file__).resolve().parents[2]
training_dir = base / "Training"

with st.sidebar:
    st.header("Inputs")
    train_csv = st.text_input("Train CSV", str(base / "data/year_2023_data.csv"))
    eval_csv = st.text_input("Eval CSV", str(base / "data/year_2024_data.csv"))
    config_yaml = st.text_input("Config YAML", str(base / "config.yaml"))
    out_path = st.text_input("Output Model", str(base / "model_output/optimal_cqf_step8.joblib"))
    horizon = st.number_input("Horizon (days)", min_value=1, max_value=30, value=5)
    run_btn = st.button("Train CQF", type="primary")

st.markdown("Uses `Training/prod_cqf.py`; saves model + predictions to `model_output/`.")

if run_btn:
    cmd = [
        sys.executable, str(training_dir / "prod_cqf.py"),
        "--train-data", str(Path(train_csv).resolve()),
        "--eval-data", str(Path(eval_csv).resolve()),
        "--config", str(Path(config_yaml).resolve()),
        "--output", str(Path(out_path).resolve()),
        "--horizon", str(int(horizon)),
    ]
    st.code(" ".join(cmd), language="bash")
    with st.status("Training CQF…", expanded=True) as status:
        try:
            proc = subprocess.run(cmd, cwd=str(base), check=True, capture_output=True, text=True)
            st.write(proc.stdout)
            if proc.stderr:
                st.write(proc.stderr)
            status.update(label="Training complete", state="complete")
        except subprocess.CalledProcessError as e:
            st.error("CQF training failed")
            st.write(e.stdout)
            st.write(e.stderr)


