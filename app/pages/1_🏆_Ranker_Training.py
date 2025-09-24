import streamlit as st
from pathlib import Path
import subprocess
import sys

st.set_page_config(page_title="Ranker Training", page_icon="🏆", layout="wide")
st.title("🏆 Ranker Training")

base = Path(__file__).resolve().parents[2]
training_dir = base / "Training"

with st.sidebar:
    st.header("Inputs")
    start_year = st.number_input("Start Year", min_value=2018, max_value=2025, value=2019)
    end_year = st.number_input("End Year", min_value=2018, max_value=2025, value=2023)
    trials = st.number_input("Optuna Trials (0=fixed params)", min_value=0, max_value=1000, value=0)
    config_path = st.text_input("Config YAML", str(base / "config.yaml"))
    run_btn = st.button("Train Ranker", type="primary")

st.markdown("Training uses `Training/prod_train_ranker.py` and writes artifacts to `model_output/`.")

log = st.empty()
if run_btn:
    cmd = [
        sys.executable, str(training_dir / "prod_train_ranker.py"),
        "--start-year", str(int(start_year)),
        "--end-year", str(int(end_year)),
        "--trials", str(int(trials)),
        "--config", str(Path(config_path).resolve()),
    ]
    st.code(" ".join(cmd), language="bash")
    with st.status("Training ranker…", expanded=True) as status:
        try:
            proc = subprocess.run(cmd, cwd=str(base), check=True, capture_output=True, text=True)
            st.write(proc.stdout)
            if proc.stderr:
                st.write(proc.stderr)
            status.update(label="Training complete", state="complete")
        except subprocess.CalledProcessError as e:
            st.error("Training failed")
            st.write(e.stdout)
            st.write(e.stderr)


