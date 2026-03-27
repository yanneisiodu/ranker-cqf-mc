#!/bin/bash
set -e

echo "=== Neural Ranker Training on Cloud Run ==="

# Print config
python -c "from cloud_config import CloudConfig; CloudConfig.print_config()"

# Check GPU
nvidia-smi || echo "WARNING: No GPU detected"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# Download data and pre-process into Parquet cache
echo ""
echo "=== Downloading data and preparing Parquet cache ==="
python -u -c "
import os, subprocess, time
import pandas as pd
from cloud_config import CloudConfig

os.makedirs(CloudConfig.DATA_DIR, exist_ok=True)
os.makedirs(CloudConfig.OUTPUT_DIR, exist_ok=True)
cache_dir = CloudConfig.DATA_DIR + '/cache'
os.makedirs(cache_dir, exist_ok=True)

from utils import load_config, prepare_model_frame

config = load_config(CloudConfig.CONFIG_PATH)
all_years = CloudConfig.train_years_list() + CloudConfig.val_years_list()

for year in all_years:
    parquet_path = f'{cache_dir}/year_{year}_prepared.parquet'
    if os.path.exists(parquet_path):
        print(f'{year}: using cached parquet')
        continue

    # Download CSV
    src = CloudConfig.gcs_data_uri(year)
    print(f'{year}: downloading from GCS...')
    subprocess.run(['gcloud', 'storage', 'cp', src, CloudConfig.DATA_DIR + '/'], check=True)

    # Process and cache as Parquet
    csv_path = f'{CloudConfig.DATA_DIR}/year_{year}_data.csv'
    print(f'{year}: processing features + targets...')
    t0 = time.time()
    frame = prepare_model_frame([csv_path], config, include_targets=True)
    frame.to_parquet(parquet_path, index=False)
    elapsed = time.time() - t0
    print(f'{year}: {len(frame):,} rows cached as parquet ({elapsed:.0f}s)')

    # Remove CSV to save disk space
    os.remove(csv_path)

print('All data prepared.')
"

# Run training from Parquet cache
echo ""
echo "=== Starting training ==="
python -u -c "
import os, glob
import pandas as pd
from cloud_config import CloudConfig
from train_neural_ranker import train_neural_ranker_from_frames
from utils import load_config

config = load_config(CloudConfig.CONFIG_PATH)
cache_dir = CloudConfig.DATA_DIR + '/cache'

# Load pre-processed Parquet files
train_frames = []
for year in CloudConfig.train_years_list():
    path = f'{cache_dir}/year_{year}_prepared.parquet'
    print(f'Loading {path}...')
    train_frames.append(pd.read_parquet(path))
train_frame = pd.concat(train_frames, ignore_index=True)
print(f'Train: {len(train_frame):,} rows')

val_frames = []
for year in CloudConfig.val_years_list():
    path = f'{cache_dir}/year_{year}_prepared.parquet'
    print(f'Loading {path}...')
    val_frames.append(pd.read_parquet(path))
val_frame = pd.concat(val_frames, ignore_index=True)
print(f'Val: {len(val_frame):,} rows')

result = train_neural_ranker_from_frames(
    train_frame=train_frame,
    val_frame=val_frame,
    config=config,
    output_dir=CloudConfig.OUTPUT_DIR,
    nrows=CloudConfig.NROWS,
)
print(f'Training complete: {result}')
"

# Upload artifacts to GCS
echo ""
echo "=== Uploading artifacts to GCS ==="
python -c "
import subprocess, glob
from cloud_config import CloudConfig

for f in glob.glob(f'{CloudConfig.OUTPUT_DIR}/*'):
    name = f.split('/')[-1]
    dst = CloudConfig.gcs_artifact_uri(name)
    print(f'Uploading {name} -> {dst}')
    subprocess.run(['gcloud', 'storage', 'cp', f, dst], check=True)

print('All artifacts uploaded.')
"

echo ""
echo "=== Done ==="
