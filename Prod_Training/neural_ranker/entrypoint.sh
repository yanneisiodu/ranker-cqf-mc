#!/bin/bash
set -e

echo "=== Neural Ranker Training on Cloud Run ==="

# Print config
python -c "from cloud_config import CloudConfig; CloudConfig.print_config()"

# Check GPU
nvidia-smi || echo "WARNING: No GPU detected"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# Download pre-processed Parquet files from GCS
echo ""
echo "=== Preparing data ==="
python -u -c "
import os, subprocess
from cloud_config import CloudConfig

cache_dir = CloudConfig.DATA_DIR + '/cache'
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(CloudConfig.OUTPUT_DIR, exist_ok=True)

target = CloudConfig.TARGET_MODE or 'net_long_return'
horizon = CloudConfig.HORIZON_DAYS or 5

suffix_map = {
    ('net_long_return', 5): 'prepared',
    ('net_delta_hedged_return', 5): 'delta_hedged_5d',
    ('net_long_return', 2): 'raw_2d',
    ('net_delta_hedged_return', 2): 'delta_hedged_2d',
}
suffix = suffix_map.get((target, horizon), 'prepared')
print(f'Target={target}, Horizon={horizon}d -> suffix={suffix}')

all_years = list(dict.fromkeys(CloudConfig.train_years_list() + CloudConfig.val_years_list()))
for year in all_years:
    src = f'{CloudConfig.GCS_BUCKET}/parquet/year_{year}_{suffix}.parquet'
    dst = f'{cache_dir}/year_{year}_{suffix}.parquet'
    if os.path.exists(dst):
        print(f'{year}: already cached')
        continue
    print(f'{year}: downloading {suffix} parquet...')
    subprocess.run(['gcloud', 'storage', 'cp', src, dst], check=True)

with open(f'{cache_dir}/.suffix', 'w') as f: f.write(suffix)
print('Data ready.')
"

# Route based on MODE
echo ""
MODE=$(python -c "from cloud_config import CloudConfig; print(CloudConfig.MODE)")

if [ "$MODE" = "optuna" ]; then
    echo "=== Running Optuna Sweep ==="
    python -u optuna_neural_sweep.py
else
    echo "=== Starting Training ==="
    python -u -c "
import os, gc
import numpy as np
import pandas as pd
from cloud_config import CloudConfig
from train_neural_ranker import train_neural_ranker_from_datasets
from utils import load_config, select_feature_columns, compute_relevance_bins, apply_relevance_bins
from simulation_engine import ExecutionConfig, filter_tradeable_raw

config = load_config(CloudConfig.CONFIG_PATH)
if CloudConfig.TARGET_MODE:
    config['data']['target_mode'] = CloudConfig.TARGET_MODE
if CloudConfig.HORIZON_DAYS:
    config['data']['horizon_days'] = CloudConfig.HORIZON_DAYS
if CloudConfig.PATIENCE != 8:
    config.setdefault('neural_ranker', {})['patience'] = CloudConfig.PATIENCE
if CloudConfig.EPOCHS != 50:
    config.setdefault('neural_ranker', {})['epochs'] = CloudConfig.EPOCHS
cache_dir = CloudConfig.DATA_DIR + '/cache'

suffix = open(f'{cache_dir}/.suffix').read().strip()
print(f'Using parquet suffix: {suffix}')

first_year = CloudConfig.train_years_list()[0]
sample = pd.read_parquet(f'{cache_dir}/year_{first_year}_{suffix}.parquet')
feature_columns, _, _ = select_feature_columns(sample, config)
num_features = [c for c in feature_columns if c != 'type'] + ['type_numeric']

# Compute stats across all training years
print('Computing feature statistics...')
sums = None; sq_sums = None; total_n = 0; all_returns = []
for year in CloudConfig.train_years_list():
    df = pd.read_parquet(f'{cache_dir}/year_{year}_{suffix}.parquet')
    df['type_numeric'] = (df['type'].str.lower() == 'call').astype(np.float32)
    vals = df[num_features].fillna(0).values.astype(np.float64)
    if sums is None:
        sums = vals.sum(axis=0); sq_sums = (vals**2).sum(axis=0)
    else:
        sums += vals.sum(axis=0); sq_sums += (vals**2).sum(axis=0)
    total_n += len(vals)
    all_returns.append(df['target_return'].values)
    del df, vals; gc.collect()
    print(f'  {year}: stats computed')

train_mean = pd.Series(sums / total_n, index=num_features)
variance = (sq_sums / total_n - (sums / total_n)**2) * total_n / (total_n - 1)
train_std = pd.Series(np.sqrt(np.maximum(variance, 0)), index=num_features).replace(0, 1)
all_returns = np.concatenate(all_returns)
edges = compute_relevance_bins(pd.Series(all_returns), n_bins=5)
del all_returns, sums, sq_sums; gc.collect()
print(f'Stats computed over {total_n:,} rows')

# Build datasets — filter on RAW columns before normalization
exec_cfg = ExecutionConfig.from_config(config)
print('Building training dataset...')
train_groups = []
for year in CloudConfig.train_years_list():
    df = pd.read_parquet(f'{cache_dir}/year_{year}_{suffix}.parquet')
    df['type_numeric'] = (df['type'].str.lower() == 'call').astype(np.float32)
    df['target_relevance'] = apply_relevance_bins(df['target_return'], edges).astype(np.float32)
    df = filter_tradeable_raw(df, exec_cfg)
    df[num_features] = (df[num_features] - train_mean) / train_std
    df[num_features] = df[num_features].fillna(0.0)
    for date in sorted(df['date'].unique()):
        day = df[df['date'] == date]
        if len(day) < 2: continue
        train_groups.append((np.nan_to_num(day[num_features].values.astype(np.float32)),
                             day['target_relevance'].values.astype(np.float32)))
    del df; gc.collect()
    print(f'  {year}: {len(train_groups)} total days')

print('Building validation dataset...')
val_groups = []
for year in CloudConfig.val_years_list():
    df = pd.read_parquet(f'{cache_dir}/year_{year}_{suffix}.parquet')
    df['type_numeric'] = (df['type'].str.lower() == 'call').astype(np.float32)
    df['target_relevance'] = apply_relevance_bins(df['target_return'], edges).astype(np.float32)
    df = filter_tradeable_raw(df, exec_cfg)
    df[num_features] = (df[num_features] - train_mean) / train_std
    df[num_features] = df[num_features].fillna(0.0)
    for date in sorted(df['date'].unique()):
        day = df[df['date'] == date]
        if len(day) < 2: continue
        val_groups.append((np.nan_to_num(day[num_features].values.astype(np.float32)),
                           day['target_relevance'].values.astype(np.float32)))
    del df; gc.collect()
    print(f'  {year}: {len(val_groups)} total days')

print(f'Train: {len(train_groups)} days, Val: {len(val_groups)} days')

result = train_neural_ranker_from_datasets(
    train_groups=train_groups,
    val_groups=val_groups,
    num_features=num_features,
    train_mean=train_mean,
    train_std=train_std,
    relevance_edges=edges,
    config=config,
    output_dir=CloudConfig.OUTPUT_DIR,
)
print(f'Training complete: {result}')
"
fi

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
