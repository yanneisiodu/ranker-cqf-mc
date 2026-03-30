#!/bin/bash
set -e

echo "=== Neural Ranker Training on Cloud Run ==="

# Print config
python -c "from cloud_config import CloudConfig; CloudConfig.print_config()"

# Check GPU
nvidia-smi || echo "WARNING: No GPU detected"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# Download data — use pre-processed parquets if default target/horizon, else download CSVs and reprocess
echo ""
echo "=== Preparing data ==="
python -u -c "
import os, subprocess, time, gc
from cloud_config import CloudConfig

cache_dir = CloudConfig.DATA_DIR + '/cache'
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(CloudConfig.OUTPUT_DIR, exist_ok=True)

# Map target/horizon to pre-processed parquet suffix
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

all_years = CloudConfig.train_years_list() + CloudConfig.val_years_list()
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

if [ "$MODE" = "sweep" ]; then
    echo "=== Running Grid Sweep ==="
    python -u sweep.py
elif [ "$MODE" = "optuna" ]; then
    echo "=== Running Optuna Sweep ==="
    python -u optuna_neural_sweep.py
elif [ "$MODE" = "pretrain" ]; then
    echo "=== Pretraining (Masked Reconstruction) ==="
    python -u -c "
import os, gc
import numpy as np
import pandas as pd
from cloud_config import CloudConfig
from pretrain import pretrain_from_groups
from utils import load_config, select_feature_columns, compute_relevance_bins, apply_relevance_bins

config = load_config(CloudConfig.CONFIG_PATH)
cache_dir = CloudConfig.DATA_DIR + '/cache'

# Compute stats
first_year = CloudConfig.train_years_list()[0]
sample = pd.read_parquet(f'{cache_dir}/year_{first_year}_prepared.parquet')
feature_columns, _, _ = select_feature_columns(sample, config)
num_features = [c for c in feature_columns if c != 'type'] + ['type_numeric']
del sample; gc.collect()

sums = None; sq_sums = None; total_n = 0; all_returns = []
for year in CloudConfig.train_years_list():
    df = pd.read_parquet(f'{cache_dir}/year_{year}_{suffix}.parquet')
    df['type_numeric'] = (df['type'].str.lower() == 'call').astype(np.float32)
    vals = df[num_features].fillna(0).values.astype(np.float64)
    if sums is None: sums = vals.sum(axis=0); sq_sums = (vals**2).sum(axis=0)
    else: sums += vals.sum(axis=0); sq_sums += (vals**2).sum(axis=0)
    total_n += len(vals); all_returns.append(df['target_return'].values)
    del df, vals; gc.collect()

train_mean = pd.Series(sums / total_n, index=num_features)
train_std = pd.Series(np.sqrt(sq_sums/total_n - (sums/total_n)**2), index=num_features).replace(0, 1)
edges = compute_relevance_bins(pd.Series(np.concatenate(all_returns)), n_bins=5)
del all_returns, sums, sq_sums; gc.collect()

# Build all groups (train + val combined for pretraining — no labels needed)
all_groups = []
all_years = CloudConfig.train_years_list() + CloudConfig.val_years_list()
for year in all_years:
    df = pd.read_parquet(f'{cache_dir}/year_{year}_{suffix}.parquet')
    df['type_numeric'] = (df['type'].str.lower() == 'call').astype(np.float32)
    df['target_relevance'] = apply_relevance_bins(df['target_return'], edges).astype(np.float32)
    if 'relative_spread' in df.columns: df = df[df['relative_spread'] <= 0.50]
    df[num_features] = (df[num_features] - train_mean) / train_std
    df[num_features] = df[num_features].fillna(0.0)
    for date in sorted(df['date'].unique()):
        day = df[df['date'] == date]
        if len(day) < 2: continue
        all_groups.append((np.nan_to_num(day[num_features].values.astype(np.float32)),
                           day['target_relevance'].values.astype(np.float32)))
    del df; gc.collect()
    print(f'  {year}: {len(all_groups)} total days')

# 90/10 split
split = int(len(all_groups) * 0.9)
train_groups = all_groups[:split]
val_groups = all_groups[split:]
print(f'Pretrain: {len(train_groups)} train days, {len(val_groups)} val days')

result = pretrain_from_groups(train_groups, val_groups, num_features, config, CloudConfig.OUTPUT_DIR)
print(f'Pretraining complete: {result}')
"
else
    echo "=== Starting Training ==="
    python -u -c "
import os, gc
import numpy as np
import pandas as pd
from cloud_config import CloudConfig
from train_neural_ranker import DailyChainDataset, train_neural_ranker_from_datasets
from utils import load_config, select_feature_columns, compute_relevance_bins, apply_relevance_bins

config = load_config(CloudConfig.CONFIG_PATH)
if CloudConfig.TARGET_MODE:
    config['data']['target_mode'] = CloudConfig.TARGET_MODE
if CloudConfig.HORIZON_DAYS:
    config['data']['horizon_days'] = CloudConfig.HORIZON_DAYS
cache_dir = CloudConfig.DATA_DIR + '/cache'

# Read parquet suffix
suffix = open(f'{cache_dir}/.suffix').read().strip()
print(f'Using parquet suffix: {suffix}')

# First pass: compute feature stats and relevance bins from first train year
first_year = CloudConfig.train_years_list()[0]
sample = pd.read_parquet(f'{cache_dir}/year_{first_year}_{suffix}.parquet')
feature_columns, _, _ = select_feature_columns(sample, config)
num_features = [c for c in feature_columns if c != 'type'] + ['type_numeric']

# Compute stats across all training years without loading all at once
print('Computing feature statistics...')
sums = None
sq_sums = None
total_n = 0
all_returns = []
for year in CloudConfig.train_years_list():
    df = pd.read_parquet(f'{cache_dir}/year_{year}_{suffix}.parquet')
    df['type_numeric'] = (df['type'].str.lower() == 'call').astype(np.float32)
    vals = df[num_features].fillna(0).values.astype(np.float64)
    if sums is None:
        sums = vals.sum(axis=0)
        sq_sums = (vals ** 2).sum(axis=0)
    else:
        sums += vals.sum(axis=0)
        sq_sums += (vals ** 2).sum(axis=0)
    total_n += len(vals)
    all_returns.append(df['target_return'].values)
    del df, vals; gc.collect()
    print(f'  {year}: stats computed')

# Use ddof=1 (sample std) to match pandas .std() in from_frames path
train_mean = pd.Series(sums / total_n, index=num_features)
variance = (sq_sums / total_n - (sums / total_n) ** 2) * total_n / (total_n - 1)  # Bessel's correction
train_std = pd.Series(np.sqrt(np.maximum(variance, 0)), index=num_features).replace(0, 1)
all_returns = np.concatenate(all_returns)
edges = compute_relevance_bins(pd.Series(all_returns), n_bins=5)
del all_returns, sums, sq_sums; gc.collect()
print(f'Stats computed over {total_n:,} rows')

# Build datasets year by year
print('Building training dataset...')
train_groups = []
for year in CloudConfig.train_years_list():
    df = pd.read_parquet(f'{cache_dir}/year_{year}_{suffix}.parquet')
    df['type_numeric'] = (df['type'].str.lower() == 'call').astype(np.float32)
    df['target_relevance'] = apply_relevance_bins(df['target_return'], edges).astype(np.float32)
    # Normalize FIRST, then filter — matches from_frames path
    df[num_features] = (df[num_features] - train_mean) / train_std
    df[num_features] = df[num_features].fillna(0.0)
    if 'relative_spread' in df.columns:
        df = df[df['relative_spread'] <= 0.50]
    for date in sorted(df['date'].unique()):
        day = df[df['date'] == date]
        if len(day) < 2:
            continue
        feats = np.nan_to_num(day[num_features].values.astype(np.float32))
        rels = day['target_relevance'].values.astype(np.float32)
        train_groups.append((feats, rels))
    del df; gc.collect()
    print(f'  {year}: {len(train_groups)} total days')

print('Building validation dataset...')
val_groups = []
for year in CloudConfig.val_years_list():
    df = pd.read_parquet(f'{cache_dir}/year_{year}_{suffix}.parquet')
    df['type_numeric'] = (df['type'].str.lower() == 'call').astype(np.float32)
    df['target_relevance'] = apply_relevance_bins(df['target_return'], edges).astype(np.float32)
    df[num_features] = (df[num_features] - train_mean) / train_std
    df[num_features] = df[num_features].fillna(0.0)
    if 'relative_spread' in df.columns:
        df = df[df['relative_spread'] <= 0.50]
    for date in sorted(df['date'].unique()):
        day = df[df['date'] == date]
        if len(day) < 2:
            continue
        feats = np.nan_to_num(day[num_features].values.astype(np.float32))
        rels = day['target_relevance'].values.astype(np.float32)
        val_groups.append((feats, rels))
    del df; gc.collect()
    print(f'  {year}: {len(val_groups)} total days')

print(f'Train: {len(train_groups)} days, Val: {len(val_groups)} days')

# Download pretrained weights if specified
pretrained_path = None
if CloudConfig.PRETRAINED_PATH:
    import subprocess
    local_pt = f'{CloudConfig.OUTPUT_DIR}/pretrained_encoder.pt'
    if CloudConfig.PRETRAINED_PATH.startswith('gs://'):
        print(f'Downloading pretrained weights from {CloudConfig.PRETRAINED_PATH}...')
        subprocess.run(['gcloud', 'storage', 'cp', CloudConfig.PRETRAINED_PATH, local_pt], check=True)
        pretrained_path = local_pt
    else:
        pretrained_path = CloudConfig.PRETRAINED_PATH
    print(f'Using pretrained weights: {pretrained_path}')

result = train_neural_ranker_from_datasets(
    train_groups=train_groups,
    val_groups=val_groups,
    num_features=num_features,
    train_mean=train_mean,
    train_std=train_std,
    relevance_edges=edges,
    config=config,
    output_dir=CloudConfig.OUTPUT_DIR,
    pretrained_path=pretrained_path,
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
