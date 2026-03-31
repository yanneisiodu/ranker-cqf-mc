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

fisher_years = [y.strip() for y in CloudConfig.FISHER_YEARS.split(',') if y.strip()] if CloudConfig.FISHER_YEARS else []
all_years = list(dict.fromkeys(CloudConfig.train_years_list() + CloudConfig.val_years_list() + fisher_years))
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
elif [ "$MODE" = "ensemble" ]; then
    echo "=== Training Expert Ensemble ==="
    python -u -c "
import os, gc
import numpy as np
import pandas as pd
from cloud_config import CloudConfig
from train_ensemble import build_groups_from_frame, select_window, select_stress_replay, train_expert, train_gate
from regime_ensemble import recency_weight, ExpertEnsemble
from neural_ranker import NeuralRankerConfig, ChainTransformer, get_device, ndcg_at_k
from train_neural_ranker import PrebuiltDataset, collate_chains, evaluate
from torch.utils.data import DataLoader
from dataclasses import asdict
from utils import load_config, select_feature_columns, compute_relevance_bins, apply_relevance_bins, save_json
import torch
from pathlib import Path

config = load_config(CloudConfig.CONFIG_PATH)
if CloudConfig.PATIENCE != 8:
    config.setdefault('neural_ranker', {})['patience'] = CloudConfig.PATIENCE
if CloudConfig.EPOCHS != 50:
    config.setdefault('neural_ranker', {})['epochs'] = CloudConfig.EPOCHS
cache_dir = CloudConfig.DATA_DIR + '/cache'
suffix = open(f'{cache_dir}/.suffix').read().strip()
device = get_device()
nr_config = NeuralRankerConfig.from_config(config)

# Get feature columns
first_year = CloudConfig.train_years_list()[0]
sample = pd.read_parquet(f'{cache_dir}/year_{first_year}_{suffix}.parquet')
feature_columns, _, _ = select_feature_columns(sample, config)
num_features = [c for c in feature_columns if c != 'type'] + ['type_numeric']
del sample; gc.collect()

# Compute stats
print('Computing feature statistics...')
sums = None; sq_sums = None; total_n = 0; all_returns = []
all_years = CloudConfig.train_years_list() + CloudConfig.val_years_list()
for year in all_years:
    df = pd.read_parquet(f'{cache_dir}/year_{year}_{suffix}.parquet')
    df['type_numeric'] = (df['type'].str.lower() == 'call').astype(np.float32)
    vals = df[num_features].fillna(0).values.astype(np.float64)
    if sums is None: sums = vals.sum(axis=0); sq_sums = (vals**2).sum(axis=0)
    else: sums += vals.sum(axis=0); sq_sums += (vals**2).sum(axis=0)
    total_n += len(vals); all_returns.append(df['target_return'].values)
    del df, vals; gc.collect()

train_mean = pd.Series(sums / total_n, index=num_features)
variance = (sq_sums / total_n - (sums / total_n)**2) * total_n / (total_n - 1)
train_std = pd.Series(np.sqrt(np.maximum(variance, 0)), index=num_features).replace(0, 1)
edges = compute_relevance_bins(pd.Series(np.concatenate(all_returns)), n_bins=5)
del all_returns, sums, sq_sums; gc.collect()

# Build all groups
from regime_ensemble import assign_regime_bucket, is_stress_day
all_groups = []; all_dates = []; all_meta = []
for year in CloudConfig.train_years_list():
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
        all_dates.append(date)
        all_meta.append({'date': date, 'regime_bucket': assign_regime_bucket(day),
                         'is_stress': is_stress_day(day),
                         'spy_momentum': day['spy_momentum'].iloc[0] if 'spy_momentum' in day.columns else 0,
                         'vix': day['vix_d_close'].iloc[0] if 'vix_d_close' in day.columns else 20,
                         'realized_vol_20d': day['realized_vol_20d'].iloc[0] if 'realized_vol_20d' in day.columns else 0,
                         'vrp_20d': day['vrp_20d'].iloc[0] if 'vrp_20d' in day.columns else 0,
                         'spy_rsi': day['spy_d_rsi'].iloc[0] if 'spy_d_rsi' in day.columns else 50,
                         'n_options': len(day)})
    del df; gc.collect()
    print(f'  {year}: {len(all_groups)} total days')

val_groups = []; val_meta = []
for year in CloudConfig.val_years_list():
    df = pd.read_parquet(f'{cache_dir}/year_{year}_{suffix}.parquet')
    df['type_numeric'] = (df['type'].str.lower() == 'call').astype(np.float32)
    df['target_relevance'] = apply_relevance_bins(df['target_return'], edges).astype(np.float32)
    if 'relative_spread' in df.columns: df = df[df['relative_spread'] <= 0.50]
    df[num_features] = (df[num_features] - train_mean) / train_std
    df[num_features] = df[num_features].fillna(0.0)
    for date in sorted(df['date'].unique()):
        day = df[df['date'] == date]
        if len(day) < 2: continue
        val_groups.append((np.nan_to_num(day[num_features].values.astype(np.float32)),
                           day['target_relevance'].values.astype(np.float32)))
        val_meta.append({'spy_momentum': day['spy_momentum'].iloc[0] if 'spy_momentum' in day.columns else 0,
                         'vix': day['vix_d_close'].iloc[0] if 'vix_d_close' in day.columns else 20,
                         'realized_vol_20d': day['realized_vol_20d'].iloc[0] if 'realized_vol_20d' in day.columns else 0,
                         'vrp_20d': day['vrp_20d'].iloc[0] if 'vrp_20d' in day.columns else 0,
                         'spy_rsi': day['spy_d_rsi'].iloc[0] if 'spy_d_rsi' in day.columns else 50,
                         'regime_bucket': assign_regime_bucket(day), 'n_options': len(day)})
    del df; gc.collect()
    print(f'  Val: {len(val_groups)} days')

end_date = max(all_dates)
actual_config = NeuralRankerConfig(input_dim=len(num_features), embed_dim=nr_config.embed_dim,
    n_heads=nr_config.n_heads, n_layers=nr_config.n_layers, dropout=nr_config.dropout,
    mlp_hidden=nr_config.mlp_hidden, learning_rate=nr_config.learning_rate,
    weight_decay=nr_config.weight_decay, warmup_epochs=nr_config.warmup_epochs, epochs=CloudConfig.EPOCHS)

out = Path(CloudConfig.OUTPUT_DIR)
out.mkdir(parents=True, exist_ok=True)
epochs = CloudConfig.EPOCHS

# Load base model for warm-starting
base_state = None
if CloudConfig.EWC_BASE_ARTIFACT:
    import subprocess as _sp
    base_local = f'{CloudConfig.OUTPUT_DIR}/base_artifact.pt'
    print(f'Downloading base model from {CloudConfig.EWC_BASE_ARTIFACT}...')
    _sp.run(['gcloud', 'storage', 'cp', CloudConfig.EWC_BASE_ARTIFACT, base_local], check=True)
    base_art = torch.load(base_local, map_location='cpu', weights_only=False)
    base_state = base_art['model_state_dict']
    print('Base model loaded for warm-starting.')

# Train experts
recent_g, recent_w = select_window(all_groups, all_dates, all_meta, end_date, CloudConfig.RECENT_MONTHS)
print(f'E_recent: {len(recent_g)} days')
rs, rn, re = train_expert('E_recent', recent_g, recent_w, val_groups, actual_config, device, epochs=epochs, base_state=base_state)
torch.save({'state_dict': rs, 'config': asdict(actual_config), 'name': 'recent', 'ndcg': rn}, out/'expert_recent.pt')

core_g, core_w = select_window(all_groups, all_dates, all_meta, end_date, CloudConfig.CORE_MONTHS)
print(f'E_core: {len(core_g)} days')
cs, cn, ce = train_expert('E_core', core_g, core_w, val_groups, actual_config, device, epochs=epochs, base_state=base_state)
torch.save({'state_dict': cs, 'config': asdict(actual_config), 'name': 'core', 'ndcg': cn}, out/'expert_core.pt')

stress_g, stress_w = select_stress_replay(all_groups, all_dates, all_meta, end_date,
    recent_months=CloudConfig.CORE_MONTHS, replay_pct=CloudConfig.STRESS_REPLAY_PCT)
ss, sn, se = train_expert('E_stress', stress_g, stress_w, val_groups, actual_config, device, epochs=epochs, base_state=base_state)
torch.save({'state_dict': ss, 'config': asdict(actual_config), 'name': 'stress', 'ndcg': sn}, out/'expert_stress.pt')

# Gate
print('Training gate...')
expert_ndcgs = {}
for name, state in [('recent', rs), ('core', cs), ('stress', ss)]:
    model = ChainTransformer(actual_config).to(device); model.load_state_dict(state); model.eval()
    ndcgs = []
    for feats, rels in val_groups:
        x = torch.from_numpy(feats).unsqueeze(0).to(device)
        pm = torch.zeros(1, feats.shape[0], dtype=torch.bool, device=device)
        with torch.no_grad(): scores = model(x, padding_mask=pm).squeeze(0).cpu().numpy()
        ndcgs.append(ndcg_at_k(scores, rels, k=20))
    expert_ndcgs[name] = np.array(ndcgs)
    del model; gc.collect()

gate_feats = np.array([[m['spy_momentum'],m['vix'],m['realized_vol_20d'],m['vrp_20d'],
                         m['spy_rsi'],m['regime_bucket'],m['n_options']] for m in val_meta], dtype=np.float32)
train_gate(expert_ndcgs, gate_feats, out)

save_json({'recent': {'ndcg': rn, 'epoch': re, 'days': len(recent_g)},
           'core': {'ndcg': cn, 'epoch': ce, 'days': len(core_g)},
           'stress': {'ndcg': sn, 'epoch': se, 'days': len(stress_g)},
           'feature_columns': num_features,
           'train_mean': train_mean.to_dict(), 'train_std': train_std.to_dict()}, out/'ensemble_summary.json')
torch.save({'feature_columns': num_features, 'relevance_edges': edges,
            'train_mean': train_mean.to_dict(), 'train_std': train_std.to_dict(),
            'config': asdict(actual_config)}, out/'ensemble_meta.pt')
print('Ensemble training complete.')
"
elif [ "$MODE" = "ewc" ]; then
    echo "=== EWC Fine-tuning ==="
    python -u -c "
import os, gc, subprocess, torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from cloud_config import CloudConfig
from ewc_finetune import finetune_ewc_from_loaders
from train_neural_ranker import PrebuiltDataset, collate_chains
from utils import load_config, select_feature_columns, compute_relevance_bins, apply_relevance_bins

config = load_config(CloudConfig.CONFIG_PATH)
cache_dir = CloudConfig.DATA_DIR + '/cache'
suffix = open(f'{cache_dir}/.suffix').read().strip()

# Download base artifact
base_local = f'{CloudConfig.OUTPUT_DIR}/base_artifact.pt'
print(f'Downloading base artifact from {CloudConfig.EWC_BASE_ARTIFACT}...')
subprocess.run(['gcloud', 'storage', 'cp', CloudConfig.EWC_BASE_ARTIFACT, base_local], check=True)

# Load artifact to get feature columns and normalization stats
artifact = torch.load(base_local, map_location='cpu', weights_only=False)
feature_columns = artifact['feature_columns']
edges = artifact['relevance_edges']
train_mean = pd.Series(artifact['train_mean'])
train_std = pd.Series(artifact['train_std'])

fisher_years = [y.strip() for y in CloudConfig.FISHER_YEARS.split(',') if y.strip()] if CloudConfig.FISHER_YEARS else CloudConfig.train_years_list()[-1:]

def build_groups(years):
    groups = []
    for year in years:
        path = f'{cache_dir}/year_{year}_{suffix}.parquet'
        df = pd.read_parquet(path)
        df['type_numeric'] = (df['type'].str.lower() == 'call').astype(np.float32)
        df['target_relevance'] = apply_relevance_bins(df['target_return'], edges).astype(np.float32)
        if 'relative_spread' in df.columns:
            df = df[df['relative_spread'] <= 0.50]
        df[feature_columns] = (df[feature_columns] - train_mean) / train_std
        df[feature_columns] = df[feature_columns].fillna(0.0)
        for date in sorted(df['date'].unique()):
            day = df[df['date'] == date]
            if len(day) < 2: continue
            groups.append((np.nan_to_num(day[feature_columns].values.astype(np.float32)),
                           day['target_relevance'].values.astype(np.float32)))
        del df; gc.collect()
        print(f'  {year}: {len(groups)} total days')
    return groups

print('Building Fisher data...')
old_groups = build_groups(fisher_years)
print('Building fine-tune data...')
new_groups = build_groups(CloudConfig.train_years_list())
print('Building validation data...')
val_groups = build_groups(CloudConfig.val_years_list())

loader_kwargs = {'batch_size': 1, 'collate_fn': collate_chains, 'num_workers': 2, 'pin_memory': True}
old_loader = DataLoader(PrebuiltDataset(old_groups), shuffle=True, **loader_kwargs)
new_loader = DataLoader(PrebuiltDataset(new_groups), shuffle=True, **loader_kwargs)
val_loader = DataLoader(PrebuiltDataset(val_groups), shuffle=False, **loader_kwargs)

print(f'Fisher: {len(old_groups)} days, Fine-tune: {len(new_groups)} days, Val: {len(val_groups)} days')
print(f'EWC lambda={CloudConfig.EWC_LAMBDA}, lr={CloudConfig.EWC_LR}, epochs={CloudConfig.EWC_EPOCHS}')

result = finetune_ewc_from_loaders(
    base_artifact_path=base_local,
    old_loader=old_loader,
    new_loader=new_loader,
    val_loader=val_loader,
    output_dir=CloudConfig.OUTPUT_DIR,
    ewc_lambda=CloudConfig.EWC_LAMBDA,
    finetune_lr=CloudConfig.EWC_LR,
    finetune_epochs=CloudConfig.EWC_EPOCHS,
    fisher_samples=CloudConfig.FISHER_SAMPLES,
)
print(f'EWC fine-tuning complete: {result}')
"
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
if CloudConfig.PATIENCE != 8:
    config.setdefault('neural_ranker', {})['patience'] = CloudConfig.PATIENCE
if CloudConfig.EPOCHS != 50:
    config.setdefault('neural_ranker', {})['epochs'] = CloudConfig.EPOCHS
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
