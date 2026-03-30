FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# System deps: gcloud CLI + C compiler for torch.compile/Triton
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gnupg apt-transport-https ca-certificates gcc && \
    echo "deb https://packages.cloud.google.com/apt cloud-sdk main" > /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    apt-get update && apt-get install -y --no-install-recommends google-cloud-cli && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Python deps — use compatible versions, don't fight the base image
RUN pip install --no-cache-dir \
    pandas \
    scikit-learn \
    xgboost \
    pyyaml \
    joblib \
    pyarrow \
    optuna

# Copy all code
COPY neural_ranker.py hierarchical_ranker.py train_neural_ranker.py pretrain.py sweep.py optuna_neural_sweep.py \
     logger.py utils.py config_tuned.yaml cloud_config.py entrypoint.sh ./

RUN chmod +x entrypoint.sh

# Verify everything imports before shipping
RUN python -c "import torch, pandas, numpy, sklearn, xgboost, yaml, joblib, pyarrow, optuna; \
    print(f'torch={torch.__version__} pandas={pandas.__version__} pyarrow={pyarrow.__version__}'); \
    from neural_ranker import ChainTransformer; \
    from train_neural_ranker import train_neural_ranker_from_frames, train_neural_ranker_from_datasets, PrebuiltDataset; \
    from pretrain import pretrain_from_groups, MaskedChainDataset, ReconstructionHead; \
    from optuna_neural_sweep import run_optuna_sweep; \
    from cloud_config import CloudConfig; \
    print('All imports OK')"

ENTRYPOINT ["./entrypoint.sh"]
