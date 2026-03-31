"""Cloud Run configuration — reads environment variables with sensible defaults."""
import os


class CloudConfig:
    # GCS
    GCS_BUCKET = os.getenv("GCS_BUCKET", "gs://neural-ranker-training-data")
    GCS_DATA_PREFIX = os.getenv("GCS_DATA_PREFIX", "data")
    GCS_ARTIFACT_PREFIX = os.getenv("GCS_ARTIFACT_PREFIX", "artifacts")

    # Training data split
    TRAIN_YEARS = os.getenv("TRAIN_YEARS", "2018,2019,2020,2021,2022,2023,2024,2025")
    VAL_YEARS = os.getenv("VAL_YEARS", "2026")

    # Local paths inside container
    DATA_DIR = os.getenv("DATA_DIR", "/app/data")
    OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/app/output")
    CONFIG_PATH = os.getenv("CONFIG_PATH", "./config_tuned.yaml")

    # Mode: "train" or "optuna"
    MODE = os.getenv("MODE", "train")

    # Optuna seed
    OPTUNA_SEED = int(os.getenv("OPTUNA_SEED", "") or "42")

    # Target overrides
    TARGET_MODE = os.getenv("TARGET_MODE", "")
    HORIZON_DAYS = int(os.getenv("HORIZON_DAYS", "") or "0")

    # Torch compile (set to "false" to disable for deterministic results)
    TORCH_COMPILE = os.getenv("TORCH_COMPILE", "true").lower() in ("true", "1", "yes")

    # Training overrides
    EPOCHS = int(os.getenv("EPOCHS", "50"))
    PATIENCE = int(os.getenv("PATIENCE", "8"))
    NROWS = int(os.getenv("NROWS", "0")) or None

    @classmethod
    def train_years_list(cls):
        return [y.strip() for y in cls.TRAIN_YEARS.split(",")]

    @classmethod
    def val_years_list(cls):
        return [y.strip() for y in cls.VAL_YEARS.split(",")]

    @classmethod
    def gcs_artifact_uri(cls, filename):
        return f"{cls.GCS_BUCKET}/{cls.GCS_ARTIFACT_PREFIX}/{filename}"

    @classmethod
    def print_config(cls):
        print("=== Cloud Config ===")
        print(f"  GCS Bucket:    {cls.GCS_BUCKET}")
        print(f"  Mode:          {cls.MODE}")
        print(f"  Train years:   {cls.TRAIN_YEARS}")
        print(f"  Val years:     {cls.VAL_YEARS}")
        print(f"  Epochs:        {cls.EPOCHS}")
        print(f"  Patience:      {cls.PATIENCE}")
        print(f"  Torch compile: {cls.TORCH_COMPILE}")
        print(f"  Nrows:         {cls.NROWS or 'all'}")
        print()
