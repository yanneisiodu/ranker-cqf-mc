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

    # Mode: "train", "sweep", or "optuna"
    MODE = os.getenv("MODE", "train")

    # Optuna seed (different per job for parallel exploration)
    OPTUNA_SEED = int(os.getenv("OPTUNA_SEED", "") or "42")

    # Pretrained weights path (GCS or local)
    PRETRAINED_PATH = os.getenv("PRETRAINED_PATH", "")

    # Target overrides
    TARGET_MODE = os.getenv("TARGET_MODE", "")  # net_long_return or net_delta_hedged_return
    HORIZON_DAYS = int(os.getenv("HORIZON_DAYS", "") or "0")  # 0 = use config default

    # Training overrides
    EPOCHS = int(os.getenv("EPOCHS", "50"))
    PATIENCE = int(os.getenv("PATIENCE", "8"))
    NROWS = int(os.getenv("NROWS", "0")) or None  # 0 = no limit

    @classmethod
    def train_years_list(cls):
        return [y.strip() for y in cls.TRAIN_YEARS.split(",")]

    @classmethod
    def val_years_list(cls):
        return [y.strip() for y in cls.VAL_YEARS.split(",")]

    @classmethod
    def train_files(cls):
        return [f"{cls.DATA_DIR}/year_{y}_data.csv" for y in cls.train_years_list()]

    @classmethod
    def val_files(cls):
        return [f"{cls.DATA_DIR}/year_{y}_data.csv" for y in cls.val_years_list()]

    @classmethod
    def gcs_data_uri(cls, year):
        return f"{cls.GCS_BUCKET}/{cls.GCS_DATA_PREFIX}/year_{year}_data.csv"

    @classmethod
    def gcs_artifact_uri(cls, filename):
        return f"{cls.GCS_BUCKET}/{cls.GCS_ARTIFACT_PREFIX}/{filename}"

    @classmethod
    def print_config(cls):
        print("=== Cloud Config ===")
        print(f"  GCS Bucket:    {cls.GCS_BUCKET}")
        print(f"  Train years:   {cls.TRAIN_YEARS}")
        print(f"  Val years:     {cls.VAL_YEARS}")
        print(f"  Data dir:      {cls.DATA_DIR}")
        print(f"  Output dir:    {cls.OUTPUT_DIR}")
        print(f"  Epochs:        {cls.EPOCHS}")
        print(f"  Patience:      {cls.PATIENCE}")
        print(f"  Nrows:         {cls.NROWS or 'all'}")
        print()
