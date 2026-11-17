import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from fraud_detection.config import Config
from fraud_detection.data import prepare, validate_transactions
from fraud_detection.inference import predict
from fraud_detection.pipeline import train
from fraud_detection.reporting import eda


def main() -> None:
    parser = argparse.ArgumentParser(description="Fraud detection data science pipeline")
    parser.add_argument("command", choices=["prepare", "eda", "train", "predict", "run"])
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--input", type=Path, help="Joined inference CSV or Parquet")
    parser.add_argument("--state", type=Path, help="Persistent online review state across batches")
    args = parser.parse_args()
    if (args.input or args.state) and args.command != "predict":
        parser.error("--input and --state are only supported by predict")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = Config.load(args.config)
    if args.command in {"prepare", "run"}:
        prepare(config)
    if args.command in {"train", "predict", "eda"} and args.input is None:
        manifest_path = Path(config.processed_dir) / "manifest.json"
        if not manifest_path.exists():
            raise ValueError("Run prepare before using processed data")
        manifest = json.loads(manifest_path.read_text())
        if manifest["row_limit"] != config.row_limit:
            raise ValueError("Processed data row_limit differs from config; rerun prepare")
    if args.command == "eda":
        frame = pd.read_parquet(Path(config.processed_dir) / "train.parquet")
        validate_transactions(frame, labeled=True)
        eda(frame, Path(config.output_dir) / "eda")
    if args.command in {"train", "run"}:
        train(config)
    if args.command in {"predict", "run"}:
        predict(config, args.input, args.state)
