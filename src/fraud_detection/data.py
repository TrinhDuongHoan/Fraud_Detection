import hashlib
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from fraud_detection.config import Config

ID = "TransactionID"
TARGET = "isFraud"
TIME = "TransactionDT"
LOGGER = logging.getLogger(__name__)
TEXT_COLUMNS = {
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceType",
    "DeviceInfo",
    *(f"M{i}" for i in range(1, 10)),
    *(f"id_{i:02d}" for i in [12, 15, 16, 23, 27, 28, 29, 30, 31, 33, 34, 35, 36, 37, 38]),
}


def normalize_columns(columns) -> list[str]:
    normalized = [re.sub(r"^id-(\d+)$", r"id_\1", column) for column in columns]
    if len(set(normalized)) != len(normalized):
        raise ValueError("Column names collide after normalization")
    return normalized


def csv_dtypes(path: Path) -> dict:
    columns = pd.read_csv(path, nrows=0).columns
    return {
        raw: (
            "string"
            if name in TEXT_COLUMNS
            else "int64"
            if name in {ID, TIME}
            else "float64"
            if name == "TransactionAmt"
            else "float32"
        )
        for raw, name in zip(columns, normalize_columns(columns), strict=True)
    }


def validate_transactions(frame: pd.DataFrame, labeled: bool) -> None:
    required = {ID, TIME, "TransactionAmt"} | ({TARGET} if labeled else set())
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if frame[ID].isna().any() or frame[ID].duplicated().any():
        raise ValueError("TransactionID must be unique and non-null")
    if (frame[ID] < 0).any() or (frame[ID] % 1 != 0).any():
        raise ValueError("TransactionID must contain nonnegative integers")
    for column in (TIME, "TransactionAmt"):
        values = frame[column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"{column} must be finite, nonnegative and non-null")
    if labeled and (frame[TARGET].isna().any() or not frame[TARGET].isin([0, 1]).all()):
        raise ValueError("isFraud must contain only 0 and 1")
    if not labeled and TARGET in frame:
        raise ValueError("Unlabeled data must not contain isFraud")


def merge_identity(transactions: pd.DataFrame, identity: pd.DataFrame) -> pd.DataFrame:
    identity = identity.rename(
        columns=dict(zip(identity.columns, normalize_columns(identity.columns), strict=True))
    )
    if identity[ID].isna().any() or identity[ID].duplicated().any():
        raise ValueError("Identity keys must be unique and non-null")
    overlap = set(transactions.columns) & set(identity.columns) - {ID}
    if overlap:
        raise ValueError(f"Transaction and identity columns overlap: {sorted(overlap)}")
    return transactions.merge(identity, on=ID, how="left", validate="one_to_one", sort=False)


def file_signature(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"file": path.name, "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def prepare(config: Config) -> dict:
    raw, destination = Path(config.raw_dir), Path(config.processed_dir)
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {"row_limit": config.row_limit, "files": [], "datasets": {}}
    for split in ("train", "test"):
        transaction_path = raw / f"{split}_transaction.csv"
        identity_path = raw / f"{split}_identity.csv"
        identity = pd.read_csv(identity_path, dtype=csv_dtypes(identity_path))
        seen = set()
        count = 0
        temporary = destination / f"{split}.parquet.tmp"
        writer = None
        try:
            chunks = pd.read_csv(
                transaction_path,
                dtype=csv_dtypes(transaction_path),
                chunksize=config.chunk_size,
                nrows=config.row_limit,
            )
            for chunk in chunks:
                validate_transactions(chunk, labeled=split == "train")
                ids = set(chunk[ID])
                if seen.intersection(ids):
                    raise ValueError("Duplicate TransactionID across chunks")
                seen.update(ids)
                merged = merge_identity(chunk, identity)
                table = pa.Table.from_pandas(merged, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
                writer.write_table(table)
                count += len(chunk)
                LOGGER.info("Prepared %s: %s rows", split, count)
            if not count:
                raise ValueError(f"Empty {split} transactions")
        except BaseException:
            if writer is not None:
                writer.close()
                writer = None
            temporary.unlink(missing_ok=True)
            raise
        finally:
            if writer is not None:
                writer.close()
        temporary.replace(destination / f"{split}.parquet")
        manifest["datasets"][split] = {"rows": count, "columns": len(table.schema)}
        manifest["files"].extend([file_signature(transaction_path), file_signature(identity_path)])
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
