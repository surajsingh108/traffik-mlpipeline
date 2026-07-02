"""Azure Blob Storage model backup and restore utilities.

Usage
-----
  python model_backup.py --backup          # backup all current model files
  python model_backup.py --list            # list available backup timestamps
  python model_backup.py --restore <ts>    # restore models from a specific timestamp

Environment
-----------
AZURE_STORAGE_CONNECTION_STRING  Azure Storage connection string
AZURE_STORAGE_CONTAINER          Blob container name (default: model-backups)
MODEL_DIR                        Local model directory (default: model)
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

try:
    from azure.storage.blob import BlobServiceClient
except ModuleNotFoundError as _err:
    BlobServiceClient = None  # type: ignore[assignment,misc]
    _AZURE_MISSING = _err


def _require_azure() -> None:
    if BlobServiceClient is None:
        raise ModuleNotFoundError(
            "Install with: pip install azure-storage-blob"
        ) from _AZURE_MISSING


CONTAINER  = os.getenv("AZURE_STORAGE_CONTAINER", "model-backups")
MODEL_DIR  = Path(os.getenv("MODEL_DIR", "model"))
MODEL_FILES = [
    "delay_model.pkl",
]

BLOB_PREFIX = "model-backups"


def _blob_client(blob_name: str):
    _require_azure()
    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
    if not conn_str:
        raise RuntimeError("AZURE_STORAGE_CONNECTION_STRING env var not set")
    service = BlobServiceClient.from_connection_string(conn_str)
    return service.get_blob_client(container=CONTAINER, blob=blob_name)


def backup_all() -> str:
    """Backup all model files to Azure Blob. Returns the timestamp string used."""
    _require_azure()
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    backed_up: list[str] = []
    skipped:   list[str] = []

    for fname in MODEL_FILES:
        local_path = MODEL_DIR / fname
        if not local_path.exists():
            skipped.append(fname)
            continue

        blob_name = f"{BLOB_PREFIX}/{fname.replace('.pkl', '')}/{ts}.pkl"
        client = _blob_client(blob_name)
        with open(local_path, "rb") as data:
            client.upload_blob(data, overwrite=True)

        backed_up.append(blob_name)
        print(f"  OK Backed up {fname} → {CONTAINER}/{blob_name}")

    if skipped:
        print(f"  SKIP (not found locally): {skipped}")

    print(f"\nBackup timestamp: {ts}")
    return ts


def list_backups() -> list[str]:
    """List available backup timestamps from Azure Blob (most recent first)."""
    _require_azure()
    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
    if not conn_str:
        raise RuntimeError("AZURE_STORAGE_CONNECTION_STRING env var not set")

    service = BlobServiceClient.from_connection_string(conn_str)
    container_client = service.get_container_client(CONTAINER)

    blobs = list(container_client.list_blobs(name_starts_with=BLOB_PREFIX))
    timestamps = sorted(
        set(
            b.name.split("/")[2].replace(".pkl", "")
            for b in blobs
            if b.name.endswith(".pkl")
        ),
        reverse=True,
    )

    if not timestamps:
        print("No backups found.")
        return []

    print("\nAvailable backup timestamps (most recent first):")
    for ts in timestamps[:20]:
        print(f"  {ts}")

    return timestamps


def restore(timestamp: str) -> None:
    """Restore all models from a given backup timestamp."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for fname in MODEL_FILES:
        model_name = fname.replace(".pkl", "")
        blob_name  = f"{BLOB_PREFIX}/{model_name}/{timestamp}.pkl"

        try:
            client = _blob_client(blob_name)
            data   = client.download_blob().readall()
        except Exception as exc:
            print(f"  SKIP {fname} not found at {blob_name}: {exc}")
            continue

        local_path = MODEL_DIR / fname
        if local_path.exists():
            local_path.rename(local_path.with_suffix(".pkl.bak"))

        local_path.write_bytes(data)
        print(f"  OK Restored {fname} from {CONTAINER}/{blob_name}")

    print("\nRestore complete. Old models saved as .pkl.bak")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backup",  action="store_true")
    parser.add_argument("--list",    action="store_true")
    parser.add_argument("--restore", type=str, metavar="TIMESTAMP")
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    if args.backup:
        backup_all()
    elif args.list:
        list_backups()
    elif args.restore:
        restore(args.restore)
    else:
        parser.print_help()
