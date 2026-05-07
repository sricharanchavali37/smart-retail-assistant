"""
upload_to_blob.py
─────────────────
Uploads local data files to Azure Blob Storage (bronze container).
This seeds the ADF pipeline with data to process.

Files uploaded:
  data/raw/retail_sales.csv          → bronze/raw/retail_sales.csv
  data/processed/retail_features.csv → bronze/processed/retail_features.csv
  data/processed/anomaly_scores.csv  → bronze/processed/anomaly_scores.csv
  backend/models/xgb_demand.pkl      → bronze/models/xgb_demand.pkl

Run:
    python scripts/upload_to_blob.py
"""

import sys
import logging
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent

UPLOAD_MAP = {
    ROOT / "data" / "raw" / "retail_sales.csv":             "raw/retail_sales.csv",
    ROOT / "data" / "processed" / "retail_features.csv":    "processed/retail_features.csv",
    ROOT / "data" / "processed" / "anomaly_scores.csv":     "processed/anomaly_scores.csv",
    ROOT / "backend" / "models" / "xgb_demand.pkl":         "models/xgb_demand.pkl",
    ROOT / "backend" / "models" / "iso_anomaly.pkl":        "models/iso_anomaly.pkl",
}

CONTAINER = "bronze"


def upload_all():
    conn_str = os.getenv("AZURE_BLOB_CONNECTION_STRING", "")
    if not conn_str:
        log.error("AZURE_BLOB_CONNECTION_STRING not set in .env")
        log.error("Get it from: Azure Portal → Storage Account → Access Keys → Connection string")
        sys.exit(1)

    try:
        from azure.storage.blob import BlobServiceClient
    except ImportError:
        log.error("Run: pip install azure-storage-blob")
        sys.exit(1)

    log.info("=" * 55)
    log.info("  UPLOADING DATA TO AZURE BLOB STORAGE")
    log.info(f"  Container: {CONTAINER}")
    log.info("=" * 55)

    client    = BlobServiceClient.from_connection_string(conn_str)
    container = client.get_container_client(CONTAINER)

    # Create container if not exists
    try:
        container.create_container()
        log.info(f"  Created container: {CONTAINER}")
    except Exception:
        log.info(f"  Container '{CONTAINER}' already exists")

    success = 0
    failed  = 0

    for local_path, blob_name in UPLOAD_MAP.items():
        if not local_path.exists():
            log.warning(f"  SKIP (not found): {local_path.name}")
            continue
        try:
            size_kb = local_path.stat().st_size / 1024
            with open(local_path, "rb") as f:
                container.upload_blob(name=blob_name, data=f, overwrite=True)
            log.info(f"  ✅  {local_path.name:<35} → blob://{CONTAINER}/{blob_name}  ({size_kb:,.0f} KB)")
            success += 1
        except Exception as e:
            log.error(f"  ❌  {local_path.name}: {e}")
            failed += 1

    log.info("\n" + "=" * 55)
    log.info(f"  Uploaded: {success}  |  Failed: {failed}")
    log.info("=" * 55)

    if success > 0:
        log.info("\n  Next steps:")
        log.info("  1. Go to Azure Portal → Data Factory")
        log.info("  2. Import adf/pipeline_definition.json")
        log.info("  3. Trigger pipeline manually → Debug")
        log.info("  4. Check Databricks for Delta tables")


if __name__ == "__main__":
    upload_all()