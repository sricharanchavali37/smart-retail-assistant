"""
blob_client.py
──────────────
Azure Blob Storage client for uploading raw CSVs and ML model artefacts.
Credentials loaded from .env

Containers used:
  bronze  - raw uploaded CSV files
  models  - serialised .pkl model files
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)

try:
    from azure.storage.blob import BlobServiceClient
    BLOB_AVAILABLE = True
except ImportError:
    BLOB_AVAILABLE = False
    log.warning("azure-storage-blob not installed. Blob features disabled.")

AZURE_BLOB_CONN = os.getenv("AZURE_BLOB_CONNECTION_STRING", "")
BRONZE_CONTAINER = "bronze"
MODELS_CONTAINER = "models"


def get_blob_client():
    if not BLOB_AVAILABLE:
        log.warning("azure-storage-blob not available.")
        return None
    if not AZURE_BLOB_CONN:
        log.warning("AZURE_BLOB_CONNECTION_STRING not set — skipping blob upload.")
        return None
    try:
        return BlobServiceClient.from_connection_string(AZURE_BLOB_CONN)
    except Exception as e:
        log.error(f"Blob client init failed: {e}")
        return None


def upload_csv(local_path: str, blob_name: str) -> bool:
    """Upload a local CSV file to the bronze container."""
    client = get_blob_client()
    if client is None:
        return False
    try:
        container = client.get_container_client(BRONZE_CONTAINER)
        try:
            container.create_container()
        except Exception:
            pass  # already exists
        with open(local_path, "rb") as f:
            container.upload_blob(name=blob_name, data=f, overwrite=True)
        log.info(f"Uploaded {local_path} -> blob://{BRONZE_CONTAINER}/{blob_name}")
        return True
    except Exception as e:
        log.error(f"upload_csv failed: {e}")
        return False


def upload_model(local_path: str, blob_name: str) -> bool:
    """Upload a model .pkl file to the models container."""
    client = get_blob_client()
    if client is None:
        return False
    try:
        container = client.get_container_client(MODELS_CONTAINER)
        try:
            container.create_container()
        except Exception:
            pass
        with open(local_path, "rb") as f:
            container.upload_blob(name=blob_name, data=f, overwrite=True)
        log.info(f"Uploaded {local_path} -> blob://{MODELS_CONTAINER}/{blob_name}")
        return True
    except Exception as e:
        log.error(f"upload_model failed: {e}")
        return False