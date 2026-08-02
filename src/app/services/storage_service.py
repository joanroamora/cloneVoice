import os
import logging
from google.cloud import storage
from app.config import settings

logger = logging.getLogger(__name__)

class StorageService:
    def __init__(self):
        self.bucket_name = settings.GCS_BUCKET_NAME
        try:
            self.client = storage.Client()
            self.bucket = self.client.bucket(self.bucket_name)
        except Exception as e:
            logger.warning(f"Could not initialize GCS client: {e}. Falling back to local mode.")
            self.client = None
            self.bucket = None

    def upload_file(self, local_path: str, gcs_blob_name: str) -> str:
        """Uploads a local file to GCS."""
        if not self.bucket:
            logger.info(f"[Local Mode] Simulated upload of {local_path} to {gcs_blob_name}")
            return f"gs://{self.bucket_name}/{gcs_blob_name}"

        blob = self.bucket.blob(gcs_blob_name)
        blob.upload_from_filename(local_path)
        gcs_uri = f"gs://{self.bucket_name}/{gcs_blob_name}"
        logger.info(f"Uploaded {local_path} -> {gcs_uri}")
        return gcs_uri

    def download_file(self, gcs_blob_name: str, local_destination: str) -> str:
        """Downloads a blob from GCS to local disk."""
        if not self.bucket:
            logger.info(f"[Local Mode] Simulated download of {gcs_blob_name} to {local_destination}")
            return local_destination

        blob = self.bucket.blob(gcs_blob_name)
        os.makedirs(os.path.dirname(local_destination), exist_ok=True)
        blob.download_to_filename(local_destination)
        logger.info(f"Downloaded gs://{self.bucket_name}/{gcs_blob_name} -> {local_destination}")
        return local_destination

    def exists(self, gcs_blob_name: str) -> bool:
        """Check if blob exists in GCS."""
        if not self.bucket:
            return False
        blob = self.bucket.blob(gcs_blob_name)
        return blob.exists()

storage_service = StorageService()
