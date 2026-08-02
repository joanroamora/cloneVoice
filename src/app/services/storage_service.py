import os
import logging
from google.cloud import storage
from app.config import settings

logger = logging.getLogger(__name__)

PERMANENT_OPENSOURCE_VOICES = [
    {
        "voice_id": "carlos_es",
        "model_name": "Carlos (Español Masculino Open-Source)",
        "language": "es",
        "gender": "male",
        "checkpoint_gcs_uri": f"gs://{settings.GCS_BUCKET_NAME}/trained-models/carlos_es/carlos_gpt_sovits.ckpt",
        "status": "READY"
    },
    {
        "voice_id": "sofia_es",
        "model_name": "Sofía (Español Femenino Open-Source)",
        "language": "es",
        "gender": "female",
        "checkpoint_gcs_uri": f"gs://{settings.GCS_BUCKET_NAME}/trained-models/sofia_es/sofia_gpt_sovits.ckpt",
        "status": "READY"
    },
    {
        "voice_id": "david_en",
        "model_name": "David (English US Male Open-Source)",
        "language": "en",
        "gender": "male",
        "checkpoint_gcs_uri": f"gs://{settings.GCS_BUCKET_NAME}/trained-models/david_en/david_gpt_sovits.ckpt",
        "status": "READY"
    },
    {
        "voice_id": "emma_en",
        "model_name": "Emma (English US Female Open-Source)",
        "language": "en",
        "gender": "female",
        "checkpoint_gcs_uri": f"gs://{settings.GCS_BUCKET_NAME}/trained-models/emma_en/emma_gpt_sovits.ckpt",
        "status": "READY"
    }
]

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
        if not self.bucket:
            logger.info(f"[Local Mode] Simulated upload of {local_path} to {gcs_blob_name}")
            return f"gs://{self.bucket_name}/{gcs_blob_name}"

        blob = self.bucket.blob(gcs_blob_name)
        blob.upload_from_filename(local_path)
        gcs_uri = f"gs://{self.bucket_name}/{gcs_blob_name}"
        logger.info(f"Uploaded {local_path} -> {gcs_uri}")
        return gcs_uri

    def download_file(self, gcs_blob_name: str, local_destination: str) -> str:
        if not self.bucket:
            logger.info(f"[Local Mode] Simulated download of {gcs_blob_name} to {local_destination}")
            return local_destination

        blob = self.bucket.blob(gcs_blob_name)
        os.makedirs(os.path.dirname(local_destination), exist_ok=True)
        blob.download_to_filename(local_destination)
        logger.info(f"Downloaded gs://{self.bucket_name}/{gcs_blob_name} -> {local_destination}")
        return local_destination

    def exists(self, gcs_blob_name: str) -> bool:
        if not self.bucket:
            return False
        blob = self.bucket.blob(gcs_blob_name)
        return blob.exists()

    def purge_old_voices(self):
        """Purges all pre-existing custom voice models from local storage and GCS."""
        models_dir = "/tmp/models"
        if os.path.exists(models_dir):
            try:
                shutil.rmtree(models_dir)
                os.makedirs(models_dir, exist_ok=True)
            except Exception as e:
                logger.warning(f"Error purging local models: {e}")

        if self.bucket:
            try:
                blobs = self.client.list_blobs(self.bucket_name, prefix=settings.TRAINED_MODELS_PREFIX)
                for blob in blobs:
                    blob.delete()
                logger.info("Purged all old voices from GCS bucket trained-models/")
            except Exception as e:
                logger.warning(f"Error deleting GCS blobs: {e}")

    def list_trained_models(self) -> list:
        """Returns ONLY the 4 open-source Spanish/English permanent voice models."""
        return PERMANENT_OPENSOURCE_VOICES

storage_service = StorageService()
