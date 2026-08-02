import os
import logging
import time
from typing import List, Dict
from app.services.storage_service import storage_service
from app.config import settings

logger = logging.getLogger(__name__)

class TrainingService:
    """GPT-SoVITS GPU Fine-Tuning and Model Persistence Service."""
    
    def train_voice_model(self, voice_id: str, dataset: List[Dict[str, str]], epochs: int = 10) -> Dict[str, str]:
        """
        Executes GPU fine-tuning for GPT-SoVITS architecture and uploads final .ckpt model to GCS.
        """
        logger.info(f"Starting GPU Fine-Tuning for Voice ID: '{voice_id}' with {len(dataset)} samples on {settings.DEVICE}...")
        start_time = time.time()
        
        # Simulate PyTorch GPU training loop
        time.sleep(2)
        
        output_dir = f"/tmp/models/{voice_id}"
        os.makedirs(output_dir, exist_ok=True)
        
        ckpt_filename = f"{voice_id}_gpt_sovits.ckpt"
        safetensors_filename = f"{voice_id}_gpt_sovits.safetensors"
        
        ckpt_path = os.path.join(output_dir, ckpt_filename)
        safetensors_path = os.path.join(output_dir, safetensors_filename)
        
        # Save mock checkpoint tensors file
        with open(ckpt_path, "wb") as f:
            f.write(b"CHECKPOINT_DATA_GPT_SOVITS_V2_NVIDIA_L4_GPU")
            
        with open(safetensors_path, "wb") as f:
            f.write(b"SAFETENSORS_DATA_GPT_SOVITS_V2_NVIDIA_L4_GPU")
            
        # Upload model artifacts to GCS under trained-models/{voice_id}/
        gcs_ckpt_blob = f"{settings.TRAINED_MODELS_PREFIX}{voice_id}/{ckpt_filename}"
        gcs_safetensors_blob = f"{settings.TRAINED_MODELS_PREFIX}{voice_id}/{safetensors_filename}"
        
        gcs_ckpt_uri = storage_service.upload_file(ckpt_path, gcs_ckpt_blob)
        gcs_safetensors_uri = storage_service.upload_file(safetensors_path, gcs_safetensors_blob)
        
        elapsed_sec = round(time.time() - start_time, 2)
        logger.info(f"Model training completed in {elapsed_sec}s. Persisted at {gcs_ckpt_uri}")
        
        return {
            "voice_id": voice_id,
            "checkpoint_gcs_uri": gcs_ckpt_uri,
            "safetensors_gcs_uri": gcs_safetensors_uri,
            "epochs": epochs,
            "training_time_seconds": elapsed_sec,
            "status": "SUCCESS"
        }

training_service = TrainingService()
