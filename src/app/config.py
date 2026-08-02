import os
from pydantic import BaseModel

class Settings(BaseModel):
    PROJECT_ID: str = os.getenv("GCP_PROJECT_ID", "clonevoice")
    GCS_BUCKET_NAME: str = os.getenv("GCS_BUCKET_NAME", "clone-voice-storage-default")
    RAW_AUDIO_PREFIX: str = "raw-audio/"
    PROCESSED_CHUNKS_PREFIX: str = "processed-chunks/"
    TRAINED_MODELS_PREFIX: str = "trained-models/"
    INFER_OUTPUTS_PREFIX: str = "infer-outputs/"
    
    # GPU & CUDA settings
    DEVICE: str = "cuda" if os.getenv("USE_CUDA", "true").lower() == "true" else "cpu"
    WHISPER_MODEL_SIZE: str = os.getenv("WHISPER_MODEL_SIZE", "base")

settings = Settings()
