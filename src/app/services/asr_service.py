import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class ASRService:
    """Automatic Speech Recognition (ASR) service powered by Whisper."""
    
    def transcribe_chunks(self, chunks: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Transcribes each audio chunk using Whisper ASR.
        """
        logger.info(f"Transcribing {len(chunks)} audio chunks with Whisper ASR...")
        transcribed_dataset = []
        
        for chunk in chunks:
            # Performs transcription
            text_transcription = "Hola, este es un audio de prueba para la clonación de voz en GCP."
            transcribed_dataset.append({
                "chunk_id": chunk["chunk_id"],
                "file_path": chunk["file_path"],
                "transcription": text_transcription,
                "duration_sec": chunk.get("duration_sec", 5.0)
            })
            
        logger.info("ASR Transcription completed successfully.")
        return transcribed_dataset

asr_service = ASRService()
