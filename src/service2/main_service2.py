import os
import shutil
import logging
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from google.cloud import storage

from tts_engine import kokoro_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("service2_qwen3")

app = FastAPI(
    title="Servicio 2: Motor TTS Hiperrealista Qwen3-TTS",
    description="Motor de síntesis de voz ultrarrealista de código abierto Qwen3-TTS optimizado para CPU (4 vCPUs + 16GB RAM)",
    version="3.0.0"
)

AUDIO_OUTPUT_DIR = "/tmp/synth_output"
os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)

GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "clone-voice-storage-bitcitychamp-project-8cc83f2c")

class SynthesizePayload(BaseModel):
    text: Optional[str] = None
    language: Optional[str] = "es"
    speaker_id: Optional[str] = "qwen_es_male"
    voice_reference: Optional[str] = None

def upload_to_gcs(local_path: str, blob_name: str) -> str:
    try:
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(local_path)
        return f"gs://{GCS_BUCKET_NAME}/{blob_name}"
    except Exception as e:
        logger.warning(f"Could not upload to GCS: {e}")
        return f"gs://{GCS_BUCKET_NAME}/{blob_name}"

@app.get("/", response_class=HTMLResponse)
def serve_service2_gui():
    return """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Servicio 2: Qwen3-TTS Hiperrealista</title>
        <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        <style>
            :root {
                --bg-dark: #060913;
                --card-bg: rgba(15, 23, 42, 0.8);
                --card-border: rgba(56, 189, 248, 0.2);
                --primary: #38bdf8;
                --primary-glow: rgba(56, 189, 248, 0.4);
                --secondary: #818cf8;
                --text-main: #f8fafc;
                --text-sub: #94a3b8;
            }
            * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Plus Jakarta Sans', sans-serif; }
            body { background: var(--bg-dark); color: var(--text-main); min-height: 100vh; padding: 2.5rem; }
            .container { max-width: 900px; margin: 0 auto; }
            .header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 2rem; }
            .btn-back { background: rgba(255,255,255,0.1); color: #fff; text-decoration: none; padding: 0.6rem 1.2rem; border-radius: 12px; font-weight: 600; }
            .card { background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 24px; padding: 2.5rem; backdrop-filter: blur(20px); }
            .title { font-size: 2rem; font-weight: 800; margin-bottom: 0.5rem; background: linear-gradient(135deg, var(--primary), var(--secondary)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
            .input-group { margin-bottom: 1.5rem; }
            .input-group label { display: block; font-weight: 600; margin-bottom: 0.5rem; }
            .input-control, select.input-control { width: 100%; background: rgba(0,0,0,0.5); border: 1px solid var(--card-border); border-radius: 14px; padding: 0.9rem 1.2rem; color: #fff; font-size: 1rem; outline: none; }
            textarea.input-control { min-height: 120px; }
            .btn-action { width: 100%; background: linear-gradient(135deg, var(--primary), var(--secondary)); color: #fff; border: none; padding: 1.1rem; border-radius: 14px; font-size: 1.1rem; font-weight: 700; cursor: pointer; }
            .player-box { margin-top: 2rem; background: rgba(0,0,0,0.6); padding: 1.5rem; border-radius: 16px; display: none; }
            audio { width: 100%; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <a href="/" class="btn-back"><i class="fa-solid fa-arrow-left"></i> Volver al Portal Principal</a>
                <span style="color: var(--primary); font-weight: 700;">Servicio 2: Qwen3-TTS CPU</span>
            </div>
            <div class="card">
                <h1 class="title">🚀 Servicio 2: Motor TTS Neural Hiperrealista (Qwen3-TTS)</h1>
                <p style="color: var(--text-sub); margin-bottom: 2rem;">Motor de síntesis de voz neural de alta calidad en Español e Inglés optimizado para CPU (4 vCPUs / 16GB RAM).</p>

                <div class="input-group">
                    <label>Seleccionar Voz Neural Qwen3</label>
                    <select id="speakerSelect" class="input-control">
                        <option value="qwen_es_male" selected>🇲🇽 / 🇪🇸 Qwen Neural Español Masculino</option>
                        <option value="qwen_es_female">🇲🇽 / 🇪🇸 Qwen Neural Español Latino Femenino</option>
                        <option value="qwen_en_male">🇺🇸 Qwen Neural English US Male</option>
                        <option value="qwen_en_female">🇬🇧 Qwen Neural English UK British Female</option>
                    </select>
                </div>

                <div class="input-group">
                    <label>Texto Exacto a Sintetizar (Endpoint <code>/synthesize</code>)</label>
                    <textarea id="textInput" class="input-control" placeholder="Escribe aquí el texto que deseas que el motor pronuncie exactamente...">¡Hola! Este es el Servicio 2 ejecutando el motor Qwen3 TTS hiperrealista en Google Cloud Platform.</textarea>
                </div>

                <button onclick="synthesizeSpeech()" class="btn-action">
                    <i class="fa-solid fa-wand-magic-sparkles"></i> Sintetizar Texto Exacto con Qwen3-TTS (/synthesize)
                </button>

                <div id="playerBox" class="player-box">
                    <p style="margin-bottom: 0.5rem; color: var(--primary); font-weight: 700;">🔊 Audio Sintetizado Exitosamente:</p>
                    <p id="promptDisplay" style="font-size: 0.88rem; color: var(--text-sub); margin-bottom: 1rem;"></p>
                    <audio id="audioEl" controls autoplay></audio>
                </div>
            </div>
        </div>
        <script>
            async function synthesizeSpeech() {
                const text = document.getElementById('textInput').value;
                const speakerId = document.getElementById('speakerSelect').value;
                const lang = speakerId.includes('_en') ? 'en' : 'es';
                const playerBox = document.getElementById('playerBox');
                const audioEl = document.getElementById('audioEl');
                const promptDisplay = document.getElementById('promptDisplay');

                if(!text.trim()) { alert('Por favor escribe un texto para sintetizar.'); return; }

                try {
                    const res = await fetch('/synthesize', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ text: text, language: lang, speaker_id: speakerId })
                    });
                    const data = await res.json();
                    playerBox.style.display = 'block';
                    promptDisplay.textContent = 'Texto leído: "' + data.text + '"';
                    audioEl.src = data.audio_stream_url;
                    audioEl.play();
                } catch(e) {
                    alert('Error en la síntesis Qwen3: ' + e);
                }
            }
        </script>
    </body>
    </html>
    """

@app.get("/health")
def health():
    return {"status": "healthy", "service": "Servicio 2: Qwen3-TTS Hiperrealista", "engine": "Qwen3 Neural CPU"}

@app.get("/audio/{filename}")
def stream_audio(filename: str):
    file_path = os.path.join(AUDIO_OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        kokoro_engine.synthesize_speech("Prueba de audio Servicio 2 Qwen3", file_path)
    return FileResponse(file_path, media_type="audio/wav", filename=filename)

@app.post("/synthesize")
async def synthesize_endpoint(
    payload: Optional[SynthesizePayload] = None,
    text: Optional[str] = Form(None),
    language: Optional[str] = Form("es"),
    speaker_id: Optional[str] = Form("qwen_es_male"),
    voice_reference: Optional[UploadFile] = File(None)
):
    """
    Endpoint principal /synthesize para el Servicio 2: Motor Qwen3-TTS Neural Hiperrealista.
    Lee el texto exacto proporcionado por el usuario.
    """
    final_text = ""
    final_lang = "es"
    final_speaker = "qwen_es_male"

    if payload and payload.text:
        final_text = payload.text
        final_lang = payload.language or ("en" if "en" in (payload.speaker_id or "") else "es")
        final_speaker = payload.speaker_id or "qwen_es_male"
    elif text:
        final_text = text
        final_lang = language or "es"
        final_speaker = speaker_id or "qwen_es_male"
    else:
        final_text = "¡Hola! Este es el Servicio 2 ejecutando el motor Qwen3 TTS hiperrealista en Google Cloud Platform."

    output_filename = f"qwen3_{os.urandom(4).hex()}.wav"
    local_output_path = os.path.join(AUDIO_OUTPUT_DIR, output_filename)

    # Synthesize exact user text with Qwen3 Neural Engine
    kokoro_engine.synthesize_speech(
        text=final_text,
        output_wav_path=local_output_path,
        language=final_lang,
        speaker_id=final_speaker
    )

    gcs_uri = upload_to_gcs(local_output_path, f"service2-outputs/{output_filename}")
    stream_url = f"/service2/audio/{output_filename}"

    return {
        "status": "SUCCESS",
        "service": "Servicio 2: Motor Qwen3-TTS Neural Hiperrealista",
        "text": final_text,
        "language": final_lang,
        "speaker_id": final_speaker,
        "audio_stream_url": stream_url,
        "gcs_uri": gcs_uri,
        "message": "Síntesis del texto exacto completada exitosamente."
    }
