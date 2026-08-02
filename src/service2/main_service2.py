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
logger = logging.getLogger("service2_kokoro")

app = FastAPI(
    title="Servicio 2: Motor TTS Hiperrealista Kokoro-82M",
    description="Motor de síntesis de voz ultrarrealista de código abierto en Español e Inglés optimizado para CPU (4 vCPUs + 16GB RAM)",
    version="2.0.0"
)

AUDIO_OUTPUT_DIR = "/tmp/synth_output"
os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)

GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "clone-voice-storage-bitcitychamp-project-8cc83f2c")

class SynthesizeRequest(BaseModel):
    text: str
    voice_reference: Optional[str] = None
    language: Optional[str] = "es"
    speed: Optional[float] = 1.0

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
        <title>Servicio 2: Kokoro-82M TTS Hiperrealista</title>
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
            body { background: var(--bg-dark); color: var(--text-main); min-height: 100vh; padding: 2rem; }
            .container { max-width: 900px; margin: 0 auto; }
            .header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 2rem; }
            .btn-back { background: rgba(255,255,255,0.1); color: #fff; text-decoration: none; padding: 0.6rem 1.2rem; border-radius: 12px; font-weight: 600; }
            .card { background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 24px; padding: 2.5rem; backdrop-filter: blur(20px); }
            .title { font-size: 1.8rem; font-weight: 800; margin-bottom: 0.5rem; background: linear-gradient(135deg, var(--primary), var(--secondary)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
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
                <span style="color: var(--primary); font-weight: 700;">Servicio 2: Kokoro-82M CPU</span>
            </div>
            <div class="card">
                <h1 class="title">🚀 Motor TTS Hiperrealista (Kokoro-82M)</h1>
                <p style="color: var(--text-sub); margin-bottom: 2rem;">Síntesis de voz ultra-natural en Español e Inglés optimizada para CPU (4 vCPUs + 16GB RAM).</p>

                <div class="input-group">
                    <label>Idioma de Síntesis</label>
                    <select id="langSelect" class="input-control">
                        <option value="es" selected>🇲🇽 / 🇪🇸 Español (Español Latino / España)</option>
                        <option value="en">🇺🇸 / 🇬🇧 English (US / UK Natural)</option>
                    </select>
                </div>

                <div class="input-group">
                    <label>Texto a Sintetizar</label>
                    <textarea id="textInput" class="input-control" placeholder="Escribe el texto que deseas convertir en voz hiperrealista...">¡Hola! Este es el Servicio 2 ejecutando el motor Kokoro-82M optimizado para CPU en Google Cloud Platform.</textarea>
                </div>

                <button onclick="synthesizeSpeech()" class="btn-action">
                    <i class="fa-solid fa-wand-magic-sparkles"></i> Sintetizar Voz Hiperrealista (/synthesize)
                </button>

                <div id="playerBox" class="player-box">
                    <p style="margin-bottom: 1rem; color: var(--primary); font-weight: 700;">🔊 Audio Generado Exitosamente:</p>
                    <audio id="audioEl" controls autoplay></audio>
                </div>
            </div>
        </div>
        <script>
            async function synthesizeSpeech() {
                const text = document.getElementById('textInput').value;
                const lang = document.getElementById('langSelect').value;
                const playerBox = document.getElementById('playerBox');
                const audioEl = document.getElementById('audioEl');

                try {
                    const res = await fetch('/synthesize', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ text: text, language: lang })
                    });
                    const data = await res.json();
                    playerBox.style.display = 'block';
                    audioEl.src = data.audio_stream_url;
                    audioEl.play();
                } catch(e) {
                    alert('Error en la síntesis: ' + e);
                }
            }
        </script>
    </body>
    </html>
    """

@app.get("/health")
def health():
    return {"status": "healthy", "service": "Servicio 2: Kokoro-82M TTS Hiperrealista", "engine": "ONNX CPU 82M"}

@app.get("/audio/{filename}")
def stream_audio(filename: str):
    file_path = os.path.join(AUDIO_OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        kokoro_engine.synthesize_speech("Prueba de audio Servicio 2", file_path)
    return FileResponse(file_path, media_type="audio/wav", filename=filename)

@app.post("/synthesize")
async def synthesize_endpoint(
    text: Optional[str] = Form(None),
    language: Optional[str] = Form("es"),
    voice_reference: Optional[UploadFile] = File(None),
    json_payload: Optional[SynthesizeRequest] = None
):
    """
    Endpoint principal /synthesize para síntesis de voz hiperrealista Kokoro-82M.
    Soporta peticiones JSON (Pydantic) y multipart Form upload con archivo de referencia de voz.
    """
    final_text = ""
    final_lang = "es"
    
    if json_payload and json_payload.text:
        final_text = json_payload.text
        final_lang = json_payload.language or "es"
    elif text:
        final_text = text
        final_lang = language or "es"
    else:
        raise HTTPException(status_code=400, detail="El parámetro 'text' es obligatorio.")

    ref_audio_path = None
    if voice_reference:
        ref_dir = "/tmp/voice_prompts"
        os.makedirs(ref_dir, exist_ok=True)
        ref_audio_path = os.path.join(ref_dir, voice_reference.filename)
        with open(ref_audio_path, "wb") as buffer:
            shutil.copyfileobj(voice_reference.file, buffer)

    output_filename = f"kokoro_{os.urandom(4).hex()}.wav"
    local_output_path = os.path.join(AUDIO_OUTPUT_DIR, output_filename)

    # Sintetizar con Kokoro-82M Engine
    kokoro_engine.synthesize_speech(
        text=final_text,
        output_wav_path=local_output_path,
        voice_reference_path=ref_audio_path,
        language=final_lang
    )

    gcs_uri = upload_to_gcs(local_output_path, f"service2-outputs/{output_filename}")
    stream_url = f"/service2/audio/{output_filename}"

    return {
        "status": "SUCCESS",
        "service": "Servicio 2: Kokoro-82M TTS Hiperrealista",
        "text": final_text,
        "language": final_lang,
        "audio_stream_url": stream_url,
        "gcs_uri": gcs_uri
    }
