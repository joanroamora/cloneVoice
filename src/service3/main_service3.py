import os
import shutil
import logging
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from google.cloud import storage

from qwen3_clone_engine import qwen3_clone_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("service3_qwen3_clone")

app = FastAPI(
    title="Servicio 3: Clonación de Voz Zero-Shot con Qwen3-TTS (GCP GPU)",
    description="Motor de clonación de voz instantánea de código abierto acelerado por GPU (NVIDIA L4 / CUDA en GCP)",
    version="3.0.0"
)

AUDIO_OUTPUT_DIR = "/tmp/qwen3_clone_outputs"
PROMPT_DIR = "/tmp/qwen3_prompts"
os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)
os.makedirs(PROMPT_DIR, exist_ok=True)

GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "clone-voice-storage-bitcitychamp-project-8cc83f2c")

class ClonePayload(BaseModel):
    text_prompt: str
    reference_audio_path: Optional[str] = None
    language: Optional[str] = "es"

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
def serve_service3_gui():
    return """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Servicio 3: Clonación de Voz Zero-Shot Qwen3-TTS</title>
        <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        <style>
            :root {
                --bg-dark: #050714;
                --card-bg: rgba(15, 23, 42, 0.85);
                --card-border: rgba(168, 85, 247, 0.25);
                --primary: #a855f7;
                --primary-glow: rgba(168, 85, 247, 0.4);
                --secondary: #6366f1;
                --text-main: #f8fafc;
                --text-sub: #94a3b8;
            }
            * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Plus Jakarta Sans', sans-serif; }
            body { background: var(--bg-dark); color: var(--text-main); min-height: 100vh; padding: 2.5rem; }
            .container { max-width: 950px; margin: 0 auto; }
            .header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 2rem; }
            .btn-back { background: rgba(255,255,255,0.1); color: #fff; text-decoration: none; padding: 0.6rem 1.2rem; border-radius: 12px; font-weight: 600; }
            .card { background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 24px; padding: 2.5rem; backdrop-filter: blur(20px); }
            .title { font-size: 2.2rem; font-weight: 800; margin-bottom: 0.5rem; background: linear-gradient(135deg, var(--primary), var(--secondary)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
            .input-group { margin-bottom: 1.5rem; }
            .input-group label { display: block; font-weight: 600; margin-bottom: 0.5rem; }
            .input-control { width: 100%; background: rgba(0,0,0,0.5); border: 1px solid var(--card-border); border-radius: 14px; padding: 0.9rem 1.2rem; color: #fff; font-size: 1rem; outline: none; }
            textarea.input-control { min-height: 110px; }
            .dropzone { border: 2px dashed var(--card-border); border-radius: 16px; padding: 2rem 1.5rem; text-align: center; background: rgba(0,0,0,0.4); cursor: pointer; }
            .dropzone i { font-size: 2.5rem; color: var(--primary); margin-bottom: 0.8rem; }
            .mic-btn { background: rgba(239,68,68,0.15); color: #ef4444; border: 1px solid rgba(239,68,68,0.3); padding: 0.7rem 1.2rem; border-radius: 12px; font-weight: 600; cursor: pointer; margin-top: 1rem; display: inline-flex; align-items: center; gap: 0.5rem; }
            .mic-btn.recording { background: #ef4444; color: #fff; }
            .btn-action { width: 100%; background: linear-gradient(135deg, var(--primary), var(--secondary)); color: #fff; border: none; padding: 1.1rem; border-radius: 14px; font-size: 1.1rem; font-weight: 700; cursor: pointer; box-shadow: 0 10px 25px -5px var(--primary-glow); }
            .player-box { margin-top: 2rem; background: rgba(0,0,0,0.6); padding: 1.5rem; border-radius: 16px; display: none; }
            audio { width: 100%; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <a href="/" class="btn-back"><i class="fa-solid fa-arrow-left"></i> Volver al Portal Principal</a>
                <span style="color: var(--primary); font-weight: 700;">Servicio 3: Qwen3-TTS GPU Clone</span>
            </div>
            <div class="card">
                <h1 class="title">⚡ Servicio 3: Clonación de Voz Zero-Shot (Qwen3-TTS GPU)</h1>
                <p style="color: var(--text-sub); margin-bottom: 2rem;">Sube una muestra de voz de referencia (3-15 segundos) o graba tu voz para clonarla instantáneamente con Qwen3-TTS.</p>

                <div class="input-group">
                    <label>1. Muestra de Audio de la Voz a Clonar (WAV / MP3)</label>
                    <div class="dropzone" onclick="document.getElementById('refInput').click()">
                        <i class="fa-solid fa-cloud-arrow-up"></i>
                        <p><b id="fileNameDisplay">Haz clic o arrastra tu muestra de audio aquí</b></p>
                        <span style="font-size: 0.85rem; color: var(--text-sub);">Formatos aceptados: .WAV, .MP3, .M4A (3 a 15 segundos)</span>
                    </div>
                    <input type="file" id="refInput" accept="audio/*" style="display: none;" onchange="handleFileSelect(this)">
                </div>

                <div class="input-group" style="text-align: center;">
                    <label>O graba tu voz con el micrófono:</label>
                    <button id="micBtn" class="mic-btn" onclick="toggleMic()">
                        <i class="fa-solid fa-microphone"></i> <span id="micBtnText">Grabar Muestra de Voz</span>
                    </button>
                    <p id="micStatus" style="font-size: 0.8rem; color: var(--text-sub); margin-top: 0.4rem;"></p>
                </div>

                <div class="input-group">
                    <label>2. Texto a Sintetizar con la Voz Clonada</label>
                    <textarea id="textInput" class="input-control" placeholder="Escribe las frases que la voz clonada expresará...">¡Hola! Esta es la demostración de clonación de voz instantánea Zero-Shot ejecutándose con el motor Qwen3-TTS en Google Cloud Platform.</textarea>
                </div>

                <button onclick="runVoiceCloning()" class="btn-action">
                    <i class="fa-solid fa-bolt"></i> Clonar Voz Instantáneamente con Qwen3-TTS (/service3/clone)
                </button>

                <div id="playerBox" class="player-box">
                    <p style="margin-bottom: 0.5rem; color: var(--primary); font-weight: 700;">🔊 Voz Clonada Sintetizada Exitosamente:</p>
                    <p id="promptTextDisplay" style="font-size: 0.88rem; color: var(--text-sub); margin-bottom: 1rem;"></p>
                    <audio id="audioEl" controls autoplay></audio>
                </div>
            </div>
        </div>
        <script>
            let recordedBlob = null, mediaRecorder, audioChunks = [];

            function handleFileSelect(input) {
                if (input.files && input.files[0]) {
                    document.getElementById('fileNameDisplay').textContent = "Muestra seleccionada: " + input.files[0].name;
                }
            }

            async function toggleMic() {
                const btn = document.getElementById('micBtn');
                const text = document.getElementById('micBtnText');
                const status = document.getElementById('micStatus');

                if (mediaRecorder && mediaRecorder.state === "recording") {
                    mediaRecorder.stop();
                    btn.classList.remove('recording');
                    text.textContent = "Grabar Muestra de Voz";
                    status.textContent = "✅ Muestra de voz grabada con éxito.";
                } else {
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    mediaRecorder = new MediaRecorder(stream);
                    audioChunks = [];
                    mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
                    mediaRecorder.onstop = () => { recordedBlob = new Blob(audioChunks, { type: 'audio/wav' }); };
                    mediaRecorder.start();
                    btn.classList.add('recording');
                    text.textContent = "Detener Grabación";
                    status.textContent = "🔴 Grabando voz de muestra...";
                }
            }

            async function runVoiceCloning() {
                const refInput = document.getElementById('refInput');
                const text = document.getElementById('textInput').value;
                const playerBox = document.getElementById('playerBox');
                const audioEl = document.getElementById('audioEl');
                const promptTextDisplay = document.getElementById('promptTextDisplay');

                if(!text.trim()) { alert('Por favor ingresa el texto a sintetizar.'); return; }

                const formData = new FormData();
                formData.append('text_prompt', text);
                formData.append('language', 'es');

                if (recordedBlob) {
                    formData.append('reference_audio', recordedBlob, 'mic_reference.wav');
                } else if (refInput.files && refInput.files[0]) {
                    formData.append('reference_audio', refInput.files[0]);
                }

                try {
                    const res = await fetch('/service3/clone', {
                        method: 'POST',
                        body: formData
                    });
                    const data = await res.json();
                    playerBox.style.display = 'block';
                    promptTextDisplay.textContent = 'Texto procesado: "' + data.text_prompt + '"';
                    audioEl.src = data.audio_stream_url;
                    audioEl.play();
                } catch(e) {
                    alert('Error procesando la clonación Qwen3: ' + e);
                }
            }
        </script>
    </body>
    </html>
    """

@app.get("/health")
def health():
    return {"status": "healthy", "service": "Servicio 3: Qwen3-TTS Zero-Shot GPU Voice Cloning"}

@app.get("/audio/{filename}")
def stream_audio(filename: str):
    file_path = os.path.join(AUDIO_OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        qwen3_clone_engine.clone_voice_from_reference("Prueba Servicio 3 Qwen3", None, file_path)
    return FileResponse(file_path, media_type="audio/wav", filename=filename)

@app.post("/clone")
async def clone_endpoint(
    text_prompt: Optional[str] = Form(None),
    language: Optional[str] = Form("es"),
    reference_audio: Optional[UploadFile] = File(None),
    payload: Optional[ClonePayload] = None
):
    """
    Endpoint principal /service3/clone para clonar cualquier voz entregada en reference_audio.
    """
    final_text = ""
    final_lang = "es"

    if payload and payload.text_prompt:
        final_text = payload.text_prompt
        final_lang = payload.language or "es"
    elif text_prompt:
        final_text = text_prompt
        final_lang = language or "es"
    else:
        final_text = "¡Hola! Esta es la voz clonada ejecutándose sobre el motor Qwen3-TTS acelerado por GPU."

    ref_audio_path = None
    if reference_audio:
        ref_audio_path = os.path.join(PROMPT_DIR, reference_audio.filename)
        with open(ref_audio_path, "wb") as buffer:
            shutil.copyfileobj(reference_audio.file, buffer)

    output_filename = f"cloned_qwen3_{os.urandom(4).hex()}.wav"
    local_output_path = os.path.join(AUDIO_OUTPUT_DIR, output_filename)

    qwen3_clone_engine.clone_voice_from_reference(
        text_prompt=final_text,
        reference_audio_path=ref_audio_path,
        output_wav_path=local_output_path,
        language=final_lang
    )

    gcs_uri = upload_to_gcs(local_output_path, f"service3-cloned/{output_filename}")
    stream_url = f"/service3/audio/{output_filename}"

    return {
        "status": "SUCCESS",
        "service": "Servicio 3: Qwen3-TTS Zero-Shot GPU Voice Cloning",
        "text_prompt": final_text,
        "language": final_lang,
        "audio_stream_url": stream_url,
        "gcs_uri": gcs_uri,
        "message": "Voz clonada exitosamente con el motor Qwen3-TTS."
    }
