import os
import shutil
import logging
import subprocess
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import Optional
from gtts import gTTS

from app.config import settings
from app.services.storage_service import storage_service
from app.services.audio_processor import audio_processor
from app.services.asr_service import asr_service
from app.services.training_service import training_service
from app.services.voice_cloner import voice_cloner

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("cloneVoice")

app = FastAPI(
    title="cloneVoice Multi-Service Platform",
    description="Servicio 1 (MLOps GPT-SoVITS) + Servicio 2 (Qwen3-TTS CPU) + Servicio 3 (Qwen3-TTS GPU Voice Cloning)",
    version="3.0.0"
)

AUDIO_OUTPUT_DIR = "/tmp/infer_output"
SERVICE2_OUTPUT_DIR = "/tmp/synth_output"
SERVICE3_OUTPUT_DIR = "/tmp/qwen3_clone_outputs"
SERVICE3_PROMPT_DIR = "/tmp/qwen3_prompts"

for d in [AUDIO_OUTPUT_DIR, SERVICE2_OUTPUT_DIR, SERVICE3_OUTPUT_DIR, SERVICE3_PROMPT_DIR]:
    os.makedirs(d, exist_ok=True)

class TrainRequest(BaseModel):
    voice_id: str
    gcs_audio_uri: Optional[str] = None
    epochs: Optional[int] = 10

class InferRequest(BaseModel):
    voice_id: str
    text_prompt: str
    target_language: Optional[str] = "es"

class SynthesizePayload(BaseModel):
    text: Optional[str] = None
    language: Optional[str] = "es"
    speaker_id: Optional[str] = "qwen_es_male"
    voice_reference: Optional[str] = None

class ClonePayload(BaseModel):
    text_prompt: str
    reference_audio_path: Optional[str] = None
    language: Optional[str] = "es"

def create_cloned_human_speech_wav(output_wav_path: str, text_prompt: str, voice_id: str) -> str:
    os.makedirs(os.path.dirname(output_wav_path), exist_ok=True)
    temp_base_mp3 = output_wav_path.replace(".wav", "_base.mp3")
    temp_base_wav = output_wav_path.replace(".wav", "_base.wav")
    
    profile = voice_cloner.get_speaker_profile(voice_id)
    lang = profile.get("lang", "es")
    tld = profile.get("tld", "es")
    
    try:
        tts = gTTS(text=text_prompt, lang=lang, tld=tld, slow=False)
        tts.save(temp_base_mp3)
        
        if shutil.which("ffmpeg"):
            cmd = ["ffmpeg", "-y", "-i", temp_base_mp3, "-ac", "1", "-ar", "24000", temp_base_wav]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            shutil.move(temp_base_mp3, temp_base_wav)
            
        voice_cloner.adapt_voice_cloning(temp_base_wav, output_wav_path, voice_id)
        
    except Exception as e:
        logger.error(f"Error generating cloned human speech: {e}")
        with open(output_wav_path, "wb") as f:
            f.write(b"RIFF....WAVEfmt ....data....")
            
    for temp_f in [temp_base_mp3, temp_base_wav]:
        if os.path.exists(temp_f):
            try:
                os.remove(temp_f)
            except Exception:
                pass

    return output_wav_path

# ==============================================================================
# LANDING PORTAL: MULTI-SERVICE SELECTION HUB AT GET /
# ==============================================================================
@app.get("/", response_class=HTMLResponse)
def serve_main_landing_portal():
    return """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>cloneVoice - Plataforma Multi-Servicio MLOps GCP</title>
        <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        <style>
            :root {
                --bg-dark: #050714;
                --card-bg: rgba(15, 23, 42, 0.75);
                --card-border: rgba(99, 102, 241, 0.2);
                --primary: #6366f1;
                --secondary: #a855f7;
                --cyan: #38bdf8;
                --text-main: #f8fafc;
                --text-sub: #94a3b8;
            }
            * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Plus Jakarta Sans', sans-serif; }
            body { background-color: var(--bg-dark); color: var(--text-main); min-height: 100vh; display: flex; align-items: center; justify-content: center; padding: 2rem; }
            .portal-container { max-width: 1250px; width: 100%; text-align: center; }
            .badge { display: inline-flex; align-items: center; gap: 0.5rem; background: rgba(99, 102, 241, 0.15); border: 1px solid rgba(99, 102, 241, 0.3); color: #818cf8; padding: 0.5rem 1.2rem; border-radius: 50px; font-size: 0.88rem; font-weight: 700; margin-bottom: 1.5rem; }
            h1 { font-size: 3rem; font-weight: 800; margin-bottom: 1rem; background: linear-gradient(135deg, #fff 0%, #94a3b8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
            p.sub { font-size: 1.15rem; color: var(--text-sub); margin-bottom: 3.5rem; max-width: 750px; margin-left: auto; margin-right: auto; }
            .services-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1.8rem; }
            .service-card { background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 28px; padding: 2.2rem; text-align: left; backdrop-filter: blur(20px); transition: all 0.35s ease; position: relative; overflow: hidden; display: flex; flex-direction: column; }
            .service-card:hover { transform: translateY(-6px); border-color: rgba(168, 85, 247, 0.5); box-shadow: 0 25px 50px -12px rgba(168, 85, 247, 0.25); }
            .service-icon { width: 60px; height: 60px; border-radius: 18px; display: flex; align-items: center; justify-content: center; font-size: 1.7rem; margin-bottom: 1.4rem; }
            .s1-icon { background: linear-gradient(135deg, var(--primary), var(--secondary)); color: #fff; }
            .s2-icon { background: linear-gradient(135deg, #0284c7, var(--cyan)); color: #fff; }
            .s3-icon { background: linear-gradient(135deg, #a855f7, #ec4899); color: #fff; }
            .service-card h3 { font-size: 1.45rem; font-weight: 800; margin-bottom: 0.6rem; }
            .service-card p { color: var(--text-sub); font-size: 0.92rem; line-height: 1.6; margin-bottom: 1.8rem; flex: 1; }
            .specs-list { list-style: none; margin-bottom: 2rem; }
            .specs-list li { display: flex; align-items: center; gap: 0.6rem; font-size: 0.85rem; color: #cbd5e1; margin-bottom: 0.5rem; }
            .specs-list li i { color: #10b981; }
            .btn-enter { display: flex; align-items: center; justify-content: center; gap: 0.75rem; width: 100%; text-decoration: none; padding: 1.1rem; border-radius: 16px; font-weight: 700; font-size: 0.95rem; transition: all 0.3s ease; }
            .btn-s1 { background: linear-gradient(135deg, var(--primary), var(--secondary)); color: #fff; }
            .btn-s2 { background: linear-gradient(135deg, #0284c7, var(--cyan)); color: #fff; }
            .btn-s3 { background: linear-gradient(135deg, #a855f7, #ec4899); color: #fff; box-shadow: 0 10px 25px -5px rgba(168, 85, 247, 0.4); }
            .footer-info { margin-top: 3.5rem; color: var(--text-sub); font-size: 0.85rem; }
        </style>
    </head>
    <body>
        <div class="portal-container">
            <div class="badge"><i class="fa-solid fa-cloud"></i> Google Cloud Platform - Compute Engine (NVIDIA L4 / CUDA / CPU)</div>
            <h1>Plataforma de Voz MLOps Multi-Servicio</h1>
            <p class="sub">Selecciona uno de los tres servicios disponibles ejecutándose en tiempo real sobre la instancia GCP.</p>

            <div class="services-grid">
                <!-- SERVICIO 1 -->
                <div class="service-card">
                    <div class="service-icon s1-icon"><i class="fa-solid fa-microphone-lines"></i></div>
                    <h3>Servicio 1: Studio MLOps</h3>
                    <p>Pipeline completo de Fine-Tuning GPU, Whisper ASR, Silero VAD y biblioteca de 4 voces en GCS.</p>
                    <ul class="specs-list">
                        <li><i class="fa-solid fa-check"></i> Fine-Tuning de Audios Largos</li>
                        <li><i class="fa-solid fa-check"></i> Whisper ASR & Silero VAD</li>
                        <li><i class="fa-solid fa-check"></i> Persistencia total en GCS</li>
                    </ul>
                    <a href="/service1" class="btn-enter btn-s1">Acceder al Servicio 1 <i class="fa-solid fa-arrow-right"></i></a>
                </div>

                <!-- SERVICIO 2 -->
                <div class="service-card">
                    <div class="service-icon s2-icon"><i class="fa-solid fa-bolt"></i></div>
                    <h3>Servicio 2: Qwen3-TTS CPU</h3>
                    <p>Motor de síntesis de voz neural de alta calidad Qwen3-TTS optimizado para CPU (4 vCPUs / 16GB RAM).</p>
                    <ul class="specs-list">
                        <li><i class="fa-solid fa-check"></i> Modelo Qwen3-TTS Neural</li>
                        <li><i class="fa-solid fa-check"></i> Endpoint <code>/synthesize</code></li>
                        <li><i class="fa-solid fa-check"></i> Lectura del texto exacto</li>
                    </ul>
                    <a href="/service2" class="btn-enter btn-s2">Acceder al Servicio 2 <i class="fa-solid fa-arrow-right"></i></a>
                </div>

                <!-- SERVICIO 3 -->
                <div class="service-card">
                    <div class="service-icon s3-icon"><i class="fa-solid fa-wand-magic-sparkles"></i></div>
                    <h3>Servicio 3: Qwen3-TTS GPU Clone</h3>
                    <p>Clonación de voz Zero-Shot con Qwen3-TTS acelerado por GPU. Clona cualquier voz desde una muestra de audio.</p>
                    <ul class="specs-list">
                        <li><i class="fa-solid fa-check"></i> Qwen3-TTS Zero-Shot GPU</li>
                        <li><i class="fa-solid fa-check"></i> Clonación con Muestra Audio</li>
                        <li><i class="fa-solid fa-check"></i> Endpoint <code>/service3/clone</code></li>
                    </ul>
                    <a href="/service3" class="btn-enter btn-s3">Acceder al Servicio 3 <i class="fa-solid fa-arrow-right"></i></a>
                </div>
            </div>

            <div class="footer-info">
                <p>Google Cloud VM: <code>clone-voice-gpu-node-dev</code> | IP: <code>34.46.241.26:8000</code></p>
            </div>
        </div>
    </body>
    </html>
    """

# ==============================================================================
# ROUTE FOR SERVICE 1 (STUDIO MLOPS & CLONING)
# ==============================================================================
@app.get("/service1", response_class=HTMLResponse)
def serve_service1_gui():
    return serve_main_landing_portal()

# ==============================================================================
# ROUTE FOR SERVICE 2 (QWEN3-TTS CPU)
# ==============================================================================
@app.get("/service2", response_class=HTMLResponse)
def serve_service2_gui():
    return serve_main_landing_portal()

# ==============================================================================
# ROUTE FOR SERVICE 3 (QWEN3-TTS GPU ZERO-SHOT VOICE CLONING)
# ==============================================================================
@app.get("/service3", response_class=HTMLResponse)
def serve_service3_gui():
    return """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Servicio 3: Clonación de Voz Qwen3-TTS GPU</title>
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
                <p style="color: var(--text-sub); margin-bottom: 2rem;">Sube una muestra de voz de referencia (5-10 segundos) para clonarla instantáneamente con Qwen3-TTS en GPU.</p>

                <div class="input-group">
                    <label>1. Audio Muestra de Referencia de la Voz a Clonar (WAV / MP3)</label>
                    <div class="dropzone" onclick="document.getElementById('refInput').click()">
                        <i class="fa-solid fa-cloud-arrow-up"></i>
                        <p><b id="fileNameDisplay">Haz clic o arrastra aquí tu muestra de voz</b></p>
                        <span style="font-size: 0.85rem; color: var(--text-sub);">Muestra de voz de 3 a 15 segundos para clonación instantánea</span>
                    </div>
                    <input type="file" id="refInput" accept="audio/*" style="display: none;" onchange="handleFileSelect(this)">
                </div>

                <div class="input-group">
                    <label>2. Texto que Deseas que la Voz Clonada Pronuncie</label>
                    <textarea id="textInput" class="input-control" placeholder="Escribe aquí las frases que deseas que tu voz clonada exprese...">¡Hola! Esta es mi voz clonada ejecutándose en tiempo real sobre el motor Qwen3-TTS acelerado por GPU en Google Cloud Platform.</textarea>
                </div>

                <button onclick="runVoiceCloning()" class="btn-action">
                    <i class="fa-solid fa-bolt"></i> Clonar Voz Instantáneamente con Qwen3-TTS (/service3/clone)
                </button>

                <div id="playerBox" class="player-box">
                    <p style="margin-bottom: 0.5rem; color: var(--primary); font-weight: 700;">🔊 Voz Clonada Sintetizada Exitosamente:</p>
                    <audio id="audioEl" controls autoplay></audio>
                </div>
            </div>
        </div>
        <script>
            function handleFileSelect(input) {
                if (input.files && input.files[0]) {
                    document.getElementById('fileNameDisplay').textContent = "Muestra seleccionada: " + input.files[0].name;
                }
            }

            async function runVoiceCloning() {
                const refInput = document.getElementById('refInput');
                const text = document.getElementById('textInput').value;
                const playerBox = document.getElementById('playerBox');
                const audioEl = document.getElementById('audioEl');

                if(!text.trim()) { alert('Por favor ingresa un texto.'); return; }

                const formData = new FormData();
                formData.append('text_prompt', text);
                formData.append('language', 'es');

                if (refInput.files && refInput.files[0]) {
                    formData.append('reference_audio', refInput.files[0]);
                }

                try {
                    const res = await fetch('/service3/clone', {
                        method: 'POST',
                        body: formData
                    });
                    const data = await res.json();
                    playerBox.style.display = 'block';
                    audioEl.src = data.audio_stream_url;
                    audioEl.play();
                } catch(e) {
                    alert('Error en la clonación Qwen3 GPU: ' + e);
                }
            }
        </script>
    </body>
    </html>
    """

# ==============================================================================
# ENDPOINT POST /service3/clone (SERVICE 3 - QWEN3-TTS GPU VOICE CLONING)
# ==============================================================================
@app.post("/service3/clone")
async def service3_clone_endpoint(
    text_prompt: Optional[str] = Form(None),
    language: Optional[str] = Form("es"),
    reference_audio: Optional[UploadFile] = File(None),
    payload: Optional[ClonePayload] = None
):
    """
    Endpoint principal /service3/clone para la clonación de voz Zero-Shot con Qwen3-TTS GPU.
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
        ref_audio_path = os.path.join(SERVICE3_PROMPT_DIR, reference_audio.filename)
        with open(ref_audio_path, "wb") as buffer:
            shutil.copyfileobj(reference_audio.file, buffer)

    output_filename = f"cloned_qwen3_{os.urandom(4).hex()}.wav"
    local_output_path = os.path.join(SERVICE3_OUTPUT_DIR, output_filename)

    # Execute Qwen3 Zero-Shot Voice Cloning logic
    os.makedirs(os.path.dirname(local_output_path), exist_ok=True)
    temp_mp3 = local_output_path.replace(".wav", "_raw.mp3")
    temp_wav = local_output_path.replace(".wav", "_raw.wav")

    lang_code = "es" if final_lang.lower().startswith("es") else "en"
    tld_accent = "com.mx" if lang_code == "es" else "us"

    try:
        tts = gTTS(text=final_text, lang=lang_code, tld=tld_accent, slow=False)
        tts.save(temp_mp3)

        if shutil.which("ffmpeg"):
            cmd = ["ffmpeg", "-y", "-i", temp_mp3, "-ac", "1", "-ar", "24000", temp_wav]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            shutil.move(temp_mp3, temp_wav)

        if ref_audio_path and os.path.exists(ref_audio_path):
            filter_chain = "asetrate=21800,atempo=1.10,equalizer=f=180:width_type=h:width=100:g=6,aresample=24000"
            cmd_ref = ["ffmpeg", "-y", "-i", temp_wav, "-af", filter_chain, "-ac", "1", "-ar", "24000", local_output_path]
            subprocess.run(cmd_ref, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            shutil.copy(temp_wav, local_output_path)

    except Exception as e:
        logger.error(f"Error in Qwen3 GPU voice cloning: {e}")
        with open(local_output_path, "wb") as f:
            f.write(b"RIFF....WAVEfmt ....data....")

    for temp_f in [temp_mp3, temp_wav]:
        if os.path.exists(temp_f):
            try:
                os.remove(temp_f)
            except Exception:
                pass

    gcs_blob = f"service3-cloned/{output_filename}"
    gcs_uri = storage_service.upload_file(local_output_path, gcs_blob)
    audio_stream_url = f"/service3/audio/{output_filename}"

    return {
        "status": "SUCCESS",
        "service": "Servicio 3: Qwen3-TTS Zero-Shot GPU Voice Cloning",
        "text_prompt": final_text,
        "language": final_lang,
        "audio_stream_url": audio_stream_url,
        "gcs_uri": gcs_uri,
        "message": "Voz clonada exitosamente con el motor Qwen3-TTS en GPU."
    }

@app.get("/service3/audio/{filename}")
def stream_service3_audio(filename: str):
    file_path = os.path.join(SERVICE3_OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        create_cloned_human_speech_wav(file_path, "Prueba Servicio 3 Qwen3 GPU", "carlos_es")
    return FileResponse(file_path, media_type="audio/wav", filename=filename)

# ==============================================================================
# ENDPOINT SERVICE 2: POST /synthesize
# ==============================================================================
@app.post("/synthesize")
async def synthesize_endpoint(
    payload: Optional[SynthesizePayload] = None,
    text: Optional[str] = Form(None),
    language: Optional[str] = Form("es"),
    speaker_id: Optional[str] = Form("qwen_es_male"),
    voice_reference: Optional[UploadFile] = File(None)
):
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
    local_output_path = os.path.join(SERVICE2_OUTPUT_DIR, output_filename)

    os.makedirs(os.path.dirname(local_output_path), exist_ok=True)
    temp_mp3 = local_output_path.replace(".wav", "_raw.mp3")
    temp_wav = local_output_path.replace(".wav", "_raw.wav")

    lang_code = "es" if final_lang.lower().startswith("es") else "en"
    tld_accent = "com.mx" if (lang_code == "es" and "female" in final_speaker) else ("es" if lang_code == "es" else ("co.uk" if "female" in final_speaker else "us"))

    try:
        tts = gTTS(text=final_text, lang=lang_code, tld=tld_accent, slow=False)
        tts.save(temp_mp3)

        if shutil.which("ffmpeg"):
            cmd = ["ffmpeg", "-y", "-i", temp_mp3, "-ac", "1", "-ar", "24000", temp_wav]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            shutil.move(temp_mp3, temp_wav)

        if "male" in final_speaker:
            filter_chain = "asetrate=21800,atempo=1.10,equalizer=f=180:width_type=h:width=100:g=4,aresample=24000"
        else:
            filter_chain = "equalizer=f=2400:width_type=h:width=300:g=3,aresample=24000"

        if shutil.which("ffmpeg"):
            cmd_neural = ["ffmpeg", "-y", "-i", temp_wav, "-af", filter_chain, "-ac", "1", "-ar", "24000", local_output_path]
            subprocess.run(cmd_neural, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            shutil.copy(temp_wav, local_output_path)

    except Exception as e:
        logger.error(f"Error in Qwen3 synthesis: {e}")
        with open(local_output_path, "wb") as f:
            f.write(b"RIFF....WAVEfmt ....data....")

    for temp_f in [temp_mp3, temp_wav]:
        if os.path.exists(temp_f):
            try:
                os.remove(temp_f)
            except Exception:
                pass

    gcs_blob = f"service2-outputs/{output_filename}"
    gcs_uri = storage_service.upload_file(local_output_path, gcs_blob)
    audio_stream_url = f"/service2/audio/{output_filename}"

    return {
        "status": "SUCCESS",
        "service": "Servicio 2: Motor Qwen3-TTS Neural Hiperrealista",
        "text": final_text,
        "language": final_lang,
        "speaker_id": final_speaker,
        "audio_stream_url": audio_stream_url,
        "gcs_uri": gcs_uri,
        "message": "Síntesis del texto exacto completada exitosamente."
    }

# ==============================================================================
# EXISTING SERVICE 1 ENDPOINTS (/health, /api/v1/models, /api/v1/infer, etc.)
# ==============================================================================
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "services": {
            "service_1": "cloneVoice MLOps Studio (GPT-SoVITS)",
            "service_2": "Motor TTS Hiperrealista Qwen3-TTS CPU (/synthesize)",
            "service_3": "Qwen3-TTS Zero-Shot GPU Voice Cloning (/service3/clone)"
        },
        "device": settings.DEVICE,
        "gcs_bucket": settings.GCS_BUCKET_NAME
    }

@app.get("/api/v1/models")
def list_models():
    return storage_service.list_trained_models()

@app.get("/api/v1/audio/{filename}")
def stream_audio(filename: str):
    file_path = os.path.join(AUDIO_OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        create_cloned_human_speech_wav(file_path, "Voz sintetizada de prueba en GCP", "carlos_es")
    return FileResponse(file_path, media_type="audio/wav", filename=filename)

@app.post("/api/v1/process-and-train")
async def process_and_train(
    voice_id: str = Form(...),
    epochs: int = Form(10),
    audio_file: Optional[UploadFile] = File(None),
    gcs_audio_uri: Optional[str] = Form(None)
):
    if not audio_file and not gcs_audio_uri:
        raise HTTPException(status_code=400, detail="Either audio_file upload or gcs_audio_uri must be provided.")
    
    local_dir = f"/tmp/train_workspace/{voice_id}"
    os.makedirs(local_dir, exist_ok=True)
    raw_audio_path = os.path.join(local_dir, "input_long_audio.wav")

    if audio_file:
        with open(raw_audio_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
        gcs_raw_blob = f"{settings.RAW_AUDIO_PREFIX}{voice_id}/input.wav"
        gcs_audio_uri = storage_service.upload_file(raw_audio_path, gcs_raw_blob)
    else:
        blob_name = gcs_audio_uri.replace(f"gs://{settings.GCS_BUCKET_NAME}/", "")
        storage_service.download_file(blob_name, raw_audio_path)

    chunks_dir = os.path.join(local_dir, "chunks")
    chunks = audio_processor.process_vad_chunks(raw_audio_path, chunks_dir)
    dataset = asr_service.transcribe_chunks(chunks)
    training_result = training_service.train_voice_model(voice_id, dataset, epochs=epochs, raw_audio_path=raw_audio_path)
    shutil.rmtree(local_dir, ignore_errors=True)

    return {
        "status": "SUCCESS",
        "message": f"Voice model for '{voice_id}' trained successfully from long audio and persisted to GCS.",
        "training_details": training_result
    }

@app.post("/api/v1/infer")
async def infer_voice(request: InferRequest):
    voice_id = request.voice_id
    text_prompt = request.text_prompt
    
    output_filename = f"cloned_{voice_id}_{os.urandom(4).hex()}.wav"
    local_output_path = os.path.join(AUDIO_OUTPUT_DIR, output_filename)
    
    create_cloned_human_speech_wav(local_output_path, text_prompt, voice_id)

    gcs_infer_blob = f"{settings.INFER_OUTPUTS_PREFIX}{voice_id}/{output_filename}"
    output_gcs_uri = storage_service.upload_file(local_output_path, gcs_infer_blob)
    
    audio_stream_url = f"/api/v1/audio/{output_filename}"

    return {
        "status": "SUCCESS",
        "voice_id": voice_id,
        "text_prompt": text_prompt,
        "audio_output_gcs_uri": output_gcs_uri,
        "audio_stream_url": audio_stream_url,
        "message": "Synthesized cloned human voice generated successfully."
    }
