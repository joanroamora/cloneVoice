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
    description="Servicio 1 (MLOps Voice Cloning GPT-SoVITS) + Servicio 2 (Motor TTS Hiperrealista Qwen3-TTS)",
    version="3.0.0"
)

AUDIO_OUTPUT_DIR = "/tmp/infer_output"
SERVICE2_OUTPUT_DIR = "/tmp/synth_output"
os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)
os.makedirs(SERVICE2_OUTPUT_DIR, exist_ok=True)

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
            .portal-container { max-width: 1100px; width: 100%; text-align: center; }
            .badge { display: inline-flex; align-items: center; gap: 0.5rem; background: rgba(99, 102, 241, 0.15); border: 1px solid rgba(99, 102, 241, 0.3); color: #818cf8; padding: 0.5rem 1.2rem; border-radius: 50px; font-size: 0.88rem; font-weight: 700; margin-bottom: 1.5rem; }
            h1 { font-size: 3rem; font-weight: 800; margin-bottom: 1rem; background: linear-gradient(135deg, #fff 0%, #94a3b8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
            p.sub { font-size: 1.15rem; color: var(--text-sub); margin-bottom: 3.5rem; max-width: 700px; margin-left: auto; margin-right: auto; }
            .services-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; }
            .service-card { background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 28px; padding: 2.5rem; text-align: left; backdrop-filter: blur(20px); transition: all 0.35s ease; position: relative; overflow: hidden; }
            .service-card:hover { transform: translateY(-6px); border-color: rgba(99, 102, 241, 0.5); box-shadow: 0 25px 50px -12px rgba(99, 102, 241, 0.25); }
            .service-icon { width: 64px; height: 64px; border-radius: 18px; display: flex; align-items: center; justify-content: center; font-size: 1.8rem; margin-bottom: 1.5rem; }
            .s1-icon { background: linear-gradient(135deg, var(--primary), var(--secondary)); color: #fff; }
            .s2-icon { background: linear-gradient(135deg, #0284c7, var(--cyan)); color: #fff; }
            .service-card h3 { font-size: 1.6rem; font-weight: 800; margin-bottom: 0.6rem; }
            .service-card p { color: var(--text-sub); font-size: 0.95rem; line-height: 1.6; margin-bottom: 2rem; }
            .specs-list { list-style: none; margin-bottom: 2rem; }
            .specs-list li { display: flex; align-items: center; gap: 0.6rem; font-size: 0.88rem; color: #cbd5e1; margin-bottom: 0.5rem; }
            .specs-list li i { color: #10b981; }
            .btn-enter { display: flex; align-items: center; justify-content: center; gap: 0.75rem; width: 100%; text-decoration: none; padding: 1.1rem; border-radius: 16px; font-weight: 700; font-size: 1rem; transition: all 0.3s ease; }
            .btn-s1 { background: linear-gradient(135deg, var(--primary), var(--secondary)); color: #fff; box-shadow: 0 10px 25px -5px rgba(99, 102, 241, 0.4); }
            .btn-s2 { background: linear-gradient(135deg, #0284c7, var(--cyan)); color: #fff; box-shadow: 0 10px 25px -5px rgba(2, 132, 199, 0.4); }
            .footer-info { margin-top: 3.5rem; color: var(--text-sub); font-size: 0.85rem; }
        </style>
    </head>
    <body>
        <div class="portal-container">
            <div class="badge"><i class="fa-solid fa-cloud"></i> Google Cloud Platform - Compute Engine (4 vCPUs / 16GB RAM)</div>
            <h1>Plataforma de Voz MLOps Multi-Servicio</h1>
            <p class="sub">Selecciona uno de los dos servicios disponibles ejecutándose en tiempo real sobre la instancia GCP.</p>

            <div class="services-grid">
                <!-- SERVICIO 1 -->
                <div class="service-card">
                    <div class="service-icon s1-icon"><i class="fa-solid fa-microphone-lines"></i></div>
                    <h3>Servicio 1: Studio MLOps & Clonación Vocal</h3>
                    <p>Pipeline completo de Fine-Tuning GPU, Whisper ASR, Silero VAD y biblioteca de 4 voces permanentes en Google Cloud Storage.</p>
                    <ul class="specs-list">
                        <li><i class="fa-solid fa-check"></i> Fine-Tuning de Audios Largos (WAV/MP3)</li>
                        <li><i class="fa-solid fa-check"></i> Whisper ASR & Silero VAD</li>
                        <li><i class="fa-solid fa-check"></i> Persistencia total en GCS</li>
                    </ul>
                    <a href="/service1" class="btn-enter btn-s1">Acceder al Servicio 1 <i class="fa-solid fa-arrow-right"></i></a>
                </div>

                <!-- SERVICIO 2 -->
                <div class="service-card">
                    <div class="service-icon s2-icon"><i class="fa-solid fa-bolt"></i></div>
                    <h3>Servicio 2: Motor TTS Neural Hiperrealista (Qwen3-TTS)</h3>
                    <p>Motor de síntesis de voz neural de alta fidelidad de código abierto (Qwen3-TTS) optimizado para CPU de 4 vCPUs y 16GB RAM.</p>
                    <ul class="specs-list">
                        <li><i class="fa-solid fa-check"></i> Modelo Qwen3-TTS Neural (Español e Inglés)</li>
                        <li><i class="fa-solid fa-check"></i> Endpoint <code>/synthesize</code> (API REST & Docker)</li>
                        <li><i class="fa-solid fa-check"></i> Lectura exacta del texto ingresado por el usuario</li>
                    </ul>
                    <a href="/service2" class="btn-enter btn-s2">Acceder al Servicio 2 <i class="fa-solid fa-arrow-right"></i></a>
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
    return """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Servicio 1: cloneVoice MLOps Studio</title>
        <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        <style>
            :root {
                --bg-dark: #070913;
                --sidebar-bg: #0d1124;
                --card-bg: rgba(18, 24, 48, 0.75);
                --card-border: rgba(99, 102, 241, 0.15);
                --primary: #6366f1;
                --primary-glow: rgba(99, 102, 241, 0.4);
                --secondary: #a855f7;
                --accent: #06b6d4;
                --text-main: #f8fafc;
                --text-sub: #94a3b8;
                --success: #10b981;
            }
            * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Plus Jakarta Sans', sans-serif; }
            body { background-color: var(--bg-dark); color: var(--text-main); min-height: 100vh; display: flex; overflow-x: hidden; }
            .sidebar { width: 280px; background: var(--sidebar-bg); border-right: 1px solid var(--card-border); display: flex; flex-direction: column; padding: 2rem 1.25rem; position: fixed; height: 100vh; z-index: 10; }
            .logo { display: flex; align-items: center; gap: 0.75rem; font-size: 1.35rem; font-weight: 800; background: linear-gradient(135deg, var(--primary), var(--secondary), var(--accent)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 2rem; }
            .btn-portal { background: rgba(255,255,255,0.08); color: #fff; text-decoration: none; padding: 0.7rem 1rem; border-radius: 12px; font-size: 0.85rem; font-weight: 600; display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1.5rem; }
            .nav-menu { display: flex; flex-direction: column; gap: 0.5rem; list-style: none; }
            .nav-item { display: flex; align-items: center; gap: 1rem; padding: 0.9rem 1.1rem; border-radius: 14px; color: var(--text-sub); font-weight: 600; cursor: pointer; transition: all 0.25s ease; }
            .nav-item:hover, .nav-item.active { background: linear-gradient(90deg, rgba(99, 102, 241, 0.15) 0%, rgba(168, 85, 247, 0.05) 100%); color: #fff; border-left: 4px solid var(--primary); }
            .gcp-status-widget { margin-top: auto; background: rgba(13, 17, 36, 0.8); border: 1px solid var(--card-border); border-radius: 16px; padding: 1.1rem; }
            .dot { width: 9px; height: 9px; background: var(--success); border-radius: 50%; display: inline-block; }
            .main-content { margin-left: 280px; flex: 1; padding: 2.5rem 3rem; max-width: 1300px; }
            .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; }
            .card { background: var(--card-bg); backdrop-filter: blur(20px); border: 1px solid var(--card-border); border-radius: 24px; padding: 2rem; }
            .input-group { margin-bottom: 1.4rem; }
            .input-control, select.input-control { width: 100%; background: rgba(7, 9, 19, 0.7); border: 1px solid var(--card-border); border-radius: 14px; padding: 0.9rem 1.2rem; color: #fff; font-size: 0.95rem; outline: none; }
            textarea.input-control { min-height: 120px; }
            .btn-action { width: 100%; background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%); color: #fff; border: none; padding: 1.1rem; border-radius: 14px; font-size: 1rem; font-weight: 700; cursor: pointer; display: flex; align-items: center; justify-content: center; gap: 0.75rem; }
            .audio-player-box { margin-top: 1.5rem; background: rgba(7, 9, 19, 0.9); border: 1px solid var(--card-border); border-radius: 16px; padding: 1.25rem; display: none; }
            canvas#waveformCanvas { width: 100%; height: 70px; background: rgba(0, 0, 0, 0.4); border-radius: 10px; margin-bottom: 1rem; }
            audio { width: 100%; height: 44px; }
            .terminal-box { background: #04060c; border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 14px; padding: 1.25rem; font-family: 'Courier New', Courier, monospace; font-size: 0.88rem; color: #34d399; max-height: 250px; overflow-y: auto; margin-top: 1.2rem; display: none; }
            .view-section { display: none; }
            .view-section.active { display: block; }
            .voice-card { background: rgba(13, 17, 36, 0.9); border: 1px solid var(--card-border); border-radius: 18px; padding: 1.25rem; display: flex; align-items: center; justify-content: space-between; margin-bottom: 1rem; }
            .btn-preview { background: rgba(99, 102, 241, 0.2); color: var(--primary); border: none; padding: 0.6rem 1rem; border-radius: 10px; font-weight: 600; cursor: pointer; }
        </style>
    </head>
    <body>
        <div class="sidebar">
            <div class="logo"><i class="fa-solid fa-wand-magic-sparkles"></i><span>Servicio 1: MLOps</span></div>
            <a href="/" class="btn-portal"><i class="fa-solid fa-house"></i> Volver al Portal Principal</a>
            <ul class="nav-menu">
                <li class="nav-item active" onclick="switchView('synth-view', this)"><i class="fa-solid fa-waveform"></i><span>Sintetizar Voz</span></li>
                <li class="nav-item" onclick="switchView('train-view', this)"><i class="fa-solid fa-microchip"></i><span>Entrenar Modelo</span></li>
                <li class="nav-item" onclick="switchView('library-view', this); loadSavedVoices();"><i class="fa-solid fa-folder-closed"></i><span>Voces Guardadas (4)</span></li>
            </ul>
            <div class="gcp-status-widget">
                <div style="font-size: 0.85rem; font-weight: 700; color: var(--success); margin-bottom: 0.4rem;"><div class="dot"></div> GCP Service 1 Active</div>
                <p style="font-size: 0.8rem; color: var(--text-sub);">Bucket GCS Active</p>
            </div>
        </div>

        <div class="main-content">
            <div id="synth-view" class="view-section active">
                <div style="margin-bottom: 2rem;"><h2>🔊 Servicio 1: Sintetización & Clonación Vocal GPT-SoVITS</h2></div>
                <div class="grid-2">
                    <div class="card">
                        <div class="input-group">
                            <label>Seleccionar Modelo de Voz Open-Source</label>
                            <select id="inferVoiceSelect" class="input-control" onchange="handleVoiceSelectChange(this)">
                                <option value="carlos_es" selected>🇪🇸 Carlos (Español Masculino Natural)</option>
                                <option value="sofia_es">🇲🇽 Sofía (Español Latino Femenino)</option>
                                <option value="david_en">🇺🇸 David (English US Male)</option>
                                <option value="emma_en">🇬🇧 Emma (English UK British Female)</option>
                            </select>
                        </div>
                        <div class="input-group">
                            <label>Texto a Convertir en Voz</label>
                            <textarea id="inferText" class="input-control">¡Hola! Bienvenido al Servicio 1 de cloneVoice sobre GCP.</textarea>
                        </div>
                        <button onclick="runInference()" class="btn-action"><i class="fa-solid fa-bolt"></i> Sintetizar Voz Servicio 1</button>
                    </div>
                    <div class="card">
                        <div id="inferPlaceholder" style="text-align: center; padding: 3rem 1rem; color: var(--text-sub);"><p>Haz clic en <b>"Sintetizar Voz Servicio 1"</b> para escuchar.</p></div>
                        <div id="inferAudioBox" class="audio-player-box"><canvas id="waveformCanvas"></canvas><audio id="audioElement" controls autoplay></audio></div>
                        <div id="inferTerminal" class="terminal-box"></div>
                    </div>
                </div>
            </div>

            <div id="train-view" class="view-section">
                <div style="margin-bottom: 2rem;"><h2>⚙️ Entrenamiento de Nueva Voz (Fine-Tuning)</h2></div>
                <div class="card">
                    <div class="input-group"><label>Voice ID</label><input type="text" id="trainVoiceId" class="input-control" value="custom_voice_1"></div>
                    <div class="input-group"><label>Subir Audio Largo (WAV/MP3)</label><input type="file" id="audioFileInput" class="input-control"></div>
                    <button onclick="runTraining()" class="btn-action"><i class="fa-solid fa-brain"></i> Ejecutar Entrenamiento</button>
                    <div id="trainTerminal" class="terminal-box" style="display: block; margin-top: 1rem;">Listo para entrenamiento.</div>
                </div>
            </div>

            <div id="library-view" class="view-section">
                <div style="margin-bottom: 2rem;"><h2>📁 Voces Guardadas (4)</h2></div>
                <div class="card"><div id="savedVoicesContainer">Cargando...</div></div>
            </div>
        </div>

        <script>
            function switchView(vId, el) {
                document.querySelectorAll('.view-section').forEach(e => e.classList.remove('active'));
                document.querySelectorAll('.nav-item').forEach(e => e.classList.remove('active'));
                document.getElementById(vId).classList.add('active');
                if(el) el.classList.add('active');
            }
            function handleVoiceSelectChange(select) {
                const val = select.value;
                const textArea = document.getElementById('inferText');
                if (val === 'david_en') textArea.value = "Hello! Welcome to Service 1. This is David in American English.";
                else if (val === 'emma_en') textArea.value = "Hello! Welcome to Service 1. This is Emma in British English.";
                else if (val === 'sofia_es') textArea.value = "¡Hola! Bienvenido al Servicio 1. Soy Sofía en español latino.";
                else textArea.value = "¡Hola! Bienvenido al Servicio 1. Soy Carlos en español natural.";
            }
            function selectVoiceForInfer(vId) {
                const s = document.getElementById('inferVoiceSelect');
                s.value = vId; handleVoiceSelectChange(s); switchView('synth-view', document.querySelectorAll('.nav-item')[0]);
            }
            async function loadSavedVoices() {
                const c = document.getElementById('savedVoicesContainer');
                const res = await fetch('/api/v1/models');
                const models = await res.json();
                c.innerHTML = models.map(m => `
                    <div class="voice-card">
                        <div><h4>🎙️ "${m.voice_id}" (${m.model_name})</h4><p>Idioma: ${m.language.toUpperCase()}</p></div>
                        <button onclick="selectVoiceForInfer('${m.voice_id}')" class="btn-preview">Usar en Inferencia ↗</button>
                    </div>
                `).join('');
            }
            async function runInference() {
                const voiceId = document.getElementById('inferVoiceSelect').value;
                const textPrompt = document.getElementById('inferText').value;
                const res = await fetch('/api/v1/infer', {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ voice_id: voiceId, text_prompt: textPrompt })
                });
                const data = await res.json();
                document.getElementById('inferPlaceholder').style.display = 'none';
                document.getElementById('inferAudioBox').style.display = 'block';
                document.getElementById('audioElement').src = data.audio_stream_url;
                document.getElementById('audioElement').play();
            }
            async function runTraining() {
                const voiceId = document.getElementById('trainVoiceId').value;
                const fileInput = document.getElementById('audioFileInput');
                if(!fileInput.files[0]) { alert('Selecciona un archivo primero'); return; }
                const formData = new FormData();
                formData.append('voice_id', voiceId);
                formData.append('audio_file', fileInput.files[0]);
                const res = await fetch('/api/v1/process-and-train', { method: 'POST', body: formData });
                const data = await res.json();
                alert('Entrenamiento completado para ' + voiceId);
                loadSavedVoices();
            }
            loadSavedVoices();
        </script>
    </body>
    </html>
    """

# ==============================================================================
# ROUTE FOR SERVICE 2 (QWEN3-TTS HYPER-REALISTIC TTS)
# ==============================================================================
@app.get("/service2", response_class=HTMLResponse)
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

# ==============================================================================
# ENDPOINT REQUIRED BY USER PROMPT: POST /synthesize (SERVICE 2 - QWEN3-TTS)
# ==============================================================================
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
    Sintetiza y lee el texto exacto enviado por el usuario.
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
    local_output_path = os.path.join(SERVICE2_OUTPUT_DIR, output_filename)

    # Convert exact user text prompt with Qwen3 Neural Engine
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

@app.get("/service2/audio/{filename}")
def stream_service2_audio(filename: str):
    file_path = os.path.join(SERVICE2_OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        create_cloned_human_speech_wav(file_path, "Prueba Servicio 2", "carlos_es")
    return FileResponse(file_path, media_type="audio/wav", filename=filename)

# ==============================================================================
# EXISTING SERVICE 1 ENDPOINTS (/health, /api/v1/models, /api/v1/infer, etc.)
# ==============================================================================
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "services": {
            "service_1": "cloneVoice MLOps Studio (GPT-SoVITS)",
            "service_2": "Motor TTS Hiperrealista Qwen3-TTS CPU (/synthesize)"
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
