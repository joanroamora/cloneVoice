# 🚀 Servicio 2: Motor TTS Hiperrealista Kokoro-82M (CPU Optimizado)

## 📌 Descripción General

El **Servicio 2** es una solución ligera e hiperrealista de síntesis de voz de código abierto (**Kokoro-82M ONNX**). Está diseñado específicamente para ejecutarse con **extrema naturalidad y bajo consumo de recursos** en instancias de desarrollo con **4 vCPUs y 16 GB de RAM** (`e2-standard-4` en GCP Compute Engine) sin requerir GPU obligatoria.

---

## 🛠️ Arquitectura y Gestión Eficiente de Memoria

- **Tamaño de Modelo**: 82M parámetros (~320 MB footprint).
- **Consumo de Memoria RAM**: < 1.2 GB (Garantiza cero saturación en la máquina de 16 GB RAM).
- **Optimización de CPU**: Ejecución basada en ONNX Runtime / PyTorch CPU wheel sin dependencias innecesarias de CUDA.
- **Formatos de Audio**: Genera salida en formato de audio estándar PCM `.wav` a 24kHz mono.

---

## 📂 Estructura del Código Fuente (`src/service2/`)

```
src/service2/
├── Dockerfile.service2         # Contenedor optimizado para CPU (Python 3.10 + PyTorch CPU)
├── requirements_service2.txt   # Dependencias exactas (FastAPI, ONNX, soundfile, PyTorch CPU)
├── tts_engine.py              # Motor Kokoro-82M de síntesis hiperrealista
└── main_service2.py           # Servicio FastAPI independiente en puerto 8001
```

---

## 🐳 Instrucciones de Despliegue con Docker (CPU Optimizado)

### 1. Construir la Imagen Docker para Servicio 2
```bash
cd /home/joanr/agentic-platforms/GCP/cloneVoice/src/service2
docker build -f Dockerfile.service2 -t kokoro-tts-service2:latest .
```

### 2. Levantar el Contenedor Docker en la Instancia GCP
```bash
docker run -d --name service2-kokoro \
  -p 8001:8001 \
  -e GCS_BUCKET_NAME="clone-voice-storage-bitcitychamp-project-8cc83f2c" \
  kokoro-tts-service2:latest
```

---

## 🧪 Pruebas de Sintetización (`/synthesize`) mediante `curl` o Python

### Opción A: Prueba HTTP POST JSON con `curl`
```bash
curl -X POST http://34.46.241.26:8000/synthesize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hola, este es una prueba del motor hiperrealista Kokoro en español.",
    "language": "es"
  }'
```

### Opción B: Prueba en Inglés con `curl`
```bash
curl -X POST http://34.46.241.26:8000/synthesize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello! This is a test of the hyper-realistic Kokoro speech synthesis engine.",
    "language": "en"
  }'
```

### Opción C: Prueba en Python con `requests`
```python
import requests

url = "http://34.46.241.26:8000/synthesize"
payload = {
    "text": "Demostración de voz hiperrealista Kokoro-82M en Google Cloud Platform.",
    "language": "es"
}

response = requests.post(url, json=payload)
data = response.json()
print("Respuesta de Sintetización:", data)
print("URL del Audio:", f"http://34.46.241.26:8000{data['audio_stream_url']}")
```
