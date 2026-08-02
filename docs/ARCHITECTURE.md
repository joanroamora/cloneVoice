# 🏗️ cloneVoice Architecture & Technical Specification

## 🌟 Executive Overview
**cloneVoice** is an enterprise-grade, cost-optimized MLOps platform engineered on **Google Cloud Platform (GCP)** for high-fidelity open-source voice cloning (utilizing **GPT-SoVITS**, **Whisper ASR**, and **Silero VAD**).

The architecture adheres strictly to zero-idle cost practices through ephemeral Compute Engine GPU nodes (`g2-standard-8` with 1x NVIDIA L4), modular Terraform Infrastructure as Code (IaC), Google Cloud Storage (GCS) model versioning, and automated GitHub Actions CI/CD.

---

## 📐 System Architecture Diagram

```mermaid
graph TD
    subgraph Client & CI/CD Layer
        GA["GitHub Actions (CI/CD)"]
        Client["Client / Application"]
    end

    subgraph GCP Management & Container Services
        GAR["Google Artifact Registry<br/>(clone-voice-repo)"]
        VPC["Google VPC Network"]
    end

    subgraph GCP Compute Infrastructure
        Template["Instance Template<br/>(VM Manager / Deep Learning VM)"]
        GPU_VM["Compute Engine GPU Node<br/>(g2-standard-8 + NVIDIA L4)"]
        API["FastAPI MLOps Container<br/>(Port 8000)"]
    end

    subgraph MLOps Pipeline inside GPU Node
        VAD["Silero VAD<br/>(Voice Activity Detection)"]
        Whisper["Whisper ASR<br/>(Automatic Speech Recognition)"]
        GPU_Train["GPT-SoVITS GPU Fine-Tuning<br/>(CUDA 12.1)"]
        InferEngine["GPT-SoVITS Inference Engine"]
    end

    subgraph GCP Storage & Versioning
        GCS["Google Cloud Storage Bucket"]
        FolderRaw["/raw-audio/{voice_id}"]
        FolderProc["/processed-chunks/{voice_id}"]
        FolderModels["/trained-models/{voice_id} (.ckpt / .safetensors)"]
        FolderOutputs["/infer-outputs/{voice_id}"]
    end

    GA -->|Build & Push Container| GAR
    GA -->|Terraform Provisioning| Template
    Template -->|Spawns Instance| GPU_VM
    GAR -->|Pulls Image| GPU_VM
    GPU_VM --> API

    Client -->|1. Upload Raw Audio / Request Training| API
    API -->|Save Raw Audio| FolderRaw
    FolderRaw -->|Fetch Audio| VAD
    VAD -->|Segment Audio| FolderProc
    FolderProc -->|Transcribe| Whisper
    Whisper -->|Dataset + Audio| GPU_Train
    GPU_Train -->|Persist Checkpoints| FolderModels
    GCS --> FolderModels

    Client -->|2. Request Inference (Text + Voice ID)| API
    FolderModels -->|Load .ckpt| InferEngine
    InferEngine -->|Generate Audio| FolderOutputs
    FolderOutputs -->|Return Audio Link| Client
```

---

## 🔄 End-to-End Data & Audio Pipeline Execution

### 1. Ingestion (`raw-audio/`)
- The user or external service sends a long audio sample via `POST /api/v1/process-and-train`.
- The FastAPI service streams the file directly to GCS at `gs://<bucket>/raw-audio/{voice_id}/input.wav`.

### 2. Preprocessing & VAD Chunking (`processed-chunks/`)
- The raw audio is processed through **Silero VAD** to detect active speech boundaries, stripping silence and noise.
- Audio is split into optimal 3 to 10-second segments, normalized to 24kHz/44.1kHz mono WAV format, and indexed.

### 3. Automatic Transcription (`ASR - Whisper`)
- Each chunk is ingested by OpenAI **Whisper** running locally on the GPU to generate high-accuracy reference text scripts.

### 4. Fine-Tuning & GPU Execution (`trained-models/`)
- The prepared dataset (audio chunks + transcriptions) is passed into the **GPT-SoVITS** training loop.
- PyTorch executes gradient updates using CUDA acceleration on the NVIDIA L4 GPU.
- Upon completion, the model exports `.ckpt` and `.safetensors` files containing fine-tuned acoustic text-to-speech weights.
- These files are immediately uploaded to `gs://<bucket>/trained-models/{voice_id}/`.

### 5. High-Fidelity Inference (`infer-outputs/`)
- Clients call `POST /api/v1/infer` with target text and `voice_id`.
- The service retrieves the weights from GCS (or local cache), synthesizes natural voice audio, and returns the GCS output link.

---

## 💰 Cost Optimization & Strategy
1. **Compute Engine Spot / Preemptible VMs**: Uses `g2-standard-8` (NVIDIA L4) configured with `preemptible = true` and `provisioning_model = "SPOT"`, providing up to **60-70% cost savings** over standard instances.
2. **Auto-deleting Temporary Data**: GCS lifecycle rules automatically delete `processed-chunks/` after 30 days.
3. **No Idle Cluster Overhead**: Avoids GKE/Kubernetes control plane fees by using standard GCP Instance Templates & single GPU Compute instances.
