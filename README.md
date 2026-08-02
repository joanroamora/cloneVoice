# 🎙️ cloneVoice - GCP MLOps Voice Cloning Platform

High fidelity open-source Voice Cloning infrastructure (GPT-SoVITS + Whisper ASR + Silero VAD) provisioned on Google Cloud Platform with Terraform & GitHub Actions.

---

## 📁 Repository Architecture & Layout

```
cloneVoice/
├── .github/
│   └── workflows/
│       └── deploy.yml            # CI/CD Pipeline (Docker build -> GAR -> Terraform apply)
├── terraform/
│   ├── main.tf                   # Root Terraform configuration
│   ├── variables.tf              # Global GCP & VM variables
│   ├── outputs.tf                # Service endpoints & IP outputs
│   └── modules/
│       ├── storage/              # GCS Bucket with structured folders & lifecycle rules
│       ├── artifact_registry/   # GCP Docker Registry for GPU images
│       ├── network/              # VPC, Subnetwork & Firewall rules
│       └── compute/              # GPU VM (g2-standard-8 + 1x NVIDIA L4) & Instance Template
├── src/
│   ├── Dockerfile                # Multi-stage Docker image with CUDA 12.1 + ffmpeg
│   ├── requirements.txt          # Python dependencies (FastAPI, PyTorch, Whisper)
│   └── app/                      # FastAPI MLOps Application
│       ├── main.py               # API Endpoints (/process-and-train, /infer)
│       └── services/             # Storage, Audio Processor (VAD), ASR & Training services
└── docs/
    └── ARCHITECTURE.md           # Technical Architecture & Pipeline Specification
```

---

## 🔑 GCP Authentication Setup

### 1. Locally with `gcloud` CLI
Run the following commands in your terminal:
```bash
# Login to GCP
gcloud auth login
gcloud auth application-default login

# Set active project
gcloud config set project clonevoice
```

### 2. Creating a Service Account Key File
If you require a local JSON credentials key file:
```bash
# Create Service Account
gcloud iam service-accounts create clonevoice-sa --display-name="cloneVoice SA"

# Grant Owner / Editor role
gcloud projects add-iam-policy-binding clonevoice \
  --member="serviceAccount:clonevoice-sa@clonevoice.iam.gserviceaccount.com" \
  --role="roles/owner"

# Create JSON Key
gcloud iam service-accounts keys create ./gcp-key.json \
  --iam-account=clonevoice-sa@clonevoice.iam.gserviceaccount.com
```
> ⚠️ `gcp-key.json` and `*.json` files are automatically ignored in [`.gitignore`](.gitignore).

---

## 🚀 Quick Deployment Guide

### Local Terraform Execution
```bash
cd terraform
cp terraform.tfvars.example terraform.tfvars
terraform init
terraform plan
terraform apply
```

### API Endpoints
- **Health Check**: `GET http://<GPU_INSTANCE_IP>:8000/health`
- **Train & Process**: `POST http://<GPU_INSTANCE_IP>:8000/api/v1/process-and-train`
- **Infer Cloned Voice**: `POST http://<GPU_INSTANCE_IP>:8000/api/v1/infer`

---

## 📄 Documentation
For detailed architectural diagrams and step-by-step pipeline flows, refer to [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).
