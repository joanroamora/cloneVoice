#!/bin/bash
set -euo pipefail

# Log execution
exec > >(tee /var/log/startup-script.log|logger -t startup-script -s 2>/dev/console) 2>&1

echo "========================================="
echo "Starting cloneVoice MLOps GPU VM Provisioning"
echo "========================================="

# 1. Enable GCP VM Manager / OS Config Agent
systemctl enable google-osconfig-agent || true
systemctl start google-osconfig-agent || true

# 2. Install Docker if not present
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    apt-get update
    apt-get install -y apt-transport-https ca-certificates curl gnupg lsb-release
    mkdir -p /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    echo "deb [arch=$$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $$(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
    apt-get update
    apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
fi

# 3. Install NVIDIA Drivers and Container Toolkit if GPU is detected
if command -v lspci &> /dev/null && lspci | grep -i nvidia; then
    echo "NVIDIA GPU Detected. Setting up CUDA & Container Toolkit..."
    if ! command -v nvidia-smi &> /dev/null; then
        apt-get update
        apt-get install -y linux-headers-$$(uname -r)
        apt-get install -y nvidia-driver-535 nvidia-dkms-535 || true
    fi

    if ! dpkg -l | grep -q nvidia-container-toolkit; then
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
        curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
          sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
          tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
        apt-get update
        apt-get install -y nvidia-container-toolkit
        nvidia-ctk runtime configure --runtime=docker
        systemctl restart docker
    fi
fi

# 4. Authenticate Docker with GCP Artifact Registry via Service Account Metadata
gcloud auth configure-docker ${GCP_REGION}-docker.pkg.dev --quiet || true

# 5. Run cloneVoice MLOps Service Container
CONTAINER_NAME="clonevoice-mlops-api"
IMAGE_URI="${CONTAINER_IMAGE}"
GCS_BUCKET="${GCS_BUCKET_NAME}"

echo "Pulling container image: $${IMAGE_URI}"
docker pull "$${IMAGE_URI}" || echo "Warning: Could not pull container image immediately, will retry if exists local fallback."

docker stop "$${CONTAINER_NAME}" || true
docker rm "$${CONTAINER_NAME}" || true

echo "Starting cloneVoice container..."
docker run -d \
    --name "$${CONTAINER_NAME}" \
    --restart always \
    --gpus all \
    -p 8000:8000 \
    -e GCS_BUCKET_NAME="$${GCS_BUCKET}" \
    -e GCP_PROJECT_ID="${GCP_PROJECT_ID}" \
    -e PORT=8000 \
    "$${IMAGE_URI}"

echo "========================================="
echo "cloneVoice GPU VM Provisioning Completed Successfully!"
echo "========================================="
