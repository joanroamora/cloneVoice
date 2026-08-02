variable "gcp_project_id" {
  description = "The GCP Project ID"
  type        = string
  default     = "clonevoice"
}

variable "gcp_region" {
  description = "The GCP region for resources"
  type        = string
  default     = "us-central1"
}

variable "gcp_zone" {
  description = "The GCP zone for the Compute Engine instance (must support NVIDIA L4 - g2-standard-8)"
  type        = string
  default     = "us-central1-a"
}

variable "environment" {
  description = "Deployment environment (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "gpu_type" {
  description = "GPU Accelerator type for NVIDIA L4"
  type        = string
  default     = "nvidia-l4"
}

variable "gpu_count" {
  description = "Number of GPUs attached to the VM"
  type        = number
  default     = 1
}

variable "machine_type" {
  description = "Machine type optimized for GPU"
  type        = string
  default     = "g2-standard-8"
}

variable "preemptible_vm" {
  description = "Whether to use preemptible/spot VM to drastically lower costs"
  type        = bool
  default     = true
}

variable "container_image" {
  description = "Fully qualified Docker image URI in GAR"
  type        = string
  default     = "us-central1-docker.pkg.dev/clonevoice/clone-voice-repo/clone-voice-mlops:latest"
}
