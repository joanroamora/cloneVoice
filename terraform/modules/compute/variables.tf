variable "project_id" {
  type        = string
  description = "GCP Project ID"
}

variable "region" {
  type        = string
  description = "GCP Region"
}

variable "zone" {
  type        = string
  description = "GCP Zone"
}

variable "environment" {
  type        = string
  description = "Environment name"
}

variable "subnet_id" {
  type        = string
  description = "VPC Subnet ID"
}

variable "bucket_name" {
  type        = string
  description = "GCS Storage Bucket Name"
}

variable "machine_type" {
  type        = string
  default     = "g2-standard-8"
  description = "Machine type with GPU support"
}

variable "gpu_type" {
  type        = string
  default     = "nvidia-l4"
  description = "GPU Accelerator type"
}

variable "gpu_count" {
  type        = number
  default     = 1
  description = "Number of GPUs"
}

variable "preemptible_vm" {
  type        = bool
  default     = true
  description = "Use spot/preemptible instance for cost optimization"
}

variable "container_image" {
  type        = string
  description = "Container image URI in Artifact Registry"
}
