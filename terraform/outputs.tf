output "gcs_bucket_name" {
  value       = module.storage.bucket_name
  description = "GCS bucket name for audio datasets and trained models"
}

output "artifact_registry_repo" {
  value       = module.artifact_registry.repository_url
  description = "Artifact Registry Docker URL"
}

output "gpu_instance_ip" {
  value       = module.compute.instance_external_ip
  description = "Public IP address of the MLOps GPU node"
}

output "gpu_instance_name" {
  value       = module.compute.instance_name
  description = "Compute instance name"
}

output "fastapi_api_endpoint" {
  value       = "http://${module.compute.instance_external_ip}:8000"
  description = "Public endpoint for cloneVoice FastAPI service"
}
