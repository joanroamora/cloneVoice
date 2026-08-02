output "repository_id" {
  value       = google_artifact_registry_repository.repo.repository_id
  description = "The ID of the Artifact Registry repository"
}

output "repository_url" {
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.repo.repository_id}"
  description = "The full URL of the Docker repository"
}
