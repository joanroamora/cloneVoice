resource "google_artifact_registry_repository" "repo" {
  location      = var.region
  repository_id = "clone-voice-repo"
  description   = "Docker repository for cloneVoice MLOps FastAPI service"
  format        = "DOCKER"
  project       = var.project_id

  labels = {
    environment = var.environment
    service     = "clone-voice-mlops"
  }
}
