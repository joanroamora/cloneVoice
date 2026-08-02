resource "random_id" "bucket_suffix" {
  byte_length = 4
}

resource "google_storage_bucket" "mlops_storage" {
  name                     = "clone-voice-storage-${var.project_id}-${random_id.bucket_suffix.hex}"
  location                 = var.region
  project                  = var.project_id
  force_destroy            = true
  uniform_bucket_level_access = true

  storage_class = "STANDARD"

  versioning {
    enabled = true
  }

  lifecycle_rule {
    action {
      type = "Delete"
    }
    condition {
      age = 30 # Delete temporary processed chunks older than 30 days
      with_state = "ANY"
      matches_prefix = ["processed-chunks/"]
    }
  }

  labels = {
    environment = var.environment
    service     = "clone-voice-mlops"
  }
}

# Create folder structure markers in GCS
resource "google_storage_bucket_object" "folder_raw_audio" {
  name    = "raw-audio/"
  content = " "
  bucket  = google_storage_bucket.mlops_storage.name
}

resource "google_storage_bucket_object" "folder_processed_chunks" {
  name    = "processed-chunks/"
  content = " "
  bucket  = google_storage_bucket.mlops_storage.name
}

resource "google_storage_bucket_object" "folder_trained_models" {
  name    = "trained-models/"
  content = " "
  bucket  = google_storage_bucket.mlops_storage.name
}

resource "google_storage_bucket_object" "folder_infer_outputs" {
  name    = "infer-outputs/"
  content = " "
  bucket  = google_storage_bucket.mlops_storage.name
}
