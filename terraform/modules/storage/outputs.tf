output "bucket_name" {
  value       = google_storage_bucket.mlops_storage.name
  description = "The name of the GCS bucket for MLOps artifacts"
}

output "bucket_url" {
  value       = google_storage_bucket.mlops_storage.url
  description = "The GCS URL of the bucket"
}
