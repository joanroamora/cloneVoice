output "instance_name" {
  value       = google_compute_instance_from_template.gpu_instance.name
  description = "Compute Engine instance name"
}

output "instance_external_ip" {
  value       = google_compute_instance_from_template.gpu_instance.network_interface[0].access_config[0].nat_ip
  description = "Public IP address of the GPU instance"
}

output "service_account_email" {
  value       = google_service_account.vm_sa.email
  description = "Email of the VM service account"
}
