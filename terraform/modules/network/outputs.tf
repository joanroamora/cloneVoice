output "network_name" {
  value       = google_compute_network.vpc_network.name
  description = "VPC Network Name"
}

output "subnet_name" {
  value       = google_compute_subnetwork.subnet.name
  description = "Subnet Name"
}

output "subnet_id" {
  value       = google_compute_subnetwork.subnet.id
  description = "Subnet ID"
}
