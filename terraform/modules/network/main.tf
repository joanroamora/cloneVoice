resource "google_compute_network" "vpc_network" {
  name                    = "clone-voice-vpc-${var.environment}"
  auto_create_subnetworks = false
  project                 = var.project_id
}

resource "google_compute_subnetwork" "subnet" {
  name          = "clone-voice-subnet-${var.region}"
  ip_cidr_range = "10.0.1.0/24"
  region        = var.region
  network       = google_compute_network.vpc_network.id
  project       = var.project_id
}

# Firewall rule for SSH (Port 22)
resource "google_compute_firewall" "allow_ssh" {
  name    = "clone-voice-allow-ssh"
  network = google_compute_network.vpc_network.name
  project = var.project_id

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["clone-voice-instance"]
}

# Firewall rule for FastAPI MLOps service (Port 8000 & 80)
resource "google_compute_firewall" "allow_fastapi" {
  name    = "clone-voice-allow-fastapi"
  network = google_compute_network.vpc_network.name
  project = var.project_id

  allow {
    protocol = "tcp"
    ports    = ["8000", "80", "443"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["clone-voice-instance"]
}
