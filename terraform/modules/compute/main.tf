# Service Account for GPU VM Instance
resource "google_service_account" "vm_sa" {
  account_id   = "clonevoice-gpu-vm-sa"
  display_name = "Service Account for cloneVoice GPU Instance"
  project      = var.project_id
}

# Grant GCS Storage Object Admin access to VM
resource "google_project_iam_member" "vm_storage_access" {
  project = var.project_id
  role    = "roles/storage.objectAdmin"
  member  = "serviceAccount:${google_service_account.vm_sa.email}"
}

# Grant Artifact Registry Reader access to VM
resource "google_project_iam_member" "vm_ar_access" {
  project = var.project_id
  role    = "roles/artifactregistry.reader"
  member  = "serviceAccount:${google_service_account.vm_sa.email}"
}

# Grant OS Config / VM Manager Admin permissions to VM
resource "google_project_iam_member" "vm_osconfig_access" {
  project = var.project_id
  role    = "roles/osconfig.osPolicyAssignmentAdmin"
  member  = "serviceAccount:${google_service_account.vm_sa.email}"
}

# GCP Instance Template for standardizing OS environment & MLOps setup
resource "google_compute_instance_template" "gpu_vm_template" {
  name_prefix  = "clone-voice-gpu-template-"
  description  = "Instance template for cloneVoice GPU MLOps node"
  machine_type = var.machine_type
  project      = var.project_id

  scheduling {
    automatic_restart   = var.preemptible_vm ? false : true
    on_host_maintenance = "TERMINATE" # Required for GPUs
    preemptible         = var.preemptible_vm
    provisioning_model  = var.preemptible_vm ? "SPOT" : "STANDARD"
  }

  disk {
    source_image = "deeplearning-platform-release/tf2-11-cu113-notebooks" # GCP Deep Learning VM image preconfigured with CUDA/Drivers
    auto_delete  = true
    boot         = true
    disk_size_gb = 100
    disk_type    = "pd-ssd"
  }

  guest_accelerator {
    type  = var.gpu_type
    count = var.gpu_count
  }

  network_interface {
    subnetwork = var.subnet_id
    access_config {
      // Ephemeral IP for outbound & inbound access
    }
  }

  metadata = {
    enable-osconfig         = "TRUE"
    enable-guest-attributes = "TRUE"
    startup-script = templatefile("${path.module}/scripts/startup.sh", {
      CONTAINER_IMAGE = var.container_image
      GCS_BUCKET_NAME = var.bucket_name
      GCP_PROJECT_ID  = var.project_id
      GCP_REGION      = var.region
    })
  }

  service_account {
    email  = google_service_account.vm_sa.email
    scopes = ["cloud-platform"]
  }

  labels = {
    environment = var.environment
    service     = "clone-voice-mlops"
    gpu         = "nvidia-l4"
  }

  lifecycle {
    create_before_destroy = true
  }
}

# Single Compute Engine GPU Instance provisioned on demand from Template
resource "google_compute_instance_from_template" "gpu_instance" {
  name                     = "clone-voice-gpu-node-${var.environment}"
  project                  = var.project_id
  zone                     = var.zone
  source_instance_template = google_compute_instance_template.gpu_vm_template.id

  tags = ["clone-voice-instance"]

  labels = {
    environment = var.environment
    managed_by  = "terraform"
  }
}
