terraform {
  required_version = ">= 1.5.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.5"
    }
  }
  # Backend config can be unlocked by providing backend.tf or -backend-config in GitHub Actions
}

provider "google" {
  project = var.gcp_project_id
  region  = var.gcp_region
  zone    = var.gcp_zone
}

# Module 1: Storage (GCS Bucket with lifecycle rules and folder structure)
module "storage" {
  source      = "./modules/storage"
  project_id  = var.gcp_project_id
  region      = var.gcp_region
  environment = var.environment
}

# Module 2: Artifact Registry (Container Registry for Docker GPU images)
module "artifact_registry" {
  source      = "./modules/artifact_registry"
  project_id  = var.gcp_project_id
  region      = var.gcp_region
  environment = var.environment
}

# Module 3: Network (VPC, Subnet, Firewall rules)
module "network" {
  source      = "./modules/network"
  project_id  = var.gcp_project_id
  region      = var.gcp_region
  environment = var.environment
}

# Module 4: Compute Engine GPU Node (Instance Template + VM Manager)
module "compute" {
  source          = "./modules/compute"
  project_id      = var.gcp_project_id
  region          = var.gcp_region
  zone            = var.gcp_zone
  environment     = var.environment
  subnet_id       = module.network.subnet_id
  bucket_name     = module.storage.bucket_name
  machine_type    = var.machine_type
  gpu_type        = var.gpu_type
  gpu_count       = var.gpu_count
  preemptible_vm  = var.preemptible_vm
  container_image = var.container_image

  depends_on = [module.network, module.storage, module.artifact_registry]
}
