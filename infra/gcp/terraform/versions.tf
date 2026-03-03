terraform {
  required_version = ">= 1.14.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 7.0"
    }
  }

  # Remote state backend using GCS
  # Uncomment and configure when GCS bucket is created
  # backend "gcs" {
  #   bucket = "miniflow-terraform-state"
  #   prefix = "staging"
  # }
}
