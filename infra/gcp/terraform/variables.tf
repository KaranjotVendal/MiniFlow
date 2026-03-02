variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "container_image" {
  description = "Container image URL"
  type        = string
}

variable "service_name" {
  description = "Name of the Cloud Run service"
  type        = string
  default     = "miniflow"
}

variable "cpu" {
  description = "CPU allocation (1 or 2)"
  type        = string
  default     = "2"
}

variable "memory" {
  description = "Memory allocation (e.g., 4Gi, 8Gi)"
  type        = string
  default     = "4Gi"
}

variable "miniflow_config" {
  description = "Path to MiniFlow config file"
  type        = string
  default     = "configs/3_TTS-to-vibevoice.yml"
}

variable "miniflow_device" {
  description = "Device for inference (cpu or cuda)"
  type        = string
  default     = "cpu"
}

variable "release_id" {
  description = "Release identifier"
  type        = string
  default     = "gcp-staging"
}
