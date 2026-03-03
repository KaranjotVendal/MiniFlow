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

# Deployment Configuration
variable "deployment_type" {
  description = "Deployment type: cpu (Cloud Run), gpu (Compute Engine), auto (GPU if available else CPU)"
  type        = string
  default     = "cpu"
}

# GPU Configuration
variable "gpu_zone" {
  description = "Zone with available GPU capacity"
  type        = string
  default     = "us-central1-b"
}

variable "gpu_type" {
  description = "GPU accelerator type (nvidia-tesla-p100, nvidia-tesla-t4, etc.)"
  type        = string
  default     = "nvidia-tesla-p100"
}

variable "gpu_count" {
  description = "Number of GPUs to attach"
  type        = number
  default     = 1
}

# Alert Configuration
variable "alert_email" {
  description = "Email address for budget alerts"
  type        = string
  default     = "karanjotgharu60@gmail.com"
}

variable "billing_account_id" {
  description = "GCP billing account ID"
  type        = string
  default     = "01413B-39FB78-6C67C1"
}
