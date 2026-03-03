# Cloud Run outputs (only when CPU deployment)
output "service_url" {
  description = "URL of the deployed Cloud Run service"
  value       = try(google_cloud_run_service.miniflow[0].status[0].url, null)
}

output "service_name" {
  description = "Name of the Cloud Run service"
  value       = try(google_cloud_run_service.miniflow[0].name, null)
}

output "service_location" {
  description = "Region where the service is deployed"
  value       = try(google_cloud_run_service.miniflow[0].location, null)
}

output "container_image" {
  description = "Container image being used"
  value       = var.container_image
}

# GPU Instance outputs (only when GPU deployment)
output "gpu_instance_url" {
  description = "URL to access MiniFlow on GPU instance"
  value       = try("http://${google_compute_instance.miniflow_gpu[0].network_interface[0].access_config[0].nat_ip}:8000", null)
}

output "gpu_instance_public_ip" {
  description = "Public IP of the GPU instance"
  value       = try(google_compute_instance.miniflow_gpu[0].network_interface[0].access_config[0].nat_ip, null)
}

output "gpu_instance_ssh_command" {
  description = "Command to SSH into the GPU instance"
  value       = try("gcloud compute ssh ${google_compute_instance.miniflow_gpu[0].name} --zone=${var.gpu_zone} --project=${var.project_id}", null)
}

output "gpu_zone_used" {
  description = "Zone where GPU instance was deployed"
  value       = var.deployment_type == "gpu" ? var.gpu_zone : null
}

# Deployment type output
output "deployment_type" {
  description = "Type of deployment (cpu or gpu)"
  value       = var.deployment_type
}
