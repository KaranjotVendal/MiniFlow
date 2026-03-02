output "service_url" {
  description = "URL of the deployed Cloud Run service"
  value       = google_cloud_run_service.miniflow.status[0].url
}

output "service_name" {
  description = "Name of the Cloud Run service"
  value       = google_cloud_run_service.miniflow.name
}

output "service_location" {
  description = "Region where the service is deployed"
  value       = google_cloud_run_service.miniflow.location
}

output "container_image" {
  description = "Container image being used"
  value       = var.container_image
}
