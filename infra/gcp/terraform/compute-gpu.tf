# Compute Engine with GPU for MiniFlow
# This provides GPU acceleration for ML inference

locals {
  gpu_instance_name = "${var.service_name}-gpu"
  gpu_zone          = "${var.region}-a"  # us-central1-a, etc.
}

# Firewall rule to allow HTTP traffic to GPU instance
resource "google_compute_firewall" "allow_miniflow_gpu" {
  name    = "${local.gpu_instance_name}-http"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["8000", "22"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = [local.gpu_instance_name]
}

# Compute Engine instance with GPU
resource "google_compute_instance" "miniflow_gpu" {
  name         = local.gpu_instance_name
  machine_type = "n1-standard-4"  # 4 vCPU, 15GB RAM
  zone         = local.gpu_zone

  tags = [local.gpu_instance_name]

  # Boot disk with Container-Optimized OS
  boot_disk {
    initialize_params {
      image = "cos-cloud/cos-stable"  # Container-Optimized OS
      size  = 50  # GB
      type  = "pd-ssd"
    }
  }

  # Attach NVIDIA T4 GPU
  guest_accelerator {
    type  = "nvidia-tesla-t4"  # 16GB GPU memory
    count = 1
  }

  # Required for GPU instances
  scheduling {
    on_host_maintenance = "TERMINATE"  # GPU requires TERMINATE
    preemptible         = true          # 70% cheaper (Spot equivalent)
    automatic_restart   = false
  }

  # Network configuration
  network_interface {
    network = "default"
    
    access_config {
      # Ephemeral public IP
      # Note: For production, use static IP
    }
  }

  # Metadata for container declaration
  # This tells Container-Optimized OS what to run
  metadata = {
    gce-container-declaration = <<EOF
spec:
  containers:
    - name: miniflow
      image: ${var.container_image}
      securityContext:
        privileged: true
      env:
        - name: MINIFLOW_DEVICE
          value: "cuda"
        - name: COQUI_TOS_AGREED
          value: "1"
        - name: MINIFLOW_CONFIG
          value: "${var.miniflow_config}"
        - name: MINIFLOW_REQUEST_TIMEOUT_SECONDS
          value: "900"
        - name: RELEASE_ID
          value: "${var.release_id}"
      resources:
        limits:
          nvidia.com/gpu: 1
      ports:
        - containerPort: 8000
          hostPort: 8000
  restartPolicy: Always
  volumes: []
EOF

    # Enable GPU driver installation
    install-nvidia-driver = "true"
  }

  # Labels for organization
  labels = {
    environment = "staging"
    managed_by  = "terraform"
    service     = "miniflow"
    type        = "gpu"
  }
}

# Output the public IP
output "gpu_instance_public_ip" {
  description = "Public IP of the GPU instance"
  value       = google_compute_instance.miniflow_gpu.network_interface[0].access_config[0].nat_ip
}

output "gpu_instance_ssh_command" {
  description = "Command to SSH into the GPU instance"
  value       = "gcloud compute ssh ${google_compute_instance.miniflow_gpu.name} --zone=${local.gpu_zone} --project=${var.project_id}"
}

output "gpu_instance_url" {
  description = "URL to access MiniFlow on GPU instance"
  value       = "http://${google_compute_instance.miniflow_gpu.network_interface[0].access_config[0].nat_ip}:8000"
}
