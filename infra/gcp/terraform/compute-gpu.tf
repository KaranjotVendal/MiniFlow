# Compute Engine with GPU for MiniFlow
# This provides GPU acceleration for ML inference

locals {
  gpu_instance_name = "${var.service_name}-gpu"
}

# Firewall rule to allow HTTP traffic to GPU instance
# Only created when deployment_type is "gpu"
resource "google_compute_firewall" "allow_miniflow_gpu" {
  count    = var.deployment_type == "gpu" ? 1 : 0
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
# Only created when deployment_type is "gpu"
resource "google_compute_instance" "miniflow_gpu" {
  count        = var.deployment_type == "gpu" ? 1 : 0
  name         = local.gpu_instance_name
  machine_type = "n1-standard-4"  # 4 vCPU, 15GB RAM
  zone         = var.gpu_zone      # Zone with available GPU capacity

  tags = [local.gpu_instance_name]

  # Boot disk with Container-Optimized OS
  boot_disk {
    initialize_params {
      image = "cos-cloud/cos-stable"  # Container-Optimized OS
      size  = 50  # GB
      type  = "pd-ssd"
    }
  }

  # Attach NVIDIA P100 GPU
  guest_accelerator {
    type  = var.gpu_type       # P100 is more available than T4
    count = var.gpu_count
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
