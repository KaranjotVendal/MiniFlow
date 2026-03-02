locals {
  service_name = var.service_name
  common_labels = {
    project     = "miniflow"
    environment = "staging"
    managed_by  = "terraform"
  }
}

# Enable required APIs
resource "google_project_service" "run" {
  service = "run.googleapis.com"

  disable_on_destroy = false
}

resource "google_project_service" "container_registry" {
  service = "containerregistry.googleapis.com"

  disable_on_destroy = false
}

# Cloud Run service
resource "google_cloud_run_service" "miniflow" {
  name     = local.service_name
  location = var.region

  template {
    spec {
      containers {
        image = var.container_image

        resources {
          limits = {
            cpu    = var.cpu
            memory = var.memory
          }
        }

        ports {
          container_port = 8000
        }

        env {
          name  = "MINIFLOW_DEVICE"
          value = var.miniflow_device
        }

        env {
          name  = "COQUI_TOS_AGREED"
          value = "1"
        }

        env {
          name  = "MINIFLOW_CONFIG"
          value = var.miniflow_config
        }

        env {
          name  = "MINIFLOW_REQUEST_TIMEOUT_SECONDS"
          value = "900"
        }

        env {
          name  = "RELEASE_ID"
          value = var.release_id
        }
      }

      # Container concurrency: how many requests per container
      container_concurrency = 10

      # Timeout for requests (in seconds)
      timeout_seconds = 900
    }

    metadata {
      labels = local.common_labels
      annotations = {
        "autoscaling.knative.dev/minScale" = "0"
        "autoscaling.knative.dev/maxScale" = "5"
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }

  depends_on = [google_project_service.run]
}

# Allow public access to the service
resource "google_cloud_run_service_iam_member" "public" {
  service  = google_cloud_run_service.miniflow.name
  location = google_cloud_run_service.miniflow.location
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# Budget alert (optional, requires billing account)
resource "google_billing_budget" "monthly" {
  count = var.project_id != "" ? 1 : 0

  billing_account = data.google_billing_account.account[0].id
  display_name    = "${local.service_name}-monthly-budget"

  amount {
    specified_amount {
      currency_code = "USD"
      units         = "50"
    }
  }

  threshold_rules {
    threshold_percent = 50
  }

  threshold_rules {
    threshold_percent = 80
  }

  threshold_rules {
    threshold_percent = 100
  }

  all_updates_rule {
    monitoring_notification_channels = [
      google_monitoring_notification_channel.email[0].id
    ]
    disable_default_iam_recipients = false
  }
}

# Notification channel for budget alerts
data "google_billing_account" "account" {
  count               = var.project_id != "" ? 1 : 0
  billing_account     = "your-billing-account-id"  # Replace with your billing account
  open                = true
}

resource "google_monitoring_notification_channel" "email" {
  count        = var.project_id != "" ? 1 : 0
  display_name = "Budget Alert Email"
  type         = "email"

  labels = {
    email_address = "your-email@example.com"  # Replace with your email
  }
}
