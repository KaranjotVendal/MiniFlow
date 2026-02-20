variable "aws_region" {
  type    = string
  default = "us-east-1"
}

variable "environment" {
  type    = string
  default = "staging"
}

variable "project_name" {
  type    = string
  default = "miniflow"
}

variable "container_image" {
  type = string
}

variable "container_port" {
  type    = number
  default = 8000
}

variable "cpu" {
  type    = number
  default = 2048
}

variable "memory" {
  type    = number
  default = 4096
}

variable "desired_count" {
  type    = number
  default = 1
}

variable "budget_alert_email" {
  type      = string
  default   = ""
  sensitive = false
}

variable "vpc_id" {
  type = string
}

variable "public_subnet_ids" {
  type = list(string)
}
