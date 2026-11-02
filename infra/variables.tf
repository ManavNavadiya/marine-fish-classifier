variable "aws_region" {
  description = "AWS region to deploy into"
  type        = string
  default     = "ap-south-1"
}

variable "project_name" {
  description = "Prefix for resource names"
  type        = string
  default     = "marine-fish-classifier"
}

variable "instance_type" {
  description = "EC2 size; t3.small has 2 GB RAM, enough for the app plus Prometheus and Grafana"
  type        = string
  default     = "t3.small"
}

variable "key_name" {
  description = "Name of the EC2 key pair for SSH"
  type        = string
  default     = "mlops-key"
}

variable "my_ip_cidr" {
  description = "Your public IP in CIDR form, e.g. 203.0.113.10/32 (only this IP may SSH in)"
  type        = string
}
