output "ecr_repository_url" {
  description = "Push/pull images here"
  value       = aws_ecr_repository.app.repository_url
}

output "instance_id" {
  description = "EC2 instance ID (used by the deploy job)"
  value       = aws_instance.app.id
}

output "public_ip" {
  description = "Server's public IP"
  value       = aws_instance.app.public_ip
}

output "app_url" {
  description = "Open this in a browser"
  value       = "http://${aws_instance.app.public_ip}"
}

output "ssh_command" {
  description = "Log in to the server"
  value       = "ssh -i ~/.ssh/${var.key_name}.pem ec2-user@${aws_instance.app.public_ip}"
}
