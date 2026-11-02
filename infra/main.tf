# ---------- Look-ups (read existing things, create nothing) ----------

data "aws_vpc" "default" {
  default = true
}

data "aws_ami" "al2023" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["al2023-ami-2023.*-x86_64"]
  }
}

# ---------- ECR: AWS's private image registry ----------

resource "aws_ecr_repository" "app" {
  name                 = var.project_name
  image_tag_mutability = "MUTABLE"
  force_delete         = true # lets "terraform destroy" remove it even if it holds images

  image_scanning_configuration {
    scan_on_push = true # AWS scans each pushed image for known vulnerabilities
  }
}

# ---------- Security group: the server's firewall ----------

resource "aws_security_group" "app" {
  name        = "${var.project_name}-sg"
  description = "Marine fish classifier: HTTP from anywhere, admin ports from my IP only"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description = "App over HTTP"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "SSH from my IP"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.my_ip_cidr]
  }

  ingress {
    description = "Grafana from my IP"
    from_port   = 3000
    to_port     = 3000
    protocol    = "tcp"
    cidr_blocks = [var.my_ip_cidr]
  }

  ingress {
    description = "Prometheus from my IP"
    from_port   = 9090
    to_port     = 9090
    protocol    = "tcp"
    cidr_blocks = [var.my_ip_cidr]
  }

  egress {
    description = "All outbound (pull images, install packages)"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# ---------- IAM: let the server pull from ECR without stored keys ----------

resource "aws_iam_role" "ec2" {
  name = "${var.project_name}-ec2-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ecr_read" {
  role       = aws_iam_role.ec2.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}

# Lets GitHub Actions run the deploy command later without SSH (AWS Systems Manager).
resource "aws_iam_role_policy_attachment" "ssm" {
  role       = aws_iam_role.ec2.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "ec2" {
  name = "${var.project_name}-ec2-profile"
  role = aws_iam_role.ec2.name
}

# ---------- EC2: the server ----------

resource "aws_instance" "app" {
  ami                         = data.aws_ami.al2023.id
  instance_type               = var.instance_type
  key_name                    = var.key_name
  vpc_security_group_ids      = [aws_security_group.app.id]
  iam_instance_profile        = aws_iam_instance_profile.ec2.name
  associate_public_ip_address = true

  root_block_device {
    volume_size = 20 # GB: app image plus room for monitoring in Phase 6
    volume_type = "gp3"
  }

  # Script that runs once on first boot: installs Docker, creates the deploy script.
  user_data = templatefile("${path.module}/user_data.sh.tpl", {
    region         = var.aws_region
    registry       = split("/", aws_ecr_repository.app.repository_url)[0]
    repository_url = aws_ecr_repository.app.repository_url
  })
  user_data_replace_on_change = true

  tags = {
    Name = "${var.project_name}-server"
  }

  # Amazon publishes new AMIs regularly; don't replace the running server
  # (and change its IP) just because a newer one exists.
  lifecycle {
    ignore_changes = [ami]
  }
}
