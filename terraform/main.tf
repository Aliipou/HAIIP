terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.12"
    }
  }
  backend "s3" {
    bucket         = "haiip-terraform-state"
    key            = "production/terraform.tfstate"
    region         = "eu-north-1"
    encrypt        = true
    dynamodb_table = "haiip-terraform-lock"
  }
}

provider "aws" {
  region = var.aws_region
  default_tags {
    tags = {
      Project     = "HAIIP"
      Environment = var.environment
      ManagedBy   = "Terraform"
      EUAIAct     = "compliant"
    }
  }
}

# ── VPC ────────────────────────────────────────────────────────────────────────

module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "haiip-vpc-${var.environment}"
  cidr = var.vpc_cidr

  azs             = var.availability_zones
  private_subnets = var.private_subnet_cidrs
  public_subnets  = var.public_subnet_cidrs

  enable_nat_gateway     = true
  single_nat_gateway     = var.environment != "production"
  enable_vpn_gateway     = false
  enable_dns_hostnames   = true
  enable_dns_support     = true

  public_subnet_tags = {
    "kubernetes.io/role/elb"                        = 1
    "kubernetes.io/cluster/${local.cluster_name}"   = "shared"
  }
  private_subnet_tags = {
    "kubernetes.io/role/internal-elb"               = 1
    "kubernetes.io/cluster/${local.cluster_name}"   = "shared"
  }
}

# ── EKS ────────────────────────────────────────────────────────────────────────

locals {
  cluster_name = "haiip-${var.environment}"
}

module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.0"

  cluster_name    = local.cluster_name
  cluster_version = var.k8s_version

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  cluster_endpoint_public_access  = true
  cluster_endpoint_private_access = true

  cluster_addons = {
    coredns     = { most_recent = true }
    kube-proxy  = { most_recent = true }
    vpc-cni     = { most_recent = true }
    aws-ebs-csi-driver = { most_recent = true }
  }

  eks_managed_node_groups = {
    api = {
      name           = "haiip-api-nodes"
      instance_types = [var.api_instance_type]
      min_size       = 2
      max_size       = 10
      desired_size   = 2
      labels = { role = "api" }
      taints = []
    }
    worker = {
      name           = "haiip-worker-nodes"
      instance_types = [var.worker_instance_type]
      min_size       = 1
      max_size       = 5
      desired_size   = 1
      labels = { role = "worker" }
    }
  }

  enable_cluster_creator_admin_permissions = true
}

# ── RDS (PostgreSQL) ───────────────────────────────────────────────────────────

module "rds" {
  source  = "terraform-aws-modules/rds/aws"
  version = "~> 6.0"

  identifier = "haiip-db-${var.environment}"

  engine               = "postgres"
  engine_version       = "15.4"
  family               = "postgres15"
  major_engine_version = "15"
  instance_class       = var.db_instance_class

  allocated_storage     = 20
  max_allocated_storage = 200
  storage_encrypted     = true

  db_name  = "haiip"
  username = "haiip_admin"
  port     = "5432"

  manage_master_user_password = true

  vpc_security_group_ids = [module.db_sg.security_group_id]
  subnet_ids             = module.vpc.private_subnets
  create_db_subnet_group = true

  maintenance_window      = "Mon:00:00-Mon:03:00"
  backup_window           = "03:00-06:00"
  backup_retention_period = 7
  deletion_protection     = var.environment == "production"

  enabled_cloudwatch_logs_exports = ["postgresql"]
  monitoring_interval             = 60
  monitoring_role_name            = "haiip-rds-monitoring"
  create_monitoring_role          = true

  tags = { Component = "database" }
}

# ── ElastiCache (Redis) ────────────────────────────────────────────────────────

resource "aws_elasticache_cluster" "redis" {
  cluster_id           = "haiip-redis-${var.environment}"
  engine               = "redis"
  node_type            = var.redis_instance_type
  num_cache_nodes      = 1
  parameter_group_name = "default.redis7"
  engine_version       = "7.0"
  port                 = 6379
  subnet_group_name    = aws_elasticache_subnet_group.redis.name
  security_group_ids   = [module.redis_sg.security_group_id]

  tags = { Component = "cache" }
}

resource "aws_elasticache_subnet_group" "redis" {
  name       = "haiip-redis-${var.environment}"
  subnet_ids = module.vpc.private_subnets
}

# ── S3 (model artifacts + audit logs) ─────────────────────────────────────────

resource "aws_s3_bucket" "models" {
  bucket = "haiip-models-${var.environment}-${data.aws_caller_identity.current.account_id}"
  tags   = { Component = "model-store" }
}

resource "aws_s3_bucket_versioning" "models" {
  bucket = aws_s3_bucket.models.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "models" {
  bucket = aws_s3_bucket.models.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "aws:kms"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "models" {
  bucket                  = aws_s3_bucket.models.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ── Security groups ────────────────────────────────────────────────────────────

module "db_sg" {
  source  = "terraform-aws-modules/security-group/aws"
  version = "~> 5.0"
  name    = "haiip-db-${var.environment}"
  vpc_id  = module.vpc.vpc_id

  ingress_with_source_security_group_id = [{
    from_port                = 5432
    to_port                  = 5432
    protocol                 = "tcp"
    source_security_group_id = module.eks.node_security_group_id
  }]
}

module "redis_sg" {
  source  = "terraform-aws-modules/security-group/aws"
  version = "~> 5.0"
  name    = "haiip-redis-${var.environment}"
  vpc_id  = module.vpc.vpc_id

  ingress_with_source_security_group_id = [{
    from_port                = 6379
    to_port                  = 6379
    protocol                 = "tcp"
    source_security_group_id = module.eks.node_security_group_id
  }]
}

# ── Data sources ───────────────────────────────────────────────────────────────

data "aws_caller_identity" "current" {}
