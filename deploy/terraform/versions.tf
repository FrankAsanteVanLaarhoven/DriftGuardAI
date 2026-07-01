terraform {
  required_version = ">= 1.6"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.60"
    }
  }

  # Remote state (uncomment and set for a real deployment; the bucket is one of the
  # versioned S3 buckets this stack creates — bootstrap it once with local state).
  # backend "s3" {
  #   bucket = "driftguard-tfstate"
  #   key    = "driftguard/terraform.tfstate"
  #   region = "eu-west-2"
  #   encrypt = true
  # }
}

provider "aws" {
  region = var.region
  default_tags {
    tags = local.tags
  }
}
