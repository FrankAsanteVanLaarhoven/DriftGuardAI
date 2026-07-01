variable "region" {
  description = "AWS region."
  type        = string
  default     = "eu-west-2"
}

variable "project" {
  description = "Project name, used as a resource name prefix."
  type        = string
  default     = "driftguard"
}

variable "cluster_name" {
  description = "EKS cluster name."
  type        = string
  default     = "driftguard"
}

variable "cluster_version" {
  description = "EKS Kubernetes version."
  type        = string
  default     = "1.30"
}

variable "vpc_cidr" {
  description = "VPC CIDR block."
  type        = string
  default     = "10.42.0.0/16"
}

variable "node_instance_types" {
  description = "EKS managed node group instance types."
  type        = list(string)
  default     = ["t3.large"]
}

variable "node_desired_size" {
  type    = number
  default = 2
}

variable "node_min_size" {
  type    = number
  default = 1
}

variable "node_max_size" {
  type    = number
  default = 4
}

variable "app_namespace" {
  description = "Kubernetes namespace the app runs in (for IRSA trust)."
  type        = string
  default     = "driftguard"
}

variable "app_service_account" {
  description = "Kubernetes service account the app uses (for IRSA trust)."
  type        = string
  default     = "driftguard"
}

variable "tags" {
  description = "Extra tags merged into every resource."
  type        = map(string)
  default     = {}
}
