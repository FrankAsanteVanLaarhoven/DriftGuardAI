output "region" {
  value = var.region
}

output "ecr_repository_url" {
  description = "Push images here (docker push <url>:<sha>)."
  value       = aws_ecr_repository.app.repository_url
}

output "eks_cluster_name" {
  value = module.eks.cluster_name
}

output "eks_cluster_endpoint" {
  value = module.eks.cluster_endpoint
}

output "update_kubeconfig_command" {
  description = "Run this to point kubectl at the new cluster."
  value       = "aws eks update-kubeconfig --region ${var.region} --name ${module.eks.cluster_name}"
}

output "dvc_bucket" {
  value = aws_s3_bucket.store["dvc"].bucket
}

output "artifacts_bucket" {
  value = aws_s3_bucket.store["artifacts"].bucket
}

output "app_irsa_role_arn" {
  description = "Annotate the app ServiceAccount with this role ARN."
  value       = module.app_irsa.iam_role_arn
}

output "app_secret_arn" {
  value = aws_secretsmanager_secret.app.arn
}
