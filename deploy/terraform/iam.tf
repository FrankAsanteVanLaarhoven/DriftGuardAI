# Least-privilege IAM for the app, assumed via IRSA (no node-wide credentials).
data "aws_iam_policy_document" "app" {
  # Read/write only the DriftGuard buckets (DVC remote + artifacts).
  statement {
    sid = "S3Artifacts"
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:DeleteObject",
      "s3:ListBucket",
    ]
    resources = concat(
      [for b in aws_s3_bucket.store : b.arn],
      [for b in aws_s3_bucket.store : "${b.arn}/*"],
    )
  }

  # Read only the app's own secret.
  statement {
    sid       = "ReadAppSecret"
    actions   = ["secretsmanager:GetSecretValue"]
    resources = [aws_secretsmanager_secret.app.arn]
  }

  # Pull images from the app's ECR repo.
  statement {
    sid = "EcrPull"
    actions = [
      "ecr:GetDownloadUrlForLayer",
      "ecr:BatchGetImage",
      "ecr:BatchCheckLayerAvailability",
    ]
    resources = [aws_ecr_repository.app.arn]
  }

  statement {
    sid       = "EcrAuth"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }
}

resource "aws_iam_policy" "app" {
  name   = "${var.project}-app"
  policy = data.aws_iam_policy_document.app.json
  tags   = local.tags
}

# IRSA role trusted by the app's Kubernetes service account only.
module "app_irsa" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.44"

  role_name = "${var.project}-app"

  oidc_providers = {
    main = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["${var.app_namespace}:${var.app_service_account}"]
    }
  }

  role_policy_arns = {
    app = aws_iam_policy.app.arn
  }

  tags = local.tags
}
