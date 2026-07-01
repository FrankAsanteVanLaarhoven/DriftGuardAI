# Application secret (e.g. MLflow backend DSN / registry token). The value is set
# out-of-band or via CI — Terraform only provisions the container, not the secret.
resource "aws_secretsmanager_secret" "app" {
  name        = "${var.project}/app"
  description = "DriftGuard runtime secrets (MLflow DSN, registry token)."
  tags        = local.tags
}

# Placeholder version so the secret exists; real values are rotated in later.
resource "aws_secretsmanager_secret_version" "app" {
  secret_id = aws_secretsmanager_secret.app.id
  secret_string = jsonencode({
    MLFLOW_TRACKING_URI = "http://mlflow.driftguard.svc.cluster.local:5000"
  })

  lifecycle {
    ignore_changes = [secret_string] # do not clobber rotated values
  }
}
