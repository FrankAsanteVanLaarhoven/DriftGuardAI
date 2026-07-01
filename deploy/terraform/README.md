# DriftGuard AWS infrastructure (Terraform)

Real, `validate`/`plan`-clean Terraform. It provisions:

- **ECR** — immutable, scan-on-push image repository (+ lifecycle policy).
- **S3** — two versioned, encrypted, private buckets (DVC remote + artifact store).
- **VPC + EKS** — managed node group across 3 AZs (community modules).
- **IAM (IRSA)** — a least-privilege role assumed only by the app's Kubernetes
  service account (`driftguard:driftguard`), scoped to its buckets, its secret, and
  its ECR repo. No node-wide credentials.
- **Secrets Manager** — the app secret container (values rotated out-of-band).

> Nothing here is applied for you. `apply` provisions real, billable AWS resources
> and requires **your** credentials. The steps below are exact; run them yourself.

## Prerequisites

```bash
aws sts get-caller-identity          # confirm you are authenticated
export AWS_REGION=eu-west-2
cp terraform.tfvars.example terraform.tfvars   # then edit
```

## Validate (no credentials needed)

```bash
terraform -chdir=deploy/terraform init -backend=false
terraform -chdir=deploy/terraform validate
```

## Plan & apply (needs credentials — real resources)

```bash
terraform -chdir=deploy/terraform init
terraform -chdir=deploy/terraform plan  -out tf.plan
terraform -chdir=deploy/terraform apply tf.plan
```

## Wire up kubectl and push the first image

```bash
# From the outputs:
aws eks update-kubeconfig --region eu-west-2 --name driftguard

ECR=$(terraform -chdir=deploy/terraform output -raw ecr_repository_url)
aws ecr get-login-password --region eu-west-2 | docker login --username AWS --password-stdin "${ECR%/*}"
docker build -t "$ECR:$(git rev-parse --short HEAD)" .
docker push  "$ECR:$(git rev-parse --short HEAD)"

# Annotate the app ServiceAccount with the IRSA role:
terraform -chdir=deploy/terraform output -raw app_irsa_role_arn
```

## Teardown

```bash
terraform -chdir=deploy/terraform destroy
```

State: bootstrap with local state, then uncomment the S3 backend in `versions.tf`
(pointing at the `driftguard-tfstate` bucket) and `terraform init -migrate-state`.
