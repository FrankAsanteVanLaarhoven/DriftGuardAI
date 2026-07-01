// DriftGuard CI/CD — declarative pipeline.
// setup -> lint -> test -> baseline-gate -> build -> trivy scan -> push ECR
//   -> deploy staging -> smoke -> HUMAN GATE -> deploy prod -> (post) auto-rollback.
// Images are tagged by commit SHA. Green is the merge/deploy gate.
pipeline {
    agent any

    options {
        timestamps()
        disableConcurrentBuilds()
        timeout(time: 60, unit: 'MINUTES')
    }

    environment {
        AWS_REGION   = 'eu-west-2'
        ECR_REPO     = credentials('driftguard-ecr-repo')   // e.g. <acct>.dkr.ecr.<region>.amazonaws.com/driftguard
        IMAGE_TAG    = "${env.GIT_COMMIT?.take(12) ?: 'dev'}"
        IMAGE        = "${ECR_REPO}:${IMAGE_TAG}"
        STAGING_NS   = 'driftguard-staging'
        PROD_NS      = 'driftguard'
        STAGING_URL  = 'http://driftguard.driftguard-staging.svc.cluster.local:8000'
    }

    stages {
        stage('setup') {
            steps { sh 'make install' }
        }

        stage('lint') {
            steps { sh 'make lint' }
        }

        stage('test') {
            steps { sh 'make test' }   // unit + integration + FALLBACK chaos test
        }

        stage('baseline-gate') {
            // Build the model, then fail closed if it regresses vs the committed baseline.
            steps {
                sh 'make data'
                sh 'make train'
                sh 'make gate'         // exit 1 blocks the build on any regression
            }
        }

        stage('build') {
            steps { sh 'docker build -t "$IMAGE" .' }
        }

        stage('trivy scan') {
            steps {
                // Fail the build on fixable HIGH/CRITICAL vulnerabilities.
                sh 'trivy image --exit-code 1 --ignore-unfixed --severity HIGH,CRITICAL "$IMAGE"'
            }
        }

        stage('push ECR') {
            steps {
                sh '''
                  aws ecr get-login-password --region "$AWS_REGION" \
                    | docker login --username AWS --password-stdin "${ECR_REPO%/*}"
                  docker push "$IMAGE"
                '''
            }
        }

        stage('deploy staging') {
            steps {
                sh '''
                  kubectl -n "$STAGING_NS" apply -f deploy/k8s/
                  kubectl -n "$STAGING_NS" set image deployment/driftguard app="$IMAGE"
                  kubectl -n "$STAGING_NS" rollout status deployment/driftguard --timeout=180s
                '''
            }
        }

        stage('smoke') {
            steps {
                // Reuses tests/test_smoke.py against the live staging service.
                sh 'SERVICE_URL="$STAGING_URL" make test -e PYTEST_ADDOPTS="tests/test_smoke.py"'
            }
        }

        stage('HUMAN GATE') {
            steps {
                timeout(time: 24, unit: 'HOURS') {
                    input message: "Promote ${IMAGE} to production?", ok: 'Promote'
                }
            }
        }

        stage('deploy prod') {
            steps {
                sh '''
                  kubectl -n "$PROD_NS" apply -f deploy/k8s/
                  kubectl -n "$PROD_NS" set image deployment/driftguard app="$IMAGE"
                  kubectl -n "$PROD_NS" rollout status deployment/driftguard --timeout=180s
                '''
            }
        }
    }

    post {
        failure {
            // Idempotent rollback: revert the prod deployment to the last good revision.
            echo 'Pipeline failed — rolling back production deployment.'
            sh 'kubectl -n "$PROD_NS" rollout undo deployment/driftguard || true'
        }
        always {
            sh 'docker image rm "$IMAGE" || true'
        }
    }
}
