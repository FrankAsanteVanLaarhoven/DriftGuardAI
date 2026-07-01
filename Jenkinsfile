// DriftGuard CI/CD — skeleton (fully wired in Phase 7).
// Gate order: setup -> lint -> test -> baseline-gate -> build -> scan -> push
//             -> deploy staging -> smoke -> HUMAN GATE -> deploy prod -> auto-rollback.
pipeline {
    agent any
    options { timestamps() }
    stages {
        stage('setup') { steps { sh 'make install' } }
        stage('lint')  { steps { sh 'make lint' } }
        stage('test')  { steps { sh 'make test' } }
    }
    post {
        always { echo 'Pipeline finished (skeleton — see Phase 7 for the full pipeline).' }
    }
}
