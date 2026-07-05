{{/* Chart name */}}
{{- define "driftguard.name" -}}
{{- .Chart.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/* Fully qualified app name */}}
{{- define "driftguard.fullname" -}}
{{- if contains .Chart.Name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name .Chart.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}

{{/* Common labels */}}
{{- define "driftguard.labels" -}}
app.kubernetes.io/name: {{ include "driftguard.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version }}
{{- end -}}

{{/*
Selector labels. Deliberately shared by the stable AND canary pods: the Service
selects on these alone, so traffic splits by replica ratio. The `track` label
distinguishes the two deployments without entering the Service selector.
*/}}
{{- define "driftguard.selectorLabels" -}}
app.kubernetes.io/name: {{ include "driftguard.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{/*
The fallback-contract probes. Fixed here, not in values: /health (liveness =
process up) and /ready (200 if ANY model can serve, so a bad primary never
pulls the pod out of rotation) are stable API — see AGENTS.md golden rule 1.
*/}}
{{- define "driftguard.probes" -}}
startupProbe:
  httpGet: { path: /health, port: http }
  failureThreshold: 30
  periodSeconds: 2
livenessProbe:
  httpGet: { path: /health, port: http }
  periodSeconds: 15
  failureThreshold: 3
readinessProbe:
  httpGet: { path: /ready, port: http }
  periodSeconds: 10
  failureThreshold: 3
{{- end -}}

{{/* Pod-level security context */}}
{{- define "driftguard.podSecurityContext" -}}
runAsNonRoot: true
runAsUser: 10001
fsGroup: 10001
seccompProfile:
  type: RuntimeDefault
{{- end -}}

{{/* Container-level security context */}}
{{- define "driftguard.containerSecurityContext" -}}
allowPrivilegeEscalation: false
readOnlyRootFilesystem: true
capabilities:
  drop: ["ALL"]
{{- end -}}
