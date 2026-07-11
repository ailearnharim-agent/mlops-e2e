{{- define "heart-disease-api.name" -}}
{{- .Chart.Name -}}
{{- end -}}

{{- define "heart-disease-api.fullname" -}}
{{- .Chart.Name -}}
{{- end -}}

{{- define "heart-disease-api.labels" -}}
app: {{ include "heart-disease-api.fullname" . }}
chart: {{ .Chart.Name }}-{{ .Chart.Version }}
release: {{ .Release.Name }}
{{- end -}}
