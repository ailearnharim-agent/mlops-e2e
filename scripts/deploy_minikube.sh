#!/usr/bin/env bash
# Build the image, load it into minikube, and deploy via Helm.
# Usage: ./scripts/deploy_minikube.sh
set -euo pipefail

cd "$(dirname "$0")/.."

IMAGE_NAME="heart-disease-api:local"
RELEASE_NAME="heart-disease-api"
CHART_PATH="k8s/helm/heart-disease-api"

echo "==> Ensuring minikube is running"
if ! minikube status >/dev/null 2>&1; then
  minikube start --driver=docker --cpus=2 --memory=4096
fi

echo "==> Enabling ingress addon"
minikube addons enable ingress

echo "==> Building image"
docker build -f docker/Dockerfile -t "${IMAGE_NAME}" .

echo "==> Loading image into minikube"
minikube image load "${IMAGE_NAME}"

echo "==> Deploying with Helm"
helm upgrade --install "${RELEASE_NAME}" "${CHART_PATH}" \
  --set image.repository=heart-disease-api \
  --set image.tag=local

echo "==> Waiting for rollout"
kubectl rollout status deployment/heart-disease-api -n heart-disease-mlops --timeout=120s

MINIKUBE_IP=$(minikube ip)
echo
echo "==> Deployment complete."
echo "Pods:"
kubectl get pods -n heart-disease-mlops
echo
echo "To reach the API:"
echo "  Option A (port-forward, recommended -- works on every OS/driver, no sudo needed):"
echo "    kubectl port-forward -n heart-disease-mlops svc/heart-disease-api-svc 8080:80"
echo "    curl http://localhost:8080/health"
echo "  Option B (Ingress): add '${MINIKUBE_IP} heart-disease.local' to /etc/hosts, then:"
echo "    curl http://heart-disease.local/health"
echo "    On macOS with the docker driver, the node IP is only reachable from inside the Docker"
echo "    Desktop VM, so this also needs 'minikube tunnel' (requires sudo) running in another"
echo "    terminal -- prefer Option A on macOS."
