#!/usr/bin/env bash
# Build the serving image and run it locally with a sample /predict request.
# Usage: ./scripts/build_and_run_docker.sh
set -euo pipefail

cd "$(dirname "$0")/.."

IMAGE_NAME="heart-disease-api:local"
CONTAINER_NAME="heart-disease-api-local"

echo "==> Building image ${IMAGE_NAME}"
docker build -f docker/Dockerfile -t "${IMAGE_NAME}" .

echo "==> Removing any existing container named ${CONTAINER_NAME}"
docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

echo "==> Running container"
docker run -d --name "${CONTAINER_NAME}" -p 8000:8000 "${IMAGE_NAME}"

echo "==> Waiting for the API to become healthy"
for i in $(seq 1 20); do
  if curl -sf http://localhost:8000/health >/dev/null; then
    echo "API is healthy"
    break
  fi
  sleep 2
  if [ "$i" -eq 20 ]; then
    echo "API did not become healthy in time" >&2
    docker logs "${CONTAINER_NAME}"
    exit 1
  fi
done

echo "==> /health"
curl -s http://localhost:8000/health
echo

echo "==> /predict (sample patient)"
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":1,"trestbps":145,"chol":233,"fbs":1,"restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,"slope":3,"ca":0,"thal":6}'
echo

echo "==> Done. Container '${CONTAINER_NAME}' is running on http://localhost:8000"
echo "    Stop it with: docker rm -f ${CONTAINER_NAME}"
