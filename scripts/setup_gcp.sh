#!/usr/bin/env bash
# One-time GCP setup: APIs, bucket, service account + IAM, Secret Manager secrets, Model Armor template.
#
# Required env: PROJECT_ID, BUCKET
# Optional env (a secret version is added for each one that is set):
#   JINA_API_KEY, GROQ_API_KEY, QDRANT_API_KEY, LANGSMITH_API_KEY
#   STREAMLIT_AUTH_FILE (path to your filled-in secrets.toml, default .streamlit/secrets.toml)
set -euo pipefail

: "${PROJECT_ID:?set PROJECT_ID}"
: "${BUCKET:?set BUCKET}"
REGION="${REGION:-us-central1}"
SA_NAME="${SA_NAME:-hr-rag-assistant}"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
TEMPLATE_ID="${MODEL_ARMOR_TEMPLATE:-hr-rag-template}"
STREAMLIT_AUTH_FILE="${STREAMLIT_AUTH_FILE:-.streamlit/secrets.toml}"

gcloud config set project "$PROJECT_ID"

echo "==> Enabling APIs"
gcloud services enable \
  run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com \
  secretmanager.googleapis.com aiplatform.googleapis.com storage.googleapis.com \
  modelarmor.googleapis.com

echo "==> Bucket gs://${BUCKET}"
gcloud storage buckets describe "gs://${BUCKET}" >/dev/null 2>&1 \
  || gcloud storage buckets create "gs://${BUCKET}" --location="$REGION" --uniform-bucket-level-access

echo "==> Service account ${SA_EMAIL}"
gcloud iam service-accounts describe "$SA_EMAIL" >/dev/null 2>&1 \
  || gcloud iam service-accounts create "$SA_NAME" --display-name="HR RAG Assistant"

for role in roles/aiplatform.user roles/modelarmor.user roles/secretmanager.secretAccessor roles/logging.logWriter; do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:${SA_EMAIL}" --role="$role" --condition=None --quiet >/dev/null
done
gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
  --member="serviceAccount:${SA_EMAIL}" --role=roles/storage.objectViewer >/dev/null

upsert_secret() {  # name, value-file
  local name="$1" file="$2"
  gcloud secrets describe "$name" >/dev/null 2>&1 \
    || gcloud secrets create "$name" --replication-policy=automatic
  gcloud secrets versions add "$name" --data-file="$file" >/dev/null
  echo "    secret ${name}: new version added"
}

echo "==> Secrets"
tmp="$(mktemp)"; trap 'rm -f "$tmp"' EXIT
for pair in "jina-api-key:JINA_API_KEY" "groq-api-key:GROQ_API_KEY" \
            "qdrant-api-key:QDRANT_API_KEY" "langsmith-api-key:LANGSMITH_API_KEY"; do
  secret="${pair%%:*}"; var="${pair##*:}"
  if [[ -n "${!var:-}" ]]; then
    printf '%s' "${!var}" > "$tmp"
    upsert_secret "$secret" "$tmp"
  fi
done
if [[ -f "$STREAMLIT_AUTH_FILE" ]]; then
  upsert_secret streamlit-auth "$STREAMLIT_AUTH_FILE"
else
  echo "    skipped streamlit-auth: ${STREAMLIT_AUTH_FILE} not found"
fi

echo "==> Model Armor template ${TEMPLATE_ID} (${REGION})"
MA_ENDPOINT="https://modelarmor.${REGION}.rep.googleapis.com/v1/projects/${PROJECT_ID}/locations/${REGION}/templates"
TOKEN="$(gcloud auth print-access-token)"
if curl -sf -H "Authorization: Bearer ${TOKEN}" "${MA_ENDPOINT}/${TEMPLATE_ID}" >/dev/null; then
  echo "    template already exists"
else
  curl -sf -X POST -H "Authorization: Bearer ${TOKEN}" -H "Content-Type: application/json" \
    "${MA_ENDPOINT}?templateId=${TEMPLATE_ID}" -d @- >/dev/null <<'JSON'
{
  "filterConfig": {
    "raiSettings": {
      "raiFilters": [
        {"filterType": "HATE_SPEECH", "confidenceLevel": "MEDIUM_AND_ABOVE"},
        {"filterType": "HARASSMENT", "confidenceLevel": "MEDIUM_AND_ABOVE"},
        {"filterType": "SEXUALLY_EXPLICIT", "confidenceLevel": "MEDIUM_AND_ABOVE"},
        {"filterType": "DANGEROUS", "confidenceLevel": "MEDIUM_AND_ABOVE"}
      ]
    },
    "piAndJailbreakFilterSettings": {"filterEnforcement": "ENABLED", "confidenceLevel": "MEDIUM_AND_ABOVE"},
    "maliciousUriFilterSettings": {"filterEnforcement": "ENABLED"},
    "sdpSettings": {"basicConfig": {"filterEnforcement": "ENABLED"}}
  }
}
JSON
  echo "    template created"
fi

echo "Done. Next: upload policies (gcloud storage cp ./policies/* gs://${BUCKET}/hr-policies/), run ingestion, then scripts/deploy.sh"
