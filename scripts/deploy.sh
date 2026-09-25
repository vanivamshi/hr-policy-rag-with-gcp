#!/usr/bin/env bash
# Build from source with Cloud Build and deploy the hr-rag-assistant Cloud Run service.
# Required env: PROJECT_ID. Non-secret settings come from deploy/env.yaml.
set -euo pipefail

: "${PROJECT_ID:?set PROJECT_ID}"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-hr-rag-assistant}"
SA_EMAIL="${SA_EMAIL:-hr-rag-assistant@${PROJECT_ID}.iam.gserviceaccount.com}"
ENV_FILE="${ENV_FILE:-deploy/env.yaml}"

[[ -f "$ENV_FILE" ]] || { echo "Missing $ENV_FILE (copy deploy/env.example.yaml)"; exit 1; }

SECRETS="/app/.streamlit/secrets.toml=streamlit-auth:latest"
SECRETS+=",JINA_API_KEY=jina-api-key:latest"
SECRETS+=",QDRANT_API_KEY=qdrant-api-key:latest"
for pair in "GROQ_API_KEY=groq-api-key" "LANGSMITH_API_KEY=langsmith-api-key"; do
  if gcloud secrets describe "${pair##*=}" --project "$PROJECT_ID" >/dev/null 2>&1; then
    SECRETS+=",${pair}:latest"
  fi
done

# --allow-unauthenticated: the app does its own Google OAuth + allow-list (Layer 1).
# --session-affinity: keeps a user's Streamlit websocket on the same instance.
gcloud run deploy "$SERVICE" \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --source . \
  --service-account "$SA_EMAIL" \
  --allow-unauthenticated \
  --session-affinity \
  --memory 2Gi --cpu 2 \
  --timeout 3600 \
  --min-instances 0 --max-instances 5 \
  --env-vars-file "$ENV_FILE" \
  --set-secrets "$SECRETS"

URL="$(gcloud run services describe "$SERVICE" --project "$PROJECT_ID" --region "$REGION" --format='value(status.url)')"
echo
echo "Deployed: $URL"
echo "Make sure redirect_uri in the streamlit-auth secret is ${URL}/oauth2callback"
echo "and that URI is listed on the OAuth client in Google Cloud Console."
