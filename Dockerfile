FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FASTEMBED_CACHE_DIR=/app/models \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_CLIENT_TOOLBAR_MODE=minimal

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
RUN pip install .

# Bake the BM25 sparse model into the image so cold starts don't download it.
RUN python -c "from fastembed import SparseTextEmbedding; SparseTextEmbedding('Qdrant/bm25', cache_dir='/app/models')"

COPY app.py ./

# /app/.streamlit/secrets.toml is mounted from Secret Manager (streamlit-auth) at deploy time,
# so nothing else is placed in /app/.streamlit (the mount would hide it).
RUN useradd --uid 1000 --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8080
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-8080} --server.address=0.0.0.0"]
