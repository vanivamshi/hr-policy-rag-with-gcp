"""Offline ingestion: HR policy files in GCS (or a local folder) -> Qdrant hybrid index.

Usage:
    python -m hr_rag.ingestion.ingest                      # uses GCS_BUCKET / GCS_PREFIX
    python -m hr_rag.ingestion.ingest --bucket my-bucket --prefix hr-policies/
    python -m hr_rag.ingestion.ingest --local-dir ./sample_docs
    python -m hr_rag.ingestion.ingest --recreate           # drop and rebuild the collection
"""

import argparse
import logging
from collections.abc import Iterator
from pathlib import Path, PurePosixPath

from hr_rag.config import get_settings
from hr_rag.factory import build_cache, build_jina, build_qdrant_client, build_store
from hr_rag.ingestion.chunking import chunk_sections
from hr_rag.ingestion.loaders import SUPPORTED_EXTENSIONS, load_document

logger = logging.getLogger("hr_rag.ingest")


def iter_gcs_files(bucket_name: str, prefix: str) -> Iterator[tuple[str, bytes]]:
    from google.cloud import storage

    client = storage.Client()
    for blob in client.list_blobs(bucket_name, prefix=prefix):
        if PurePosixPath(blob.name).suffix.lower() in SUPPORTED_EXTENSIONS:
            yield f"gs://{bucket_name}/{blob.name}", blob.download_as_bytes()


def iter_local_files(directory: Path) -> Iterator[tuple[str, bytes]]:
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path.relative_to(directory).as_posix(), path.read_bytes()


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bucket", default=settings.gcs_bucket)
    parser.add_argument("--prefix", default=settings.gcs_prefix)
    parser.add_argument("--local-dir", type=Path, help="Ingest from a local folder instead of GCS")
    parser.add_argument("--recreate", action="store_true", help="Drop and recreate the collection")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.local_dir:
        files = iter_local_files(args.local_dir)
    elif args.bucket:
        files = iter_gcs_files(args.bucket, args.prefix)
    else:
        parser.error("Provide --bucket (or GCS_BUCKET) or --local-dir")

    client = build_qdrant_client(settings)
    jina = build_jina(settings)
    store = build_store(settings, client, jina)
    store.ensure_collection(recreate=args.recreate)

    total_files = total_chunks = failures = 0
    for source, data in files:
        try:
            sections = load_document(source, data)
            chunks = chunk_sections(sections, source, settings.chunk_size, settings.chunk_overlap)
            if not chunks:
                logger.warning("No extractable text in %s (scanned PDF?) - skipped", source)
                continue
            store.delete_source(source)  # replace any previous version of this document
            store.upsert(chunks)
            total_files += 1
            total_chunks += len(chunks)
            logger.info("Indexed %s: %d chunks", source, len(chunks))
        except Exception:  # noqa: BLE001 - keep going, report at the end
            failures += 1
            logger.exception("Failed to ingest %s", source)

    # Cached answers may reference stale policy text after a re-index.
    if settings.cache_enabled and total_files:
        build_cache(settings, client, jina).clear()
        logger.info("Cleared semantic cache collection %s", settings.cache_collection)

    logger.info("Done: %d files, %d chunks, %d failures", total_files, total_chunks, failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
