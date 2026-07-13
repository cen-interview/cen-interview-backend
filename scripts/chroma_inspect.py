"""Inspect the local Chroma evidence collection.

Examples:
    uv run python scripts/chroma_inspect.py --summary
    uv run python scripts/chroma_inspect.py --user-id 1 --limit 10
    uv run python scripts/chroma_inspect.py --user-id test-user --embeddings
    uv run python scripts/chroma_inspect.py --export-csv
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import Counter
from typing import Any

import chromadb


DEFAULT_PATH = ".chroma/evidence"
DEFAULT_COLLECTION = "evidence_chunks"


def main() -> None:
    """Run the Chroma inspection command."""
    args = _parse_args()
    collection = _get_collection(path=args.path, name=args.collection)

    if args.summary:
        _print_summary(collection)
        return

    if args.export_csv:
        _export_csv(
            collection,
            output_path=args.export_csv,
            user_id=args.user_id,
            embedding_head_count=args.embedding_head_count,
        )
        return

    _print_records(
        collection,
        user_id=args.user_id,
        limit=args.limit,
        show_embeddings=args.embeddings,
        full_text=args.full_text,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect local Chroma evidence data.")
    parser.add_argument(
        "--path",
        default=os.getenv("EVIDENCE_CHROMA_PATH", DEFAULT_PATH),
        help=f"Chroma persistence path. Default: {DEFAULT_PATH}",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help=f"Chroma collection name. Default: {DEFAULT_COLLECTION}",
    )
    parser.add_argument(
        "--user-id",
        default=None,
        help="Filter by stored metadata user_id.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of records to print.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print total count and metadata distribution only.",
    )
    parser.add_argument(
        "--embeddings",
        action="store_true",
        help="Print embedding dimension and first values.",
    )
    parser.add_argument(
        "--full-text",
        action="store_true",
        help="Print full document text instead of a preview.",
    )
    parser.add_argument(
        "--export-csv",
        nargs="?",
        const="chroma_evidence_export.csv",
        default=None,
        help="Export records to CSV. Default file: chroma_evidence_export.csv",
    )
    parser.add_argument(
        "--embedding-head-count",
        type=int,
        default=20,
        help="Number of leading embedding values to include in CSV export.",
    )
    return parser.parse_args()


def _get_collection(path: str, name: str) -> Any:
    client = chromadb.PersistentClient(path=path)
    try:
        return client.get_collection(name)
    except Exception as exc:
        available = [
            collection.name if hasattr(collection, "name") else str(collection)
            for collection in client.list_collections()
        ]
        raise SystemExit(
            f"Collection not found: {name}\n"
            f"path: {path}\n"
            f"available collections: {available}"
        ) from exc


def _print_summary(collection: Any) -> None:
    result = collection.get(include=["metadatas"])
    metadatas = result.get("metadatas", [])

    print(f"total: {collection.count()}")
    print(f"shown: {len(metadatas)}")
    print()
    print("by user_id:")
    for user_id, count in Counter(meta.get("user_id") for meta in metadatas).most_common():
        print(f"  {user_id}: {count}")

    print()
    print("by topic:")
    for topic, count in Counter(meta.get("topic") for meta in metadatas).most_common():
        print(f"  {topic}: {count}")

    print()
    print("by doc_type:")
    for doc_type, count in Counter(meta.get("doc_type") for meta in metadatas).most_common():
        print(f"  {doc_type}: {count}")


def _print_records(
    collection: Any,
    *,
    user_id: str | None,
    limit: int,
    show_embeddings: bool,
    full_text: bool,
) -> None:
    include = ["documents", "metadatas"]
    if show_embeddings:
        include.append("embeddings")

    kwargs: dict[str, Any] = {
        "limit": limit,
        "include": include,
    }
    if user_id is not None:
        kwargs["where"] = {"user_id": user_id}

    result = collection.get(**kwargs)
    ids = result.get("ids", [])
    documents = result.get("documents", [])
    metadatas = result.get("metadatas", [])
    embeddings = result.get("embeddings") if show_embeddings else None

    print(f"total: {collection.count()}")
    print(f"shown: {len(ids)}")

    for index, record_id in enumerate(ids):
        document = documents[index]
        metadata = metadatas[index]
        print()
        print(f"--- {index} {record_id} ---")
        print(f"meta: {metadata}")

        if embeddings is not None:
            embedding = embeddings[index]
            print(f"embedding_dim: {len(embedding)}")
            print(f"embedding_head: {list(embedding[:8])}")

        text = document if full_text else _preview(document)
        print("text:")
        print(text)


def _export_csv(
    collection: Any,
    *,
    output_path: str,
    user_id: str | None,
    embedding_head_count: int,
) -> None:
    """Export Chroma documents, metadata, and embedding heads to CSV."""
    kwargs: dict[str, Any] = {
        "include": ["documents", "metadatas", "embeddings"],
    }
    if user_id is not None:
        kwargs["where"] = {"user_id": user_id}

    result = collection.get(**kwargs)
    ids = result.get("ids", [])
    documents = result.get("documents", [])
    metadatas = result.get("metadatas", [])
    embeddings = result.get("embeddings")
    if embeddings is None:
        embeddings = []

    fieldnames = [
        "id",
        "user_id",
        "topic",
        "doc_type",
        "source_type",
        "source_url",
        "week",
        "date",
        "confidence",
        "chunk_id",
        "embedding_dim",
        "embedding_head",
        "text",
    ]

    with open(output_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record_id, document, metadata, embedding in zip(
            ids,
            documents,
            metadatas,
            embeddings,
        ):
            writer.writerow(
                {
                    "id": record_id,
                    "user_id": metadata.get("user_id"),
                    "topic": metadata.get("topic"),
                    "doc_type": metadata.get("doc_type"),
                    "source_type": metadata.get("source_type"),
                    "source_url": metadata.get("source_url"),
                    "week": metadata.get("week"),
                    "date": metadata.get("date"),
                    "confidence": metadata.get("confidence"),
                    "chunk_id": metadata.get("chunk_id"),
                    "embedding_dim": len(embedding),
                    "embedding_head": ", ".join(
                        str(float(value)) for value in embedding[:embedding_head_count]
                    ),
                    "text": document,
                }
            )

    print(f"exported: {len(ids)}")
    print(f"path: {output_path}")


def _preview(text: str, max_chars: int = 800) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


if __name__ == "__main__":
    main()
