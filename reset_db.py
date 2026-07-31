#!/usr/bin/env python
"""Reset or clean the Qdrant vector database.

Usage
-----
List available collections:

    python reset_db.py

Delete a specific collection:

    python reset_db.py --collection-name captions_lfm2.5

Delete a specific collection and skip confirmation:

    python reset_db.py -y --collection-name captions_lfm2.5

Wipe the entire database (all collections + data directory):

    python reset_db.py --full

Specify a custom database path:

    python reset_db.py --vector-db-path ./my_db_dir
"""

from __future__ import annotations

import argparse
import sys

from database import VectorStore
from models import COLLECTION_NAMES


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean / reset the Qdrant vector database.",
    )
    parser.add_argument(
        "--vector-db-path",
        default="./video_captions_db",
        help="Local directory where the Qdrant database is stored.",
    )
    parser.add_argument(
        "--qdrant-url",
        default=None,
        help="URL of a running Qdrant server (overrides local storage).",
    )
    parser.add_argument(
        "--collection-name",
        default=None,
        help=(
            "Name of the collection to delete. "
            "If omitted and --full is not set, lists available collections. "
            f"Known collections: {', '.join(COLLECTION_NAMES.values())}."
        ),
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Delete ALL collections and remove the database directory on disk.",
    )
    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Skip the confirmation prompt.",
    )
    args = parser.parse_args()

    try:
        store = VectorStore(
            collection_name=args.collection_name or "",
            db_path=args.vector_db_path,
            qdrant_url=args.qdrant_url,
        )
    except Exception as e:
        print(f"Error initializing vector store: {e}")
        sys.exit(1)

    available = store.list_collections()

    try:
        if args.full:
            if not available:
                print("No collections found — nothing to reset.")
                return
            action = "DELETE ALL collections: " + ", ".join(available)
        elif args.collection_name:
            action = f"DELETE collection '{args.collection_name}'"
        else:
            # No collection specified, no --full — just list and exit.
            if not available:
                print("No collections found in the database.")
            else:
                print("Available collections:")
                for name in available:
                    note = ""
                    for model_type, coll in COLLECTION_NAMES.items():
                        if coll == name:
                            note = f"  ({model_type})"
                            break
                    print(f"  {name}{note}")
            return

        if not args.yes:
            answer = input(f"This will {action}. Continue? [y/N] ").strip().lower()
            if answer not in ("y", "yes"):
                print("Aborted.")
                sys.exit(0)

        if args.full:
            store.reset_database()
            print("Database has been fully reset.")
        else:
            store.collection_name = args.collection_name
            deleted = store.delete_collection()
            if not deleted:
                print(f"Collection '{args.collection_name}' does not exist — nothing to delete.")
            else:
                print("Done.")
    finally:
        store.close()


if __name__ == "__main__":
    main()
