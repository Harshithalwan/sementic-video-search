#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from qdrant_client import QdrantClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query the video caption vector database for matching scenes."
    )
    parser.add_argument(
        "--query",
        required=True,
        type=str,
        help="The search query (e.g., 'a person waving' or 'empty room').",
    )
    parser.add_argument(
        "--vector-db-path",
        default="./video_captions_db",
        help="Local directory where the Qdrant database is persisted.",
    )
    parser.add_argument(
        "--qdrant-url",
        default=None,
        help="URL of a running Qdrant server (if not using local embedded mode).",
    )
    parser.add_argument(
        "--collection-name",
        default="captions",
        help="Name of the Qdrant collection to search in.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of search results to return.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Initialize client
    try:
        if args.qdrant_url:
            print(f"Connecting to Qdrant server at {args.qdrant_url}...")
            client = QdrantClient(url=args.qdrant_url)
        else:
            db_path = Path(args.vector_db_path).resolve()
            if not db_path.exists():
                print(f"Error: Local database directory '{db_path}' does not exist.")
                print("Make sure you run the captioning pipeline first to generate and store captions.")
                sys.exit(1)
            print(f"Opening local Qdrant database at {db_path}...")
            client = QdrantClient(path=str(db_path))
    except Exception as e:
        print(f"Error initializing Qdrant client: {e}")
        sys.exit(1)

    # 2. Check if collection exists
    try:
        collections = client.get_collections()
        collection_names = [c.name for c in collections.collections]
        if args.collection_name not in collection_names:
            print(f"Error: Collection '{args.collection_name}' not found in the database.")
            print(f"Available collections: {collection_names}")
            sys.exit(1)
    except Exception as e:
        print(f"Error reading database collections: {e}")
        sys.exit(1)

    # 3. Perform query
    print(f"Searching for: '{args.query}' (top {args.top_k} matches)...")
    try:
        # qdrant-client[fastembed] query method automatically embeds the query text
        results = client.query(
            collection_name=args.collection_name,
            query_text=args.query,
            limit=args.top_k
        )

        if not results:
            print("No matching captions found.")
            return

        print("\n" + "=" * 80)
        print(f"{'Score':<8} | {'Timestamp':<10} | {'Frame':<7} | {'Caption'}")
        print("-" * 80)
        for r in results:
            # client.query returns results where metadata is in payload
            payload = r.metadata or {}
            timestamp = payload.get("timestamp", "N/A")
            frame_idx = payload.get("frame_index", "N/A")
            caption = r.document or "No text"
            score = f"{r.score:.4f}"
            print(f"{score:<8} | {timestamp:<10} | {frame_idx:<7} | {caption}")
        print("=" * 80 + "\n")
    except Exception as e:
        print(f"Error executing search query: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
