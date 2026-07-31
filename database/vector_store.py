"""Vector database abstraction (save + query) using Qdrant."""

from __future__ import annotations

import sys
import uuid
from pathlib import Path
from typing import Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Filter


class VectorStore:
    """Thin wrapper around Qdrant for saving and querying video captions.

    The ``save_caption`` method accepts an open-ended ``metadata`` dict so
    additional fields (model name, resolution, confidence, …) can be stored
    alongside each caption without changing the call-site signature.
    """

    def __init__(
        self,
        collection_name: str = "captions",
        db_path: Optional[str] = None,
        qdrant_url: Optional[str] = None,
    ) -> None:
        self.collection_name = collection_name
        self._closed = False

        if qdrant_url:
            print(f"Connecting to Qdrant server at {qdrant_url}...")
            self._client = QdrantClient(url=qdrant_url)
        else:
            resolved = Path(db_path or "./video_captions_db").resolve()
            print(f"Initializing local Qdrant database at {resolved}...")
            self._client = QdrantClient(path=str(resolved))

        print(f"Using vector database collection: '{self.collection_name}'")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release the underlying Qdrant client and its storage lock.

        Idempotent: safe to call multiple times or after the store is no
        longer needed. In local (embedded) mode this releases the exclusive
        file lock on the storage folder so other clients/processes can open it.
        """
        if not getattr(self, "_closed", False):
            self._client.close()
            self._closed = True

    def __enter__(self) -> "VectorStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_caption(
        self,
        caption: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Embed and persist a caption with optional metadata.

        Args:
            caption: The caption text to store.
            metadata: Arbitrary key-value pairs stored alongside the
                embedding (e.g. ``timestamp``, ``frame_index``, ``source``,
                ``model_id``, …).  Defaults to an empty dict.

        Returns:
            The UUID assigned to the stored point.
        """
        caption_id = str(uuid.uuid4())
        self._client.add(
            collection_name=self.collection_name,
            documents=[caption],
            metadata=[metadata or {}],
            ids=[caption_id],
        )
        return caption_id

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        query_filter: Optional[Filter] = None,
    ) -> list:
        """Semantic search over stored captions.

        Args:
            query_text: Free-text search query.
            top_k: Number of results to return.
            query_filter: Optional Qdrant Filter for metadata filtering
                (e.g. by video_id, video_name, timestamp range).

        Returns:
            A list of Qdrant query result objects.
        """
        return self._client.query(
            collection_name=self.collection_name,
            query_text=query_text,
            query_filter=query_filter,
            limit=top_k,
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def delete_collection(self) -> bool:
        """Delete the configured collection from the database.

        Returns:
            True if the collection existed and was deleted, False otherwise.
        """
        if not self.collection_exists():
            return False
        self._client.delete_collection(collection_name=self.collection_name)
        return True

    def reset_database(self) -> None:
        """Delete ALL collections in the database."""
        for name in self.list_collections():
            self._client.delete_collection(collection_name=name)

    def list_collections(self) -> list[str]:
        """Return the names of all collections in the database."""
        collections = self._client.get_collections()
        return [c.name for c in collections.collections]

    def collection_exists(self) -> bool:
        """Check whether the configured collection exists."""
        return self.collection_name in self.list_collections()

    def ensure_collection_exists(self) -> None:
        """Exit with an error message if the collection is missing."""
        if not self.collection_exists():
            available = self.list_collections()
            print(
                f"Error: Collection '{self.collection_name}' not found in the database."
            )
            print(f"Available collections: {available}")
            sys.exit(1)

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    def list_video_names(self) -> list[str]:
        """Return all unique video names stored in the collection."""
        if not self.collection_exists():
            return []
        try:
            results = self._client.scroll(
                collection_name=self.collection_name,
                limit=10000,
                with_payload=["video_name"],
            )
            names = sorted({r.payload.get("video_name", "") for r in results[0] if r.payload})
            return [n for n in names if n]
        except Exception:
            return []
