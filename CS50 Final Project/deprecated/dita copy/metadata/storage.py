"""Persistent storage for DITA content metadata."""

from typing import Dict, Optional, Any, Generator, List, Tuple, Union, Set
from contextlib import contextmanager
from dataclasses import dataclass
import sqlite3
import os
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging
from uuid import uuid4
import time  # Added for retry backoff mechanism

from ..models.types import (
    MetadataTransaction,
    ProcessingPhase,
    ElementType,
    ContentScope,
)
from ..event_manager import EventManager, EventType
from ..utils.cache import ContentCache, CacheEntryType
from ..utils.logger import DITALogger

class MetadataStorage:
    """
    Handles persistent metadata storage with transactional support.
    Maintains separation between transient and persistent data.
    """

    MAX_RETRIES = 5  # Maximum retries for handling database contention

    def __init__(
        self,
        db_path: Union[str, Path],
        cache: ContentCache,
        event_manager: EventManager,
        logger: Optional[DITALogger] = None
    ):
        """Initialize storage."""
        self.db_path = Path(db_path)
        self.cache = cache
        self.event_manager = event_manager
        self.logger = logger or logging.getLogger(__name__)

        # Ensure the parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_db()

        # Transaction tracking
        self._active_transactions: Dict[str, MetadataTransaction] = {}
        self._transaction_metadata: Dict[str, Dict[str, Any]] = {}

    def _init_db(self) -> None:
        """Initialize database with schema."""
        try:
            current_dir = Path(__file__).parent
            schema_path = current_dir.parent / 'models' / 'metadata.sql'

            if not schema_path.exists():
                raise FileNotFoundError(f"Schema file not found: {schema_path}")

            schema = schema_path.read_text()

            with self._get_db() as conn:
                conn.executescript(schema)

        except Exception as e:
            self.logger.error(f"Database initialization failed: {str(e)}")
            raise

    @contextmanager
    def _get_db(self) -> Generator[sqlite3.Connection, None, None]:
        """Get database connection with context management."""
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA journal_mode=WAL;")  # Enable Write-Ahead Logging mode
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except sqlite3.OperationalError as e:
            conn.rollback()
            if "database is locked" in str(e):
                self.logger.warning("Database locked; retrying...")
                raise e
            raise
        finally:
            conn.close()

    def _retry_operation(self, func, *args, **kwargs):
        """Retry database operations to handle lock errors."""
        for attempt in range(self.MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and attempt < self.MAX_RETRIES - 1:
                    self.logger.warning(f"Retrying database operation (attempt {attempt + 1})...")
                    time.sleep(0.1)  # Retry delay
                else:
                    raise

    @contextmanager
    def transaction(
        self,
        content_id: str
    ) -> Generator[MetadataTransaction, None, None]:
        """Create metadata transaction for atomic updates."""
        transaction_id = f"txn_{content_id}_{uuid4().hex[:8]}"
        try:
            current = self._retry_operation(self.get_metadata, content_id) or {}
            transaction = MetadataTransaction(content_id=content_id, updates=current.copy())
            self._active_transactions[transaction_id] = transaction
            yield transaction
            self._retry_operation(self.commit_transaction, transaction_id)
        except Exception as e:
            self.rollback_transaction(transaction_id)
            raise
        finally:
            self._active_transactions.pop(transaction_id, None)

    def commit_transaction(self, transaction_id: str) -> None:
        """Commit a metadata transaction."""
        transaction = self._active_transactions.get(transaction_id)
        if not transaction or transaction.is_committed:
            return

        def commit():
            with self._get_db() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """DELETE FROM metadata WHERE content_id = ?""",
                    (transaction.content_id,)
                )
                for key, value in transaction.updates.items():
                    cursor.execute(
                        """INSERT INTO metadata (content_id, metadata_type, metadata_value)
                           VALUES (?, ?, ?)""",
                        (transaction.content_id, key, str(value))
                    )
                transaction.is_committed = True

        self._retry_operation(commit)
        self.cache.invalidate(
            key=f"metadata_{transaction.content_id}",
            entry_type=CacheEntryType.METADATA
        )
        self.event_manager.emit(
            EventType.CACHE_INVALIDATE,
            element_id=transaction.content_id
        )

    def rollback_transaction(self, transaction_id: str) -> None:
        """Roll back a metadata transaction."""
        transaction = self._active_transactions.pop(transaction_id, None)
        if not transaction:
            return
        self.cache.invalidate(
            key=f"metadata_{transaction.content_id}",
            entry_type=CacheEntryType.METADATA
        )

    def get_metadata(self, content_id: str) -> Dict[str, Any]:
        """Retrieve metadata for a given content ID."""
        cache_key = f"metadata_{content_id}"
        cached = self.cache.get(cache_key, CacheEntryType.METADATA)
        if cached:
            return cached

        def fetch():
            with self._get_db() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """SELECT metadata_type, metadata_value FROM metadata WHERE content_id = ?""",
                    (content_id,)
                )
                return {row["metadata_type"]: row["metadata_value"] for row in cursor.fetchall()}

        metadata = self._retry_operation(fetch) or {}
        self.cache.set(
            key=cache_key,
            data=metadata,
            entry_type=CacheEntryType.METADATA,
            element_type=ElementType.UNKNOWN,
            phase=ProcessingPhase.DISCOVERY
        )
        return metadata


    def invalidate_keys_batch(self, keys_to_invalidate: Set[str]) -> None:
        """Batch invalidate multiple keys with safety checks."""
        try:
            if not keys_to_invalidate:
                return

            with self._get_db() as conn:
                cursor = conn.cursor()
                for key in keys_to_invalidate:
                    # Delete metadata entries
                    cursor.execute(
                        "DELETE FROM metadata WHERE content_id = ?",
                        (f"key_{key}",)
                    )

                    # Invalidate cache
                    cache_key = f"metadata_key_{key}"
                    self.cache.invalidate(cache_key, entry_type=CacheEntryType.METADATA)

        except Exception as e:
            self.logger.error(f"Error in batch key invalidation: {str(e)}")

    def _get_key_dependencies(self, key: str) -> Set[str]:
        """Get all keys that depend on the given key."""
        try:
            with self._get_db() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """SELECT content_id FROM metadata
                       WHERE metadata_value LIKE ?""",
                    (f"%{key}%",)
                )
                return {row[0] for row in cursor.fetchall()}
        except Exception as e:
            self.logger.error(f"Error getting key dependencies: {str(e)}")
            return set()

    def store_bulk_metadata(self, metadata_entries: List[Tuple[str, Dict[str, Any]]]) -> None:
        """Store multiple metadata entries in bulk."""
        def store():
            with self._get_db() as conn:
                cursor = conn.cursor()
                for content_id, metadata in metadata_entries:
                    cursor.execute(
                        "DELETE FROM metadata WHERE content_id = ?",
                        (content_id,)
                    )
                    for key, value in metadata.items():
                        cursor.execute(
                            """INSERT INTO metadata (content_id, metadata_type, metadata_value)
                               VALUES (?, ?, ?)""",
                            (content_id, key, str(value))
                        )
        self._retry_operation(store)
        for content_id, _ in metadata_entries:
            self.cache.invalidate(
                key=f"metadata_{content_id}",
                entry_type=CacheEntryType.METADATA
            )

    def cleanup(self) -> None:
        """Clean up storage resources."""
        for transaction_id in list(self._active_transactions.keys()):
            self.rollback_transaction(transaction_id)
        self._active_transactions.clear()
        self.logger.debug("Metadata storage cleanup completed")
