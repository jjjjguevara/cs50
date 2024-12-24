"""Persistent storage for DITA content metadata."""

from typing import Dict, Optional, Any, Generator, Set, List, Tuple
from contextlib import contextmanager
import sqlite3
from datetime import datetime
from pathlib import Path
import json
import logging
from uuid import uuid4

from ..models.types import (
    MetadataTransaction,
    ProcessingPhase,
    ElementType,
    ContentScope,
    ValidationResult,
    ValidationMessage,
    ValidationSeverity
)
from ..event_manager import EventManager, EventType
from ..utils.cache import ContentCache
from ..utils.logger import DITALogger

class MetadataStorage:
    """
    Handles persistent metadata storage with transactional support.
    Maintains separation between transient and persistent data.
    """

    def __init__(
        self,
        db_path: Path,
        cache: ContentCache,
        event_manager: EventManager,
        logger: Optional[DITALogger] = None
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.db_path = db_path
        self.cache = cache
        self.event_manager = event_manager

        # Transaction tracking
        self._active_transactions: Dict[str, sqlite3.Connection] = {}
        self._transaction_metadata: Dict[str, Dict[str, Any]] = {}

        # Initialize database
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database connection and schema."""
        try:
            with self._get_db() as conn:
                # Load schema from metadata.sql
                schema_path = Path(__file__).parent.parent / "metadata.sql"
                schema = schema_path.read_text()
                conn.executescript(schema)
                self.logger.debug("Database schema initialized")

        except Exception as e:
            self.logger.error(f"Database initialization failed: {str(e)}")
            raise

    @contextmanager
    def _get_db(self) -> Generator[sqlite3.Connection, None, None]:
        """Get database connection with context management."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.row_factory = sqlite3.Row
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise
        finally:
            conn.close()

    @contextmanager
    def transaction(
        self,
        content_id: str
    ) -> Generator[MetadataTransaction, None, None]:
        """Create metadata transaction for atomic updates."""
        transaction_id = f"txn_{content_id}_{uuid4().hex[:8]}"

        try:
            # Create isolated connection
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row

            # Start transaction
            self._active_transactions[transaction_id] = conn

            # Get current metadata
            current = self.get_metadata(content_id)

            # Create transaction object
            transaction = MetadataTransaction(
                content_id=content_id,
                updates=current.copy()
            )

            # Store transaction metadata
            self._transaction_metadata[transaction_id] = {
                'content_id': content_id,
                'started_at': datetime.now().isoformat(),
                'original': current
            }

            yield transaction

            # Auto-commit if no explicit commit/rollback
            self.commit_transaction(transaction_id)

        except Exception as e:
            self.rollback_transaction(transaction_id)
            raise

        finally:
            # Cleanup
            if transaction_id in self._active_transactions:
                self._active_transactions[transaction_id].close()
                del self._active_transactions[transaction_id]
            if transaction_id in self._transaction_metadata:
                del self._transaction_metadata[transaction_id]

    def commit_transaction(self, transaction_id: str) -> None:
        """Commit a metadata transaction."""
        try:
            if transaction_id not in self._active_transactions:
                raise ValueError(f"No active transaction: {transaction_id}")

            conn = self._active_transactions[transaction_id]
            metadata = self._transaction_metadata[transaction_id]

            # Get final updates
            content_id = metadata['content_id']
            updates = metadata.get('updates', {})

            # Store metadata
            conn.execute(
                """
                INSERT OR REPLACE INTO content_metadata
                (content_id, metadata, updated_at)
                VALUES (?, ?, ?)
                """,
                (content_id, json.dumps(updates), datetime.now().isoformat())
            )
            conn.commit()

            # Invalidate cache
            self.cache.invalidate(f"metadata_{content_id}")

            # Emit event
            self.event_manager.emit(
                EventType.CACHE_INVALIDATE,
                element_id=content_id
            )

        except Exception as e:
            self.logger.error(f"Transaction commit failed: {str(e)}")
            raise

    def rollback_transaction(self, transaction_id: str) -> None:
        """Rollback a metadata transaction."""
        try:
            if transaction_id not in self._active_transactions:
                return

            conn = self._active_transactions[transaction_id]
            conn.rollback()

            metadata = self._transaction_metadata[transaction_id]
            content_id = metadata['content_id']

            # Invalidate cache
            self.cache.invalidate(f"metadata_{content_id}")

        except Exception as e:
            self.logger.error(f"Transaction rollback failed: {str(e)}")
            raise

    def get_metadata(
        self,
        content_id: str,
        scope: Optional[ContentScope] = None
    ) -> Dict[str, Any]:
        """
        Retrieve metadata for content with optional scope filtering.

        Args:
            content_id: Content identifier
            scope: Optional scope to filter metadata

        Returns:
            Dict containing metadata
        """
        try:
            # Check cache first
            cache_key = f"metadata_{content_id}"
            if cached := self.cache.get(cache_key):
                return cached

            with self._get_db() as conn:
                query = """
                    SELECT metadata
                    FROM content_metadata
                    WHERE content_id = ?
                """

                if scope:
                    query += " AND scope = ?"
                    params = (content_id, scope.value)
                else:
                    params = (content_id,)

                result = conn.execute(query, params).fetchone()

                if result:
                    metadata = json.loads(result[0])

                    # Cache result
                    self.cache.set(
                        key=cache_key,
                        data=metadata,
                        element_type=ElementType.UNKNOWN,
                        phase=ProcessingPhase.DISCOVERY
                    )

                    return metadata

                return {}

        except Exception as e:
            self.logger.error(f"Error retrieving metadata: {str(e)}")
            return {}

    def store_bulk_metadata(
        self,
        metadata_entries: List[Tuple[str, Dict[str, Any]]]
    ) -> None:
        """Store metadata for multiple content items."""
        try:
            with self._get_db() as conn:
                timestamp = datetime.now().isoformat()

                # Prepare all entries
                entries = [
                    (
                        content_id,
                        json.dumps(metadata),
                        timestamp
                    )
                    for content_id, metadata in metadata_entries
                ]

                # Bulk insert
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO content_metadata
                    (content_id, metadata, updated_at)
                    VALUES (?, ?, ?)
                    """,
                    entries
                )

                # Invalidate cache for all entries
                for content_id, _ in metadata_entries:
                    self.cache.invalidate(f"metadata_{content_id}")

        except Exception as e:
            self.logger.error(f"Error storing bulk metadata: {str(e)}")
            raise

    def cleanup(self) -> None:
        """Clean up storage resources."""
        try:
            # Roll back any active transactions
            for transaction_id in list(self._active_transactions.keys()):
                self.rollback_transaction(transaction_id)

            self._active_transactions.clear()
            self._transaction_metadata.clear()

            self.logger.debug("Metadata storage cleanup completed")

        except Exception as e:
            self.logger.error(f"Cleanup failed: {str(e)}")
            raise
