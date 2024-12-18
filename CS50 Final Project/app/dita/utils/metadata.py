# app/dita/utils/metadata.py

from typing import(
    ContextManager,
    Generator,
    Optional,
    Callable,
    TypeVar,
    Tuple,
    Dict,
    List,
    Any,
    Set,
)
from datetime import datetime
from contextlib import contextmanager
from dataclasses import dataclass, field
import sqlite3
from pathlib import Path
import logging
import yaml
from uuid import uuid4


from lxml import etree
from lxml.etree import _Element
import frontmatter
import json
import re
from enum import Enum
from app.dita.models.types import(
    PathLike,
    ElementType,
    ProcessingPhase,
    ProcessingState,
    ContentType,
    MetadataField,
    MetadataTransaction,
    ValidationSeverity,
    ValidationMessage,
    ValidationResult,
    YAMLFrontmatter,
    KeyDefinition,
    LogContext
)

T = TypeVar('T')

# Global config
from app_config import DITAConfig

from app.dita.utils.cache import ContentCache
from app.dita.event_manager import EventManager, EventType



class MetadataSchema:
    """Base class for metadata schemas."""
    def __init__(self, schema_type: str):
        self.schema_type = schema_type
        self.required_fields: Set[str] = set()
        self.field_types: Dict[str, type] = {}
        self.validators: Dict[str, List[Callable]] = {}

    def add_field(
        self,
        name: str,
        field_type: type,
        required: bool = False,
        validators: Optional[List[Callable]] = None
    ) -> None:
        """Add field to schema."""
        self.field_types[name] = field_type
        if required:
            self.required_fields.add(name)
        if validators:
            self.validators[name] = validators



class MetadataHandler:
    def __init__(
        self,
        event_manager: EventManager,
        content_cache: ContentCache,
        config: Optional[DITAConfig] = None
    ):
        """
        Initialize metadata handler with event management and caching.

        Args:
            event_manager: Event management system
            content_cache: Content caching system
            config: Optional configuration settings
        """
        self.logger = logging.getLogger(__name__)
        self.event_manager = event_manager
        self.content_cache = content_cache
        self.config = config


        # Database connection
        self._conn: Optional[sqlite3.Connection] = None

        # Transaction management
        self._active_transactions: Dict[str, MetadataTransaction] = {}
        self._transaction_locks: Set[str] = set()

        # Cache management
        self._dirty_keys: Set[str] = set()
        self._invalidation_queue: List[str] = []

        # Initialize cache for metadata validation results
        self._validation_cache: Dict[str, bool] = {}


        # Validation schemas
        self._schemas: Dict[str, MetadataSchema] = {}
        self._init_schemas()

        # Validation cache
        self._validation_results: Dict[str, ValidationResult] = {}

    def _init_schemas(self) -> None:
        """Initialize metadata schemas."""
        # Base content schema
        base_schema = MetadataSchema("base")
        base_schema.add_field("title", str, required=True)
        base_schema.add_field("content_type", str, required=True)
        base_schema.add_field("created_at", datetime, required=True)
        base_schema.add_field("updated_at", datetime, required=True)
        self._schemas["base"] = base_schema

        # Topic schema
        topic_schema = MetadataSchema("topic")
        topic_schema.add_field("topic_type", str, required=True)
        topic_schema.add_field("short_desc", str)
        topic_schema.add_field("prerequisites", list)
        topic_schema.add_field("related_topics", list)
        self._schemas["topic"] = topic_schema

        # Map schema
        map_schema = MetadataSchema("map")
        map_schema.add_field("topics", list, required=True)
        map_schema.add_field("toc_enabled", bool)
        map_schema.add_field("index_numbers_enabled", bool)
        self._schemas["map"] = map_schema

    #############################
    # Metadata Validation methods
    #############################

    def validate_metadata(
        self,
        metadata: Dict[str, Any],
        schema_type: str = "base"
    ) -> ValidationResult:
        """
        Validate metadata against schema.

        Args:
            metadata: Metadata to validate
            schema_type: Type of schema to use

        Returns:
            ValidationResult with validation details
        """
        try:
            # Check cache
            cache_key = f"validation_{hash(json.dumps(metadata))}_{schema_type}"
            if cached := self._validation_results.get(cache_key):
                return cached

            messages: List[ValidationMessage] = []
            schema = self._schemas.get(schema_type)

            if not schema:
                raise ValueError(f"Unknown schema type: {schema_type}")

            # Check required fields
            for field in schema.required_fields:
                if field not in metadata:
                    messages.append(ValidationMessage(
                        path=field,
                        message=f"Required field '{field}' is missing",
                        severity=ValidationSeverity.ERROR,
                        code="missing_required"
                    ))

            # Validate field types
            for field, value in metadata.items():
                if field in schema.field_types:
                    expected_type = schema.field_types[field]
                    if not isinstance(value, expected_type):
                        messages.append(ValidationMessage(
                            path=field,
                            message=f"Field '{field}' should be type {expected_type.__name__}",
                            severity=ValidationSeverity.ERROR,
                            code="invalid_type"
                        ))

            # Run custom validators
            for field, validators in schema.validators.items():
                if field in metadata:
                    value = metadata[field]
                    for validator in validators:
                        try:
                            validator(value)
                        except Exception as e:
                            messages.append(ValidationMessage(
                                path=field,
                                message=str(e),
                                severity=ValidationSeverity.ERROR,
                                code="validation_failed"
                            ))

            # Create result
            result = ValidationResult(
                is_valid=not any(
                    msg.severity == ValidationSeverity.ERROR
                    for msg in messages
                ),
                messages=messages
            )

            # Cache result
            self._validation_results[cache_key] = result
            return result

        except Exception as e:
            self.logger.error(f"Validation failed: {str(e)}")
            return ValidationResult(
                is_valid=False,
                messages=[ValidationMessage(
                    path="",
                    message=str(e),
                    severity=ValidationSeverity.ERROR,
                    code="validation_error"
                )]
            )

    def batch_validate(
        self,
        items: List[Tuple[Dict[str, Any], str]]
    ) -> Dict[str, ValidationResult]:
        """
        Validate multiple metadata items in batch.

        Args:
            items: List of (metadata, schema_type) tuples

        Returns:
            Dict mapping content IDs to validation results
        """
        results = {}

        for metadata, schema_type in items:
            content_id = metadata.get("content_id", str(hash(json.dumps(metadata))))
            results[content_id] = self.validate_metadata(metadata, schema_type)

        return results

    def _validate_update(
        self,
        current: Dict[str, Any],
        updates: Dict[str, Any],
        schema_type: str
    ) -> List[str]:
        """
        Validate metadata updates for conflicts.

        Args:
            current: Current metadata state
            updates: Proposed updates
            schema_type: Schema type to validate against

        Returns:
            List of conflict messages
        """
        conflicts = []

        # Validate complete metadata after merge
        merged = self._merge_metadata(current, updates)
        result = self.validate_metadata(merged, schema_type)

        if not result.is_valid:
            conflicts.extend([msg.message for msg in result.messages])

        return conflicts

    def _get_current_metadata(self, content_id: str) -> Dict[str, Any]:
        """
        Get current metadata state for content.

        Args:
            content_id: Content identifier

        Returns:
            Current metadata state
        """
        try:
            if self._conn is None:
                raise RuntimeError("Database connection not initialized")

            cursor = self._conn.execute("""
                SELECT metadata
                FROM metadata_store
                WHERE content_id = ?
            """, (content_id,))

            if row := cursor.fetchone():
                return json.loads(row[0])
            return {}

        except Exception as e:
            self.logger.error(f"Error getting current metadata: {str(e)}")
            raise


    @contextmanager
    def transaction(self, content_id: str) -> Generator[MetadataTransaction, None, None]:
        """
        Create a managed transaction context.

        Args:
            content_id: Content identifier for the transaction

        Yields:
            MetadataTransaction for the operation
        """
        if content_id in self._transaction_locks:
            raise RuntimeError(f"Content {content_id} is locked by another transaction")

        transaction = MetadataTransaction(content_id=content_id, updates={})
        self._transaction_locks.add(content_id)
        self._active_transactions[content_id] = transaction

        try:
            yield transaction

            if transaction.updates:
                self._commit_transaction(transaction)

        except Exception as e:
            self._rollback_transaction(transaction)
            raise
        finally:
            self._transaction_locks.remove(content_id)
            self._active_transactions.pop(content_id, None)

    def _commit_transaction(self, transaction: MetadataTransaction) -> None:
        """
        Commit a metadata transaction.

        Args:
            transaction: Transaction to commit
        """
        try:
            if self._conn is None:
                raise RuntimeError("Database connection not initialized")

            # Start database transaction
            with self._conn:
                # Apply updates
                current_metadata = self._get_current_metadata(transaction.content_id)
                merged_metadata = self._merge_metadata(
                    current_metadata,
                    transaction.updates
                )

                # Store updated metadata
                self._conn.execute("""
                    INSERT INTO metadata_store (content_id, metadata, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(content_id) DO UPDATE SET
                    metadata = excluded.metadata,
                    updated_at = excluded.updated_at
                """, (
                    transaction.content_id,
                    json.dumps(merged_metadata),
                    transaction.timestamp.isoformat()
                ))

            # Invalidate cache
            self.invalidate_metadata(transaction.content_id)

            # Mark transaction as committed
            transaction.is_committed = True

            # Emit event
            self.event_manager.emit(
                EventType.STATE_CHANGE,
                element_id=transaction.content_id,
                old_state=ProcessingState.PROCESSING,
                new_state=ProcessingState.COMPLETED
            )

        except Exception as e:
            self.logger.error(f"Transaction commit failed: {str(e)}")
            raise

    def _rollback_transaction(self, transaction: MetadataTransaction) -> None:
        """Rollback a failed transaction."""
        try:
            if self._conn is not None:
                self._conn.rollback()

            # Emit event
            self.event_manager.emit(
                EventType.ERROR,
                error_type="transaction_rollback",
                element_id=transaction.content_id,
                message="Transaction rollback"
            )

        except Exception as e:
            self.logger.error(f"Transaction rollback failed: {str(e)}")
            raise

    def invalidate_metadata(self, content_id: str) -> None:
        """
        Invalidate metadata cache for content.

        Args:
            content_id: Content identifier to invalidate
        """
        try:
            # Add to dirty set
            self._dirty_keys.add(content_id)

            # Add to invalidation queue
            self._invalidation_queue.append(content_id)

            # Remove from cache
            cache_key = f"metadata_{content_id}"
            self.content_cache.invalidate(cache_key)

            # Emit event
            self.event_manager.emit(
                EventType.CACHE_INVALIDATE,
                element_id=content_id
            )

        except Exception as e:
            self.logger.error(f"Cache invalidation failed: {str(e)}")
            raise

    def _merge_metadata(
        self,
        current: Dict[str, Any],
        updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge metadata updates with conflict resolution.

        Args:
            current: Current metadata state
            updates: New updates to apply

        Returns:
            Merged metadata dictionary
        """
        merged = current.copy()

        for key, value in updates.items():
            if key in current and isinstance(current[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                merged[key] = self._merge_metadata(current[key], value)
            else:
                # For non-dict values, newer value wins
                merged[key] = value

        return merged

    def process_batch(
        self,
        updates: List[Tuple[str, Dict[str, Any]]]
    ) -> List[str]:
        """
        Process batch metadata updates.

        Args:
            updates: List of (content_id, updates) tuples

        Returns:
            List of failed content IDs
        """
        failed_ids = []

        for content_id, metadata in updates:
            try:
                with self.transaction(content_id) as txn:
                    txn.updates = metadata
            except Exception as e:
                self.logger.error(f"Batch update failed for {content_id}: {str(e)}")
                failed_ids.append(content_id)

        return failed_ids

    def _process_invalidation_queue(self) -> None:
        """Process pending cache invalidations."""
        try:
            while self._invalidation_queue:
                content_id = self._invalidation_queue.pop(0)

                # Remove from cache
                cache_key = f"metadata_{content_id}"
                self.content_cache.invalidate(cache_key)

                # Emit event
                self.event_manager.emit(
                    EventType.CACHE_INVALIDATE,
                    element_id=content_id
                )

        except Exception as e:
            self.logger.error(f"Error processing invalidation queue: {str(e)}")
            raise


    ##########################################################################
    # Schema management methods
    ##########################################################################


    def register_schema(
        self,
        name: str,
        schema: MetadataSchema
    ) -> None:
        """
        Register a new metadata schema.

        Args:
            name: Schema name
            schema: Schema definition
        """
        try:
            if name in self._schemas:
                raise ValueError(f"Schema {name} already exists")

            self._schemas[name] = schema

            # Clear validation cache
            self._validation_results.clear()

            # Emit event
            self.event_manager.emit(
                EventType.STATE_CHANGE,
                old_state=ProcessingState.PROCESSING,
                new_state=ProcessingState.COMPLETED
            )

        except Exception as e:
            self.logger.error(f"Schema registration failed: {str(e)}")
            raise

    def extend_schema(
        self,
        base_name: str,
        extension_name: str,
        additional_fields: Dict[str, Tuple[type, bool, Optional[List[Callable]]]]
    ) -> None:
        """
        Extend an existing schema.

        Args:
            base_name: Name of base schema
            extension_name: Name for new schema
            additional_fields: Dict of field_name -> (type, required, validators)
        """
        try:
            if base_name not in self._schemas:
                raise ValueError(f"Base schema {base_name} not found")

            if extension_name in self._schemas:
                raise ValueError(f"Schema {extension_name} already exists")

            # Create new schema from base
            base_schema = self._schemas[base_name]
            new_schema = MetadataSchema(extension_name)

            # Copy base fields
            new_schema.required_fields = base_schema.required_fields.copy()
            new_schema.field_types = base_schema.field_types.copy()
            new_schema.validators = base_schema.validators.copy()

            # Add new fields
            for field_name, (field_type, required, validators) in additional_fields.items():
                new_schema.add_field(field_name, field_type, required, validators)

            # Register new schema
            self.register_schema(extension_name, new_schema)

        except Exception as e:
            self.logger.error(f"Schema extension failed: {str(e)}")
            raise

    ##########################################################################
    # Metadata extraction methods
    ##########################################################################


    def _init_db_connection(self) -> sqlite3.Connection:
            """Initialize SQLite connection with proper settings."""
            try:
                db_path = Path('metadata.db')  # You might want to make this configurable
                conn = sqlite3.connect(str(db_path))
                conn.row_factory = sqlite3.Row
                return conn
            except Exception as e:
                self.logger.error(f"Failed to initialize database connection: {str(e)}")
                raise

    def configure(self, config: DITAConfig) -> None:
            """
            Configure metadata handler with provided settings.

            Args:
                config: Configuration object containing settings
            """
            try:
                if self._conn is not None:
                    self._conn.close()

                self._conn = sqlite3.connect(str(config.metadata_db_path))
                self._conn.row_factory = sqlite3.Row

                # Update configuration
                self.config = config

                self.logger.debug("Metadata handler configuration completed")

                # Emit configuration event
                self.event_manager.emit(
                    EventType.STATE_CHANGE,
                    old_state=ProcessingState.PENDING,
                    new_state=ProcessingState.PROCESSING
                )

            except Exception as e:
                self.logger.error(f"Metadata handler configuration failed: {str(e)}")
                raise

    def extract_metadata(
        self,
        file_path: Path,
        content_id: str,
        heading_id: Optional[str] = None,
        map_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Extract metadata with event tracking and caching.

        Args:
            file_path: Path to content file
            content_id: Unique content identifier
            heading_id: Optional heading identifier
            map_metadata: Optional parent map metadata

        Returns:
            Dict containing extracted metadata
        """
        try:
            # Check cache first
            cache_key = f"metadata_{content_id}"
            if cached := self.content_cache.get(cache_key):
                return cached

            # Start metadata extraction
            self.event_manager.emit(
                EventType.PHASE_START,
                element_id=content_id,
                phase=ProcessingPhase.DISCOVERY
            )

            # Determine content type
            content_type = self._determine_content_type(file_path)

            # Extract metadata based on content type
            metadata = self._extract_content_metadata(
                file_path=file_path,
                content_id=content_id,
                content_type=content_type,
                heading_id=heading_id
            )

            # Apply map metadata if provided
            if map_metadata:
                metadata.update(map_metadata)

            # Enrich metadata
            metadata = self._enrich_metadata(metadata, map_metadata)

            # Validate metadata
            validation_result = self.validate_metadata(metadata)
            if not validation_result.is_valid:
                error_messages = [msg.message for msg in validation_result.messages]
                raise ValueError(
                    f"Invalid metadata for {content_id}: {'; '.join(error_messages)}"
                )

            # Cache results
            self.content_cache.set(
                cache_key,
                metadata,
                ElementType.UNKNOWN,  # Use appropriate type based on content
                ProcessingPhase.DISCOVERY
            )

            # Complete extraction
            self.event_manager.emit(
                EventType.PHASE_END,
                element_id=content_id,
                phase=ProcessingPhase.DISCOVERY
            )

            return metadata

        except Exception as e:
            self.logger.error(f"Error extracting metadata from {file_path}: {str(e)}")
            self.event_manager.emit(
                EventType.ERROR,
                error_type="metadata_extraction",
                message=str(e),
                context=str(file_path)
            )
            return {}

    def store_metadata(
        self,
        content_id: str,
        metadata: Dict[str, Any]
    ) -> None:
        """
        Store metadata with event tracking.

        Args:
            content_id: Content identifier
            metadata: Metadata to store
        """
        try:
            # Start storage phase
            self.event_manager.emit(
                EventType.PHASE_START,
                element_id=content_id,
                phase=ProcessingPhase.DISCOVERY
            )

            # Ensure database connection
            if self._conn is None:
                raise RuntimeError("Database connection not initialized")

            # Store metadata
            with self._conn as conn:
                conn.execute("""
                    INSERT INTO metadata_store (content_id, metadata)
                    VALUES (?, ?)
                    ON CONFLICT(content_id) DO UPDATE SET
                    metadata = excluded.metadata
                """, (content_id, json.dumps(metadata)))

            # Complete storage
            self.event_manager.emit(
                EventType.PHASE_END,
                element_id=content_id,
                phase=ProcessingPhase.DISCOVERY
            )

        except Exception as e:
            self.logger.error(f"Error storing metadata for {content_id}: {str(e)}")
            self.event_manager.emit(
                EventType.ERROR,
                error_type="metadata_storage",
                message=str(e),
                context=content_id
            )
            raise

    def _enrich_metadata(
        self,
        metadata: Dict[str, Any],
        map_metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Enrich metadata with configuration-based defaults.

        Args:
            metadata: Base metadata
            map_metadata: Optional parent map metadata

        Returns:
            Enriched metadata
        """
        enriched = metadata.copy()

        # Add feature flags from config
        if self.config:
            enriched.setdefault("feature_flags", self.config.features)

        # Add default metadata fields
        enriched.setdefault("prerequisites", [])
        enriched.setdefault("related_topics", [])

        # Merge map metadata if provided
        if map_metadata:
            enriched["context"] = map_metadata.get("context", {})
            enriched["feature_flags"].update(
                map_metadata.get("feature_flags", {})
            )

        # Validate enriched metadata
        validation_result = self.validate_metadata(enriched)
        if not validation_result.is_valid:
            self.logger.warning(
                f"Enriched metadata validation failed: "
                f"{'; '.join(msg.message for msg in validation_result.messages)}"
            )

        return enriched



    def _determine_content_type(self, file_path: Path) -> ContentType:
        """
        Determine content type from file extension.

        Args:
            file_path: Path to content file

        Returns:
            ContentType: Determined content type
        """
        suffix = file_path.suffix.lower()
        if suffix == '.ditamap':
            return ContentType.MAP
        elif suffix == '.dita':
            return ContentType.DITA
        elif suffix == '.md':
            return ContentType.MARKDOWN
        else:
            return ContentType.UNKNOWN

    def add_index_entry(
        self,
        topic_id: str,
        term: str,
        entry_type: str,
        target_id: Optional[str] = None
    ) -> None:
        """Add an index entry."""
        try:
            if self._conn is None:
                raise RuntimeError("Database connection not initialized")

            with self._conn as conn:
                conn.execute("""
                    INSERT INTO index_entries (topic_id, term, type, target_id)
                    VALUES (?, ?, ?, ?)
                """, (topic_id, term, entry_type, target_id))

        except Exception as e:
            self.logger.error(f"Error adding index entry: {str(e)}")
            raise

    def add_conditional_attribute(
        self,
        name: str,
        attr_type: str,
        scope: str,
        description: Optional[str] = None
    ) -> None:
        """Add a conditional processing attribute."""
        try:
            if self._conn is None:
                raise RuntimeError("Database connection not initialized")

            with self._conn as conn:
                conn.execute("""
                    INSERT INTO conditional_attributes
                    (name, attribute_type, scope, description)
                    VALUES (?, ?, ?, ?)
                """, (name, attr_type, scope, description))

        except Exception as e:
            self.logger.error(f"Error adding conditional attribute: {str(e)}")
            raise

    def _extract_content_metadata(
        self,
        file_path: Path,
        content_id: str,
        content_type: ContentType,
        heading_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract metadata based on content type.

        Args:
            file_path: Path to content file
            content_id: Content identifier
            content_type: Type of content
            heading_id: Optional heading identifier

        Returns:
            Dict containing extracted metadata
        """
        try:
            if content_type == ContentType.DITA:
                # Initialize XML parser if needed
                parser = etree.XMLParser(
                    recover=True,
                    remove_blank_text=True,
                    resolve_entities=False,
                    dtd_validation=False,
                    load_dtd=False,
                    no_network=True
                )

                tree = etree.parse(str(file_path), parser)

                metadata = {
                    'content_type': content_type.value,
                    'content_id': content_id,
                    'heading_id': heading_id,
                    'source_file': str(file_path),
                    'processed_at': datetime.now().isoformat()
                }

                # Extract metadata from othermeta elements
                for othermeta in tree.xpath('.//othermeta'):
                    name = othermeta.get('name')
                    content = othermeta.get('content')
                    if name and content:
                        if name in ['index-numbers', 'append-toc']:
                            metadata[name] = content.lower() == 'true'
                        else:
                            metadata[name] = content

                return metadata

            elif content_type == ContentType.MARKDOWN:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    metadata, _ = self.extract_yaml_metadata(content)
                    metadata.update({
                        'content_type': content_type.value,
                        'content_id': content_id,
                        'heading_id': heading_id,
                        'source_file': str(file_path),
                        'processed_at': datetime.now().isoformat()
                    })
                    return metadata

            else:
                self.logger.warning(f"Unsupported content type: {content_type}")
                return {
                    'content_type': content_type.value,
                    'content_id': content_id,
                    'heading_id': heading_id,
                    'source_file': str(file_path),
                    'processed_at': datetime.now().isoformat()
                }

        except Exception as e:
            self.logger.error(f"Error extracting content metadata: {str(e)}")
            raise



    def _check_metadata_flag(self, prolog: _Element, flag_name: str) -> bool:
        """Check if a metadata flag is set to true"""
        for meta in prolog.findall('.//othermeta'):
            name = meta.get('name')
            content = meta.get('content')
            if name == flag_name and isinstance(content, str):
                return content.lower() == 'true'
        return False

    def extract_yaml_metadata(
        self,
        content: str,
        parent_flags: Optional[Dict[str, bool]] = None
    ) -> YAMLFrontmatter:
        """
        Extract and validate YAML frontmatter.

        Args:
            content: Content string
            parent_flags: Optional parent feature flags

        Returns:
            Structured YAML frontmatter data
        """
        try:
            # Extract YAML block
            if not content.startswith('---\n'):
                return YAMLFrontmatter(
                    feature_flags={},
                    relationships={},
                    context={},
                    raw_data={}
                )

            end_idx = content.find('\n---\n', 4)
            if end_idx == -1:
                raise ValueError("Invalid YAML frontmatter format")

            # Parse YAML
            yaml_content = content[4:end_idx]
            raw_data = yaml.safe_load(yaml_content)

            # Extract feature flags with inheritance
            feature_flags = parent_flags.copy() if parent_flags else {}
            if yaml_flags := raw_data.get('features', {}):
                feature_flags.update(yaml_flags)

            # Extract relationships
            relationships = {
                'prerequisites': raw_data.get('prerequisites', []),
                'related_topics': raw_data.get('related_topics', []),
                'children': raw_data.get('children', []),
                'parents': raw_data.get('parents', [])
            }

            # Extract context
            context = {
                'scope': raw_data.get('scope', 'local'),
                'processing-role': raw_data.get('processing-role', 'normal'),
                'format': raw_data.get('format'),
                'type': raw_data.get('type', 'topic'),
                'platform': raw_data.get('platform'),
                'audience': raw_data.get('audience'),
                'props': raw_data.get('props', {})
            }

            # Log extraction
            self._log_operation(
                'yaml_extraction',
                str(hash(content)),
                {
                    'features_count': len(feature_flags),
                    'relationships_count': sum(len(r) for r in relationships.values()),
                    'has_context': bool(context)
                }
            )

            return YAMLFrontmatter(
                feature_flags=feature_flags,
                relationships=relationships,
                context=context,
                raw_data=raw_data
            )

        except Exception as e:
            self.logger.error(json.dumps({
                "error": "yaml_extraction_failed",
                "message": str(e),
                "content_hash": str(hash(content))
            }))
            raise

    def store_key_definition(
        self,
        key_def: KeyDefinition
    ) -> None:
        """
        Store DITA key definition.

        Args:
            key_def: Key definition to store
        """
        try:
            if self._conn is None:
                raise RuntimeError("Database connection not initialized")

            with self._conn:
                # Store key definition
                self._conn.execute("""
                    INSERT INTO key_definitions (
                        key_id, href, scope, processing_role,
                        metadata, source_map, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(key_id) DO UPDATE SET
                    href = excluded.href,
                    scope = excluded.scope,
                    processing_role = excluded.processing_role,
                    metadata = excluded.metadata,
                    source_map = excluded.source_map
                """, (
                    key_def.key,
                    key_def.href,
                    key_def.scope,
                    key_def.processing_role,
                    json.dumps(key_def.metadata),
                    key_def.source_map,
                    datetime.now().isoformat()
                ))

            # Log operation
            self._log_operation(
                'key_definition_stored',
                key_def.key,
                {
                    'href': key_def.href,
                    'scope': key_def.scope,
                    'source_map': key_def.source_map
                }
            )

        except Exception as e:
            self.logger.error(json.dumps({
                "error": "key_definition_store_failed",
                "key": key_def.key,
                "message": str(e)
            }))
            raise

    def get_key_definitions(
        self,
        map_id: str,
        scope: Optional[str] = None
    ) -> Dict[str, KeyDefinition]:
        """
        Get key definitions with scope handling.

        Args:
            map_id: Map identifier
            scope: Optional scope filter

        Returns:
            Dict of key definitions
        """
        try:
            if self._conn is None:
                raise RuntimeError("Database connection not initialized")

            query = """
                SELECT k.*, m.metadata as map_metadata
                FROM key_definitions k
                JOIN maps m ON k.source_map = m.map_id
                WHERE k.source_map = ?
            """
            params = [map_id]

            if scope:
                query += " AND k.scope = ?"
                params.append(scope)

            cursor = self._conn.execute(query, params)
            key_defs = {}

            for row in cursor:
                key_defs[row['key_id']] = KeyDefinition(
                    key=row['key_id'],
                    href=row['href'],
                    scope=row['scope'],
                    processing_role=row['processing_role'],
                    metadata=json.loads(row['metadata']),
                    source_map=row['source_map']
                )

            # Log retrieval
            self._log_operation(
                'key_definitions_retrieved',
                map_id,
                {
                    'count': len(key_defs),
                    'scope': scope
                }
            )

            return key_defs

        except Exception as e:
            self.logger.error(json.dumps({
                "error": "key_definitions_retrieval_failed",
                "map_id": map_id,
                "message": str(e)
            }))
            raise


    def get_content_relationships(
        self,
        content_id: str
    ) -> List[Dict[str, Any]]:
        """
        Get all relationships for a content element.

        Args:
            content_id: Content identifier

        Returns:
            List of relationship dictionaries
        """
        try:
            if self._conn is None:
                raise RuntimeError("Database connection not initialized")

            with self._conn as conn:
                cur = conn.execute("""
                    SELECT
                        r.target_id,
                        r.relationship_type as type,
                        r.scope,
                        r.metadata,
                        r.created_at
                    FROM content_relationships r
                    WHERE r.source_id = ?
                    ORDER BY r.created_at ASC
                """, (content_id,))

                relationships = []
                for row in cur:
                    relationships.append({
                        'target_id': row['target_id'],
                        'type': row['type'],
                        'scope': row['scope'],
                        'metadata': json.loads(row['metadata'] or '{}'),
                        'created_at': row['created_at']
                    })

                return relationships

        except Exception as e:
            self.logger.error(f"Error getting relationships for {content_id}: {str(e)}")
            raise

    def store_content_relationships(
        self,
        content_id: str,
        relationships: List[Dict[str, Any]]
    ) -> None:
        """
        Store content relationships in the database.

        Args:
            content_id: Content identifier
            relationships: List of relationship dictionaries
        """
        try:
            if self._conn is None:
                raise RuntimeError("Database connection not initialized")

            with self._conn as conn:
                # First, clear existing relationships
                conn.execute("""
                    DELETE FROM content_relationships
                    WHERE source_id = ?
                """, (content_id,))

                # Insert new relationships
                for rel in relationships:
                    conn.execute("""
                        INSERT INTO content_relationships (
                            source_id,
                            target_id,
                            relationship_type,
                            scope,
                            metadata,
                            created_at
                        ) VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        content_id,
                        rel['target_id'],
                        rel['type'],
                        rel['scope'],
                        json.dumps(rel.get('metadata', {})),
                        datetime.now().isoformat()
                    ))

                # Invalidate cache
                self.event_manager.emit(
                    EventType.CACHE_INVALIDATE,
                    element_id=content_id
                )

        except Exception as e:
            self.logger.error(f"Error storing relationships for {content_id}: {str(e)}")
            raise


    def cleanup(self) -> None:
        """Clean up resources with event tracking."""
        try:
            # Close database connection
            if self._conn is not None:
                self._conn.close()
                self._conn = None

            # Clear caches
            self._validation_cache.clear()

            # Emit cleanup event
            self.event_manager.emit(
                EventType.STATE_CHANGE,
                old_state=ProcessingState.PROCESSING,
                new_state=ProcessingState.COMPLETED
            )

            self.logger.debug("Metadata handler cleanup completed")

            # Clear validation caches
            self._validation_results.clear()

        except Exception as e:
            self.logger.error(f"Cleanup failed: {str(e)}")
            raise




 # DEPRECATE - MOVE TO TRANSFORMER
 # def get_strategy_metadata(self, strategy: str, content_id: str) -> Dict[str, Any]:
 #    """Retrieves required metadata based on BaseTransformer strategy."""

 #    strategy_queries = {
 #        # Inject strategies
 #        "latex": """
 #            SELECT content_id, math_content
 #            FROM topic_elements
 #            WHERE element_type = 'latex' AND content_id = ?
 #        """,

 #         # Media and key definitions
 #         "image": """
 #             SELECT te.element_id, te.content_hash,
 #                     kd.href, kd.alt, kd.placement, kd.scale,
 #                     kd.width, kd.height, kd.align, kd.outputclass
 #             FROM topic_elements te
 #             JOIN element_context ec ON te.element_id = ec.element_id
 #             LEFT JOIN key_definitions kd ON te.keyref = kd.keys
 #             WHERE te.element_type = 'image' AND te.topic_id = ?
 #         """,

 #         "video": """
 #             SELECT te.element_id, te.content_hash,
 #                     kd.href, kd.width, kd.height,
 #                     kd.controls, kd.autoplay, kd.loop,
 #                     kd.poster, kd.preload, kd.outputclass
 #             FROM topic_elements te
 #             JOIN element_context ec ON te.element_id = ec.element_id
 #             LEFT JOIN key_definitions kd ON te.keyref = kd.keys
 #             WHERE te.element_type = 'video' AND te.topic_id = ?
 #         """,

 #         "audio": """
 #             SELECT te.element_id, te.content_hash,
 #                     kd.href, kd.controls, kd.autoplay,
 #                     kd.loop, kd.preload, kd.outputclass
 #             FROM topic_elements te
 #             JOIN element_context ec ON te.element_id = ec.element_id
 #             LEFT JOIN key_definitions kd ON te.keyref = kd.keys
 #             WHERE te.element_type = 'audio' AND te.topic_id = ?
 #         """,

 #         "iframe": """
 #             SELECT te.element_id, te.content_hash,
 #                     kd.href, kd.width, kd.height,
 #                     kd.sandbox, kd.allow, kd.outputclass
 #             FROM topic_elements te
 #             JOIN element_context ec ON te.element_id = ec.element_id
 #             LEFT JOIN key_definitions kd ON te.keyref = kd.keys
 #             WHERE te.element_type = 'iframe' AND te.topic_id = ?
 #         """,

 #        "topic_section": """
 #            SELECT t.id, t.title, t.content_type, t.specialization_type,
 #                   tm.features, tm.prerequisites
 #            FROM topics t
 #            LEFT JOIN content_metadata tm ON t.id = tm.content_id
 #            WHERE t.id = ?
 #        """,

 #        # Append strategies
 #        "heading_attributes": """
 #            SELECT hi.id, hi.text, hi.level, hi.sequence_number,
 #                   hi.path_fragment
 #            FROM heading_index hi
 #            WHERE hi.topic_id = ?
 #            ORDER BY hi.sequence_number
 #        """,

 #        "toc": """
 #            SELECT hi.id, hi.text, hi.level
 #            FROM heading_index hi
 #            WHERE hi.map_id = (
 #                SELECT root_map_id FROM topics WHERE id = ?
 #            )
 #            ORDER BY hi.sequence_number
 #        """,

 #        "bibliography": """
 #            SELECT citation_data
 #            FROM citations
 #            WHERE topic_id = ?
 #            ORDER BY citation_data->>'$.author'
 #        """,

 #        "glossary": """
 #            SELECT term, definition
 #            FROM topic_elements
 #            WHERE topic_id = ? AND element_type = 'dlentry'
 #        """,

 #        # Swap strategies
 #        "topic_version": """
 #            SELECT version, revision_history
 #            FROM content_items
 #            WHERE id = ?
 #        """,

 #        "topic_type": """
 #            SELECT t.type_id, tt.name, tt.base_type
 #            FROM topics t
 #            JOIN topic_types tt ON t.type_id = tt.type_id
 #            WHERE t.id = ?
 #        """
 #    }

 #    query = strategy_queries.get(strategy.lstrip("_"))
 #    if not query:
 #        return {}

 #    with self._conn as conn:
 #        cur = conn.execute(query, (content_id,))
 #        result = cur.fetchall()
 #        return {
 #            "content_id": content_id,
 #            "strategy": strategy,
 #            "data": [dict(row) for row in result]
 #        }
