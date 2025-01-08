from typing import Dict, Set, Optional, Any, List
from pathlib import Path
import logging

from ..utils.cache import ContentCache, CacheEntryType
from ..utils.logger import DITALogger
from ..models.types import ElementType, ProcessingPhase, ProcessingRuleType

class SchemaResolver:
    """Handles schema resolution with inheritance support."""

    def __init__(
        self,
        schema_registry: Dict[str, Dict[str, Any]],
        dtd_schemas: Dict[str, Dict[str, Any]],
        cache: ContentCache,
        logger: Optional[DITALogger] = None
    ):
        """Initialize schema resolver.

        Args:
            schema_registry: Central schema storage
            dtd_schemas: DTD-derived schemas
            cache: Cache system
            logger: Optional logger
        """
        self.schema_registry = schema_registry
        self.dtd_schemas = dtd_schemas
        self.cache = cache
        self.logger = logger or DITALogger(name=__name__)

        # Track resolution state
        self._resolution_stack: List[str] = []
        self._max_resolution_depth = 10

    def _check_resolution_depth(self, schema_name: str) -> None:
        """Check resolution depth and detect cycles.

        Args:
            schema_name: Schema being resolved

        Raises:
            ValueError: If max depth exceeded or cycle detected
        """
        if schema_name in self._resolution_stack:
            raise ValueError(f"Circular schema reference detected: {schema_name}")

        if len(self._resolution_stack) >= self._max_resolution_depth:
            raise ValueError(f"Maximum resolution depth exceeded for {schema_name}")

        self._resolution_stack.append(schema_name)

    def get_schema(
        self,
        name: str,
        include_inherited: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Get schema with inheritance resolution.

        Args:
            name: Schema name
            include_inherited: Whether to include inherited schemas

        Returns:
            Optional[Dict[str, Any]]: Resolved schema if found
        """
        try:
            # Check cache first
            cache_key = f"schema_{name}_{include_inherited}"
            if cached := self.cache.get(
                key=cache_key,
                entry_type=CacheEntryType.CONTENT
            ):
                return cached

            # Get base schema
            base_schema = self.schema_registry.get(name)
            if not base_schema:
                return None

            if not include_inherited:
                return base_schema

            # Resolve inheritance
            resolved = self._resolve_schema_inheritance(name)

            # Cache result
            if resolved:
                self.cache.set(
                    key=cache_key,
                    data=resolved,
                    entry_type=CacheEntryType.CONTENT,
                    element_type=ElementType.UNKNOWN,
                    phase=ProcessingPhase.DISCOVERY
                )

            return resolved

        except Exception as e:
            self.logger.error(f"Error getting schema {name}: {str(e)}")
            return None

    def get_dtd_schema(
        self,
        dtd_name: str,
        include_inherited: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Get schema derived from DTD.

        Args:
            dtd_name: DTD schema name
            include_inherited: Whether to include inherited schemas

        Returns:
            Optional[Dict[str, Any]]: Resolved DTD schema if found
        """
        try:
            if dtd_schema := self.dtd_schemas.get(dtd_name):
                if not include_inherited:
                    return dtd_schema['schema']

                # Include inherited schemas
                return self._resolve_dtd_inheritance(dtd_name)
            return None

        except Exception as e:
            self.logger.error(f"Error getting DTD schema: {str(e)}")
            return None

    def get_element_schema(
        self,
        element_type: ElementType,
        processing_type: ProcessingRuleType
    ) -> Optional[Dict[str, Any]]:
        """Get element-specific schema with inheritance.

        Args:
            element_type: Type of element
            processing_type: Type of processing

        Returns:
            Optional[Dict[str, Any]]: Element schema if found
        """
        try:
            # Get processing schema
            if processing_schema := self.get_schema("processing"):
                # Get rule type section
                if rule_section := processing_schema.get("rules", {}).get(processing_type.value):
                    # Look for element-specific rules
                    return rule_section.get(element_type.value)

            return None

        except Exception as e:
            self.logger.error(f"Error getting element schema: {str(e)}")
            return None

    def _resolve_schema_inheritance(
        self,
        schema_name: str,
        visited: Optional[Set[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """Resolve schema inheritance with cycle detection.

        Args:
            schema_name: Schema to resolve
            visited: Set of visited schemas

        Returns:
            Optional[Dict[str, Any]]: Resolved schema
        """
        try:
            if visited is None:
                visited = set()

            self._check_resolution_depth(schema_name)

            try:
                # Get base schema
                base_schema = self.schema_registry.get(schema_name)
                if not base_schema:
                    return None

                # Start with base schema
                resolved = base_schema.copy()

                # Process parent schemas
                if parent_refs := base_schema.get('extends', []):
                    for parent in parent_refs:
                        if parent_schema := self._resolve_schema_inheritance(
                            parent, visited | {schema_name}
                        ):
                            # Merge parent into resolved
                            resolved = self._merge_with_parent(parent_schema, resolved)

                return resolved

            finally:
                self._resolution_stack.pop()

        except Exception as e:
            self.logger.error(
                f"Error resolving schema inheritance for {schema_name}: {str(e)}"
            )
            return None

    def _resolve_dtd_inheritance(
        self,
        dtd_name: str,
        visited: Optional[Set[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """Resolve DTD inheritance chain.

        Args:
            dtd_name: DTD schema name
            visited: Set of visited DTDs

        Returns:
            Optional[Dict[str, Any]]: Resolved DTD schema
        """
        try:
            if visited is None:
                visited = set()

            self._check_resolution_depth(dtd_name)

            try:
                # Get base DTD schema
                if dtd_info := self.dtd_schemas.get(dtd_name):
                    base_schema = dtd_info['schema'].copy()

                    # Process inheritance chain
                    if inheritance := dtd_info.get('inheritance', []):
                        for parent in inheritance:
                            if parent_schema := self._resolve_dtd_inheritance(
                                parent, visited | {dtd_name}
                            ):
                                # Merge parent into base
                                base_schema = self._merge_with_parent(
                                    parent_schema,
                                    base_schema
                                )

                    return base_schema

                return None

            finally:
                self._resolution_stack.pop()

        except Exception as e:
            self.logger.error(
                f"Error resolving DTD inheritance for {dtd_name}: {str(e)}"
            )
            return None

    def _merge_with_parent(
        self,
        parent: Dict[str, Any],
        child: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge parent schema into child schema.

        Args:
            parent: Parent schema
            child: Child schema

        Returns:
            Dict[str, Any]: Merged schema
        """
        result = parent.copy()
        for key, value in child.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_with_parent(result[key], value)
            else:
                result[key] = value
        return result
