"""DTD resolution and entity handling."""
from typing import (
    Dict,
    Set,
    Optional,
    Any,
    List,
    Tuple,
    Union,
    TypeVar,
    overload,
    Protocol,
    BinaryIO,
    TYPE_CHECKING
)
from pathlib import Path
from lxml import etree
from io import BufferedIOBase
from os import PathLike
from dataclasses import dataclass
from datetime import datetime
import logging
import re

if TYPE_CHECKING:
    from .dtd_models import ValidationContext, DTDParsingResult
    from .dtd_specializer import SpecializationInfo

# Type variables for lxml's generic types
_AnyStr = TypeVar("_AnyStr", str, bytes)

from app.dita.utils.logger import DITALogger

from ..types.schema.dtd.entities import DTDReference

class _ResolverInputDocument(Protocol):
    """Protocol for lxml's input document type."""
    def read(self) -> bytes: ...
    def close(self) -> None: ...



class DTDResolver(etree.Resolver):
    """Handles DTD resolution and loading."""

    def __init__(self, base_path: Path, logger: Optional[DITALogger] = None):
        self.base_path = Path(base_path)
        self.logger = logger or DITALogger(name=__name__)
        super().__init__()

        # Track loaded DTDs and their dependencies
        self._loaded_dtds: Dict[str, DTDReference] = {}
        self._dependencies: Dict[str, Set[str]] = {}
        self._entity_cache: Dict[str, str] = {}

        # Track external entities
        self._external_entities: Dict[str, Tuple[str, str]] = {}

        # Initialize catalog
        self._catalog: Dict[str, Path] = {}
        self._load_catalog()


    def resolve(
            self,
            system_url: str,
            public_id: str,
            context: Any
        ) -> Optional[Any]:
            """
            Resolve DTD references using catalog.

            Args:
                system_url: System identifier
                public_id: Public identifier
                context: Resolution context

            Returns:
                Optional resolved file
            """
            try:
                # Try catalog resolution
                if public_id and (resolved := self._catalog.get(public_id)):
                    dtd_path = resolved
                elif system_url and (resolved := self._catalog.get(system_url)):
                    dtd_path = resolved
                else:
                    self.logger.warning(f"No catalog entry for DTD: {system_url}")
                    return None

                if dtd_path.exists():
                    self.logger.debug(f"Resolved {system_url} to {dtd_path}")
                    return self.resolve_filename(str(dtd_path), context)

                self.logger.warning(f"DTD file not found: {dtd_path}")
                return None

            except Exception as e:
                self.logger.error(f"Error resolving DTD {system_url}: {str(e)}")
                return None

    def resolve_filename(  # type: ignore[override]
        self,
        filename: _AnyStr,
        context: Any
    ) -> Any:
        """
        Resolve a filename to an input source.

        Args:
            filename: File path to resolve
            context: Resolution context

        Returns:
            Input document for lxml
        """
        try:
            return super().resolve_filename(filename, context)
        except Exception as e:
            self.logger.error(f"Error resolving filename {filename}: {str(e)}")
            raise


    def load_dtd_content(self, dtd_path: Path) -> str:
            """Load DTD file content."""
            try:
                with open(dtd_path, 'r') as f:
                    return f.read()
            except Exception as e:
                self.logger.error(f"Error loading DTD content: {str(e)}")
                raise

    def _load_catalog(self) -> None:
        """Load DTD catalog if available."""
        try:
            catalog_path = self.base_path / 'catalog.xml'
            if not catalog_path.exists():
                self.logger.warning(f"Catalog file not found: {catalog_path}")
                return

            self.logger.debug(f"Loading catalog from: {catalog_path}")

            parser = etree.XMLParser(remove_blank_text=True)
            tree = etree.parse(str(catalog_path), parser)
            root = tree.getroot()

            # Process catalog entries
            namespace = "{urn:oasis:names:tc:entity:xmlns:xml:catalog}"

            # Parse public mappings
            for public in root.findall(f".//{namespace}public"):
                public_id = public.get("publicId")
                uri = public.get("uri")
                if public_id and uri:
                    resolved_path = self.base_path / uri
                    self._catalog[public_id] = resolved_path
                    self.logger.debug(f"Added public mapping: {public_id} -> {resolved_path}")

            # Parse system mappings
            for system in root.findall(f".//{namespace}system"):
                system_id = system.get("systemId")
                uri = system.get("uri")
                if system_id and uri:
                    resolved_path = self.base_path / uri
                    self._catalog[system_id] = resolved_path
                    self.logger.debug(f"Added system mapping: {system_id} -> {resolved_path}")

            self.logger.debug(f"Successfully loaded {len(self._catalog)} catalog entries")

        except Exception as e:
            self.logger.error(f"Error loading catalog: {str(e)}")
            self._catalog = {}  # Reset on error

    def resolve_dtd(self, dtd_path: Path, context_path: Optional[Path] = None) -> DTDReference:
        """Resolve DTD path to file reference."""
        try:
            dtd_str = str(dtd_path)
            if dtd_str in self._loaded_dtds:
                return self._loaded_dtds[dtd_str]

            # Handle public/system identifiers
            if dtd_str.startswith('PUBLIC'):
                return self._resolve_public_dtd(dtd_str)
            if dtd_str.startswith('SYSTEM'):
                return self._resolve_system_dtd(dtd_str)

            # Handle direct file paths
            resolved_path = self._resolve_path(str(dtd_path), context_path)  # Convert to str here
            if not resolved_path.exists():
                raise FileNotFoundError(f"DTD not found: {dtd_path}")

            reference = DTDReference(
                path=dtd_path,  # Keep as Path object
                type='system',
                resolved_path=resolved_path,
                last_modified=datetime.fromtimestamp(resolved_path.stat().st_mtime)
            )

            self._loaded_dtds[dtd_str] = reference
            return reference

        except Exception as e:
            self.logger.error(f"Error resolving DTD {dtd_path}: {str(e)}")
            raise

    def _resolve_public_dtd(self, dtd_identifier: str) -> DTDReference:
        """Resolve public DTD identifier."""
        # Parse PUBLIC identifier
        parts = dtd_identifier.split(maxsplit=2)
        if len(parts) < 3:
            raise ValueError(f"Invalid PUBLIC identifier: {dtd_identifier}")

        public_id = parts[1].strip('"\'')
        system_id = parts[2].strip('"\'') if len(parts) > 2 else None

        # Check catalog
        if public_id in self._catalog:
            resolved_path = self._catalog[public_id]
        elif system_id:
            resolved_path = self._resolve_path(system_id)
        else:
            raise ValueError(f"Unable to resolve PUBLIC identifier: {public_id}")

        reference = DTDReference(
            path=Path(dtd_identifier),
            type='public',
            public_id=public_id,
            system_id=system_id,
            resolved_path=resolved_path,
            last_modified=datetime.fromtimestamp(resolved_path.stat().st_mtime)
        )

        self._loaded_dtds[dtd_identifier] = reference
        return reference

    def _resolve_system_dtd(self, dtd_identifier: str) -> DTDReference:
        """Resolve system DTD identifier."""
        # Parse SYSTEM identifier
        parts = dtd_identifier.split(maxsplit=1)
        if len(parts) < 2:
            raise ValueError(f"Invalid SYSTEM identifier: {dtd_identifier}")

        system_id = parts[1].strip('"\'')

        # Check catalog first
        if system_id in self._catalog:
            resolved_path = self._catalog[system_id]
        else:
            resolved_path = self._resolve_path(system_id)

        if not resolved_path.exists():
            raise FileNotFoundError(f"DTD not found: {system_id}")

        reference = DTDReference(
            path=Path(dtd_identifier),
            type='system',
            system_id=system_id,
            resolved_path=resolved_path,
            last_modified=datetime.fromtimestamp(resolved_path.stat().st_mtime)
        )

        self._loaded_dtds[dtd_identifier] = reference
        return reference

    def _resolve_path(self, path_str: str, context_path: Optional[Path] = None) -> Path:
        """Resolve string path to absolute Path."""
        try:
            if path_str.startswith('/'):
                return Path(path_str).resolve()
            if context_path:
                return (context_path.parent / path_str).resolve()
            return (self.base_path / path_str).resolve()
        except Exception as e:
            raise ValueError(f"Error resolving path {path_str}: {str(e)}")
