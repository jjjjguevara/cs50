"""DTD specialization and inheritance handling."""
from typing import Dict, Set, Optional, Any, List, Tuple, TYPE_CHECKING
from pathlib import Path
import logging
from dataclasses import dataclass, field
from datetime import datetime
from .dtd_models import DTDAttribute

if TYPE_CHECKING:
    from .dtd_models import DTDAttribute, ValidationContext

@dataclass
class SpecializationInfo:
    """Information about a DTD specialization."""
    base_type: str
    specialized_type: str
    inheritance_path: List[str]
    attributes: Dict[str, DTDAttribute]
    constraints: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

class DTDSpecializationHandler:
    """Handles DTD specialization and inheritance relationships."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

        # Track specialization relationships
        self._specializations: Dict[str, SpecializationInfo] = {}
        self._inheritance_chains: Dict[str, List[str]] = {}
        self._constraint_modules: Dict[str, Dict[str, Any]] = {}

        # Cache validated specializations
        self._validated_specs: Set[str] = set()

    def register_specialization(
        self,
        base_type: str,
        specialized_type: str,
        attributes: Dict[str, Any],
        constraints: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register a DTD specialization.

        Args:
            base_type: Base DTD type
            specialized_type: Specialized type name
            attributes: Specialized attributes
            constraints: Optional constraints
            metadata: Optional metadata
        """
        try:
            # Build inheritance path
            inheritance_path = self._build_inheritance_path(base_type, specialized_type)

            # Create specialization info
            spec_info = SpecializationInfo(
                base_type=base_type,
                specialized_type=specialized_type,
                inheritance_path=inheritance_path,
                attributes=attributes,
                constraints=constraints or {},
                metadata=metadata or {}
            )

            # Store specialization
            self._specializations[specialized_type] = spec_info

            # Update inheritance chains
            self._update_inheritance_chains(spec_info)

        except Exception as e:
            self.logger.error(
                f"Error registering specialization {specialized_type}: {str(e)}"
            )
            raise

    def _build_inheritance_path(
        self,
        base_type: str,
        specialized_type: str
    ) -> List[str]:
        """Build inheritance path from base to specialized type."""
        inheritance_path = [base_type]

        # Check existing chains for intermediate types
        for chain in self._inheritance_chains.values():
            if base_type in chain and specialized_type in chain:
                start_idx = chain.index(base_type)
                end_idx = chain.index(specialized_type)
                if start_idx < end_idx:
                    return chain[start_idx:end_idx + 1]

        inheritance_path.append(specialized_type)
        return inheritance_path

    def _update_inheritance_chains(self, spec_info: SpecializationInfo) -> None:
        """Update inheritance chains with new specialization."""
        chain_key = f"{spec_info.base_type}_{spec_info.specialized_type}"
        self._inheritance_chains[chain_key] = spec_info.inheritance_path

        # Update related chains
        for existing_key, chain in self._inheritance_chains.items():
            if spec_info.base_type in chain:
                # Insert specialized type after base type
                base_idx = chain.index(spec_info.base_type)
                new_chain = chain[:base_idx + 1]
                new_chain.extend(spec_info.inheritance_path[1:])
                new_chain.extend(chain[base_idx + 1:])
                self._inheritance_chains[existing_key] = new_chain

    def validate_specialization(
        self,
        specialized_type: str,
        schema_rules: Dict[str, Any]
    ) -> bool:
        """
        Validate specialization against schema rules.

        Args:
            specialized_type: Type to validate
            schema_rules: Validation rules from schema

        Returns:
            bool: True if valid, False otherwise
        """
        try:
            if specialized_type in self._validated_specs:
                return True

            spec_info = self._specializations.get(specialized_type)
            if not spec_info:
                self.logger.error(f"Unknown specialization: {specialized_type}")
                return False

            # Validate inheritance
            if not self._validate_inheritance(spec_info, schema_rules):
                return False

            # Validate attributes
            if not self._validate_attributes(spec_info, schema_rules):
                return False

            # Validate constraints
            if not self._validate_constraints(spec_info, schema_rules):
                return False

            self._validated_specs.add(specialized_type)
            return True

        except Exception as e:
            self.logger.error(f"Error validating specialization {specialized_type}: {str(e)}")
            return False

    def _validate_inheritance(
        self,
        spec_info: SpecializationInfo,
        schema_rules: Dict[str, Any]
    ) -> bool:
        """Validate inheritance relationships."""
        base_rules = schema_rules.get('inheritance', {})

        try:
            # Check base type exists
            if spec_info.base_type not in base_rules:
                self.logger.error(f"Unknown base type: {spec_info.base_type}")
                return False

            # Validate inheritance path
            current = spec_info.base_type
            for ancestor in spec_info.inheritance_path[1:]:
                if ancestor not in base_rules.get(current, {}).get('allowed_specializations', []):
                    self.logger.error(
                        f"Invalid specialization path: {current} -> {ancestor}"
                    )
                    return False
                current = ancestor

            return True

        except Exception as e:
            self.logger.error(f"Inheritance validation error: {str(e)}")
            return False

    def _validate_attributes(
        self,
        spec_info: SpecializationInfo,
        schema_rules: Dict[str, Any]
    ) -> bool:
        """Validate specialized attributes."""
        attribute_rules = schema_rules.get('attributes', {})

        try:
            # Get base type attributes
            base_attrs = attribute_rules.get(spec_info.base_type, {})

            # Validate required attributes are present
            for attr, rules in base_attrs.items():
                if rules.get('required', False):
                    if attr not in spec_info.attributes:
                        self.logger.error(
                            f"Missing required attribute: {attr} in {spec_info.specialized_type}"
                        )
                        return False

            # Validate attribute types and values
            for attr, value in spec_info.attributes.items():
                if attr in base_attrs:
                    if not self._validate_attribute_value(
                        attr, value, base_attrs[attr]
                    ):
                        return False

            return True

        except Exception as e:
            self.logger.error(f"Attribute validation error: {str(e)}")
            return False

    def _validate_attribute_value(
        self,
        attr: str,
        value: Any,
        rules: Dict[str, Any]
    ) -> bool:
        """Validate single attribute value."""
        try:
            # Type validation
            expected_type = rules.get('type')
            if expected_type:
                if not self._check_type(value, expected_type):
                    self.logger.error(
                        f"Invalid type for {attr}: expected {expected_type}"
                    )
                    return False

            # Value constraints
            if allowed_values := rules.get('allowed_values'):
                if value not in allowed_values:
                    self.logger.error(
                        f"Invalid value for {attr}: {value} not in {allowed_values}"
                    )
                    return False

            # Pattern validation
            if pattern := rules.get('pattern'):
                if not self._validate_pattern(value, pattern):
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Value validation error: {str(e)}")
            return False

    def _validate_constraints(
        self,
        spec_info: SpecializationInfo,
        schema_rules: Dict[str, Any]
    ) -> bool:
        """Validate specialization constraints."""
        constraint_rules = schema_rules.get('constraints', {})

        try:
            for constraint_name, constraint in spec_info.constraints.items():
                # Get constraint rules
                rules = constraint_rules.get(constraint_name)
                if not rules:
                    self.logger.error(f"Unknown constraint: {constraint_name}")
                    return False

                # Validate against rules
                if not self._validate_constraint_definition(
                    constraint, rules
                ):
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Constraint validation error: {str(e)}")
            return False

    def _validate_constraint_definition(
        self,
        constraint: Dict[str, Any],
        rules: Dict[str, Any]
    ) -> bool:
        """Validate single constraint definition."""
        try:
            # Check required fields
            for field in rules.get('required_fields', []):
                if field not in constraint:
                    self.logger.error(f"Missing required field in constraint: {field}")
                    return False

            # Validate constraint type
            if constraint_type := rules.get('type'):
                if not self._check_type(constraint.get('type'), constraint_type):
                    return False

            # Validate constraint values
            if value_rules := rules.get('values', {}):
                if not self._validate_constraint_values(
                    constraint.get('values', {}),
                    value_rules
                ):
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Constraint definition error: {str(e)}")
            return False

    def _validate_constraint_values(
        self,
        values: Dict[str, Any],
        rules: Dict[str, Any]
    ) -> bool:
        """Validate constraint values."""
        try:
            for field, value in values.items():
                if field_rules := rules.get(field):
                    # Type validation
                    if value_type := field_rules.get('type'):
                        if not self._check_type(value, value_type):
                            return False

                    # Range validation
                    if 'min' in field_rules and value < field_rules['min']:
                        return False
                    if 'max' in field_rules and value > field_rules['max']:
                        return False

            return True

        except Exception as e:
            self.logger.error(f"Constraint values error: {str(e)}")
            return False

    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check value matches expected type."""
        type_mapping = {
            'string': str,
            'number': (int, float),
            'integer': int,
            'boolean': bool,
            'array': list,
            'object': dict
        }

        expected = type_mapping.get(expected_type)
        if not expected:
            return True  # Skip unknown types

        return isinstance(value, expected)

    def _validate_pattern(self, value: str, pattern: str) -> bool:
        """Validate value against regex pattern."""
        import re
        try:
            return bool(re.match(pattern, str(value)))
        except Exception as e:
            self.logger.error(f"Pattern validation error: {str(e)}")
            return False

    def get_base_type(self, specialized_type: str) -> Optional[str]:
        """Get base type for specialization."""
        if spec_info := self._specializations.get(specialized_type):
            return spec_info.base_type
        return None

    def get_inheritance_chain(self, specialized_type: str) -> List[str]:
        """Get complete inheritance chain."""
        if spec_info := self._specializations.get(specialized_type):
            return spec_info.inheritance_path.copy()
        return []

    def get_specializations(self, base_type: str) -> List[str]:
        """Get all specializations of a base type."""
        specialized = []
        for spec_info in self._specializations.values():
            if spec_info.base_type == base_type:
                specialized.append(spec_info.specialized_type)
        return specialized

    def merge_constraints(
        self,
        base_constraints: Dict[str, Any],
        specialized_constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge base and specialized constraints."""
        merged = base_constraints.copy()

        for name, constraint in specialized_constraints.items():
            if name in merged:
                # Merge existing constraint
                if isinstance(merged[name], dict) and isinstance(constraint, dict):
                    merged[name] = self._deep_merge(merged[name], constraint)
            else:
                # Add new constraint
                merged[name] = constraint

        return merged

    def _deep_merge(
        self,
        base: Dict[str, Any],
        extension: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()

        for key, value in extension.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def verify_constraints(self, specialized_type: str) -> bool:
        """Verify all constraints are satisfied."""
        if spec_info := self._specializations.get(specialized_type):
            for constraint in spec_info.constraints.values():
                if not self._verify_constraint(constraint):
                    return False
            return True
        return False

    def _verify_constraint(self, constraint: Dict[str, Any]) -> bool:
        """Verify single constraint is satisfied."""
        constraint_type = constraint.get('type')

        if constraint_type == 'domain':
            return self._verify_domain_constraint(constraint)
        elif constraint_type == 'structural':
            return self._verify_structural_constraint(constraint)
        elif constraint_type == 'attribute':
            return self._verify_attribute_constraint(constraint)

        return True  # Unknown constraint types pass by default

    def _verify_domain_constraint(self, constraint: Dict[str, Any]) -> bool:
        """Verify domain constraint."""
        if modules := constraint.get('modules'):
            for module in modules:
                if not self._verify_module_compatibility(module):
                    return False
        return True

    def _verify_structural_constraint(self, constraint: Dict[str, Any]) -> bool:
        """Verify structural constraint."""
        if elements := constraint.get('elements'):
            for element in elements:
                if not self._verify_element_compatibility(element):
                    return False
        return True

    def _verify_attribute_constraint(self, constraint: Dict[str, Any]) -> bool:
        """Verify attribute constraint."""
        if attributes := constraint.get('attributes'):
            for attr in attributes:
                if not self._verify_attribute_compatibility(attr):
                    return False
        return True

    def _verify_module_compatibility(self, module: Dict[str, Any]) -> bool:
        """Verify module compatibility."""
        # Implementation depends on specific module compatibility rules
        return True

    def _verify_element_compatibility(self, element: Dict[str, Any]) -> bool:
        """Verify element compatibility."""
        # Implementation depends on specific element compatibility rules
        return True

    def _verify_attribute_compatibility(self, attribute: Dict[str, Any]) -> bool:
        """Verify attribute compatibility."""
        # Implementation depends on specific attribute compatibility rules
        return True

    def get_specialization_info(self, element_tag: str) -> Optional[SpecializationInfo]:
        """Get specialization information for an element."""
        try:
            return self._specializations.get(element_tag)
        except Exception as e:
            self.logger.error(f"Error getting specialization info: {str(e)}")
            return None
