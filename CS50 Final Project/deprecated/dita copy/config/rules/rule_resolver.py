"""Rule resolution and composition for DITA processing."""
from typing import Dict, List, Optional, Any, Set, Union
from dataclasses import dataclass, field
from datetime import datetime
import logging

from ...models.types import (
    ProcessingRuleType,
    ElementType,
    ProcessingContext,
    ValidationResult,
    ValidationMessage,
    ValidationSeverity,
    ProcessingPhase,
    ContentScope
)

from ...event_manager import EventManager, EventType
from ...utils.cache import ContentCache, CacheEntryType
from ...utils.logger import DITALogger

@dataclass
class ResolvedRule:
    """Represents a fully resolved processing rule."""
    rule_id: str
    element_type: ElementType
    rule_type: ProcessingRuleType
    config: Dict[str, Any]
    conditions: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_rules: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

class RuleResolver:
    """
    Handles processing rule resolution and composition.
    Manages rule inheritance and context-based resolution.
    """

    def __init__(
        self,
        event_manager: EventManager,
        cache: ContentCache,
        logger: Optional[DITALogger] = None
    ):
        """Initialize rule resolver."""
        self.event_manager = event_manager
        self.cache = cache
        self.logger = logger or DITALogger(name=__name__)

        # Rule storage
        self._rule_registry: Dict[str, Dict[str, Any]] = {}
        self._rule_inheritance: Dict[str, List[str]] = {}
        self._rule_compositions: Dict[str, List[str]] = {}
        self._rule_conditions: Dict[str, Dict[str, Any]] = {}

        # Register event handlers
        self.event_manager.subscribe(
            EventType.RULE_UPDATED,
            self._handle_rule_update
        )

    def resolve_rule(
        self,
        element_type: ElementType,
        rule_type: ProcessingRuleType,
        context: Optional[ProcessingContext] = None
    ) -> Optional[ResolvedRule]:
        """
        Resolve processing rule based on type and context.

        Args:
            element_type: Type of element
            rule_type: Type of rule to resolve
            context: Optional processing context

        Returns:
            Optional[ResolvedRule]: Resolved rule if found
        """
        try:
            # Get base rule
            rule_id = self._get_rule_id(element_type, rule_type)
            if not rule_id:
                return None

            # Create cache key and check cache if context provided
            if context:
                cache_key = f"rule_{element_type.value}_{rule_type.value}_{context.context_id}"
                if cached := self.cache.get(cache_key, CacheEntryType.CONFIG):
                    return cached

            # Get inherited rules
            inherited_rules = self._get_inherited_rules(rule_id)

            # Compose rules
            resolved = self._compose_rules(
                rule_id,
                inherited_rules,
                context
            )

            # Cache if context provided and we have a cache key
            if context and resolved:
                cache_key = f"rule_{element_type.value}_{rule_type.value}_{context.context_id}"
                self.cache.set(
                    key=cache_key,
                    data=resolved,
                    entry_type=CacheEntryType.CONFIG,
                    element_type=element_type,
                    phase=context.state_info.phase
                )

            return resolved

        except Exception as e:
            self.logger.error(f"Error resolving rule: {str(e)}")
            return None

    def _get_rule_id(
        self,
        element_type: ElementType,
        rule_type: ProcessingRuleType
    ) -> Optional[str]:
        """Get base rule ID for element and rule type."""
        try:
            # Handle DITA map specifically
            if element_type == ElementType.MAP:
                return "structure.map"
            elif element_type == ElementType.DITAMAP:
                return "structure.map"

            # Get rules registry
            rules = self._rule_registry.get(rule_type.value, {})

            # Try exact type match first
            for rule_id, rule in rules.items():
                if rule.get('element_type') == element_type.value:
                    return rule_id

            # Fall back to default rules
            return "default.unknown"

        except Exception as e:
            self.logger.error(f"Error getting rule ID: {str(e)}")
            return None

    def _get_inherited_rules(self, rule_id: str) -> List[str]:
        """Get list of inherited rule IDs."""
        try:
            inherited = []
            current = rule_id

            while current and current in self._rule_inheritance:
                parent = self._rule_inheritance[current][0]  # Get first parent
                inherited.append(parent)
                current = parent

            return inherited

        except Exception as e:
            self.logger.error(f"Error getting inherited rules: {str(e)}")
            return []

    def _compose_rules(
        self,
        base_rule_id: str,
        inherited_ids: List[str],
        context: Optional[ProcessingContext]
    ) -> Optional[ResolvedRule]:
        """
        Compose rules through inheritance chain.

        Args:
            base_rule_id: Base rule ID
            inherited_ids: List of inherited rule IDs
            context: Optional processing context

        Returns:
            Optional[ResolvedRule]: Composed rule
        """
        try:
            # Get base rule
            base_rule = self._rule_registry.get(base_rule_id)
            if not base_rule:
                return None

            # Start with base configuration
            composed_config = base_rule.get('config', {}).copy()
            composed_conditions = base_rule.get('conditions', {}).copy()
            source_rules = [base_rule_id]

            # Apply inherited rules
            for inherited_id in inherited_ids:
                if inherited_rule := self._rule_registry.get(inherited_id):
                    # Merge configurations
                    self._merge_configs(composed_config, inherited_rule.get('config', {}))

                    # Merge conditions
                    self._merge_conditions(composed_conditions, inherited_rule.get('conditions', {}))

                    source_rules.append(inherited_id)

            # Apply context-specific rules if provided
            if context:
                self._apply_context_rules(composed_config, composed_conditions, context)

            return ResolvedRule(
                rule_id=base_rule_id,
                element_type=ElementType(base_rule['element_type']),
                rule_type=ProcessingRuleType(base_rule['rule_type']),
                config=composed_config,
                conditions=composed_conditions,
                source_rules=source_rules,
                metadata=base_rule.get('metadata', {})
            )

        except Exception as e:
            self.logger.error(f"Error composing rules: {str(e)}")
            return None

    def _merge_configs(
        self,
        base: Dict[str, Any],
        extension: Dict[str, Any]
    ) -> None:
        """Merge configuration dictionaries."""
        for key, value in extension.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_configs(base[key], value)
            else:
                base[key] = value

    def _merge_conditions(
        self,
        base: Dict[str, Any],
        extension: Dict[str, Any]
    ) -> None:
        """Merge condition dictionaries."""
        for key, value in extension.items():
            if key in base:
                if isinstance(base[key], list) and isinstance(value, list):
                    base[key].extend(value)
                elif isinstance(base[key], dict) and isinstance(value, dict):
                    self._merge_conditions(base[key], value)
                else:
                    base[key] = value
            else:
                base[key] = value

    def _apply_context_rules(
        self,
        config: Dict[str, Any],
        conditions: Dict[str, Any],
        context: ProcessingContext
    ) -> None:
        """Apply context-specific rule modifications."""
        try:
            # Apply scope-specific modifications
            if context.scope == ContentScope.EXTERNAL:
                config['external_processing'] = True
                conditions['requires_validation'] = True

            # Apply phase-specific modifications
            if context.state_info.phase == ProcessingPhase.VALIDATION:
                conditions['strict_validation'] = True

            # Apply feature-based modifications
            for feature, enabled in context.features.items():
                if enabled and feature in self._rule_conditions:
                    feature_rules = self._rule_conditions[feature]
                    self._merge_configs(config, feature_rules.get('config', {}))
                    self._merge_conditions(conditions, feature_rules.get('conditions', {}))

        except Exception as e:
            self.logger.error(f"Error applying context rules: {str(e)}")

    def _handle_rule_update(self, **event_data: Any) -> None:
        """Handle rule update events."""
        try:
            if rule_id := event_data.get('rule_id'):
                # Invalidate caches for this rule and its dependents
                self.cache.invalidate_by_pattern(f"rule_{rule_id}*")

                # Find and invalidate dependent rules
                dependents = self._find_dependent_rules(rule_id)
                for dependent in dependents:
                    self.cache.invalidate_by_pattern(f"rule_{dependent}*")

        except Exception as e:
            self.logger.error(f"Error handling rule update: {str(e)}")

    def _find_dependent_rules(self, rule_id: str) -> Set[str]:
        """Find rules that depend on the given rule."""
        dependents = set()
        for child, parents in self._rule_inheritance.items():
            if rule_id in parents:
                dependents.add(child)
        return dependents

    def cleanup(self) -> None:
        """Clean up resolver resources."""
        try:
            self._rule_registry.clear()
            self._rule_inheritance.clear()
            self._rule_compositions.clear()
            self._rule_conditions.clear()
            self.cache.invalidate_by_pattern("rule_*")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
            raise
