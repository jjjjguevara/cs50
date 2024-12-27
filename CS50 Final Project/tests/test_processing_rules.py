import pytest
import json
from app.dita.models.types import DITAElementType, MDElementType

CONFIG_FILES = {
    "processing_rules": "app/dita/configs/processing_rules.json",
    "dita_processing_rules": "app/dita/configs/dita_processing_rules.json",
    "attribute_schema": "app/dita/configs/attribute_schema.json"
}

REQUIRED_FIELDS = ["element_type", "operation", "action"]
VALID_OPERATIONS = {"transform", "validate"}

@pytest.fixture(scope="module")
def load_configs():
    configs = {}
    for name, path in CONFIG_FILES.items():
        with open(path, "r") as f:
            configs[name] = json.load(f)
    return configs

def validate_rule(rule_id, rule_data):
    errors = []

    # Check required fields
    for field in REQUIRED_FIELDS:
        if field not in rule_data:
            errors.append(f"Missing required field: {field}")

    # Validate element_type
    element_type = rule_data.get("element_type")
    if element_type and element_type not in DITAElementType._value2member_map_ and element_type not in MDElementType._value2member_map_:
        errors.append(f"Invalid element_type: {element_type}")

    # Validate operation
    operation = rule_data.get("operation")
    if operation and operation not in VALID_OPERATIONS:
        errors.append(f"Invalid operation: {operation}")

    return errors

def test_processing_rules(load_configs):
    processing_rules = load_configs["processing_rules"]["rules"]
    all_errors = []

    for rule_type, rules in processing_rules.items():
        for rule_id, rule_data in rules.items():
            if isinstance(rule_data, dict):
                errors = validate_rule(rule_id, rule_data)
                if errors:
                    all_errors.append(f"Rule ID: {rule_id}, Errors: {', '.join(errors)}")

    assert not all_errors, f"Validation errors found:\n" + "\n".join(all_errors)

def test_attribute_schema_validity(load_configs):
    attribute_schema = load_configs["attribute_schema"]
    dita_processing_rules = load_configs["dita_processing_rules"]

    # Ensure mappings exist
    mappings = attribute_schema["hierarchy"]["levels"]["config_files"]["sources"][1]["resolution_rules"]["element_types"].get("mapping", {})
    if isinstance(mappings, str):
        pytest.fail(f"'mapping' is expected to be a dictionary but found a string: {mappings}")
    for element, mapping in mappings.items():
        assert mapping in dita_processing_rules["element_type_mapping"].values(), f"Missing mapping for {element} in dita_processing_rules"

def test_required_fields(load_configs):
    attribute_schema = load_configs["attribute_schema"]
    required_patterns = attribute_schema.get("validation", {}).get("required_attributes", {})
    for element, attrs in required_patterns.items():
        assert attrs, f"Required attributes missing for {element}"
