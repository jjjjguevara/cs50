-- metadata.sql

-- Index entries
CREATE TABLE index_entries (
    entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_id TEXT REFERENCES topic_metadata (topic_id),
    term TEXT NOT NULL,
    type TEXT CHECK (
        type IN ('primary', 'secondary', 'see', 'see-also')
    ) NOT NULL,
    target_id TEXT, -- For cross-references
    sort_key TEXT,
    UNIQUE (topic_id, term, type)
);

-- Topic Types and Specializations
CREATE TABLE topic_types (
    type_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE, -- concept, task, reference, etc.
    base_type TEXT, -- For specializations (base topic type)
    description TEXT,
    schema_file TEXT, -- Path to DTD/schema for validation
    is_custom BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert default DITA topic types
INSERT INTO
    topic_types (name, description)
VALUES
    ('topic', 'Generic DITA topic'),
    ('concept', 'Conceptual information'),
    ('task', 'Task or procedure'),
    ('reference', 'Reference information'),
    ('glossary', 'Glossary entries'),
    ('troubleshooting', 'Troubleshooting information');

-- Topics table with type tracking
CREATE TABLE topics (
    topic_id TEXT PRIMARY KEY,
    type_id INTEGER REFERENCES topic_types (type_id),
    title TEXT NOT NULL,
    short_desc TEXT,
    file_path TEXT NOT NULL,
    content_type TEXT CHECK (content_type IN ('dita', 'markdown')) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    version TEXT DEFAULT '1.0',
    status TEXT CHECK (
        status IN ('draft', 'review', 'published', 'archived')
    ),
    language TEXT DEFAULT 'en',
    content_hash TEXT,
    parent_topic_id TEXT REFERENCES topics (topic_id), -- For nested topics
    root_map_id TEXT REFERENCES maps (map_id), -- Primary map context
    specialization_type TEXT -- For custom specializations
);

-- Context Tracking
CREATE TABLE context_hierarchy (
    hierarchy_id INTEGER PRIMARY KEY AUTOINCREMENT,
    map_id TEXT REFERENCES maps (map_id),
    topic_id TEXT REFERENCES topics (topic_id),
    parent_id TEXT REFERENCES topics (topic_id), -- Parent topic
    level INTEGER NOT NULL, -- Nesting level
    sequence_num INTEGER NOT NULL, -- Order within level
    heading_number TEXT, -- Generated heading number
    context_path TEXT NOT NULL, -- Full path (e.g., "map1/topic1/subtopic2")
    UNIQUE (map_id, context_path)
);

-- Topic Relationships (beyond simple hierarchy)
CREATE TABLE topic_relationships (
    relationship_id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_topic_id TEXT REFERENCES topics (topic_id),
    target_topic_id TEXT REFERENCES topics (topic_id),
    relationship_type TEXT NOT NULL, -- prerequisites, related-links, child, etc.
    weight INTEGER DEFAULT 0, -- For ordering/priority
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Topic Elements (for fine-grained context tracking)
CREATE TABLE topic_elements (
    element_id TEXT PRIMARY KEY,
    topic_id TEXT REFERENCES topics (topic_id),
    element_type TEXT NOT NULL, -- p, section, codeph, etc.
    parent_element_id TEXT REFERENCES topic_elements (element_id),
    sequence_num INTEGER NOT NULL, -- Order within parent
    content_hash TEXT, -- For change tracking
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Element Context
CREATE TABLE element_context (
    context_id INTEGER PRIMARY KEY AUTOINCREMENT,
    element_id TEXT REFERENCES topic_elements (element_id),
    context_type TEXT NOT NULL, -- body, abstract, prereq, etc.
    parent_context TEXT, -- Parent context type
    level INTEGER, -- Nesting level
    xpath TEXT, -- Full XPath to element
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Topic Type Requirements (for validation)
CREATE TABLE topic_type_requirements (
    requirement_id INTEGER PRIMARY KEY AUTOINCREMENT,
    type_id INTEGER REFERENCES topic_types (type_id),
    element_name TEXT NOT NULL, -- Required element
    min_occurs INTEGER DEFAULT 1, -- Minimum occurrences
    max_occurs INTEGER, -- Maximum occurrences (NULL for unbounded)
    parent_element TEXT, -- Required parent element
    description TEXT
);

-- Processing Contexts
CREATE TABLE processing_contexts (
    context_id INTEGER PRIMARY KEY AUTOINCREMENT,
    content_id TEXT NOT NULL, -- topic_id or map_id
    content_type TEXT CHECK (content_type IN ('topic', 'map')),
    phase TEXT NOT NULL, -- discovery, validation, transformation, etc.
    state TEXT NOT NULL, -- pending, processing, completed, error
    parent_context_id INTEGER REFERENCES processing_contexts (context_id),
    features JSON, -- Enabled features for this context
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);

-- Modified maps table with additional context fields
CREATE TABLE maps (
    map_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    file_path TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    version TEXT DEFAULT '1.0',
    status TEXT CHECK (
        status IN ('draft', 'review', 'published', 'archived')
    ),
    language TEXT DEFAULT 'en',
    toc_enabled BOOLEAN DEFAULT TRUE,
    index_numbers_enabled BOOLEAN DEFAULT TRUE,
    context_root TEXT, -- Base context path
    processing_context_id INTEGER REFERENCES processing_contexts (context_id)
);

-- Conditional_attributes for context-aware processing
CREATE TABLE conditional_attributes (
    attribute_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT,
    attribute_type TEXT CHECK (attribute_type IN ('content_toggle', 'version_toggle')) NOT NULL,
    scope TEXT CHECK (scope IN ('global', 'map', 'topic', 'element')) NOT NULL,
    is_toggle BOOLEAN DEFAULT FALSE,
    context_dependent BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(name, scope)

-- Conditional values
CREATE TABLE conditional_values (
    value_id INTEGER PRIMARY KEY AUTOINCREMENT,
    attribute_id INTEGER REFERENCES conditional_attributes(attribute_id),
    value TEXT NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(attribute_id, value)
);

-- Create additional indexes for context-based queries
CREATE INDEX idx_topic_hierarchy ON context_hierarchy (map_id, topic_id, level);

CREATE INDEX idx_topic_elements ON topic_elements (topic_id, element_type);

CREATE INDEX idx_element_context ON element_context (element_id, context_type);

CREATE INDEX idx_processing_context ON processing_contexts (content_id, phase, state);
