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
    INSERT INTO
        topic_types (name, description)
    VALUES
        ('topic', 'Generic DITA topic'),
        ('concept', 'Conceptual information'),
        ('task', 'Task or procedure'),
        ('reference', 'Reference information'),
        ('glossary', 'Glossary entries'),
        ('troubleshooting', 'Troubleshooting information');
);

-- Table: status
CREATE TABLE content_items (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    file_path TEXT NOT NULL,
    version TEXT DEFAULT '1.0',
    status TEXT CHECK (
        status IN ('draft', 'review', 'published', 'archived')
    ),
    language TEXT DEFAULT 'en',
    content_hash TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- Revised topics table with feature flags and flexible metadata
CREATE TABLE topics (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    path TEXT NOT NULL,
    type_id INTEGER REFERENCES topic_types(type_id),
    content_type TEXT CHECK (content_type IN ('dita', 'markdown')),
    short_desc TEXT,
    parent_topic_id TEXT REFERENCES topics(id),
    root_map_id TEXT REFERENCES maps(id),
    specialization_type TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    published_version TEXT,
    status TEXT CHECK (status IN ('draft', 'review', 'published')),
    language TEXT DEFAULT 'en',
    feature_flags JSON DEFAULT NULL,   -- Feature toggles (e.g., enable_cross_refs)
    prerequisites JSON DEFAULT NULL,   -- Prerequisite topics
    related_topics JSON DEFAULT NULL,  -- Relationships with other topics
    custom_metadata JSON DEFAULT NULL  -- For future extensibility
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

-- Heading Metadata
CREATE TABLE heading_hierarchy (
    hierarchy_id INTEGER PRIMARY KEY AUTOINCREMENT,
    heading_id TEXT NOT NULL,
    parent_id TEXT,
    level INTEGER NOT NULL,
    sequence_num INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (heading_id) REFERENCES heading_index(id),
    FOREIGN KEY (parent_id) REFERENCES heading_index(id)
);

CREATE INDEX idx_heading_hierarchy ON heading_hierarchy(heading_id, parent_id);

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


-- Revised maps table with feature flags and relational metadata
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
    processing_context_id INTEGER REFERENCES processing_contexts (context_id),
    feature_flags JSON DEFAULT NULL,   -- Dynamic feature toggles (e.g., TOC, heading numbering)
    prerequisites JSON DEFAULT NULL,   -- Prerequisite maps/topics
    related_topics JSON DEFAULT NULL,  -- Relationships with other maps/topics
    custom_metadata JSON DEFAULT NULL  -- For future extensibility
);

-- Replace the three dataclasses with a simpler flags system:
CREATE TABLE content_flags (
    flag_id INTEGER PRIMARY KEY,
    content_id TEXT NOT NULL,
    name TEXT NOT NULL,
    value TEXT NOT NULL,  -- Can be 'true'/'false' for toggles or specific values
    scope TEXT CHECK (scope IN ('global', 'map', 'topic', 'element')),
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(content_id, name)
);

-- For more complex conditions/relationships
CREATE TABLE content_conditions (
    condition_id INTEGER PRIMARY KEY,
    content_id TEXT NOT NULL,
    condition_type TEXT NOT NULL, -- e.g., 'version', 'platform', 'audience'
    operator TEXT NOT NULL,       -- e.g., 'equals', 'greater_than', 'includes'
    value TEXT NOT NULL,
    active BOOLEAN DEFAULT TRUE,
    UNIQUE(content_id, condition_type, value)
);

-- Conditional values
CREATE TABLE conditional_values (
    value_id INTEGER PRIMARY KEY AUTOINCREMENT,
    attribute_id INTEGER REFERENCES conditional_attributes (attribute_id),
    value TEXT NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (attribute_id, value)
);

-- Key definitions
CREATE TABLE key_definitions (
    key_id TEXT PRIMARY KEY,
    map_id TEXT NOT NULL,
    keys TEXT NOT NULL,
    href TEXT NOT NULL,
    alt TEXT,
    placement TEXT,
    scale TEXT,
    props TEXT,
    audience TEXT,
    platform TEXT,
    product TEXT,
    otherprops TEXT,
    conref TEXT,
    keyref TEXT,
    rev TEXT,
    outputclass TEXT,
    align TEXT,
    scalefit TEXT,
    width TEXT,
    height TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (map_id) REFERENCES maps(map_id)
);

CREATE INDEX idx_keydef_map ON key_definitions(map_id);
CREATE INDEX idx_keydef_keys ON key_definitions(keys);


-- references.sql
CREATE TABLE heading_index (
    id TEXT PRIMARY KEY,
    topic_id TEXT NOT NULL,
    map_id TEXT NOT NULL,
    text TEXT NOT NULL,
    level INTEGER NOT NULL,
    sequence_number TEXT,     -- For numbered headings
    path_fragment TEXT,       -- URL fragment
    FOREIGN KEY (topic_id) REFERENCES topics(id),
    FOREIGN KEY (map_id) REFERENCES maps(id)
);

-- Table for content references, extended with metadata and relation_type
CREATE TABLE content_references (
    ref_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    ref_type TEXT CHECK (ref_type IN ('internal', 'external', 'web')),
    text TEXT,
    href TEXT NOT NULL,
    valid BOOLEAN DEFAULT true,  -- For reference validation
    relation_type TEXT,          -- Tracks relation (e.g., prerequisite, related)
    metadata JSON DEFAULT NULL,  -- Flexible metadata for reference details
    FOREIGN KEY (source_id) REFERENCES topics(id),
    -- target_id foreign key intentionally omitted for external refs
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for references
CREATE INDEX idx_heading_topic ON heading_index(topic_id);
CREATE INDEX idx_heading_map ON heading_index(map_id);
CREATE INDEX idx_ref_source ON content_references(source_id);
CREATE INDEX idx_ref_target ON content_references(target_id);

-- Create additional indexes for context-based queries
CREATE INDEX idx_topic_hierarchy ON context_hierarchy (map_id, topic_id, level);

CREATE INDEX idx_topic_elements ON topic_elements (topic_id, element_type);

CREATE INDEX idx_element_context ON element_context (element_id, context_type);

CREATE INDEX idx_processing_context ON processing_contexts (content_id, phase, state);
