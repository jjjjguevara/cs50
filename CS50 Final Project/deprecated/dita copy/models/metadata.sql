-- metadata.sql
-- Core content tables
CREATE TABLE IF NOT EXISTS content_items (
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

-- Metadata table for persistent metadata storage
CREATE TABLE IF NOT EXISTS metadata (
    id TEXT PRIMARY KEY,
    content_id TEXT NOT NULL,
    metadata_type TEXT,
    updates JSON DEFAULT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    conflicts JSON DEFAULT NULL,
    FOREIGN KEY (content_id) REFERENCES content_items (id)
);

-- Topic management
CREATE TABLE IF NOT EXISTS topic_types (
    type_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    base_type TEXT,
    description TEXT,
    schema_file TEXT,
    is_custom BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS topics (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    path TEXT NOT NULL,
    type_id INTEGER REFERENCES topic_types (type_id),
    content_type TEXT CHECK (content_type IN ('dita', 'markdown')),
    short_desc TEXT,
    parent_topic_id TEXT REFERENCES topics (id),
    root_map_id TEXT REFERENCES maps (id),
    specialization_type TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    published_version TEXT,
    status TEXT CHECK (status IN ('draft', 'review', 'published')),
    language TEXT DEFAULT 'en',
    feature_flags JSON DEFAULT NULL,
    prerequisites JSON DEFAULT NULL,
    related_topics JSON DEFAULT NULL,
    custom_metadata JSON DEFAULT NULL
);

-- Map management
CREATE TABLE IF NOT EXISTS maps (
    id TEXT PRIMARY KEY,
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
    context_root TEXT,
    processing_context_id INTEGER REFERENCES processing_contexts (context_id),
    feature_flags JSON DEFAULT NULL,
    prerequisites JSON DEFAULT NULL,
    related_topics JSON DEFAULT NULL,
    custom_metadata JSON DEFAULT NULL
);

-- Relationship tracking
CREATE TABLE IF NOT EXISTS content_relationships (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    relationship_type TEXT NOT NULL,
    scope TEXT NOT NULL,
    metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (source_id) REFERENCES content_items (id),
    FOREIGN KEY (target_id) REFERENCES content_items (id)
);

-- Key management
CREATE TABLE IF NOT EXISTS key_definitions (
    id TEXT PRIMARY KEY,
    href TEXT,
    scope TEXT CHECK (scope IN ('local', 'peer', 'external')) NOT NULL,
    processing_role TEXT CHECK (processing_role IN ('resource-only', 'normal')) NOT NULL,
    metadata JSON,
    source_map TEXT NOT NULL REFERENCES maps (id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Context tracking
CREATE TABLE IF NOT EXISTS processing_contexts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content_id TEXT NOT NULL,
    content_type TEXT CHECK (content_type IN ('topic', 'map')),
    phase TEXT NOT NULL,
    state TEXT NOT NULL,
    parent_context_id INTEGER REFERENCES processing_contexts (id),
    features JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);

-- Heading tracking
CREATE TABLE IF NOT EXISTS heading_index (
    id TEXT PRIMARY KEY,
    topic_id TEXT NOT NULL REFERENCES topics (id),
    map_id TEXT NOT NULL REFERENCES maps (id),
    text TEXT NOT NULL,
    level INTEGER NOT NULL,
    sequence_number TEXT,
    path_fragment TEXT
);

-- Create indices for performance
CREATE INDEX IF NOT EXISTS idx_content_status ON content_items (status);

CREATE INDEX IF NOT EXISTS idx_topics_map ON topics (root_map_id);

CREATE INDEX IF NOT EXISTS idx_topics_parent ON topics (parent_topic_id);

CREATE INDEX IF NOT EXISTS idx_rel_source ON content_relationships (source_id);

CREATE INDEX IF NOT EXISTS idx_rel_target ON content_relationships (target_id);

CREATE INDEX IF NOT EXISTS idx_key_scope ON key_definitions (scope, source_map);

CREATE INDEX IF NOT EXISTS idx_processing_context ON processing_contexts (content_id, phase, state);

CREATE INDEX IF NOT EXISTS idx_heading_topic ON heading_index (topic_id);

CREATE INDEX IF NOT EXISTS idx_heading_map ON heading_index (map_id);

-- Insert default topic types
INSERT
OR IGNORE INTO topic_types (name, description)
VALUES
    ('topic', 'Generic DITA topic'),
    ('concept', 'Conceptual information'),
    ('task', 'Task or procedure'),
    ('reference', 'Reference information'),
    ('glossary', 'Glossary entries'),
    ('troubleshooting', 'Troubleshooting information');
