
# Main features

## List of features

### DITA-XML powered features
- Versatile internal-link handling
  - In DITA, cross-references between topics in the same map are typically handled using `xref` elements with a `topic ID` and `heading ID`.
  - Our processor implementation handles a `.dita` name for its target topic and the `heading ID` for its target heading.
  - For example `[content standards metadata](content-standard#metadata)` will take you to the [metadata](content-standard#metadata) heading in this file, because the `content-standard.md` topic is part of our article (map).


The topic filename (without extension)
The heading ID (which your processor generates from heading text)
- Structured content via DITA XML
  - `.ditamap` transformation with `.dita` and `.md` support
- YAML frontmatter parsing and transformation into XML metadata.


Our current processor logic turns whatever topics are referenced in the ditamaps (.dita or .md) and renders them as article page in HTML.
Our goal is to append our interactive "artifacts" which are react components, as part of the body of the article.
We have 2 interactive components so far. One is `app/dita/artifacts/components/brownian.jsx` which is specific to the article `app/dita/artifacts/components/brownian.jsx` and the other is
 `app/static/js/components/layout/ScrollspyContent.jsx`.

 Scrollspy is actually a Bootstrap component. Our transformation logic should take whatever is in `app/dita/topics/artifacts/brownian-art.dita` to feed Scrollspy with text content to display.

 This ensures Scrollspy can be reused for other artifacts.

 It's also important to know from our implementation that there's a logic for handling headings. Our processor adds sequential index numbers to the headings. These are also passed to the Table of Contents component dynamically.

**Current Flow:**
1. The ditamap (`brownian-motion.ditamap`) references:
   - A markdown article (`brownian-motion.md`) for the main content
   - A React component (`brownian.jsx`) as an artifact to be injected
   - Content for the Scrollspy (`brownian-art.dita`)

2. The processor:
   - Converts the markdown to HTML
   - Should inject the React component at the specified target heading
   - Should feed the DITA content to Scrollspy


Let's restructure our code to handle any DITA ID related code in a single method. Our method should output
1. A unique identifier for every `.dita` and `.md` file it processes from a `.ditamap`
2. A unique identifier for any artifact that will be injected into the `.ditamap`


Let's also develop a separate method to handle everything related to Heading processing.

1. It goes through every heading in every `.dita` and `.md` file from the `.ditamap` and:
2. Assigns a `heading_id` to every heading, consisting of the first 4 words of the heading in lowercase (at maximum).
For example, since the first heading that appears in `brownian-motion.ditamap` is "INVESTIGATIONS ON THE THEORY OF THE BROWNIAN MOVEMENT" its corresponding heading_id would be `investigations-on-the-theory`
3. Catches any collisions of headings with the same name
For example, if another heading in `brownian-motion.ditamap` is found to be "INVESTIGATIONS ON THE THEORY OF THE BROWNIAN MOVEMENT" its corresponding heading_id would be `investigations-on-the-theory-2`


Let's also make sure that the `.ditamap` part of the string in our URLS is removed.

`http://127.0.0.1:5001/entry/brownian-motion` should direct to the article that was rendered from processing `brownian-motion.ditamap`


## Metadata handling

We should handle 3 different cases of metadata for now: `.ditamap` `.dita` and `.md`

The metadata in `.dita` and `.ditamap` uses the standard XML tags from the DITA standard.

The metadata in `.md` is provided as YAML frontmatter.

If metadata is detected in the content, our utility should extract this metadata and store it in our database as JSON information for easier handling.

Metadata should be saved in the database with their corresponding topic, map and heading (if any)

The fields in our metadata tables should allows us to toggle different aspects of the content.

Example 1: `room-acoustics.dita` shows a typical Scientific Journal entry table.

Having the tag `<othermeta name="journal-entry" content="True"/>` would make the table to be appended on top of the article as a SQL query.

Example 2: (when we add the bibliography component),

Having the YAML key value `Bibliography: True` should toggle a table that extracts the references found on the document by querying the SQL database for citations/references.

We will implement this toggle functionality in the `processor.py`, just consider these requirements when creating the logic for our `metadata.py` utility



Usage in processor:
```python
# In processor.py
from .utils.metadata import MetadataHandler

class DITAProcessor:
    def __init__(self):
        # ... other initialization ...
        self.metadata_handler = MetadataHandler()

    def transform_to_html(self, input_path: Path) -> HTMLString:
        try:
            # Extract metadata first
            metadata = self.metadata_handler.extract_metadata(
                input_path,
                content_id=input_path.stem
            )

            # Get toggleable features
            features = self.metadata_handler.get_toggleable_features(metadata)

            # Transform content
            html_content = self._transform_content(input_path, features)

            # Prepare metadata for database
            metadata_fields = self.metadata_handler.prepare_for_database(metadata)

            # Here you would save metadata_fields to database

            return html_content

        except Exception as e:
            return self._create_error_html(e, input_path)
```



> `processor.py` Current methods

1. **Initialization**
   - Basic setup and configuration

2. **Primary Entry Points**
   - Main transformation methods
   - First point of contact for external calls

3. **Core Processing**
   - Critical transformation logic
   - Content and artifact handling

4. **Path Handling**
   - File resolution and topic management

5. **Parsing and Error Handling**
   - XML parsing and error management



# 1. INITIALIZATION

class DITAProcessor:
    """DITA content processor for HTML transformation"""

```python
    def __init__():
        """Initialize processor with utilities and configurations"""
        pass
```

1. **Clear Organization**:
   - Paths initialization
   - Utility initialization
   - Parser configurations

2. **Proper Integration**:
   - All utilities are initialized and available
   - Clear separation of concerns
   - Each utility handles its specific domain

3. **Robust Error Handling**:
   - Directory creation verification
   - Setup validation
   - Detailed logging

4. **Type Safety**:
   - Type hints for clarity
   - Clear type aliases
   - Proper imports



# 2. PRIMARY ENTRY POINTS

```python
    def transform(self, input_path: Path) -> HTMLString:
        """Main entry point for transforming any DITA content to HTML"""
        # Renamed from transform_to_html
        pass
```

1. **Clear Processing Chain**:
   - Input validation
   - ID generation
   - Metadata extraction
   - Content processing
   - Artifact injection
   - Final formatting

2. **Proper Utility Integration**:
   - Uses ID Handler for content identification
   - Uses Metadata Handler for feature detection
   - Uses Heading Handler for heading management
   - Uses Artifact handlers for interactive content

3. **Robust Error Handling**:
   - Input validation
   - Try-except blocks
   - Detailed logging
   - Graceful fallbacks

4. **Metadata Integration**:
   - Extracts and processes metadata
   - Adds metadata attributes to HTML
   - Handles toggleable features

5. **Clean HTML Structure**:
   - Consistent wrapper elements
   - Proper class hierarchy
   - Data attributes for JavaScript hooks

```python
    def transform_map(self, map_path: Path) -> HTMLString:
        """Transform DITA map and its referenced topics"""
        # Renamed from _transform_map_to_html
        pass
```

1. **Structured Processing**:
   - Clear separation of map, topic, and metadata processing
   - Maintains heading hierarchy
   - Handles metadata features

2. **Utility Integration**:
   - Uses ID Handler for consistent IDs
   - Uses Metadata Handler for feature detection
   - Uses Heading Handler for section numbering

3. **Feature Handling**:
   - Journal table generation
   - Abstract section handling
   - Metadata-driven toggles

4. **Clean HTML Structure**:
   - Consistent class naming
   - Data attributes for JavaScript hooks
   - Proper nesting and hierarchy

5. **Error Handling**:
   - Per-topic error handling
   - Detailed logging
   - Graceful fallbacks



# 3. CORE PROCESSING

```python
    def process_topic(self, topic_path: Path, section_numbers: Dict) -> HTMLString:
        """Process single topic with section numbering"""
        # Renamed from _transform_topic_with_numbering
        pass
```

1. **File Type Handling**:
   - Separate processors for Markdown and DITA
   - Consistent heading processing across types
   - Proper section numbering

2. **Heading Processing**:
   - Consistent ID generation
   - Section numbering
   - Anchor links
   - Proper classes

3. **Content Structure**:
   - Clean HTML output
   - Consistent formatting
   - Proper nesting

4. **DITA Element Processing**:
   - Handles common DITA elements
   - Consistent styling
   - Bootstrap-compatible classes

5. **Error Handling**:
   - Per-element error handling
   - Detailed logging
   - Graceful fallbacks


```python
    def process_content(self, content_path: Path) -> HTMLString:
        """Process raw content to HTML"""
        # Renamed from _process_topic_content
        pass
```

1. **File Type Handling**:
   - Separate processors for Markdown and DITA
   - Consistent heading processing across types
   - Proper section numbering

2. **Heading Processing**:
   - Consistent ID generation
   - Section numbering
   - Anchor links
   - Proper classes

3. **Content Structure**:
   - Clean HTML output
   - Consistent formatting
   - Proper nesting

4. **DITA Element Processing**:
   - Handles common DITA elements
   - Consistent styling
   - Bootstrap-compatible classes

5. **Error Handling**:
   - Per-element error handling
   - Detailed logging
   - Graceful fallbacks

```python
    def inject_artifacts(self, html: str, artifacts: List[Dict]) -> HTMLString:
        """Inject interactive artifacts into HTML content"""
        # Renamed from _inject_artifacts
        pass
```

1. **Consistent Context Handling**:
   - Created a base context dictionary in `render_artifact`
   - Passed proper context to both rendering methods

2. **Type Safety**:
   - Properly typed context as `Dict[str, Any]`
   - Consistent context structure across components

3. **Target Heading**:
   - Added target heading to both component types
   - Maintained in context dictionary

4. **Error Handling**:
   - Proper error logging
   - Type-safe context handling
   - Consistent return types


# 4. PATH HANDLING
```python
    def resolve_path(self, map_path: Path, href: str) -> Optional[Path]:
        """Resolve topic reference path relative to map"""
        # Renamed from _resolve_topic_path
        pass
```

1. **Multiple Resolution Strategies**:
   - Relative to map file
   - From topics directory
   - From maps directory
   - In content subdirectories

2. **Security Features**:
   - Path traversal prevention
   - Safe path validation
   - Clear logging of attempts

3. **Cross-reference Support**:
   - Handles topic IDs in references
   - Supports cross-map references
   - Maintains reference context

4. **Robust Error Handling**:
   - Multiple fallback strategies
   - Detailed logging
   - Clear error messages

5. **Path Normalization**:
   - Consistent path formats
   - Platform independence
   - Clean references

Usage example:
```python
# In transform_map method
for topicref in tree.xpath(".//topicref"):
    href = topicref.get('href')
    if href:
        topic_path = self.resolve_path(map_path, href)
        if topic_path:
            # Process topic
            ...
        else:
            self.logger.warning(f"Could not resolve topic: {href}")
```


```python
    def get_topic(self, topic_id: str) -> Optional[Path]:
        """Get topic path from ID"""
        # Renamed from get_topic_path
        pass
```

This implementation:

1. **Multiple Search Strategies**:
   - Map directory search
   - Direct path resolution
   - Topic tree traversal
   - Metadata-based search

2. **Flexible ID Handling**:
   - Handles various ID formats
   - Supports subdirectories
   - Multiple file extensions

3. **Metadata Integration**:
   - Searches by metadata fields
   - Supports both DITA and Markdown
   - Extensible for more fields

4. **Robust Error Handling**:
   - Strategy-specific error handling
   - Detailed logging
   - Graceful fallbacks

5. **Performance Considerations**:
   - Ordered search strategies
   - Early returns
   - Efficient path checking




# 5. PARSING AND ERROR HANDLING

```python
    def parse_dita(self, input_path: Path) -> etree.Element:
        """Parse DITA file into XML tree"""
        # Renamed from _parse_dita_file
        pass
```

1. **Robust Parsing**:
   - Strict parsing first
   - Fallback to recovery mode
   - Detailed error logging

2. **Structure Validation**:
   - Document type detection
   - Required element checking
   - Reference validation

3. **Error Handling**:
   - Contextual error messages
   - Line number information
   - Visual error location

4. **Reference Management**:
   - Href validation
   - Conref validation
   - Path resolution

5. **Security Features**:
   - Entity resolution disabled
   - DTD loading disabled
   - Network access disabled


```python
    def handle_error(self, error: Exception, context: Path) -> HTMLString:
        """Create HTML error message"""
        # Renamed from _create_error_html
        pass
```

1. **Error Type Handling**:
   - XML syntax errors
   - Missing file errors
   - Validation errors
   - General errors

2. **User-Friendly Messages**:
   - Clear error descriptions
   - Visual error indicators
   - Helpful suggestions
   - Retry options

3. **Developer Features**:
   - Debug information
   - Stack traces
   - Error context
   - Timestamp information

4. **Visual Design**:
   - Consistent styling
   - Clear hierarchy
   - Responsive layout
   - Interactive elements

5. **Security**:
   - HTML escaping
   - Safe context handling
   - Development mode checks
