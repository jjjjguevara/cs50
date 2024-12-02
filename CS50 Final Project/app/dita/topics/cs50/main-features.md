
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


```python
# `processor.py` Current methods

class DITAProcessor:
    """DITA content processor for HTML transformation"""

    # 1. INITIALIZATION
    def __init__():
        """Initialize processor with utilities and configurations"""
        pass

    # 2. PRIMARY ENTRY POINTS
    def transform(self, input_path: Path) -> HTMLString:
        """Main entry point for transforming any DITA content to HTML"""
        # Renamed from transform_to_html
        pass

    def transform_map(self, map_path: Path) -> HTMLString:
        """Transform DITA map and its referenced topics"""
        # Renamed from _transform_map_to_html
        pass

    # 3. CORE PROCESSING
    def process_topic(self, topic_path: Path, section_numbers: Dict) -> HTMLString:
        """Process single topic with section numbering"""
        # Renamed from _transform_topic_with_numbering
        pass

    def process_content(self, content_path: Path) -> HTMLString:
        """Process raw content to HTML"""
        # Renamed from _process_topic_content
        pass

    def inject_artifacts(self, html: str, artifacts: List[Dict]) -> HTMLString:
        """Inject interactive artifacts into HTML content"""
        # Renamed from _inject_artifacts
        pass

    # 4. PATH HANDLING
    def resolve_path(self, map_path: Path, href: str) -> Optional[Path]:
        """Resolve topic reference path relative to map"""
        # Renamed from _resolve_topic_path
        pass

    def get_topic(self, topic_id: str) -> Optional[Path]:
        """Get topic path from ID"""
        # Renamed from get_topic_path
        pass

    # 5. PARSING AND ERROR HANDLING
    def parse_dita(self, input_path: Path) -> etree.Element:
        """Parse DITA file into XML tree"""
        # Renamed from _parse_dita_file
        pass

    def handle_error(self, error: Exception, context: Path) -> HTMLString:
        """Create HTML error message"""
        # Renamed from _create_error_html
        pass
```

This organization:

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


```
