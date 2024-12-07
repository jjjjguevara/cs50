
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
