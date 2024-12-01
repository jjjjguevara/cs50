
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
