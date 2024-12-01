
# Content standard

## Structured content with DITA XML

- Maps
- Topics
	- Concept
		- Descriptions
	- Reference
		- Specifications
		- Tables
		- Numbers
	- Task
		- Intructions
		- Guides
		- Checklists
		- Setup
- Content reuse
	- Conref

## Metadata

- YAML frontmatter
- Conditional actions

## Heading handling

- Our implementation automatically appends sequential index numbers to our headings
- We use a heading ID utility to truncate long headings and avoid collisions with same-name headings

## Artifact handling

- Interactive elements are called artifacts
- Artifacts are appended to a heading_ID using `.dita` tags within a topic reference

```xml
<topicgroup>
        <topicmeta>
            <navtitle>Article Body</navtitle>
        </topicmeta>
        <topicref
            href="../topics/articles/brownian-motion.md"
            format="markdown"
            type="concept">
            <topicmeta>
                <data name="artifacts">
                    <data name="simulator"
                          href="../topics/artifacts/brownian-art.dita"
                          target-heading="investigations-on-the-theory-of-the-brownian-movement"/>
                </data>
            </topicmeta>
        </topicref>
    </topicgroup>
```



## Style guide


## Streamlined authoring structure
- Markdown conversion and parsing
