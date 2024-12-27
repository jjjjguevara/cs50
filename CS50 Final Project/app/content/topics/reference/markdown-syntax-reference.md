---
title: Markdown Syntax Reference
type: reference
authors:
  - Technical Documentation Team
institution: Documentation Institute
publication-date: 2024-11-28
categories:
  - Reference
  - Documentation
keywords:
  - markdown
  - syntax
  - reference
  - documentation
---

# Markdown Syntax Reference

## Headers

Here's how to create headers:

```markdown
# H1 Header
## H2 Header
### H3 Header
#### H4 Header
##### H5 Header
###### H6 Header
```

And here's how they look:

# H1 Header
## H2 Header
### H3 Header
#### H4 Header
##### H5 Header
###### H6 Header

## Emphasis

Here's how to emphasize text:

```markdown
*Italic text* or _italic text_
**Bold text** or __bold text__
***Bold and italic*** or ___bold and italic___
~~Strikethrough text~~
```

And here's how it looks:

*Italic text* or _italic text_

**Bold text** or __bold text__

***Bold and italic*** or ___bold and italic___

~~Strikethrough text~~

## Lists

### Unordered Lists

Here's how to create unordered lists:

```markdown
* Item 1
* Item 2
  * Subitem 2.1
  * Subitem 2.2
    * Subsubitem 2.2.1

- Alternative item 1
- Alternative item 2
  - Subitem 2.1
```

And here's how they look:

* Item 1
* Item 2
  * Subitem 2.1
  * Subitem 2.2
    * Subsubitem 2.2.1

- Alternative item 1
- Alternative item 2
  - Subitem 2.1

### Ordered Lists

Here's how to create ordered lists:

```markdown
1. First item
2. Second item
   1. Subitem 2.1
   2. Subitem 2.2
      1. Subsubitem 2.2.1
```

And here's how they look:

1. First item
2. Second item
   1. Subitem 2.1
   2. Subitem 2.2
      1. Subsubitem 2.2.1

### Task Lists

Here's how to create task lists:

```markdown
- [x] Completed task
- [ ] Incomplete task
  - [x] Completed subtask
  - [ ] Incomplete subtask
```

And here's how they look:

- [x] Completed task
- [ ] Incomplete task
  - [x] Completed subtask
  - [ ] Incomplete subtask

## Links and Images

### Links

Here's how to create links:

```markdown
[Link text](https://example.com)
[Link with title](https://example.com "Link title")
[Reference link][reference]

[reference]: https://example.com
```

And here's how they look:

[Link text](https://example.com)
[Link with title](https://example.com "Link title")
[Reference link][reference]

[reference]: https://example.com

### Images

Here's how to embed images:

```markdown
![Alt text](/api/placeholder/400/300)
![Alt text with title](/api/placeholder/400/300 "Image title")
![Reference image][image-ref]

[image-ref]: /api/placeholder/400/300
```

And here's how they look:

![Alt text](/api/placeholder/400/300)
![Alt text with title](/api/placeholder/400/300 "Image title")
![Reference image][image-ref]

[image-ref]: /api/placeholder/400/300

## Code

### Inline Code

Here's how to use inline code:

```markdown
Use `code` in your text
```

And here's how it looks:

Use `code` in your text

### Code Blocks

Here's how to create code blocks:

````markdown
```python
def hello_world():
    print("Hello, world!")
```
````

And here's how it looks:

```python
def hello_world():
    print("Hello, world!")
```

## Tables

Here's how to create tables:

```markdown
| Header 1 | Header 2 | Header 3 |
|----------|:--------:|----------:|
| Left     | Center   | Right     |
| aligned  | aligned  | aligned   |
```

And here's how they look:

| Header 1 | Header 2 | Header 3 |
|----------|:--------:|----------:|
| Left     | Center   | Right     |
| aligned  | aligned  | aligned   |

## Blockquotes

Here's how to create blockquotes:

```markdown
> This is a blockquote
> Continued on next line
>
> > Nested blockquote
> > > Third level
```

And here's how they look:

> This is a blockquote
> Continued on next line
>
> > Nested blockquote
> > > Third level

## Horizontal Rules

Here's how to create horizontal rules:

```markdown
Above the line

---

Below the line
```

And here's how they look:

Above the line

---

Below the line

## Definition Lists

Here's how to create definition lists:

```markdown
Term 1
: Definition 1
: Another definition 1

Term 2
: Definition 2
```

And here's how they look:

Term 1
: Definition 1
: Another definition 1

Term 2
: Definition 2

## Footnotes

Here's how to create footnotes:

```markdown
Here's a sentence with a footnote[^1].
Here's another with a labeled footnote[^label].

[^1]: This is the first footnote.
[^label]: This is a labeled footnote.
```

And here's how they look:

Here's a sentence with a footnote[^1].
Here's another with a labeled footnote[^label].

[^1]: This is the first footnote.
[^label]: This is a labeled footnote.

## Custom Attributes

Here's how to add custom attributes:

```markdown
This paragraph has a class and ID.
{: .custom-class #custom-id }

[This link has a class](https://example.com){: .link-class }
```

And here's how they look:

This paragraph has a class and ID.
{: .custom-class #custom-id }

[This link has a class](https://example.com){: .link-class }

## Math Expressions

Here's how to write math expressions:

```markdown
Inline math: $E = mc^2$

Block math:
$$
\frac{n!}{k!(n-k)!} = \binom{n}{k}
$$
```

And here's how they look:

Inline math: $E = mc^2$

Block math:
$$
\frac{n!}{k!(n-k)!} = \binom{n}{k}
$$

## Keyboard Keys

Here's how to show keyboard keys:

```markdown
Press <kbd>Ctrl</kbd> + <kbd>Alt</kbd> + <kbd>Delete</kbd>
```

And here's how it looks:

Press <kbd>Ctrl</kbd> + <kbd>Alt</kbd> + <kbd>Delete</kbd>

## Special Characters

Here's how to handle special characters:

```markdown
Copyright &copy; 2024
&alpha;, &beta;, &gamma;
```

And here's how they look:

Copyright &copy; 2024
&alpha;, &beta;, &gamma;

Note: Not all Markdown features might be available in our current implementation. Check the processor configuration for supported extensions.

- [Python Frontmatter](https://python-frontmatter.readthedocs.io/en/latest/)
- [Python Markdown](https://python-markdown.github.io/)


[//]: # (End of document)
