
# Specifications

## List of requirements

1. Compressed Setup (Nov 25-27)
   - Rapid initial setup and planning
   - Parallel environment setup and schema design

2. Parallel Backend & Frontend (Nov 26 - Dec 6)
   - Backend team starts immediately after initial setup
   - Frontend team begins while backend is in progress
   - Overlapping development phases

3. Integration & Testing (Dec 6-11)
   - Condensed testing phases
   - Critical path testing only
   - Parallel bug fixing

4. Final Sprint (Dec 11-12)
   - Documentation and deployment
   - Final fixes and polish

Key strategies for this compressed timeline:
- Multiple teams working in parallel
- Focus on core functionality first
- Daily standups to catch blocking issues
- Continuous integration throughout development
- Documentation happening alongside development

## Project files

```
project-root/
├── .env                    # Environment variables
├── requirements.txt        # Python dependencies
├── config.py              # Flask configuration
├── .gitignore             # Git ignore rules
├── run.py                 # Application entry point
└── app/
    ├── __init__.py        # Flask app initialization
    ├── models.py          # Database models
    ├── routes.py          # Route definitions
    ├── templates/         # Jinja2 templates
    │   └── base.html
    └── static/            # Static files
        ├── css/
        └── js/
```





## Metadata usage
- Frontmatter and Dita metadata can be used to toggle page elements
- Bibliography table
- Default language
- Heading numbering



## Front-end

Key Bootstrap features to know:

1. Grid System:
- Uses `container`, `row`, and `col-*` classes
- 12-column system
- Responsive breakpoints: sm, md, lg, xl, xxl

2. Components:
- Navigation bars
- Cards
- Buttons
- Forms
- Modals
- Carousel
- Alerts

3. Utility Classes:
- Spacing: `m-*` (margin), `p-*` (padding)
- Colors: `text-*`, `bg-*`
- Display: `d-*`
- Flexbox: `d-flex`, `justify-content-*`
- Text alignment: `text-center`, `text-start`, `text-end`

4. Responsive Design:
```html
<!-- Responsive images -->
<img src="..." class="img-fluid">

<!-- Responsive tables -->
<div class="table-responsive">
    <table class="table">...</table>
</div>

<!-- Hide/show elements -->
<div class="d-none d-md-block">Shows only on medium screens and up</div>
```

### Page features

#### Image-loading (to be implemented)


##### Alt Text Validation
- Alt text is alternative text that describes an image for screen readers and when images fail to load
- Validation would ensure every image has meaningful alt text
- Could check for:
  - Empty alt text
  - Generic text like "image" or "picture"
  - Appropriate length (not too short or long)
- Important for accessibility and SEO

##### Image Captions
- Text that appears below images to provide additional context or description
- Different from alt text as they're visible to all users
- Can include:
  - Description of the image
  - Source/credit information
  - Additional context
- Usually styled differently from regular text (smaller, italics, etc.)

##### Lazy-loading
- Instead of loading all images when the page loads, images load only as they come into view
- Benefits:
  - Faster initial page load
  - Reduced bandwidth usage
  - Better performance on slower connections
- Implemented using the `loading="lazy"` attribute in HTML5
- Especially useful for long pages with many images

##### Responsive Image Sizes
- Makes images adapt to different screen sizes and devices
- Can include:
  - Multiple image sizes for different devices
  - Automatic scaling based on container width
  - Different crop ratios for mobile/desktop
- Uses HTML features like:
  - srcset attribute for different image resolutions
  - sizes attribute to specify image sizes for different viewports
  - picture element for art direction

##### Lightbox Functionality
- When users click an image, it opens in a larger overlay view
- Features typically include:
  - Darkened background
  - Larger view of the image
  - Close button
  - Navigation if there are multiple images
  - Zoom capabilities
  - Optional image descriptions or captions
- Enhances the viewing experience for detailed images



### Components

#### Top navbar
1. We should have a navigation bar at the top that extends only for the extension of the article body from left to right. That is, it shouldn't cover the whole browser window from left to right.
2. When we scroll down, the navigation bar should dissappear. When we scroll up, the navigation bar should reappear.
3. The contents of the navbar flow from left to right, with a HOME button leftmost, then a button called "Articles" and the two current pop-up dropdown menus Browse and About.
4. The navbar should also integrate a search bar, which we will integrate later (lets skip this part for now)

#### Left sidebar
1. This is a navigation bar that displays the contents of the category you're in. For example, if you clicked "Articles" in the top navbar, you will see a list of all the subcategories of articles and the contents of the list (the articles)
2. This list should be rendered dynamically as a query. We will develop the logic of the list later.
3. This sidebar should emerge from the leftmost part of the web browser and have its own independent page scrolling.
4. This sidebar should be collapsible.


#### Right sidebar
1. This is a small clickable Table of Contents that's adjacent to the body of the content in the page.
2. It should rest immediately top right of the body of the content.
3. The titles in this Table of Contents should be aligned right.
4. When you scroll down on the article, this sidebar should remain in place at the top right and always visible, to allow the user to navigate to any section of the page at any point.


#### Bibliography

Bibliography component specifications
- Our current implementation renders a `.ditamap` as an article by pulling multiple `.dita` and `.md` topics together and rendering them in a single page.
- The `.ditamap` topics are joined sequentially by appearance. Meaning that if the `.ditamap` file has `foo.md` `bar.md` `baz.md` it would result in an entry page `foo+bar+baz` within our `academic.html` template
- Each topic may have references.
- The bibliography block should be located at the bottom of the article.
- Our Bibliography block should list these references sequentially and in order of appearance, with reciprocal linking behaviour. If you click the reference link on the bottom it takes you to the inline quote, and viceversa.
- To achieve the sequential footnote/reference behavior, we must implement a parsing logic that goes through the target files and detects instances of references, then appends a numerical ascending value starting from 1.
- This logic should also ensure that if `foo.md` and `bar.md` are joined, and `foo.md` had 4 references , then `bar.md` references should start from 5.
- This should apply both to the Bibliography table and to the article body.
- References in `md` are a single line that starts with `[^6]:` (or any number) and it's followed by reference text e.g. `Page, M. J., McKenzie, J. E., Bossuyt, P. M., Boutron, I., Hoffmann, T. C., Mulrow, C. D., ... & Moher, D. (2021). The PRISMA 2020 statement: An updated guideline for reporting systematic reviews. *BMJ*, 372.`
	1. In-line links to a reference are `[6]` (the number corresponding to the reference)
	2. Clicking on both the in-line link `[6]` or the reference `[^6]:` should lead to one another's location on the page.
- References in `.dita` are XML references.
	1. DITA XML in-line references should be converted to HTML using our logic and standardized with the `.md` format.
	2. This means that an XML footnote would still have an in-line link next to it `[7]` and the reference on the Biblography block will appear as `[^7]: ...Bibliography text...`  in the Bibliography block.
- Once all references are identified, they will be enclosed in an XML `<reference>` tag to hide them from the content body and populate the bibliography table dynamically.


1. This is a block element on the grid of the webapp that gets conditionally appended to the bottom of the article body whenever a "footnote"  is detected on the `.md` or `.dita` files that are sent to the processor.
2. The DITA processor will look for instances of references to populate the Biblography block on both `.md` or `.dita (XML)` .

5.

Example of XML footnote syntax.
```XML
        <p>The acquisition of musical expertise represents one of the most complex forms of human skill development<fn id="fn1">Ericsson, K. A. (2008). Deliberate practice and acquisition of expert performance: A general overview. Academic Emergency Medicine, 15(11), 988-994.</fn>. Professional musicians typically begin training early in life, accumulating over 10,000 hours of deliberate practice by early adulthood<fn id="fn2">Hambrick, D. Z., &amp; Tucker-Drob, E. M. (2015). The genetics of music accomplishment: Evidence for gene-environment correlation and interaction. Psychonomic Bulletin &amp; Review, 22(1), 112-120.</fn>.</p>
```



## Abstract

Recent advances in neuroimaging techniques have revolutionized our understanding of musical expertise[^1]. This comprehensive review examines the neural correlates of musical training, focusing on structural and functional brain changes observed in professional musicians[^2]. Through meta-analysis of 157 studies conducted between 2010 and 2024, we present evidence for experience-dependent plasticity in auditory, motor, and executive function networks.

## Introduction

The acquisition of musical expertise represents one of the most complex forms of human skill development[^3]. Professional musicians typically begin training early in life, accumulating over 10,000 hours of deliberate practice by early adulthood[^4]. This intensive training regimen leads to remarkable behavioral adaptations, including enhanced auditory discrimination, superior motor control, and improved cognitive flexibility[^5].

[^1]: Zatorre, R. J., & Salimpoor, V. N. (2013). From perception to pleasure: Music and its neural substrates. _Proceedings of the National Academy of Sciences_, 110(Supplement 2), 10430-10437. [^2]: Schlaug, G. (2015). Musicians and music making as a model for the study of brain plasticity. _Progress in Brain Research_, 217, 37-55. [^3]: Ericsson, K. A. (2008). Deliberate practice and acquisition of expert performance: A general overview. _Academic Emergency Medicine_, 15(11), 988-994. [^4]: Hambrick, D. Z., & Tucker-Drob, E. M. (2015). The genetics of music accomplishment: Evidence for gene-environment correlation and interaction. _Psychonomic Bulletin & Review_, 22(1), 112-120. [^5]: Herholz, S. C., & Zatorre, R. J. (2012). Musical training as a framework for brain plasticity: Behavior, function, and structure. _Neuron_, 76(3), 486-502.



In its current implementation, this flask app takes multiple DITA topics in `.md` or `.dita` formats, and uses `.ditamap` files as path reference to assemble the topics into a single HTML article page usind a template called `academic.html`.

It also has some other react components for navigation and dynamic table of contents generation.

Help me create a helper program called `citations.py` that will parse the resulting HTML page for XML or Markdown format citations. Then convert those citations into JSON and store them to be used for a new component we will create called `Bibliography.jsx` (don't worry about it right now)

(We should use a helper python program to keep a better separation of concerns.)

An example of the expected behavior is as follows.

`processor.py` renders the page and asks `citations.py` to parse the content for citations.

If the resulting page renders:

```markdown
# 1. Abstract



Recent advances in neuroimaging techniques have revolutionized our understanding of musical expertise[^1]. This comprehensive review examines the neural correlates of musical training, focusing on structural and functional brain changes observed in professional musicians[^2]. Through meta-analysis of 157 studies conducted between 2010 and 2024, we present evidence for experience-dependent plasticity in auditory, motor, and executive function networks.



# 2. Introduction



The acquisition of musical expertise represents one of the most complex forms of human skill development[^3]. Professional musicians typically begin training early in life, accumulating over 10,000 hours of deliberate practice by early adulthood[^4]. This intensive training regimen leads to remarkable behavioral adaptations, including enhanced auditory discrimination, superior motor control, and improved cognitive flexibility[^5].



[^1]: Zatorre, R. J., & Salimpoor, V. N. (2013). From perception to pleasure: Music and its neural substrates. _Proceedings of the National Academy of Sciences_, 110(Supplement 2), 10430-10437. [^2]: Schlaug, G. (2015). Musicians and music making as a model for the study of brain plasticity. _Progress in Brain Research_, 217, 37-55. [^3]: Ericsson, K. A. (2008). Deliberate practice and acquisition of expert performance: A general overview. _Academic Emergency Medicine_, 15(11), 988-994. [^4]: Hambrick, D. Z., & Tucker-Drob, E. M. (2015). The genetics of music accomplishment: Evidence for gene-environment correlation and interaction. _Psychonomic Bulletin & Review_, 22(1), 112-120. [^5]: Herholz, S. C., & Zatorre, R. J. (2012). Musical training as a framework for brain plasticity: Behavior, function, and structure. _Neuron_, 76(3), 486-502.
```

Then `citations.py` should return:

```JSON
{
  "references": [
    {
      "type": "article-journal",
      "id": "cite-1",
      "number": "1",
      "author": [
        {
          "family": "Zatorre",
          "given": "R. J."
        },
        {
          "family": "Salimpoor",
          "given": "V. N."
        }
      ],
      "issued": {
        "year": 2013
      },
      "title": "From perception to pleasure: Music and its neural substrates",
      "container-title": "Proceedings of the National Academy of Sciences",
      "volume": "110",
      "issue": "Supplement 2",
      "page": "10430-10437"
    },
    {
      "type": "book-chapter",
      "id": "cite-2",
      "number": "2",
      "author": [
        {
          "family": "Schlaug",
          "given": "G."
        }
      ],
      "issued": {
        "year": 2015
      },
      "title": "Musicians and music making as a model for the study of brain plasticity",
      "container-title": "Progress in Brain Research",
      "volume": "217",
      "page": "37-55"
    },
    {
      "type": "article-journal",
      "id": "cite-3",
      "number": "3",
      "author": [
        {
          "family": "Ericsson",
          "given": "K. A."
        }
      ],
      "issued": {
        "year": 2008
      },
      "title": "Deliberate practice and acquisition of expert performance: A general overview",
      "container-title": "Academic Emergency Medicine",
      "volume": "15",
      "issue": "11",
      "page": "988-994"
    },
    {
      "type": "article-journal",
      "id": "cite-4",
      "number": "4",
      "author": [
        {
          "family": "Hambrick",
          "given": "D. Z."
        },
        {
          "family": "Tucker-Drob",
          "given": "E. M."
        }
      ],
      "issued": {
        "year": 2015
      },
      "title": "The genetics of music accomplishment: Evidence for gene-environment correlation and interaction",
      "container-title": "Psychonomic Bulletin & Review",
      "volume": "22",
      "issue": "1",
      "page": "112-120"
    },
    {
      "type": "article-journal",
      "id": "cite-5",
      "number": "5",
      "author": [
        {
          "family": "Herholz",
          "given": "S. C."
        },
        {
          "family": "Zatorre",
          "given": "R. J."
        }
      ],
      "issued": {
        "year": 2012
      },
      "title": "Musical training as a framework for brain plasticity: Behavior, function, and structure",
      "container-title": "Neuron",
      "volume": "76",
      "issue": "3",
      "page": "486-502"
    }
  ],
  "schema": "https://resource.citationstyles.org/schema/v1/csl-data.json"
}
```




## Database implementation

### Database structure

```python
# app/models.py
from app import db

class DitaMap(db.Model):
    """Model for DITA maps."""
    id = db.Column(db.String(255), primary_key=True)
    title = db.Column(db.String(255), nullable=False)
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Topic(db.Model):
    """Model for individual topics."""
    id = db.Column(db.String(255), primary_key=True)
    map_id = db.Column(db.String(255), db.ForeignKey('dita_map.id'))
    title = db.Column(db.String(255), nullable=False)
    content = db.Column(db.Text, nullable=False)
    type = db.Column(db.String(50))  # concept, task, reference, etc.
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
```

**API Routes:**
```python
# app/routes/api.py
from flask import Blueprint, jsonify, request
from app.models import DitaMap, Topic
from app import db

api = Blueprint('api', __name__)

@api.route('/ditamaps', methods=['GET'])
def get_ditamaps():
    """Get all DITA maps."""
    maps = DitaMap.query.all()
    return jsonify({
        'success': True,
        'maps': [{'id': m.id, 'title': m.title} for m in maps]
    })

@api.route('/ditamaps/<map_id>', methods=['GET'])
def get_ditamap(map_id):
    """Get a specific DITA map and its content."""
    dita_map = DitaMap.query.get_or_404(map_id)
    return jsonify({
        'success': True,
        'map': {
            'id': dita_map.id,
            'title': dita_map.title,
            'content': dita_map.content
        }
    })

@api.route('/topics/<topic_id>', methods=['GET'])
def get_topic(topic_id):
    """Get a specific topic."""
    topic = Topic.query.get_or_404(topic_id)
    return jsonify({
        'success': True,
        'topic': {
            'id': topic.id,
            'title': topic.title,
            'content': topic.content,
            'type': topic.type
        }
    })
```

**Frontend API Integration:**
```javascript
// app/static/js/utils/api.js
const API_BASE_URL = '/api';

export const api = {
    // Get all DITA maps
    async getDitaMaps() {
        try {
            const response = await fetch(`${API_BASE_URL}/ditamaps`);
            if (!response.ok) throw new Error('Network response was not ok');
            return await response.json();
        } catch (error) {
            console.error('Error fetching DITA maps:', error);
            throw error;
        }
    },

    // Get specific DITA map
    async getDitaMap(mapId) {
        try {
            const response = await fetch(`${API_BASE_URL}/ditamaps/${mapId}`);
            if (!response.ok) throw new Error('Network response was not ok');
            return await response.json();
        } catch (error) {
            console.error(`Error fetching DITA map ${mapId}:`, error);
            throw error;
        }
    },

    // Get specific topic
    async getTopic(topicId) {
        try {
            const response = await fetch(`${API_BASE_URL}/topics/${topicId}`);
            if (!response.ok) throw new Error('Network response was not ok');
            return await response.json();
        } catch (error) {
            console.error(`Error fetching topic ${topicId}:`, error);
            throw error;
        }
    }
};
```

**Usage in React Components:**
```jsx
// app/static/js/components/navigation/SideNav.jsx
import React, { useState, useEffect } from 'react';
import { api } from '@/utils/api';

const SideNav = () => {
    const [ditaMaps, setDitaMaps] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        const fetchDitaMaps = async () => {
            try {
                setLoading(true);
                const response = await api.getDitaMaps();
                if (response.success) {
                    setDitaMaps(response.maps);
                }
            } catch (err) {
                setError('Failed to load DITA maps');
                console.error(err);
            } finally {
                setLoading(false);
            }
        };

        fetchDitaMaps();
    }, []);

    // ... rest of component
};
```

### Data flow

#### When the application loads:

  - The SideNav component mounts
  - It calls the `/api/ditamaps` endpoint
  - The backend queries the database for all DITA maps
  - Returns the data to the frontend
  - React updates the UI with the map list

#### When a map is selected:

- User clicks a map in the sidebar
- Frontend calls `/api/ditamaps/<map_id>`
- Backend retrieves the specific map and its content
- Content is rendered in the main view

3. Database Updates:
   - Maps and topics are processed from source files
   - Content is parsed and stored in the database
   - Timestamps track creation and updates
   - Relationships maintained between maps and topics

**Key Features:**
1. **Separation of Concerns:**
   - Database models handle data structure
   - API routes handle data access
   - Frontend utilities handle API communication
   - React components handle UI/UX

2. **Error Handling:**
   - Try/catch blocks in API calls
   - Error states in components
   - User feedback for loading/error states

3. **Performance:**
   - Database indexes on frequently queried fields
   - Efficient queries using relationships
   - Frontend caching of API responses

4. **Security:**
   - Input validation
   - SQL injection prevention via SQLAlchemy
   - CSRF protection
   - API error handling

5. **Maintainability:**
   - Clear separation of backend/frontend code
   - Consistent API response format
   - Centralized API utility
   - Component-based architecture

This structure allows for:
- Easy addition of new features
- Scalable data management
- Clear data flow
- Efficient state management
- Responsive user experience
