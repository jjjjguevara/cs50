# Implementation summary

- Our project consists of a DITA XML powered webapp with Flask/SQL backend and JavaScript (react)/CSS (bootstrap) frontend.
- It features a robust transformation pipeline that takes `.ditamaps` and renders them into full `HTML` web page articles.

## Transformation logic

- The transformation phases are defined in `types.py`
```python
class ProcessingPhase(Enum):
    """Phases of the processing pipeline"""
    DISCOVERY = "discovery" # Initial parsing to classify the contents of files
    VALIDATION = "validation" # Validation of inputs and decision-making stage
    TRANSFORMATION = "transformation" # Initializtion of transformation pipelines
    ENRICHMENT = "enrichment" # Injection of LaTeX, artifacts, media, etc
    ASSEMBLY = "assembly" # HTML rendering 
```

- The transformation pipelines handle `.ditamap`, `.dita` and `.md` files at the `DISCOVERY`, `VALIDATION` and `TRANSFORMATION` stages, as well as inline `LaTeX` elements, multimedia elements, and custom `artifact` objects at the `ENRICHMENT` stage. Final assembly into HTML content must ALWAYS and ONLY be handled by `processor.py` at the `ASSEMBLY` stage.
- The workflow looks like this:
    - `processor.py` is our main orchestrator, and it is engaged whenever an endpoint defined on `routes.py` requests to see an "entry" or `.ditamap` file
    - `processor.py` will then engage the content discovery route defined in `dita_parser.py` to read the `.ditamap` file and determine what type of processing must be done.
    - Once it detects `.dita` or `.md` files it calls `dita_parser.py` again to inspect the contents of the `.dita` `.md` files from the `.ditamap` `hrefs`
    - The `dita_parser.py` parses the files and keep tracks of every element it finds there with definitions from `md_elements.py`, `dita_elements.py`, `id_handler.py` and `types.py`.
    - The parser also reaches to `latex_extension.py` for parsing LaTeX equations
    - After parsing the files, the `dita_parser.py` serves the XML/md element information it finds back to `processor.py` for handling.
    - `processor.py` keeps track of the elements received by the parser and determines the processing route based on the input
	    - The `base_processor.py` is called to pre-process the received dita topic and element information.
        - If `.md` content is sent back by the parser, then `base_processor.py` will engage with the markdown pipeline of `md_transform.py` and `md_elements.py`
        - If `.dita` content is sent back by the parser, then `base_processor.py` will engage with the dita pipeline of `dita_transform.py` and `dita_elements.py`
        - If LaTeX equation blocks are found in the content, `base_processor.py` will engage with the LaTeX pipeline of `katex_renderer.py` `latex_processor.py` `latex_validator.py`
        - If `media` elements are found in the content,  `base_processor.py` will engage with the media feature sub-pipeline. 
        - If `artifact` objects are found in the content, `base_processor.py` will engage with the artifacts feature sub-pipeline. (TO BE IMPLEMENTED)
    - Once processing is done by the processing pipelines, `processor.py` will fully assemble the `.ditamap` page in proper order
        - `processor.py` also uses `heading.py` to keep the map/topic hierarchical relationship intact regardless of the format type, and as an utility for appending indexing numbers and anchor links.
        - The final HTML rendering is aided by `html_helpers.py`
    - `processor.py` will conditionally determine which elements are sent for processing based on their element type and element ID
        - These conditions are yet to be defined, but we've already defined some contextual methods, classes, and types for expanding on this functionality.
        - Our only components currently leveraging this functionality are the Table Of Contents and the Index number appending feature of `heading.py`.
	- All data structures / data type classes passed from one file to another are comprehensively structured in `models/types.py`, which includes processing stages, states, context types, element types, metadata, etc
	- Implementation-specific configurations are handled by `config_manager.py` and `/app_config.py`
	- Error handling is centralized in `processor.py` and initialized by `_handle_processing_error()`. 
	- Logging for everything is handled by `logger.py`



## Separation of concerns
- The project uses several dedicated files to achieve maximum modularity and separation of concerns.
- The core file structure looks like this:
```txt
├── README.md                   # Project documentation
├── DESIGN.md                   # Design documentation
├── app
│   ├── __init__.py             # Flask app initialization (if not already present)
│   ├── config.py               # Project-wide configuration settings
│   ├── config_manager.py       # Configuration manager for DITA settings
│   ├── models.py               # Data models (e.g., topics, maps, processed content, etc.)
│   ├── routes.py               # Flask routes (API endpoints)
│   ├── static                  # Frontend static assets
│   │   ├── css                 # Stylesheets (global and component-specific)
│   │   ├── fonts               # Custom fonts
│   │   ├── img                 # Image assets for frontend
│   │   └── js                  # Frontend JavaScript (React components and utilities)
│   │       ├── components      # UI React components
│   │       └── utils           # JavaScript utilities (helpers for React or client-side code)
│   ├── templates               # Flask templates (Jinja2 files)
│   ├── dita                    # DITA-specific directories and files
│   │   ├── dtd                 # DITA definitions (XML schemas, DTDs)
│   │   ├── maps                # DITA map files (main "article" pages)
│   │   ├── topics              # DITA topic files (subsections of the main articles)
│   │   ├── artifacts           # Custom interactive DITA artifacts
│   │   ├── metadata            # Metadata related to DITA files
│   │   ├── models              # DITA models (e.g., for maps, topics, elements)
│   │   ├── processors          # DITA processors (handling parsing and processing of DITA content)
│   │   ├── transformers        # DITA transformers (handling transformation logic, e.g., to HTML)
│   │   ├── processor.py        # Main DITA processor (the orchestrator for DITA processing)
│   │   ├── utils               # Utilities and extensions for DITA processing
│   │   │   ├── html_helpers.py # HTML rendering helpers
│   │   │   └── types.py        # Various data types (e.g., for elements, topics, etc.)
│   ├── instance                # Flask instance folder (for runtime files like database)
│   │   └── app.db              # SQLite database for Flask app
│   ├── verify-paths.js         # Path verification script (used for verifying file structure integrity)
├── run.py                      # Entry point for running the Flask app
├── requirements.txt            # Python dependencies
├── jsconfig.json               # Configuration file for JavaScript/TypeScript (optional)
└── vite.config.js              # Configuration file for Vite (for frontend bundling and dev server)
```


- I reached this final structure after a lenghty process of trial and error. At first I tried to have most of the central processing in a single file called `processor.py`, but this was really buggy. The problem with this initial approach was that implementing components and additional features only resulted in broken pages and frustration. My code was all over the place. I also hadn't realized the complexities of handling element IDs and headings while assembling HTML. Eventually while testing, I figured out I would do better by leaving the HTML transformation to `processor.py` but I wasn't sure how to approach the rest.
- I stopped developing the project for a day or so to go back to the drawing board and understand better the tools I was using and what other Python features where out there to accomplish what I was trying to do. Then I came back after my research and decided to break the whole process into as many single function *subprocesses* I could, without driving the complexity too far.
- This last approach was still not the best. I didn't have a reasonable debugging strategy nor enough error handling for single components. because I didn't fully understand the transformation logic and didn't have a reasonable approach for debugging. Then after going to the drawing board again, I added better debugging and cleanup until the project became what it is now.
- All of the files are thoroughly commented, and the methods used have clear self-descriptive names.
- You can use the `processor.py` file as an index to explore what each *subprocessing* file and method are doing.
- The code has the following sections (denoted by # blocks) after initial imports and declarations:
	- **Initialization** - Main initialization methods
	- **State management** - Keeps track and validates processing stages and phases
	- **Context processing** - Methods to handle context queues from the parsed content (to leverage DITA Information Types and other topic/map attributes)
	- **Map-level processing** - Handles parsing and processing of files at the Map level
	- **Topic-level processing** - Handles parsing and processing of files at the Topic level
	- **Element-level processing** - Handles parsing and processing of files at the individual Element level
	- **Conditionally processed elements and features** - Handles the conditions under which elements and features are added to the content, as well as the injection logic
	- **Pipeline management** - Focuses on the orchestration of all pipelines
	- **Heading processing** - Handles heading processing and keeps track of the father/child relationship between maps/topics
	- **Final HTML assembly** - Focuses on the (post-transformation) final assembly of HTML to be served to the front-end
	- **Transformer Routes Management** - Validation logic, keeps track of the transformation routes and processing types to avoid *collisions*  and inconsistent processing
	- **Error handling and Debugging** - Focused error handling logic for the whole program, enhances debugging workflow with `logger.py`
	- **Cleanup and Waste Management** - General cleanup for resources and states 


## Areas of opportunity and current challenges

- Still pending is a UI overhaul with a better design more akin to an online magazine or science journal.
- Indexing and content aggregation on the website still requires more toughtful design.
- There are still many traces of bad design decisions from earlier in the code, as well as signs of fractured debugging strategies that need to be cleaned up.
- There are still some deprecated methods and duplicated functionalities in my codebase that need to be removed.
- The naming conventions of files may not be the best. There's room for a better file name system.
- The directories don't have the best structure. There's room for better file organization.

# Practical applications

- As mentioned in the project introduction, I focused on making a platform for sharing and publishing scientific articles. I envisioned this would make a powerful way for collaborating and keeping track of every change and development. I believe my project is a good proof of concept for that. However, I designed the structure with flexibility in mind, so there's nothing stopping us from repurposing this project for a news website, a technical documentation website, or even a website for educational content.

- For example, you could use a map with a structure like the one below to schedule publications on a news website, show different contributors and versions of the same text, denote urgency, condition versions of the text depending on location or audience profile, condition any element on the article, do tracebacks based on the users' interactions with the page to conduct user research , etc.

```XML
# EXAMPLE OF A METADATA TABLE FOR A NEWS WEBSITE
<metadata>
    <!-- General Article Metadata -->
    <othermeta name="headline" content="Global Climate Summit: Key Outcomes and Impacts"/>
    <othermeta name="slug" content="global-climate-summit-2024"/>
    <othermeta name="publication-date" content="2024-12-01"/>
    <othermeta name="last-edited" content="2024-12-05T09:30:00"/> <!-- Extracted from Git -->
    <othermeta name="author-primary" content="Jane Doe"/>
    <othermeta name="author-contributor" content="John Smith"/>
    <othermeta name="editor" content="Alice Johnson"/>
    <category>Environment</category>
    <category>Climate Change</category>
    
    <!-- Keywords for SEO -->
    <keywords>
        <keyword>climate summit</keyword>
        <keyword>global warming</keyword>
        <keyword>Paris Agreement</keyword>
        <keyword>environmental policy</keyword>
    </keywords>
    
    <!-- Audience Targeting -->
    <audience>General Public</audience>
    <audience>Policymakers</audience>
    <audience>Environmental Activists</audience>
    
    <!-- Regional Content Customization -->
    <othermeta name="region" content="Global"/>
    <othermeta name="language" content="en-US"/>
    
    <!-- Content Pipeline Metadata -->
    <othermeta name="priority" content="high"/> <!-- Indicates importance in newsfeed -->
    <othermeta name="embargo-until" content="2024-12-01T06:00:00"/> <!-- Scheduled release -->
    <othermeta name="breaking-news" content="no"/> <!-- Indicates non-urgent article -->
    
    <!-- Media Attachments -->
    <othermeta name="image-version" content="default"/> <!-- Default for light/dark mode -->
    <othermeta name="video-embed" content="https://newsagency.com/videos/climate-summit.mp4"/>
    <othermeta name="pdf-download" content="climate-summit-report.pdf"/> <!-- Supporting document -->

    <!-- Distribution Channels -->
    <othermeta name="channel-web" content="yes"/>
    <othermeta name="channel-print" content="no"/>
    <othermeta name="channel-app" content="yes"/>
    
    <!-- Conditional Content Delivery -->
    <othermeta name="region-specific-content" content="yes"/> <!-- Tailored versions per region -->
    <othermeta name="ads-enabled" content="yes"/> <!-- Ad-supported content -->
    
    <!-- Analytics and Tracking -->
    <othermeta name="utm-source" content="homepage-banner"/>
    <othermeta name="utm-medium" content="referral"/>
    <othermeta name="utm-campaign" content="climate-awareness"/>
</metadata>
```
