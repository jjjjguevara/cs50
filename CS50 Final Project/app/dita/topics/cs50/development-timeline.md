# Roadmap

## Development chart

```mermaid
gantt
    title DITA Web Application Development Roadmap (Nov 25 - Dec 12, 2024)
    dateFormat YYYY-MM-DD
    axisFormat %B %d

    section Project Setup
    Project Planning & Requirements    :2024-11-25, 2d
    Environment & Folder Setup        :2024-11-25, 1d
    Database Schema Design           :2024-11-26, 2d

    section Backend Development
    DITA Processing Pipeline        :2024-11-26, 5d
    Database Implementation         :2024-11-26, 4d
    API Development                :2024-11-28, 5d
    XSLT Transformations           :2024-11-28, 4d

    section Frontend Development
    Component Design               :2024-11-27, 3d
    Core Components Development    :2024-11-28, 5d
    Page Layouts & Routing        :2024-12-02, 4d
    Dynamic Content Integration   :2024-12-04, 3d

    section Integration & Testing
    Backend-Frontend Integration  :2024-12-06, 3d
    Unit Testing                 :2024-12-07, 2d
    Integration Testing          :2024-12-09, 2d
    Final Testing & Fixes        :2024-12-11, 2d

    section Deployment
    Documentation & Deployment    :2024-12-11, 2d
```

## TODO
- [x] Add LaTeX support
- [ ] Develop image-loading features per specification
- [ ] Develop interactive content functionality
- [ ] Develop XML metadata usage structure to toggle UI elements
- [ ] Develop logic to store pages in the Output folder and replace them only if a change to the dita topics or dita map is detected
- [ ] Develop Bibliography component with topic parser and JSON support
- [ ] Develop PDF conversion logic with Pandoc
