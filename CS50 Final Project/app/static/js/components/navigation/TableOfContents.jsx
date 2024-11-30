// app/static/js/components/navigation/TableOfContents.jsx
import React, { useEffect, useState } from "react";

const TableOfContents = () => {
  const [headings, setHeadings] = useState([]);

  useEffect(() => {
    const findHeadings = () => {
      const mapContent = document.querySelector(".map-content");

      if (mapContent) {
        const headingElements = mapContent.querySelectorAll(
          "h1.content-title, " + // Main title
            ".dita-content h1, " + // Topic titles
            ".dita-content h2, " + // Section titles
            ".dita-content h3, " + // Subsection titles
            ".markdown-content h1, " +
            ".markdown-content h2, " +
            ".markdown-content h3",
        );

        const headingsData = Array.from(headingElements).map((heading) => {
          // Remove the anchor element before getting the text
          const headingText = heading.cloneNode(true);
          const anchor = headingText.querySelector(".heading-anchor");
          if (anchor) {
            anchor.remove();
          }
          const text = headingText.textContent.trim();

          let level = 1;
          if (heading.classList.contains("content-title")) {
            level = 1;
          } else if (heading.tagName === "H1") {
            level = 2;
          } else if (heading.tagName === "H2") {
            level = 3;
          } else if (heading.tagName === "H3") {
            level = 4;
          }

          const id = heading.id || generateId(text);
          if (!heading.id) {
            heading.id = id;
          }

          return {
            id,
            text,
            level,
          };
        });

        setHeadings(headingsData);
      }
    };

    // Initial heading check
    findHeadings();

    // Set up MutationObserver to monitor changes
    const observer = new MutationObserver(findHeadings);
    observer.observe(document.body, {
      childList: true,
      subtree: true,
    });

    // Cleanup observer on component unmount
    return () => observer.disconnect();
  }, []);

  const generateId = (text) => {
    return text
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/(^-|-$)/g, "");
  };

  return (
    <nav className="toc-nav" aria-label="Table of contents">
      <div className="toc-header">
        <h2>Contents</h2>
      </div>
      <ul className="toc-list">
        {headings.map((heading) => (
          <li key={heading.id} className={`toc-item level-${heading.level}`}>
            <a href={`#${heading.id}`}>{heading.text}</a>
          </li>
        ))}
      </ul>
    </nav>
  );
};

export default TableOfContents;
