// app/static/js/components/navigation/TableOfContents.jsx
import React, { useEffect, useState } from "react";

const TableOfContents = () => {
  const [headings, setHeadings] = useState([]);

  useEffect(() => {
    const extractHeadings = () => {
      const mapContent = document.querySelector(".map-content");
      console.log("Map content found:", mapContent); // Debug log

      if (mapContent) {
        const headingElements = mapContent.querySelectorAll(
          "h1.content-title, " +
            ".dita-content h1.text-2xl, " +
            ".dita-content h2.text-xl, " +
            ".markdown-content h1.text-2xl, " +
            ".markdown-content h2.text-xl",
        );

        console.log("Found headings:", headingElements.length); // Debug log
        console.log("Heading elements:", Array.from(headingElements)); // Debug log

        const headingsData = Array.from(headingElements).map((heading) => {
          const id = heading.id || generateId(heading.textContent);
          if (!heading.id) {
            heading.id = id; // Assign ID if missing
          }

          let level = 1;
          if (heading.classList.contains("content-title")) {
            level = 1;
          } else if (heading.classList.contains("text-2xl")) {
            level = 2;
          } else if (heading.classList.contains("text-xl")) {
            level = 3;
          }

          return {
            id,
            text: heading.textContent,
            level,
          };
        });

        console.log("Processed headings:", headingsData); // Debug log
        setHeadings(headingsData);
      }
    };

    // Initial heading extraction
    extractHeadings();

    // Set up MutationObserver
    const observer = new MutationObserver(() => {
      extractHeadings();
    });

    observer.observe(document.body, {
      childList: true,
      subtree: true,
    });

    // Cleanup observer
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
