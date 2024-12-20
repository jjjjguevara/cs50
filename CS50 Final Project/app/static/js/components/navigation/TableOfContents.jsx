import React, { useState, useEffect } from "react";

const TableOfContents = () => {
  const [headings, setHeadings] = useState([]);

  useEffect(() => {
    const findHeadings = () => {
      // Change this to target where your content actually is
      const mapContent = document.getElementById("article-wrapper");
      console.log("Looking for headings in:", mapContent);

      if (mapContent) {
        const headingElements = mapContent.querySelectorAll(
          "h1.content-title, " +
            ".dita-content h1, " +
            ".dita-content h2, " +
            ".markdown-content h1, " +
            ".markdown-content h2",
        );

        const headingsData = Array.from(headingElements).map((heading) => {
          // Keep the ID and anchor functionality, but remove pilcrow from display text
          const headingText = heading.textContent.replace("¶", "").trim();

          let level = 1;
          if (heading.classList.contains("content-title")) {
            level = 1;
          } else if (heading.tagName === "H1") {
            level = 2;
          } else if (heading.tagName === "H2") {
            level = 3;
          }

          // Use the heading's existing ID
          const id = heading.id;

          return {
            id,
            text: headingText,
            level,
          };
        });

        console.log("Found headings:", headingsData);
        setHeadings(headingsData);
      }
    };

    findHeadings();

    const observer = new MutationObserver(findHeadings);
    observer.observe(document.body, {
      childList: true,
      subtree: true,
    });

    return () => observer.disconnect();
  }, []);

  return (
    <nav className="toc-nav" aria-label="Table of contents">
      <div className="toc-header">
        <h2>Table of Contents</h2>
      </div>
      <ul className="toc-list">
        {headings.map((heading) => (
          <li
            key={heading.id}
            className={`toc-item toc-level-${heading.level}`}
          >
            <a href={`#${heading.id}`}>{heading.text}</a>
          </li>
        ))}
      </ul>
    </nav>
  );
};
export default TableOfContents;
