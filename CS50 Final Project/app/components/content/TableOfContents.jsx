import React, { useState, useEffect } from "react";
import { ChevronRight, ChevronDown } from "lucide-react";

const TableOfContents = ({ content }) => {
  const [toc, setToc] = useState([]);
  const [expanded, setExpanded] = useState(true);

  useEffect(() => {
    // Parse content and extract headings
    const parseHeadings = () => {
      const parser = new DOMParser();
      const doc = parser.parseFromString(content, "text/html");
      const headings = doc.querySelectorAll("h2, h3, h4");

      const tocItems = Array.from(headings).map((heading) => ({
        id: heading.id,
        text: heading.textContent,
        level: parseInt(heading.tagName.charAt(1)),
        children: [],
      }));

      buildTocHierarchy(tocItems);
    };

    if (content) {
      parseHeadings();
    }
  }, [content]);

  const buildTocHierarchy = (items) => {
    const hierarchy = [];
    const stack = [{ level: 1, children: hierarchy }];

    items.forEach((item) => {
      while (item.level <= stack[stack.length - 1].level) {
        stack.pop();
      }
      stack[stack.length - 1].children.push(item);
      stack.push(item);
    });

    setToc(hierarchy);
  };

  const renderTocItem = (item) => (
    <li key={item.id} className={`toc-item level-${item.level}`}>
      <a href={`#${item.id}`} className="toc-link">
        {item.text}
      </a>
      {item.children?.length > 0 && (
        <ul className="toc-sublist">{item.children.map(renderTocItem)}</ul>
      )}
    </li>
  );

  return (
    <nav className="table-of-contents">
      <button className="toc-toggle" onClick={() => setExpanded(!expanded)}>
        {expanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
        Table of Contents
      </button>

      {expanded && <ul className="toc-list">{toc.map(renderTocItem)}</ul>}
    </nav>
  );
};

export default TableOfContents;
