// app/static/js/src/academic.jsx
import React, { useState, useEffect } from "react";
import { createRoot } from "react-dom/client";
import SideNav from "../components/navigation/SideNav";
import TableOfContents from "../components/navigation/TableOfContents";
import Bibliography from "../components/bibliography/bibliography.jsx";
import "../utils/ReactWebComponentWrapper";
import "../utils/componentRegistry";

const Academic = () => {
  useEffect(() => {
    // Initialize web components
    const artifacts = document.querySelectorAll(".artifact-wrapper");
    artifacts.forEach((artifact) => {
      // Force web component update
      artifact.connectedCallback?.();
    });

    // Find bibliography section and extract references
    const bibSection = document.querySelector(".bibliography-section");
    if (bibSection) {
      try {
        const refs = JSON.parse(bibSection.dataset.references);
        setReferences(refs);
        // Remove the data element as we no longer need it
        bibSection.removeAttribute("data-references");
      } catch (e) {
        console.error("Error parsing references:", e);
      }
    }
  }, []);

  return (
    <div className="academic-layout">
      <div id="header-content"></div>
      <div id="main-nav"></div>
      <div id="article-sidebar"></div>
      <div id="article-content">
        <div className="content-wrapper">
          <div className="article-content">
            <div
              dangerouslySetInnerHTML={{
                __html: window.initialContent || "",
              }}
            />
            {references.length > 0 && <Bibliography references={references} />}
          </div>
          <TableOfContents />
        </div>
      </div>
      <div id="footer-content"></div>
    </div>
  );
};

// Mount components when DOM is fully loaded
document.addEventListener("DOMContentLoaded", () => {
  // Mount SideNav
  const sidebarContainer = document.getElementById("root");
  if (sidebarContainer) {
    const sidebarRoot = createRoot(sidebarContainer);
    sidebarRoot.render(<SideNav />);
  }

  // Mount TableOfContents
  const tocContainer = document.getElementById("toc-root");
  if (tocContainer) {
    const tocRoot = createRoot(tocContainer);
    tocRoot.render(<TableOfContents />);
  }

  // Mount Bibliography
  const bibContainer = document.getElementById("bibliography-root");
  if (bibContainer) {
    const bibRoot = createRoot(bibContainer);
    bibRoot.render(<Bibliography />);
    console.log("Bibliography mounted to:", bibContainer); // Debug log
  } else {
    console.log("Bibliography container not found"); // Debug log
  }
});

// Add debug logging for references data
const debugReferences = () => {
  const bibSection = document.querySelector(".bibliography-section");
  if (bibSection) {
    console.log("Found bibliography section:", bibSection);
    console.log("References data:", bibSection.dataset.references);
  } else {
    console.log("No bibliography section found");
  }
};

// Run debug after a short delay to ensure content is loaded
setTimeout(debugReferences, 1000);

export default Academic;
