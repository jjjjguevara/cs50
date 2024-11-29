// app/static/js/src/academic.jsx
import React from "react";
import { createRoot } from "react-dom/client";
import SideNav from "../components/navigation/SideNav";
import TableOfContents from "../components/navigation/TableOfContents";

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
    console.log("TOC mounted to:", tocContainer); // Debug log
  } else {
    console.log("TOC container not found"); // Debug log
  }
});
