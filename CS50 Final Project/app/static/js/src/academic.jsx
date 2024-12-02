// app/static/js/src/academic.jsx
import React, { useEffect } from "react";
import { createRoot } from "react-dom/client";
import ErrorBoundary from "@components/ErrorBoundary";
import SideNav from "../components/navigation/SideNav";
import TableOfContents from "../components/navigation/TableOfContents";
import Bibliography from "../components/bibliography/bibliography.jsx";
import "../utils/ReactWebComponentWrapper";
import "../utils/componentRegistry";
import "@utils/modulePolyfill";

// Ensure React is globally available
if (typeof window !== "undefined" && !window.React) {
  window.React = React;
}

// Initialize web components
const initializeArtifacts = () => {
  console.log("Initializing artifacts...");
  const artifacts = document.querySelectorAll(".artifact-wrapper");
  console.log("Found artifacts:", artifacts.length);

  artifacts.forEach((artifact) => {
    const wrapper = artifact.querySelector("web-component-wrapper");
    if (wrapper) {
      const componentName = wrapper.getAttribute("component");
      console.log("Initializing component:", componentName);
      // Force web component to initialize
      wrapper.connectedCallback?.();
    }
  });
};

// Debug initialization
console.log("React version:", React.version);
console.log("ReactDOM version:", ReactDOM.version);
console.log("Recharts available:", !!window.Recharts);
console.log("Component registry:", window.ReactComponents);

const Academic = () => {
  useEffect(() => {
    // Initialize web components
    initializeArtifacts();

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
  // Mount SideNav with ErrorBoundary
  const sidebarContainer = document.getElementById("root");
  if (sidebarContainer) {
    const sidebarRoot = createRoot(sidebarContainer);
    sidebarRoot.render(
      <ErrorBoundary>
        <React.StrictMode>
          <SideNav />
        </React.StrictMode>
      </ErrorBoundary>,
    );
  }

  // Mount TableOfContents with ErrorBoundary
  const tocContainer = document.getElementById("toc-root");
  if (tocContainer) {
    const tocRoot = createRoot(tocContainer);
    tocRoot.render(
      <ErrorBoundary>
        <React.StrictMode>
          <TableOfContents />
        </React.StrictMode>
      </ErrorBoundary>,
    );
  }

  // Mount Bibliography with ErrorBoundary
  const bibContainer = document.getElementById("bibliography-root");
  if (bibContainer) {
    const bibRoot = createRoot(bibContainer);
    bibRoot.render(
      <ErrorBoundary>
        <React.StrictMode>
          <Bibliography />
        </React.StrictMode>
      </ErrorBoundary>,
    );
  }

  // Initialize artifacts
  initializeArtifacts();
});

export default Academic;
