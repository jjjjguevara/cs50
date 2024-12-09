// app/static/js/src/academic.jsx
import React from "react";
import { createRoot } from "react-dom/client";
import ErrorBoundary from "@components/ErrorBoundary";
import TopNav from "../components/navigation/TopNav";
import SideNav from "../components/navigation/SideNav";
import TableOfContents from "../components/navigation/TableOfContents";
import "../utils/modulePolyfill";

const Academic = () => {
  return (
    <div className="academic-layout">
      <TopNav />
      <div className="d-flex flex-column flex-lg-row">
        <aside className="sidebar left-sidebar">
          <SideNav />
        </aside>
        <main className="main-content container-fluid">
          <div className="row">
            <article className="col-lg-9">
              <div id="article-content" />
            </article>
            <aside className="col-md-3 right-sidebar">
              <TableOfContents />
            </aside>
          </div>
        </main>
      </div>
    </div>
  );
};

// Mount main app
const rootElement = document.getElementById("app-root");
if (rootElement) {
  const root = createRoot(rootElement);
  root.render(
    <ErrorBoundary>
      <React.StrictMode>
        <Academic />
      </React.StrictMode>
    </ErrorBoundary>,
  );
}

export default Academic;
