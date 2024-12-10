// app/static/js/src/academic.jsx
import React from "react";
import { createRoot } from "react-dom/client";
import ErrorBoundary from "../components/common/ErrorBoundary";
import TopNav from "../components/navigation/TopNav";
import SideNav from "../components/navigation/SideNav";
import TableOfContents from "../components/navigation/TableOfContents";

const Academic = () => {
  return (
    <div className="academic-layout">
      {/* Bootstrap Navbar */}
      <nav className="navbar navbar-expand-lg navbar-light bg-light fixed-top">
        <div className="container-fluid">
          <TopNav />
        </div>
      </nav>

      {/* Main Content Area */}
      <div className="main-wrapper">
        {/* Left Sidebar */}
        <aside className="sidebar-left">
          <SideNav />
        </aside>

        {/* Content Area */}
        <main className="content-main">
          <div className="content-container">
            <div
              id="article-content"
              className="article-content"
              dangerouslySetInnerHTML={{
                __html: window.__INITIAL_CONTENT__ || "",
              }}
            />
          </div>
        </main>

        {/* Right Sidebar (TOC) */}
        <aside className="sidebar-right">
          <TableOfContents />
        </aside>
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
