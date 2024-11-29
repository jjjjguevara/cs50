import React from "react";
import ReactDOM from "react-dom/client";
import AcademicLayout from "../components/AcademicLayout";
import TopNav from "../components/navigation/TopNav";
import SideNav from "../components/navigation/SideNav";
import SearchBar from "../components/navigation/SearchBar";
import ArticleHeader from "../components/content/ArticleHeader";
import ArticleContent from "../components/content/ArticleContent";
import TableOfContents from "../components/content/TableOfContents";
import FooterNav from "../components/footer/FooterNav";

// Wrap components with error boundaries
const renderComponent = (Component, elementId) => {
  const container = document.getElementById(elementId);
  if (container) {
    const root = ReactDOM.createRoot(container);
    root.render(
      <React.StrictMode>
        <ErrorBoundary>
          <Component />
        </ErrorBoundary>
      </React.StrictMode>,
    );
  }
};

// Error Boundary Component
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true };
  }

  componentDidCatch(error, errorInfo) {
    console.error("React Error:", error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return <div className="error">Something went wrong.</div>;
    }
    return this.props.children;
  }
}

// Wrap the entire app with SearchProvider
const App = () => (
  <SearchProvider>
    <div className="academic-layout">
      {renderComponent(TopNav, "header-content")}
      {renderComponent(SearchBar, "main-nav")}
      {renderComponent(SideNav, "article-sidebar")}
      <div id="article-content">
        <ArticleHeader />
        <TableOfContents />
        <ArticleContent />
      </div>
      {renderComponent(FooterNav, "footer-content")}
    </div>
  </SearchProvider>
);

// Mount the main app
const rootElement = document.createElement("div");
document.body.appendChild(rootElement);
const root = ReactDOM.createRoot(rootElement);
root.render(
  <React.StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </React.StrictMode>,
);

// Get the topic ID from the URL if available
const getTopicId = () => {
  const path = window.location.pathname;
  const matches = path.match(/\/entry\/(.+)/);
  return matches ? matches[1] : null;
};

// Mount components
document.addEventListener("DOMContentLoaded", () => {
  // Create roots for each mount point
  const mountComponent = (id, Component, props = {}) => {
    const container = document.getElementById(id);
    if (container) {
      const root = ReactDOM.createRoot(container);
      root.render(
        <React.StrictMode>
          <Component {...props} />
        </React.StrictMode>,
      );
    }
  };

  const topicId = getTopicId();

  // Mount each component
  mountComponent("header-content", TopNav);
  mountComponent("main-nav", SearchBar);
  mountComponent("article-sidebar", SideNav);

  // Only mount content components if we have a topic ID
  if (topicId) {
    mountComponent("article-header", ArticleHeader);
    mountComponent("table-of-contents", TableOfContents);
    mountComponent("article-content", ArticleContent, { topicId });
  }

  mountComponent("footer-content", FooterNav);
});
