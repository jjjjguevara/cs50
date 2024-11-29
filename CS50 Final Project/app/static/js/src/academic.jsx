import React from "react";
import ReactDOM from "react-dom/client";
import TopNav from "../components/navigation/TopNav";
import SideNav from "../components/navigation/SideNav";
import SearchBar from "../components/navigation/SearchBar";
import ArticleHeader from "../components/content/ArticleHeader";
import ArticleContent from "../components/content/ArticleContent";
import TableOfContents from "../components/content/TableOfContents";
import FooterNav from "../components/footer/FooterNav";
import { SearchProvider } from "../search";

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
