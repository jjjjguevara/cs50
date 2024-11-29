import React from "react";
import ReactDOM from "react-dom/client";
import AcademicLayout from "../components/AcademicLayout";
import TopNav from "../components/navigation/TopNav";
import SideNav from "../components/navigation/SideNav";
import SearchBar from "../components/navigation/SearchBar";
import ArticleHeader from "../components/content/ArticleHeader";
import TableOfContents from "../components/content/TableOfContents";
import FooterNav from "../components/footer/FooterNav";

// Initialize all React roots after DOM is loaded
document.addEventListener("DOMContentLoaded", () => {
  // Mount Header Components
  const headerContent = ReactDOM.createRoot(
    document.getElementById("header-content"),
  );
  headerContent.render(
    <React.StrictMode>
      <TopNav />
    </React.StrictMode>,
  );

  // Mount Search
  const mainNav = ReactDOM.createRoot(document.getElementById("main-nav"));
  mainNav.render(
    <React.StrictMode>
      <SearchBar />
    </React.StrictMode>,
  );

  // Mount Sidebar Navigation
  const sidebarNav = ReactDOM.createRoot(
    document.getElementById("article-sidebar"),
  );
  sidebarNav.render(
    <React.StrictMode>
      <SideNav />
    </React.StrictMode>,
  );

  // Let the server handle main article content rendering; removed ArticleContent mount

  // Mount Footer
  const footerContent = ReactDOM.createRoot(
    document.getElementById("footer-content"),
  );
  footerContent.render(
    <React.StrictMode>
      <FooterNav />
    </React.StrictMode>,
  );

  // Removed both map event listeners as they are no longer needed
});
