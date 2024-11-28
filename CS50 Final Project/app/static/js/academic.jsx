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

// Mount Main Content
const articleContent = ReactDOM.createRoot(
  document.getElementById("article-content"),
);
articleContent.render(
  <React.StrictMode>
    <AcademicLayout>
      <ArticleHeader />
      <TableOfContents />
      <ArticleContent />
    </AcademicLayout>
  </React.StrictMode>,
);

// Mount Footer
const footerContent = ReactDOM.createRoot(
  document.getElementById("footer-content"),
);
footerContent.render(
  <React.StrictMode>
    <FooterNav />
  </React.StrictMode>,
);
