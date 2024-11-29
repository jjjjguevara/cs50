// app/static/js/src/academic.jsx
import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import TopNav from "../components/navigation/TopNav";
import SideNav from "../components/navigation/SideNav";
import SearchBar from "../components/navigation/SearchBar";
import FooterNav from "../components/footer/FooterNav";
import AcademicLayout from "../components/AcademicLayout";

const Academic = () => {
  const [isNavVisible, setIsNavVisible] = useState(true);
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);
  const [lastScroll, setLastScroll] = useState(0);

  useEffect(() => {
    const handleScroll = () => {
      const currentScroll = window.pageYOffset;

      if (currentScroll > lastScroll && currentScroll > 100) {
        setIsNavVisible(false);
      } else {
        setIsNavVisible(true);
      }

      setLastScroll(currentScroll);
    };

    window.addEventListener("scroll", handleScroll);

    return () => {
      window.removeEventListener("scroll", handleScroll);
    };
  }, [lastScroll]);

  const toggleSidebar = () => {
    setIsSidebarCollapsed(!isSidebarCollapsed);
  };

  return (
    <AcademicLayout>
      <div
        className={`navbar navbar-expand-lg navbar-light bg-white border-bottom sticky-top ${!isNavVisible ? "nav-hidden" : ""}`}
      >
        <TopNav />
        <SearchBar />
      </div>

      <div className="d-flex">
        <div
          className={`left-sidebar ${isSidebarCollapsed ? "collapsed" : ""}`}
        >
          <button
            className="btn btn-link sidebar-toggle"
            onClick={toggleSidebar}
          >
            <i
              className={`bi bi-${isSidebarCollapsed ? "chevron-right" : "chevron-left"}`}
            ></i>
          </button>
          <SideNav />
        </div>

        <div className={`main-content ${isSidebarCollapsed ? "expanded" : ""}`}>
          <div className="container-fluid">
            <div className="row">
              <div className="col-md-9">
                <div id="article-content">
                  {/* Content will be rendered here */}
                </div>
              </div>

              <div className="col-md-3">
                <div className="right-sidebar">
                  <div className="toc-header">Contents</div>
                  <div className="toc-content">
                    {/* Table of Contents will be generated here */}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <FooterNav />
    </AcademicLayout>
  );
};

// Mount the React application
const container = document.getElementById("root");
const root = createRoot(container);
root.render(<Academic />);

export default Academic;
