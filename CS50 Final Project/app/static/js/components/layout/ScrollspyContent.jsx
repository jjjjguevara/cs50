import React from "react";
import PropTypes from "prop-types";
import { Pause, Play, RefreshCcw } from "@utils/icons";

const ScrollspyContent = ({
  sections,
  navTitle,
  navId = "section-nav",
  contentClassName = "content-wrapper",
  sidebarClassName = "sidebar-nav",
}) => {
  return (
    <div className="row">
      {/* Navigation Sidebar */}
      <div className="col-md-3">
        <nav
          id={navId}
          className={`navbar navbar-light bg-light flex-column align-items-stretch p-3 sticky-top ${sidebarClassName}`}
        >
          {navTitle && <span className="navbar-brand h6 mb-3">{navTitle}</span>}
          <nav className="nav nav-pills flex-column">
            {sections.map((section) => (
              <a key={section.id} className="nav-link" href={`#${section.id}`}>
                {section.title}
              </a>
            ))}
          </nav>
        </nav>
      </div>

      {/* Content Area */}
      <div className="col-md-9">
        <div
          className={contentClassName}
          data-bs-spy="scroll"
          data-bs-target={`#${navId}`}
          data-bs-offset="0"
          tabIndex="0"
        >
          {sections.map((section) => (
            <section key={section.id} id={section.id}>
              <h4>{section.title}</h4>
              <div className="section-content">{section.content}</div>
            </section>
          ))}
        </div>
      </div>
    </div>
  );
};

ScrollspyContent.propTypes = {
  sections: PropTypes.arrayOf(
    PropTypes.shape({
      id: PropTypes.string.isRequired,
      title: PropTypes.string.isRequired,
      content: PropTypes.node.isRequired,
    }),
  ).isRequired,
  navTitle: PropTypes.string,
  navId: PropTypes.string,
  contentClassName: PropTypes.string,
  sidebarClassName: PropTypes.string,
};

export default ScrollspyContent;
