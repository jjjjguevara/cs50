import React from "react";
import { FileText, BookOpen, Wrench, ArrowUp } from "lucide-react";

const SideNav = () => {
  return (
    <nav className="article-nav">
      <div className="nav-section">
        <h3 className="nav-title">Entry Navigation</h3>
        <ul className="nav-list">
          <li>
            <a href="#toc" className="nav-item">
              <FileText size={16} />
              Entry Contents
            </a>
          </li>
          <li>
            <a href="#bibliography" className="nav-item">
              <BookOpen size={16} />
              Bibliography
            </a>
          </li>
          <li>
            <a href="#academic-tools" className="nav-item">
              <Wrench size={16} /> {/* Changed from Tool to Wrench */}
              Academic Tools
            </a>
          </li>
        </ul>
      </div>

      <div className="nav-section mt-6">
        <h3 className="nav-title">Other Resources</h3>
        <ul className="nav-list">
          <li>
            <a href="/friends/preview" className="nav-item">
              Preview PDF
            </a>
          </li>
          <li>
            <a href="#cite-this-entry" className="nav-item">
              How to Cite
            </a>
          </li>
        </ul>
      </div>

      <button
        onClick={() => window.scrollTo({ top: 0, behavior: "smooth" })}
        className="back-to-top"
      >
        <ArrowUp size={16} />
        Back to Top
      </button>
    </nav>
  );
};

export default SideNav;
