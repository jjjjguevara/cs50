import React from "react";
import { Book, Menu, ChevronDown } from "lucide-react";

const TopNav = () => {
  return (
    <nav className="top-nav">
      {/* Logo Section */}
      {/* <div className="site-logo">
        <img src="/static/img/logos/logo.png" alt="Site Logo" />
        <h1 className="site-title">Documentation</h1>
      </div>
       */}
      <div className="site-logo">
        <h1 className="site-title">Documentation</h1>
      </div>

      {/* Main Navigation */}
      <div className="nav-menu">
        <div className="dropdown">
          <button className="flex items-center gap-2">
            <Book size={18} />
            Browse
            <ChevronDown size={14} />
          </button>
          <ul className="dropdown-menu">
            <li>
              <a href="/contents">Table of Contents</a>
            </li>
            <li>
              <a href="/new">What's New</a>
            </li>
            <li>
              <a href="/random">Random Entry</a>
            </li>
            <li>
              <a href="/archives">Archives</a>
            </li>
          </ul>
        </div>

        <div className="dropdown">
          <button className="flex items-center gap-2">
            <Menu size={18} />
            About
            <ChevronDown size={14} />
          </button>
          <ul className="dropdown-menu">
            <li>
              <a href="/about">About Us</a>
            </li>
            <li>
              <a href="/editorial">Editorial Information</a>
            </li>
            <li>
              <a href="/board">Editorial Board</a>
            </li>
            <li>
              <a href="/cite">How to Cite</a>
            </li>
            <li>
              <a href="/contact">Contact</a>
            </li>
          </ul>
        </div>
      </div>
    </nav>
  );
};

export default TopNav;
