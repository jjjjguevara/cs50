import React, { useEffect } from "react";
import { Book, Menu, ChevronDown } from "lucide-react";
import { api } from "@/utils/api";

const TopNav = () => {
  useEffect(() => {
    const currentPath = window.location.pathname;
    console.log("Current path:", currentPath); // Debug log

    if (currentPath === "/articles") {
      console.log("On articles route, attempting redirect"); // Debug log

      const loadFirstDitamap = async () => {
        try {
          console.log("Fetching ditamaps..."); // Debug log
          const response = await api.get("/api/ditamaps");
          console.log("API Response:", response.data); // Debug log

          if (
            response.data?.success &&
            response.data.maps &&
            response.data.maps.length > 0
          ) {
            const firstMap = response.data.maps[0];
            const redirectUrl = `/entry/${firstMap.id}.ditamap`;
            console.log("Redirecting to:", redirectUrl); // Debug log
            window.location.href = redirectUrl;
          } else {
            console.error("No ditamaps found");
          }
        } catch (err) {
          console.error("Error loading ditamaps:", err);
        }
      };

      loadFirstDitamap();
    }
  }, []);

  return (
    <nav className="top-nav">
      <div className="site-logo">
        <h1 className="site-title">Documentation</h1>
      </div>

      <div className="nav-menu">
        <div className="nav-item">
          <a
            href="/articles"
            onClick={(e) => {
              e.preventDefault();
              const loadFirstDitamap = async () => {
                try {
                  console.log("Fetching ditamaps from click handler..."); // Debug log
                  const response = await api.get("/api/ditamaps");
                  console.log("Click handler API Response:", response.data); // Debug log

                  if (
                    response.data?.success &&
                    response.data.maps &&
                    response.data.maps.length > 0
                  ) {
                    const firstMap = response.data.maps[0];
                    const redirectUrl = `/entry/${firstMap.id}.ditamap`;
                    console.log("Click handler redirecting to:", redirectUrl); // Debug log
                    window.location.href = redirectUrl;
                  } else {
                    console.error("No ditamaps found");
                  }
                } catch (err) {
                  console.error("Error loading ditamaps:", err);
                }
              };
              loadFirstDitamap();
            }}
          >
            Articles
          </a>
        </div>

        <div className="dropdown">
          <button>
            <Book size={18} />
            <span>Browse</span>
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
          <button>
            <Menu size={18} />
            <span>About</span>
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
