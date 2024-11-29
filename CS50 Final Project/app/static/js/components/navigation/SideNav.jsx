// app/static/js/components/navigation/SideNav.jsx
import React, { useState, useEffect } from "react";
import {
  BookOpen,
  Wrench,
  ArrowUp,
  Book,
  ChevronDown,
  ChevronRight,
} from "lucide-react";
import { api } from "@/utils/api";

const SideNav = () => {
  const [isArticlesOpen, setIsArticlesOpen] = useState(true);
  const [ditaMaps, setDitaMaps] = useState([]);
  const [selectedMap, setSelectedMap] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchDitaMaps = async () => {
      try {
        setLoading(true);
        const response = await api.get("/ditamaps");
        console.log("DitaMaps response:", response); // Debug log

        if (response.data?.success && response.data.maps) {
          setDitaMaps(response.data.maps);
        } else {
          throw new Error("Invalid response format");
        }
      } catch (err) {
        console.error("Error loading ditamaps:", err);
        setError("Failed to load articles");
      } finally {
        setLoading(false);
      }
    };

    fetchDitaMaps();
  }, []);

  const handleMapSelect = (mapId) => {
    // Instead of dispatching a renderMap event, we should navigate to the map entry
    window.location.href = `/entry/${mapId}.ditamap`;
  };

  return (
    <nav className="article-nav">
      <div className="nav-section">
        <button
          className="nav-title-button"
          onClick={() => setIsArticlesOpen(!isArticlesOpen)}
        >
          <div className="flex items-center gap-2">
            <Book size={16} />
            <span>Articles</span>
          </div>
          {isArticlesOpen ? (
            <ChevronDown size={16} />
          ) : (
            <ChevronRight size={16} />
          )}
        </button>

        {isArticlesOpen && (
          <div className="articles-list">
            {loading ? (
              <div className="nav-loading">Loading articles...</div>
            ) : error ? (
              <div className="nav-error">{error}</div>
            ) : (
              <ul className="nav-list">
                {ditaMaps.map((map) => (
                  <li key={map.id}>
                    <button
                      onClick={() => handleMapSelect(map.id)}
                      className={`nav-item ${selectedMap === map.id ? "active" : ""}`}
                    >
                      <Book size={14} />
                      <span>{map.title}</span>
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}
      </div>

      {/* Rest of your nav sections */}
    </nav>
  );
};

export default SideNav;
