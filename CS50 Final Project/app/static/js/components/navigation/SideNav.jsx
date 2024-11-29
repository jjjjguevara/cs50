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
import MapView from "../MapView";

const SideNav = () => {
  const [isArticlesOpen, setIsArticlesOpen] = useState(true);
  const [ditaMaps, setDitaMaps] = useState([]);
  const [selectedMap, setSelectedMap] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Fetch available .ditamap files
  useEffect(() => {
    const fetchDitaMaps = async () => {
      try {
        const response = await api.get("/ditamaps"); // New endpoint to list .ditamap files
        if (response.data && response.data.maps) {
          setDitaMaps(response.data.maps);
        }
        setError(null);
      } catch (err) {
        setError("Failed to load articles");
        console.error("Error loading ditamaps:", err);
      } finally {
        setLoading(false);
      }
    };

    fetchDitaMaps();
  }, []);

  const handleArticleClick = (articleId) => {
    // Remove any file extension before navigation
    const cleanId = articleId.replace(/\.(dita|md)$/, "");
    window.location.href = `/entry/${cleanId}`;
  };

  const handleMapSelect = async (mapId) => {
    try {
      // Dispatch the renderMap event instead of directly manipulating DOM
      const event = new CustomEvent("renderMap", {
        detail: {
          mapId,
          onSelectTopic: (topicId) => {
            window.location.href = `/entry/${topicId}`;
          },
        },
      });
      window.dispatchEvent(event);

      // Update selected state
      setSelectedMap(mapId);
    } catch (error) {
      console.error("Error loading map:", error);
      setError("Failed to load map content");
    }
  };

  return (
    <nav className="article-nav">
      {/* DITA Maps Section */}
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

      {/* Rest of the component remains the same */}
      {/* ... Tools Section ... */}
      {/* ... Other Resources Section ... */}
      {/* ... Back to Top Button ... */}
    </nav>
  );
};

export default SideNav;
