import React, { useState, useEffect } from "react";
import {
  BookOpen,
  Wrench,
  ArrowUp,
  Book,
  ChevronDown,
  ChevronRight,
  Building,
  LibraryBig,
} from "lucide-react";
import { api } from "@/utils/api";
import { Pause, Play, RefreshCcw } from "@utils/icons";

const SideNav = () => {
  const [ditaMaps, setDitaMaps] = useState([]);
  const [selectedMap, setSelectedMap] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchDitaMaps = async () => {
      try {
        setLoading(true);
        const response = await api.get("/ditamaps");
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
    window.location.href = `/entry/${mapId}.ditamap`;
  };

  return (
    <>
      {/* Floating Button */}
      <button
        className="btn btn-dark position-fixed d-flex align-items-center justify-content-center"
        type="button"
        data-bs-toggle="offcanvas"
        data-bs-target="#sidebarOffcanvas"
        aria-controls="sidebarOffcanvas"
        style={{
          top: "70px", // Position below navbar
          left: "20px",
          zIndex: 1030,
          width: "40px",
          height: "40px",
          borderRadius: "50%",
        }}
      >
        <LibraryBig size={40} />
      </button>

      {/* Offcanvas Sidebar */}
      <div
        className="offcanvas offcanvas-start"
        data-bs-scroll="true"
        data-bs-backdrop="false"
        tabIndex="-1"
        id="sidebarOffcanvas"
        aria-labelledby="sidebarOffcanvasLabel"
        style={{
          top: "56px", // Start below navbar
          height: "calc(100vh - 56px)", // Adjust height to account for navbar
        }}
      >
        <div className="offcanvas-header">
          <h5 className="offcanvas-title" id="sidebarOffcanvasLabel">
            Articles
          </h5>
          <button
            type="button"
            className="btn-close"
            data-bs-dismiss="offcanvas"
            aria-label="Close"
          ></button>
        </div>

        <div className="offcanvas-body">
          <div className="list-group list-group-flush">
            {loading ? (
              <div className="text-center p-3">
                <div className="spinner-border text-primary" role="status">
                  <span className="visually-hidden">Loading...</span>
                </div>
              </div>
            ) : error ? (
              <div className="alert alert-danger" role="alert">
                {error}
              </div>
            ) : (
              ditaMaps.map((map) => (
                <a
                  key={map.id}
                  href={`/entry/${map.id}.ditamap`}
                  onClick={(e) => {
                    e.preventDefault();
                    handleMapSelect(map.id);
                  }}
                  className={`list-group-item list-group-item-action ${
                    selectedMap === map.id ? "active" : ""
                  }`}
                >
                  <div className="d-flex w-100 justify-content-between">
                    <h6 className="mb-1">{map.title}</h6>
                  </div>
                  {map.groups &&
                    map.groups.map((group, index) => (
                      <small
                        key={index}
                        className={`d-block ${selectedMap === map.id ? "" : "text-body-secondary"}`}
                      >
                        {group.navtitle}
                      </small>
                    ))}
                </a>
              ))
            )}
          </div>
        </div>
      </div>
    </>
  );
};

export default SideNav;
