// app/static/js/components/MapView.jsx
import React, { useState, useEffect } from "react";
import {
  Book,
  FileText,
  ChevronRight,
  ChevronDown,
  FolderClosed,
  FolderOpen,
} from "lucide-react";
import { api } from "../utils/api";

export default function MapView({ mapId, onSelectTopic }) {
  const [map, setMap] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [expandedGroups, setExpandedGroups] = useState({});
  const [selectedTopicId, setSelectedTopicId] = useState(null);

  useEffect(() => {
    if (mapId) {
      setLoading(true);
      api
        .get(`/map/${mapId}`)
        .then((response) => {
          if (response.data && response.data.success) {
            console.log("Received map data:", response.data.map);
            setMap(response.data.map);
            // Auto-expand first group
            if (
              response.data.map.groups &&
              response.data.map.groups.length > 0
            ) {
              const initialExpanded = {};
              response.data.map.groups.forEach((_, index) => {
                initialExpanded[index] = true;
              });
              setExpandedGroups(initialExpanded);
            }
          }
          setLoading(false);
        })
        .catch((err) => {
          console.error("Error loading map:", err);
          setError(err.message);
          setLoading(false);
        });
    }
  }, [mapId]);

  const handleTopicClick = (href) => {
    // Extract topic ID from href
    const topicId = href
      .split("/")
      .pop()
      .replace(/\.(dita|md)$/, "");
    setSelectedTopicId(topicId);
    if (onSelectTopic) {
      onSelectTopic(topicId);
    }
  };

  const toggleGroup = (groupIndex) => {
    setExpandedGroups((prev) => ({
      ...prev,
      [groupIndex]: !prev[groupIndex],
    }));
  };

  if (loading) return <div className="map-loading">Loading map content...</div>;
  if (error) return <div className="map-error">Error: {error}</div>;
  if (!map) return <div className="map-empty">No map selected</div>;

  return (
    <div className="map-content">
      <h1 className="map-title">{map.title}</h1>
      <div className="map-groups">
        {map.groups.map((group, groupIndex) => (
          <div key={groupIndex} className="map-group">
            <button
              onClick={() => toggleGroup(groupIndex)}
              className="group-header"
            >
              {expandedGroups[groupIndex] ? (
                <FolderOpen size={16} className="group-icon" />
              ) : (
                <FolderClosed size={16} className="group-icon" />
              )}
              <span className="group-title">{group.navtitle}</span>
              {expandedGroups[groupIndex] ? (
                <ChevronDown size={14} className="group-chevron" />
              ) : (
                <ChevronRight size={14} className="group-chevron" />
              )}
            </button>

            {expandedGroups[groupIndex] && group.topics && (
              <div className="group-topics">
                {group.topics.map((topic, topicIndex) => {
                  // Extract just the filename from href for display
                  const displayName = topic.href
                    .split("/")
                    .pop()
                    .replace(/\.(dita|md)$/, "");

                  return (
                    <button
                      key={topicIndex}
                      onClick={() => handleTopicClick(topic.href)}
                      className={`topic-item ${
                        selectedTopicId === topic.id ? "selected" : ""
                      }`}
                    >
                      <FileText size={14} className="topic-icon" />
                      <span className="topic-title">
                        {topic.title || displayName}
                      </span>
                    </button>
                  );
                })}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
