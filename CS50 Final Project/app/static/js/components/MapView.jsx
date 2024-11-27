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

export default function MapView({ onSelectTopic }) {
  const [maps, setMaps] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [expandedMaps, setExpandedMaps] = useState({});
  const [expandedGroups, setExpandedGroups] = useState({});
  const [selectedTopicId, setSelectedTopicId] = useState(null);

  useEffect(() => {
    api
      .get("/maps")
      .then((data) => {
        console.log("Received maps:", data);
        setMaps(data);
        setLoading(false);

        // If there's only one map, expand it automatically
        if (data.length === 1) {
          setExpandedMaps({ [data[0].id]: true });
        }
      })
      .catch((err) => {
        console.error("Error loading maps:", err);
        setError(err.message);
        setLoading(false);
      });
  }, []);

  const handleTopicClick = (topic) => {
    setSelectedTopicId(topic.id);
    // Extract topic ID from href if it's a full path
    const topicId = topic.href
      ? topic.href.split("/").pop().replace(".dita", "")
      : topic.id;
    onSelectTopic(topicId);
  };

  const toggleMap = (mapId) => {
    setExpandedMaps((prev) => ({
      ...prev,
      [mapId]: !prev[mapId],
    }));
  };

  const toggleGroup = (groupId) => {
    setExpandedGroups((prev) => ({
      ...prev,
      [groupId]: !prev[groupId],
    }));
  };

  if (loading) return <div className="p-4 text-gray-600">Loading maps...</div>;
  if (error) return <div className="p-4 text-red-500">Error: {error}</div>;

  return (
    <div className="space-y-2">
      {maps.map((map) => (
        <div key={map.id} className="border-b last:border-b-0">
          <button
            onClick={() => toggleMap(map.id)}
            className="w-full p-3 flex items-center hover:bg-gray-50 text-left"
          >
            <Book className="text-blue-500 mr-2" size={18} />
            <span className="flex-grow font-medium">{map.title}</span>
            {expandedMaps[map.id] ? (
              <ChevronDown size={16} className="text-gray-400" />
            ) : (
              <ChevronRight size={16} className="text-gray-400" />
            )}
          </button>

          {expandedMaps[map.id] && map.groups && (
            <div className="pb-2">
              {map.groups.map((group, groupIndex) => (
                <div key={groupIndex} className="border-l border-gray-200 ml-4">
                  <button
                    onClick={() => toggleGroup(`${map.id}-${groupIndex}`)}
                    className="w-full p-2 pl-4 flex items-center hover:bg-gray-50 text-left"
                  >
                    {expandedGroups[`${map.id}-${groupIndex}`] ? (
                      <FolderOpen size={16} className="text-blue-500 mr-2" />
                    ) : (
                      <FolderClosed size={16} className="text-blue-500 mr-2" />
                    )}
                    <span className="text-sm font-medium">
                      {group.navtitle}
                    </span>
                    {expandedGroups[`${map.id}-${groupIndex}`] ? (
                      <ChevronDown
                        size={14}
                        className="ml-auto text-gray-400"
                      />
                    ) : (
                      <ChevronRight
                        size={14}
                        className="ml-auto text-gray-400"
                      />
                    )}
                  </button>

                  {expandedGroups[`${map.id}-${groupIndex}`] &&
                    group.topics && (
                      <div className="pl-8 py-1 space-y-1">
                        {group.topics.map((topic, topicIndex) => (
                          <button
                            key={topicIndex}
                            onClick={() => handleTopicClick(topic)}
                            className={`w-full p-2 text-left text-sm flex items-center hover:bg-gray-50
                            ${selectedTopicId === topic.id ? "text-blue-600 bg-blue-50" : "text-gray-600"}`}
                          >
                            <FileText size={14} className="mr-2" />
                            <span>{topic.title || topic.id}</span>
                          </button>
                        ))}
                      </div>
                    )}
                </div>
              ))}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
