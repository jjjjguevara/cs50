import React, { useState, useEffect } from "react";
import MapView from "./MapView";
import { api } from "../utils/api";
import "../../css/content.css";
import {
  ChevronRight,
  ChevronDown,
  BookOpen,
  FileText,
  FolderOpen,
  Folder,
  Book,
  Files,
} from "lucide-react";

export default function DocumentationInterface() {
  // Initialize state variables
  const [topics, setTopics] = useState({});
  const [selectedTopic, setSelectedTopic] = useState(null);
  const [selectedMap, setSelectedMap] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [isSidebarOpen, setSidebarOpen] = useState(true);
  const [expandedCategories, setExpandedCategories] = useState({
    acoustics: true,
    articles: true,
    audio: true,
  });
  const [viewMode, setViewMode] = useState("topics");

  const getFileIcon = (topicPath) => {
    const extension = topicPath.split(".").pop();
    return extension === "md" ? (
      <FileText size={14} className="mr-2 text-green-600" />
    ) : (
      <FileText size={14} className="mr-2" />
    );
  };

  useEffect(() => {
    const fetchTopics = async () => {
      try {
        setLoading(true);
        const response = await api.get("/topics");

        if (response.data && response.data.topics) {
          const grouped = response.data.topics.reduce((acc, topic) => {
            const type = topic.type || "uncategorized";
            if (!acc[type]) {
              acc[type] = [];
            }
            acc[type].push(topic);
            return acc;
          }, {});

          setTopics(grouped);
        } else {
          throw new Error("Invalid response format");
        }
      } catch (err) {
        console.error("Error fetching topics:", err);
        setError(err.message || "Failed to load topics");
      } finally {
        setLoading(false);
      }
    };

    fetchTopics();
  }, []);

  useEffect(() => {
    if (viewMode === "maps" && !selectedMap) {
      setSelectedMap("audio-engineering");
    }
  }, [viewMode]);

  const loadTopic = async (topicId) => {
    try {
      setLoading(true);
      if (!topicId) {
        console.warn("No topic ID provided");
        return;
      }

      const cleanId = topicId
        .replace(/\.(dita|md)$/, "")
        .split("/")
        .pop();

      const response = await api.get(`/view/${cleanId}`);
      const content = response.data || response;

      setSelectedTopic({
        id: cleanId,
        content: typeof content === "string" ? content : content.html || "",
        isMarkdown: topicId.endsWith(".md"),
      });
    } catch (err) {
      console.error("Error loading topic:", err);
      setError(`Error loading topic: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const toggleCategory = (category) => {
    setExpandedCategories((prev) => ({
      ...prev,
      [category]: !prev[category],
    }));
  };

  const handleViewModeChange = (mode) => {
    setViewMode(mode);
    if (mode === "topics") {
      setSelectedMap(null);
      setSelectedTopic(null);
    } else {
      setSelectedTopic(null);
      setSelectedMap("audio-engineering");
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-gray-600">Loading documentation...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-red-500 p-4 flex items-center justify-center">
        <div>Error: {error}</div>
      </div>
    );
  }

  return (
    <div className="flex h-screen bg-gray-100">
      {/* Sidebar */}
      <div
        className={`${
          isSidebarOpen ? "w-64" : "w-16"
        } bg-white shadow-lg transition-all duration-300 flex flex-col`}
      >
        {/* Sidebar Header */}
        <div className="p-4 border-b flex items-center justify-between">
          {isSidebarOpen && <h2 className="font-semibold">Documentation</h2>}
          <button
            onClick={() => setSidebarOpen(!isSidebarOpen)}
            className="p-1 hover:bg-gray-100 rounded"
          >
            <ChevronRight
              className={`transform transition-transform ${
                isSidebarOpen ? "rotate-180" : ""
              }`}
            />
          </button>
        </div>

        {/* View Mode Toggle */}
        {isSidebarOpen && (
          <div className="p-2 border-b">
            <div className="flex rounded-lg overflow-hidden">
              <button
                onClick={() => handleViewModeChange("topics")}
                className={`flex-1 px-3 py-2 text-sm flex items-center justify-center gap-2 ${
                  viewMode === "topics"
                    ? "bg-blue-500 text-white"
                    : "bg-gray-100 text-gray-600"
                }`}
              >
                <Files size={16} />
                Topics
              </button>
              <button
                onClick={() => handleViewModeChange("maps")}
                className={`flex-1 px-3 py-2 text-sm flex items-center justify-center gap-2 ${
                  viewMode === "maps"
                    ? "bg-blue-500 text-white"
                    : "bg-gray-100 text-gray-600"
                }`}
              >
                <Book size={16} />
                Maps
              </button>
            </div>
          </div>
        )}

        {/* Sidebar Content */}
        <div className="overflow-y-auto flex-1">
          {viewMode === "topics" ? (
            // Topics View
            Object.entries(topics).map(([category, categoryTopics]) => (
              <div key={category} className="border-b">
                <button
                  onClick={() => toggleCategory(category)}
                  className="w-full p-3 flex items-center hover:bg-gray-50"
                >
                  {expandedCategories[category] ? (
                    <FolderOpen size={18} />
                  ) : (
                    <Folder size={18} />
                  )}
                  {isSidebarOpen && (
                    <>
                      <span className="ml-2 text-sm font-medium capitalize">
                        {category}
                      </span>
                      {expandedCategories[category] ? (
                        <ChevronDown size={16} className="ml-auto" />
                      ) : (
                        <ChevronRight size={16} className="ml-auto" />
                      )}
                    </>
                  )}
                </button>

                {expandedCategories[category] && isSidebarOpen && (
                  <div className="pl-6">
                    {categoryTopics.map((topic) => (
                      <button
                        key={topic.id}
                        onClick={() => loadTopic(topic.id)}
                        className={`w-full p-2 text-left text-sm flex items-center hover:bg-gray-50 ${
                          selectedTopic?.id === topic.id
                            ? "bg-blue-50 text-blue-600"
                            : ""
                        }`}
                      >
                        {getFileIcon(topic.path)}
                        <span className="flex-1">{topic.title}</span>
                        {topic.path?.endsWith(".md") && (
                          <span className="text-xs text-green-600 px-1.5 py-0.5 rounded-full bg-green-50">
                            MD
                          </span>
                        )}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            ))
          ) : (
            // Maps View
            <MapView mapId={selectedMap} onSelectTopic={loadTopic} />
          )}
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-auto p-6">
        {selectedTopic ? (
          <div
            className={`prose max-w-none ${
              selectedTopic.isMarkdown ? "markdown-content" : "dita-content"
            }`}
          >
            <div dangerouslySetInnerHTML={{ __html: selectedTopic.content }} />
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center h-full text-gray-500">
            <BookOpen size={48} />
            <p className="mt-4">Select a topic to view its content</p>
          </div>
        )}
      </div>
    </div>
  );
}
