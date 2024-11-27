import React, { useState, useEffect, createRef } from "react";
import { api } from "../utils/api";
import {
  ChevronRight,
  ChevronDown,
  BookOpen,
  FileText,
  FolderOpen,
  Folder,
} from "lucide-react";

export default function DocumentationInterface() {
  // Initialize state variables
  const [topics, setTopics] = useState({});
  const [selectedTopic, setSelectedTopic] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [isSidebarOpen, setSidebarOpen] = useState(true);
  const [expandedCategories, setExpandedCategories] = useState({
    acoustics: true,
    articles: true,
    audio: true,
  });

  // Fetch topics from our API
  useEffect(() => {
    setLoading(true);
    api
      .get("/topics")
      .then((data) => {
        console.log("Received topics:", data); // Debug log
        const grouped = data.reduce((acc, topic) => {
          if (!acc[topic.type]) {
            acc[topic.type] = [];
          }
          acc[topic.type].push(topic);
          return acc;
        }, {});
        setTopics(grouped);
        setLoading(false);
      })
      .catch((err) => {
        console.error("Error fetching topics:", err); // Debug log
        setError(err.message);
        setLoading(false);
      });
  }, []);

  // Fetch topic content
  const loadTopic = async (topicId) => {
    try {
      setLoading(true);
      const content = await api.getText(`/view/${topicId}`);
      setSelectedTopic({ id: topicId, content });
    } catch (err) {
      console.error("Error loading topic:", err); // Debug log
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

        {/* Sidebar Content */}
        <div className="overflow-y-auto flex-1">
          {Object.entries(topics).map(([category, categoryTopics]) => (
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
                      <FileText size={14} className="mr-2" />
                      {topic.title}
                    </button>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-auto p-6">
        {selectedTopic ? (
          <div
            className="prose max-w-none"
            dangerouslySetInnerHTML={{ __html: selectedTopic.content }}
          />
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
