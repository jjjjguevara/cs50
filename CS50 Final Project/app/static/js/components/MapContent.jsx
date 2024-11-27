import React, { useState, useEffect } from "react";
import { api } from "../utils/api";

export default function MapContent({ mapId }) {
  const [mapData, setMapData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (mapId) {
      setLoading(true);
      console.log("Fetching map:", mapId); // Debug log
      api
        .get(`/maps/${mapId}`)
        .then((data) => {
          console.log("Received map data:", data); // Debug log
          setMapData(data);
          setLoading(false);
        })
        .catch((err) => {
          console.error("Error loading map:", err);
          setError(err.message);
          setLoading(false);
        });
    }
  }, [mapId]);

  if (loading) {
    return (
      <div className="flex justify-center items-center h-full">
        <div className="text-gray-600">Loading map content...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-red-500 p-4">
        <h3 className="font-bold">Error Loading Map</h3>
        <p>{error}</p>
      </div>
    );
  }

  if (!mapData) {
    return (
      <div className="flex justify-center items-center h-full">
        <div className="text-gray-500">No map selected</div>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto py-8">
      <h1 className="text-3xl font-bold mb-8 text-gray-900">{mapData.title}</h1>

      {mapData.groups?.map((group, groupIndex) => (
        <div key={groupIndex} className="mb-12">
          <h2 className="text-2xl font-semibold mb-6 text-gray-800 border-b pb-2">
            {group.navtitle}
          </h2>

          {group.topics?.length > 0 ? (
            <div className="space-y-8">
              {group.topics.map((topic, topicIndex) => (
                <div key={topicIndex}>
                  {topic.content ? (
                    <div
                      className="prose max-w-none"
                      dangerouslySetInnerHTML={{ __html: topic.content }}
                    />
                  ) : (
                    <p className="text-gray-500">
                      No content available for this topic
                    </p>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <p className="text-gray-500">No topics in this group</p>
          )}
        </div>
      ))}
    </div>
  );
}
