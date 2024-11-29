import React, { useEffect, useState } from "react";
import { api } from "@utils/api";

export default function ArticleContent({ topicId }) {
  const [content, setContent] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchContent = async () => {
      if (!topicId) return;

      try {
        setLoading(true);
        const response = await api.get(`/view/${topicId}`);
        setContent(response.data);
        setError(null);
      } catch (err) {
        setError("Failed to load article content");
        console.error("Error loading content:", err);
      } finally {
        setLoading(false);
      }
    };

    fetchContent();
  }, [topicId]);

  if (loading) {
    return <div className="loading">Loading content...</div>;
  }

  if (error) {
    return <div className="error">{error}</div>;
  }

  if (!content) {
    return <div className="no-content">No content available</div>;
  }

  return (
    <div className="content-wrapper">
      <article className="article-content">
        <div
          className="content-body"
          dangerouslySetInnerHTML={{ __html: content }}
        />
      </article>
    </div>
  );
}
