import React, { useEffect, useState } from "react";
import { api } from "../../static/js/utils/api";

const ArticleContent = ({ topicId }) => {
  const [content, setContent] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchContent = async () => {
      try {
        setLoading(true);
        const response = await api.get(`/view/${topicId}`);
        setContent(response);
        setError(null);
      } catch (err) {
        setError("Failed to load article content");
        console.error("Error loading content:", err);
      } finally {
        setLoading(false);
      }
    };

    if (topicId) {
      fetchContent();
    }
  }, [topicId]);

  if (loading) {
    return <div className="loading-spinner">Loading content...</div>;
  }

  if (error) {
    return <div className="error-message">{error}</div>;
  }

  return (
    <article className="article-content">
      <div
        className="content-body"
        dangerouslySetInnerHTML={{ __html: content }}
      />

      <div className="academic-tools">
        <h2>Academic Tools</h2>
        <ul>
          <li>
            <a href="#" className="tool-link">
              How to cite this entry
            </a>
          </li>
          <li>
            <a href="#" className="tool-link">
              Preview PDF version
            </a>
          </li>
          <li>
            <a href="#" className="tool-link">
              Look up topics
            </a>
          </li>
        </ul>
      </div>
    </article>
  );
};

export default ArticleContent;
