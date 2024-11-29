import React from "react";
import { Calendar, Users, Tag } from "lucide-react";

const ArticleHeader = ({ metadata }) => {
  const { title, authors, publicationDate, lastUpdated, categories, doi } =
    metadata || {};

  return (
    <header className="article-header">
      <h1 className="article-title">{title}</h1>

      <div className="article-meta">
        {/* Publication Info */}
        <div className="meta-section">
          <div className="meta-item">
            <Calendar size={16} className="meta-icon" />
            <span>First published: {publicationDate}</span>
            {lastUpdated && (
              <span className="text-gray-500">
                (Last updated: {lastUpdated})
              </span>
            )}
          </div>

          {authors && (
            <div className="meta-item">
              <Users size={16} className="meta-icon" />
              <span>{authors.join(", ")}</span>
            </div>
          )}
        </div>

        {/* Categories */}
        {categories && categories.length > 0 && (
          <div className="meta-section">
            <Tag size={16} className="meta-icon" />
            <div className="category-tags">
              {categories.map((category, index) => (
                <span key={index} className="category-tag">
                  {category}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* DOI */}
        {doi && (
          <div className="meta-section">
            <span className="text-sm text-gray-500">DOI: {doi}</span>
          </div>
        )}
      </div>
    </header>
  );
};

export default ArticleHeader;
