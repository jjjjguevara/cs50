import React from "react";
import { Search, Loader } from "lucide-react";
import { useSearch } from "@/search";
import { Pause, Play, RefreshCcw } from "@utils/icons";

const SearchBar = () => {
  const { searchQuery, setSearchQuery, searchResults, isLoading, error } =
    useSearch();

  return (
    <div className="search-container">
      <div className="search-input-wrapper">
        <input
          type="text"
          className="search-input"
          placeholder="Search entries..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
        />
        {isLoading ? (
          <Loader className="search-icon animate-spin" />
        ) : (
          <Search className="search-icon" />
        )}
      </div>

      {/* Search Results Dropdown */}
      {searchQuery && (
        <div className="search-results">
          {error ? (
            <div className="search-error">{error}</div>
          ) : searchResults.length > 0 ? (
            <ul className="results-list">
              {searchResults.map((result) => (
                <li key={result.id}>
                  <a href={`/entry/${result.id}`} className="result-item">
                    <h4 className="result-title">{result.title}</h4>
                    <p className="result-preview">{result.preview}</p>
                  </a>
                </li>
              ))}
            </ul>
          ) : (
            <div className="no-results">No matches found</div>
          )}
        </div>
      )}
    </div>
  );
};

export default SearchBar;
