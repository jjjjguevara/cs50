import React, { useState, useEffect, useCallback } from "react";
// Update the import path to use the alias you defined in vite.config.js
import { api } from "@utils/api";

/**
 * @typedef {Object} SearchContextType
 * @property {string} searchQuery - Current search query
 * @property {Function} setSearchQuery - Function to update search query
 * @property {Array} searchResults - Array of search results
 * @property {boolean} isLoading - Loading state
 * @property {string|null} error - Error message if any
 */

/** @type {React.Context<SearchContextType>} */
const SearchContext = React.createContext(null);

/**
 * @typedef {Object} SearchProviderProps
 * @property {React.ReactNode} children - Child components
 */

/**
 * Provider component for search functionality
 * @param {SearchProviderProps} props
 */
const SearchProvider = ({ children }) => {
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const performSearch = useCallback(async (query) => {
    if (!query.trim()) {
      setSearchResults([]);
      return;
    }

    setIsLoading(true);
    try {
      const response = await api.get(`/search?q=${encodeURIComponent(query)}`);
      setSearchResults(response.data);
      setError(null);
    } catch (err) {
      console.error("Search error:", err);
      setError("Failed to perform search");
      setSearchResults([]);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const debounce = (func, wait) => {
    let timeout;
    return (...args) => {
      clearTimeout(timeout);
      timeout = setTimeout(() => func.apply(this, args), wait);
    };
  };

  const debouncedSearch = useCallback(
    debounce((query) => performSearch(query), 300),
    [performSearch],
  );

  useEffect(() => {
    debouncedSearch(searchQuery);
  }, [searchQuery, debouncedSearch]);

  const contextValue = {
    searchQuery,
    setSearchQuery,
    searchResults,
    isLoading,
    error,
  };

  return (
    <SearchContext.Provider value={contextValue}>
      {children}
    </SearchContext.Provider>
  );
};

/**
 * Hook to use search context
 * @returns {SearchContextType}
 * @throws {Error} When used outside of SearchProvider
 */
export const useSearch = () => {
  const context = React.useContext(SearchContext);
  if (!context) {
    throw new Error("useSearch must be used within a SearchProvider");
  }
  return context;
};

export { SearchContext, SearchProvider };
