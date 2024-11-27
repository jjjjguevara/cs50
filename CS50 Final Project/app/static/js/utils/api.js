import { config } from "../config";

export const api = {
  async fetch(endpoint, options = {}) {
    const url = `${config.apiBaseUrl}${endpoint}`;
    try {
      const response = await fetch(url, {
        ...options,
        headers: {
          "Content-Type": "application/json",
          ...options.headers,
        },
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return response;
    } catch (error) {
      console.error(`API Error (${endpoint}):`, error);
      throw error;
    }
  },

  async get(endpoint) {
    return this.fetch(endpoint).then((res) => res.json());
  },

  async getText(endpoint) {
    return this.fetch(endpoint).then((res) => res.text());
  },
};
