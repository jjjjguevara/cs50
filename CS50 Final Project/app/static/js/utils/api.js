import axios from "axios";

// Create axios instance with default config
const api = axios.create({
  baseURL: "/api",
  timeout: 10000,
  headers: {
    "Content-Type": "application/json",
  },
});

// Add response interceptor for error handling
api.interceptors.response.use(
  (response) => {
    const contentType = response.headers["content-type"];
    if (contentType && contentType.includes("text/html")) {
      return response.data; // Return HTML content directly
    }
    return response; // Return full response for JSON
  },
  (error) => {
    return Promise.reject(error);
  },
);

// API methods
export const getTopics = async () => {
  try {
    const response = await api.get("/topics");
    return response.data;
  } catch (error) {
    console.error("Error fetching topics:", error);
    throw error;
  }
};

export const searchTopics = async (query) => {
  try {
    const response = await api.get("/search", { params: { q: query } });
    return response.data;
  } catch (error) {
    console.error("Error searching topics:", error);
    throw error;
  }
};

export const getTopicContent = async (topicId) => {
  try {
    const response = await api.get(`/view/${topicId}`);
    return response.data;
  } catch (error) {
    console.error("Error fetching topic content:", error);
    throw error;
  }
};

export const getDitaMaps = async () => {
  try {
    const response = await api.get("/api/ditamaps");
    return response.data;
  } catch (error) {
    console.error("Error fetching ditamaps:", error);
    throw error;
  }
};

export { api };
