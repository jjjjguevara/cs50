import React from "react";
import ReactDOM from "react-dom";

// Ensure React and ReactDOM are globally available
if (typeof window !== "undefined") {
  if (!window.React) window.React = React;
  if (!window.ReactDOM) window.ReactDOM = ReactDOM;
}

export { React, ReactDOM };
