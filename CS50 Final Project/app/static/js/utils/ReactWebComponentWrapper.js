import React from "/static/dist/react";
import ReactDOM from "/static/dist/react-dom";
import componentRegistry from "./componentRegistry";

class ReactWebComponentWrapper extends HTMLElement {
  connectedCallback() {
    console.log("Web component connected");
    const componentName = this.getAttribute("component");
    const Component = window.ReactComponents[componentName];

    if (Component) {
      const root = createRoot(this);
      root.render(React.createElement(Component, {}));
      this._root = root;
    } else {
      console.error(`Component ${componentName} not found in registry`);
    }
  }

  disconnectedCallback() {
    if (this._root) {
      this._root.unmount();
    }
  }
}

export default ReactWebComponentWrapper;
