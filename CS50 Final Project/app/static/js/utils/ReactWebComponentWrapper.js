class ReactWebComponentWrapper extends HTMLElement {
  connectedCallback() {
    const componentName = this.getAttribute("component");
    const contextData = JSON.parse(this.getAttribute("context") || "{}");

    // Get the React component from a registry
    const Component = window.ReactComponents[componentName];
    if (Component) {
      ReactDOM.render(React.createElement(Component, contextData), this);
    }
  }

  disconnectedCallback() {
    ReactDOM.unmountComponentAtNode(this);
  }
}

customElements.define("web-component-wrapper", ReactWebComponentWrapper);
