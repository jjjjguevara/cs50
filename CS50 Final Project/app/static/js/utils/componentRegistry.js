import BrownianMotion3D from "@artifacts/components/brownian";

const components = {
  brownian: BrownianMotion3D,
};

// Register components
export function registerComponents() {
  if (typeof window !== "undefined") {
    window.ReactComponents = window.ReactComponents || {};
    Object.entries(components).forEach(([name, component]) => {
      console.log(`Registering component: ${name}`);
      console.log("Loading component registry");
      window.ReactComponents[name] = component;
    });
  }
}

// Call registration
registerComponents();

export default components;
