import BrownianMotion3D from "../../dita/artifacts/components/brownian.jsx";

const components = {
  brownian: BrownianMotion3D,
};

// Register components
export function registerComponents() {
  if (typeof window !== "undefined") {
    window.ReactComponents = window.ReactComponents || {};
    Object.entries(components).forEach(([name, component]) => {
      console.log(`Registering component: ${name}`);
      window.ReactComponents[name] = component;
    });
  }
}

// Call registration
registerComponents();

export default components;
