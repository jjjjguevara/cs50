// app/static/js/utils/paths.js
export const paths = {
  components: "../components",
  utils: "../utils",
  src: "../src",
  artifacts: "../../dita/artifacts",
};

export const getComponentPath = (name) => `${paths.components}/${name}`;
export const getUtilPath = (name) => `${paths.utils}/${name}`;
