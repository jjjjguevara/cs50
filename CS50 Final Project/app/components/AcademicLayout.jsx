import React from "react";
import { SearchProvider } from "../static/js/search";

const AcademicLayout = ({ children }) => {
  return (
    <SearchProvider>
      <div className="academic-layout">{children}</div>
    </SearchProvider>
  );
};

export default AcademicLayout;
