import React from "react";
import { SearchProvider } from "@/search";

const AcademicLayout = ({ children }) => {
  return (
    <SearchProvider>
      <div className="academic-layout">{children}</div>
    </SearchProvider>
  );
};

export default AcademicLayout;
