import React from "react";
import { SearchProvider } from "@/search";
import { Pause, Play, RefreshCcw } from "@utils/icons";

const AcademicLayout = ({ children }) => {
  return (
    <SearchProvider>
      <div className="academic-layout">{children}</div>
    </SearchProvider>
  );
};

export default AcademicLayout;
