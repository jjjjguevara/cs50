import React, { useEffect, useState } from "react";

const Bibliography = () => {
  const [references, setReferences] = useState([]);

  useEffect(() => {
    const bibSection = document.querySelector(".bibliography-section");
    if (!bibSection) return;

    try {
      // Get references data
      const refsData = bibSection.getAttribute("data-references");
      if (!refsData) return;

      // Parse references
      const refs = JSON.parse(refsData);
      if (Array.isArray(refs) && refs.length > 0) {
        // Sort by ID numerically
        const sortedRefs = refs.sort((a, b) => parseInt(a.id) - parseInt(b.id));
        setReferences(sortedRefs);
      }
    } catch (e) {
      console.error("Error loading references:", e);
    }
  }, []);

  // Only render if we have references
  if (!references.length) return null;

  return (
    <div className="bibliography-container mt-8 pt-8 border-t-2 border-gray-200">
      <h2 className="text-2xl font-bold mb-4">References</h2>
      <ol className="bibliography-list space-y-4">
        {references.map((ref) => (
          <li
            key={ref.id}
            id={`ref-${ref.id}`}
            className="bibliography-item flex gap-4"
          >
            <div className="reference-number text-gray-500 flex-shrink-0">
              <a href={`#cite-${ref.id}`} className="hover:text-blue-600">
                [{ref.id}]
              </a>
            </div>
            <div
              className="reference-text"
              // Remove markdown formatting
              dangerouslySetInnerHTML={{
                __html: ref.text.replace(/\*(.*?)\*/g, "$1"),
              }}
            />
          </li>
        ))}
      </ol>
    </div>
  );
};

export default Bibliography;
