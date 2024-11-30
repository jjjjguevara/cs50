import React, { useEffect, useState } from "react";

const Bibliography = () => {
  const [references, setReferences] = useState([]);
  const [debug, setDebug] = useState({});

  useEffect(() => {
    const bibSection = document.querySelector(".bibliography-section");
    console.log("Looking for bibliography section:", bibSection);

    if (bibSection?.dataset.references) {
      console.log("Raw references data:", bibSection.dataset.references);
      try {
        // Decode HTML entities before parsing JSON
        const decodedData = bibSection.dataset.references
          .replace(/&quot;/g, '"')
          .replace(/&apos;/g, "'")
          .replace(/&lt;/g, "<")
          .replace(/&gt;/g, ">");

        console.log("Decoded references data:", decodedData);

        const refs = JSON.parse(decodedData);

        if (Array.isArray(refs) && refs.length > 0) {
          const sortedRefs = refs.sort(
            (a, b) => parseInt(a.id) - parseInt(b.id),
          );
          setReferences(sortedRefs);
          setDebug({ success: true, count: refs.length });
        } else {
          setDebug({ error: "No references found in data" });
        }
      } catch (e) {
        console.error("Error parsing references:", e);
        setDebug({
          error: e.message,
          raw: bibSection.dataset.references,
          type: typeof bibSection.dataset.references,
        });
      }
    } else {
      setDebug({ error: "No bibliography section or references found" });
    }
  }, []);

  // Debug output
  if (debug.error) {
    return (
      <div className="bibliography-debug text-sm text-gray-600 mt-4 p-4 bg-gray-100 rounded">
        <h3 className="font-bold">Debug Info</h3>
        <pre className="whitespace-pre-wrap overflow-auto">
          {JSON.stringify(debug, null, 2)}
        </pre>
      </div>
    );
  }

  if (!references.length) return null;

  return (
    <div className="bibliography-container mt-8 pt-8 border-t-2 border-gray-200">
      <h2 className="text-2xl font-bold mb-4">References</h2>
      <div className="bibliography-list space-y-4">
        {references.map((ref) => (
          <div
            key={ref.id}
            id={`ref-${ref.id}`}
            className="bibliography-item flex"
          >
            <div className="reference-number mr-4 text-gray-500 flex-shrink-0">
              <a href={`#cite-${ref.id}`} className="hover:text-blue-600">
                [{ref.id}]
              </a>
            </div>
            <div className="reference-text flex-1">{ref.text}</div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Bibliography;
