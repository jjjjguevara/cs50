const ReferenceDebug = () => {
  const [debug, setDebug] = useState({
    bibliographySection: null,
    references: null,
    error: null,
  });

  useEffect(() => {
    const bibSection = document.querySelector(".bibliography-section");
    setDebug((prev) => ({
      ...prev,
      bibliographySection: bibSection
        ? {
            exists: true,
            hasDataset: !!bibSection.dataset,
            hasReferences: !!bibSection.dataset.references,
            rawReferences: bibSection.dataset.references,
          }
        : null,
    }));

    if (bibSection?.dataset.references) {
      try {
        const refs = JSON.parse(bibSection.dataset.references);
        setDebug((prev) => ({
          ...prev,
          references: refs,
        }));
      } catch (e) {
        setDebug((prev) => ({
          ...prev,
          error: e.message,
        }));
      }
    }
  }, []);

  return (
    <div
      className="reference-debug"
      style={{
        padding: "1rem",
        margin: "1rem",
        border: "1px solid #ccc",
        borderRadius: "4px",
      }}
    >
      <h3>Reference Debug Info</h3>
      <pre style={{ whiteSpace: "pre-wrap" }}>
        {JSON.stringify(debug, null, 2)}
      </pre>
    </div>
  );
};

export default ReferenceDebug;
