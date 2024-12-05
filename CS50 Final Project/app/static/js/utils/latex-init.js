// app/static/js/utils/latex-init.js
(function () {
  const processedEquations = new Set();

  const renderLatex = () => {
    const equations = document.querySelectorAll("[data-latex-source]");
    console.log("Found equations:", equations.length);

    equations.forEach((elem, index) => {
      // Create unique identifier for each equation
      const latex = elem.getAttribute("data-latex-source");
      const uniqueId = `${index}-${latex}`;

      if (!processedEquations.has(uniqueId)) {
        try {
          console.log(`Processing equation ${index}:`, {
            latex: latex,
            processed: processedEquations.size,
            uniqueId: uniqueId,
          });

          katex.render(latex, elem, {
            throwOnError: false,
            displayMode: elem.classList.contains("math-block"),
            output: "html",
            macros: {
              "\\RR": "\\mathbb{R}",
              "\\N": "\\mathbb{N}",
            },
            trust: true,
          });

          // Mark as processed
          processedEquations.add(uniqueId);
          elem.setAttribute("data-processed", "true");
          elem.setAttribute("data-equation-id", uniqueId);

          console.log(`Successfully rendered equation ${index}`);
        } catch (e) {
          console.error(`Error rendering equation ${index}:`, e);
        }
      } else {
        console.log(`Equation ${index} already processed`);
      }
    });
  };

  // Wait for both DOM and KaTeX
  const initialize = () => {
    if (typeof katex === "undefined") {
      console.log("Waiting for KaTeX...");
      setTimeout(initialize, 100);
      return;
    }

    console.log("KaTeX loaded, starting render");
    renderLatex();

    // Watch for content changes
    const observer = new MutationObserver((mutations) => {
      let needsUpdate = false;
      for (const mutation of mutations) {
        if (
          mutation.type === "childList" &&
          mutation.target.querySelector(
            "[data-latex-source]:not([data-processed])",
          )
        ) {
          needsUpdate = true;
          break;
        }
      }
      if (needsUpdate) {
        console.log("New equations found, re-rendering");
        renderLatex();
      }
    });

    observer.observe(document.body, {
      childList: true,
      subtree: true,
    });
  };

  // Start initialization when DOM is ready
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initialize);
  } else {
    initialize();
  }
})();
