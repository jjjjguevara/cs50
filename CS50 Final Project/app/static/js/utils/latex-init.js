// app/static/js/utils/latex-init.js
document.addEventListener("DOMContentLoaded", function () {
  console.log("LaTeX init starting");

  const renderLatex = () => {
    const equations = document.querySelectorAll("[data-latex-source]");
    console.log("Found equations:", equations.length);

    equations.forEach((elem, index) => {
      if (!elem.hasAttribute("data-processed")) {
        try {
          const latex = elem.getAttribute("data-latex-source");
          console.log(`Processing equation ${index}:`, latex);

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

          elem.setAttribute("data-processed", "true");
          console.log(`Successfully rendered equation ${index}`);
        } catch (e) {
          console.error(`Error rendering equation ${index}:`, e);
          console.error("LaTeX content:", latex);
        }
      }
    });
  };

  // Initial render with delay
  setTimeout(() => {
    console.log("Running initial LaTeX render");
    renderLatex();
  }, 500);

  // Watch for content changes
  const observer = new MutationObserver((mutations) => {
    for (const mutation of mutations) {
      if (mutation.type === "childList") {
        console.log("Content changed, re-rendering LaTeX");
        renderLatex();
        break;
      }
    }
  });

  observer.observe(document.body, {
    childList: true,
    subtree: true,
  });
});
