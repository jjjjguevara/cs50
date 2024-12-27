# app/dita/utils/latex/latex_processor.py

from typing import List, Optional, Dict, Any
from pathlib import Path
import logging

from app.dita.models.types import (
    LaTeXEquation,
    ProcessedEquation,
    ProcessingError,
    ProcessingState
)
from .latex_validator import LaTeXValidator
from .katex_renderer import KaTeXRenderer

class LaTeXProcessor:
    """Main orchestrator for LaTeX processing pipeline."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.validator = LaTeXValidator()
        self.renderer = KaTeXRenderer()

    def process_equations(self, equations: List[LaTeXEquation]) -> List[ProcessedEquation]:
        """Process LaTeX equations through validation and rendering pipeline."""
        try:
            self.logger.debug(f"Processing {len(equations)} LaTeX equations")
            processed_equations = []

            for equation in equations:
                try:
                    # Validate equation
                    if not self.validator.validate_equation(equation):
                        continue

                    # Store original content
                    processed = ProcessedEquation(
                        id=equation.id,
                        original=equation.content,
                        rendered="",  # Will be filled later
                        placeholder=f"equation_{equation.id}",  # Generate placeholder
                        is_block=equation.is_block
                    )

                    processed_equations.append(processed)

                except Exception as e:
                    self.logger.error(f"Failed to process equation {equation.id}: {str(e)}")
                    continue

            return processed_equations

        except Exception as e:
            self.logger.error(f"LaTeX processing failed: {str(e)}")
            raise ProcessingError(
                error_type="latex_processing",
                message=f"LaTeX processing failed: {str(e)}",
                context="equation_processing"
            )
