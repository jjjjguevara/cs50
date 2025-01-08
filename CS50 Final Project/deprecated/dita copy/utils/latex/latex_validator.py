# app/dita/utils/latex/latex_validator.py

import re
import logging
from typing import List, Set

from app.dita.models.types import LaTeXEquation

class LaTeXValidator:
    """Validates LaTeX equations for safety and correctness."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._init_validation_rules()

    def _init_validation_rules(self):
        """Initialize validation rules and patterns."""
        # Disallowed commands for security
        self.disallowed_commands = {
            r'\input', r'\include', r'\write', r'\read',
            r'\openin', r'\openout', r'\catcode', r'\let'
        }

        # Allowed math environments
        self.allowed_environments = {
            'equation', 'align', 'matrix', 'pmatrix',
            'bmatrix', 'cases', 'gather'
        }

    def validate_equation(self, equation: LaTeXEquation) -> bool:
        """
        Validate a single LaTeX equation.

        Args:
            equation: LaTeX equation to validate

        Returns:
            True if equation is valid
        """
        try:
            content = equation.content

            # Check for disallowed commands
            if any(cmd in content for cmd in self.disallowed_commands):
                self.logger.warning(f"Equation {equation.id} contains disallowed commands")
                return False

            # Validate environment syntax
            if not self._validate_environments(content):
                self.logger.warning(f"Equation {equation.id} has invalid environments")
                return False

            # Validate basic syntax
            if not self._validate_syntax(content):
                self.logger.warning(f"Equation {equation.id} has invalid syntax")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Validation error for equation {equation.id}: {str(e)}")
            return False

    def _validate_environments(self, content: str) -> bool:
        """Validate LaTeX environments."""
        try:
            # Find all environments
            begin_matches = re.findall(r'\\begin\{(\w+)\}', content)
            end_matches = re.findall(r'\\end\{(\w+)\}', content)

            # Check environment balance
            if len(begin_matches) != len(end_matches):
                return False

            # Check allowed environments
            return all(env in self.allowed_environments for env in begin_matches)

        except Exception as e:
            self.logger.error(f"Environment validation error: {str(e)}")
            return False

    def _validate_syntax(self, content: str) -> bool:
        """Validate basic LaTeX syntax."""
        try:
            # Check bracket balance
            if not self._check_bracket_balance(content):
                return False

            # Check basic structure
            if not self._check_basic_structure(content):
                return False

            return True

        except Exception as e:
            self.logger.error(f"Syntax validation error: {str(e)}")
            return False

    def _check_bracket_balance(self, content: str) -> bool:
        """Check if brackets are properly balanced."""
        stack = []
        brackets = {'(': ')', '[': ']', '{': '}'}

        for char in content:
            if char in brackets.keys():
                stack.append(char)
            elif char in brackets.values():
                if not stack:
                    return False
                if char != brackets[stack.pop()]:
                    return False

        return len(stack) == 0

    def _check_basic_structure(self, content: str) -> bool:
        """Check basic LaTeX structure."""
        # Must not be empty
        if not content.strip():
            return False

        # Basic command syntax
        if content.count('\\') != content.count(' \\'):
            if not re.match(r'^[^\\]*(?:\\[a-zA-Z]+[^\\]*)*$', content):
                return False

        return True
