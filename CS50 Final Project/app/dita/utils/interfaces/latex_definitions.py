from abc import ABC, abstractmethod
from typing import List, Dict, Optional

class LaTeXProcessor(ABC):
    @abstractmethod
    def process_equation(self, latex: str, block: bool = False) -> str:
        pass

    @abstractmethod
    def validate_equation(self, latex: str) -> bool:
        pass

    @abstractmethod
    def extract_equations(self, content: str) -> List[Dict[str, str]]:
        pass
