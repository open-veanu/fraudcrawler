from abc import ABC, abstractmethod
import logging
from pydantic import BaseModel
from typing import Any, Dict, List, Sequence, TypeAlias

from fraudcrawler.base.base import ProductItem

logger = logging.getLogger(__name__)


UserInputs: TypeAlias = Dict[str, List[str]]


class ClassificationResult(BaseModel):
    """Model for classification results."""

    result: int


class Workflow(ABC):
    """Abstract base class for independent processing workflows."""

    def __init__(
        self,
        name: str,
    ):
        """Abstract base class for defining a classification workflow.

        Args:
            name: Name of the classification workflow.
        """
        self.name = name

    @abstractmethod
    async def run(self, product: ProductItem) -> Any:
        """Runs the workflow."""
        pass


class Processor:
    """Processing product items for a set of classification workflows."""

    def __init__(self, workflows: Sequence[Workflow]):
        """Initializes the Processor.

        Args:
            workflows: Sequence of workflows for classification of product items.
        """
        if not self._are_unique(workflows=workflows):
            raise ValueError(
                f"Workflow names are not unique: {[wf.name for wf in workflows]}"
            )
        self._workflows = workflows

    @staticmethod
    def _are_unique(workflows: Sequence[Workflow]) -> bool:
        """Tests if the workflows have unique names."""
        return len(workflows) == len(set([wf.name for wf in workflows]))

    async def run(self, product: ProductItem) -> Dict[str, Any]:
        """Run the processing step for multiple workflows and return all results together with workflow.name.

        Args:
            product: The product item to process.
        """
        results = {}
        for wf in self._workflows:
            try:
                res = await wf.run(product=product)
            except Exception as e:
                raise type(e)(f'Error while running workflow="{wf.name}": {e}') from e

            results[wf.name] = res
        return results
