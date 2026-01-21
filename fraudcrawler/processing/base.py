from abc import ABC, abstractmethod
import logging
from pydantic import BaseModel
from typing import Any, Dict, List, Sequence, TypeAlias

from tenacity import RetryCallState

from fraudcrawler.base.base import ProductItem

logger = logging.getLogger(__name__)


Context: TypeAlias = Dict[str, str]
UserInputs: TypeAlias = Dict[str, List[str]]


class ClassificationResult(BaseModel):
    """Model for classification results."""

    result: int
    input_tokens: int = 0
    output_tokens: int = 0


class TmpResult(BaseModel):
    """Model for tmp results."""

    result: Any
    input_tokens: int = 0
    output_tokens: int = 0


WorkflowResult: TypeAlias = ClassificationResult | TmpResult | None


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

    def _log_before(self, context: Context, retry_state: RetryCallState) -> None:
        """Context aware logging before the request is made."""
        if retry_state:
            logger.debug(
                f"Workflow={self.name} retry-call within context={context} (Attempt {retry_state.attempt_number})."
            )
        else:
            logger.debug(f"retry_state is {retry_state}; not logging before.")

    def _log_before_sleep(self, context: Context, retry_state: RetryCallState) -> None:
        """Context aware logging before sleeping after a failed request."""
        if retry_state and retry_state.outcome:
            logger.warning(
                f"Attempt {retry_state.attempt_number} of workflow={self.name} "
                f"retry-call within context={context} "
                f"failed with error: {retry_state.outcome.exception()}. "
                f"Retrying in {retry_state.upcoming_sleep:.0f} seconds."
            )

    @abstractmethod
    async def run(self, product: ProductItem) -> WorkflowResult:
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

    async def run(self, product: ProductItem) -> ProductItem:
        """Run the processing step for multiple workflows and return all results together with workflow.name.

        Args:
            product: The product item to process.
        """
        for wf in self._workflows:
            try:
                logger.info(
                    f'Running workflow="{wf.name}" for product with url="{product.url_resolved}".'
                )
                res = await wf.run(product=product)
            except Exception:
                logger.error(
                    f'Error while running workflow="{wf.name}" for product with url="{product.url_resolved}"',
                    exc_info=True,
                )
                continue

            # Update the product item
            inp_tok = out_tok = 0
            if isinstance(res, ClassificationResult):
                logger.debug(
                    f'result from workflow="{wf.name}" added to product.classifications'
                )
                product.classifications[wf.name] = int(res.result)
                inp_tok = res.input_tokens
                out_tok = res.output_tokens

            elif isinstance(res, TmpResult):
                logger.debug(f'result from workflow="{wf.name}" added to product.tmp')
                product.tmp[wf.name] = res
                inp_tok = res.input_tokens
                out_tok = res.output_tokens

            elif res is None:
                logger.debug(
                    f'result from workflow="{wf.name}" is `None` and therefore not stored'
                )

            else:
                logger.warning(
                    f'result from workflow="{wf.name}" return type={type(res)} is not allowed; '
                    f"must either be of type `ClassificationResult`, "
                    f"`TmpResult`, or `None`; not type={type(res)}"
                )

            if inp_tok > 0 or out_tok > 0:
                logger.debug(
                    f'result from workflow="{wf.name}" used input_tokens={inp_tok}, output_tokens={out_tok}'
                )
                product.usage[wf.name] = {
                    "input_tokens": inp_tok,
                    "output_tokens": out_tok,
                }

        return product
