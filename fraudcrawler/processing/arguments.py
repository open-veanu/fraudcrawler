from pydantic import BaseModel, Field
from typing import Any, Dict, List

from fraudcrawler.base.base import ProductItem


class ProcessingArgs(BaseModel):
    """Sets up the arguments for the processing pipeline step."""

    product: ProductItem = Field(description="ProductItem object to be processed.")
    user_input: Dict[str, Any] | None = Field(default=None, description="Additional (optional) user input.")
