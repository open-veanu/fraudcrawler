from pydantic import BaseModel, Field
from typing import List

from fraudcrawler.base.base import Prompt


class ProcessingConfig(BaseModel):
    """Sets up the processing pipeline step."""

    prompts: List[Prompt] = Field(
        description="The list of prompts to use for classification."
    )
