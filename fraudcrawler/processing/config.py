from pydantic import BaseModel, Field
from typing import List

from fraudcrawler import Prompt


class ProcessingConfig(BaseModel):
    """Sets up the processing pipeline step."""

    prompts: List[Prompt] = Field(
        description="The list of prompt to use for classification."
    )
