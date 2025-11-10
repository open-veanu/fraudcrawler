from pydantic import BaseModel, Field
from typing import Any, Dict, List, TypeAlias

from fraudcrawler.base.base import ProductItem


ProcessingArgs: TypeAlias = Dict[str, str | List[str]]
