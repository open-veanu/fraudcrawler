from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum
import logging
from pydantic import BaseModel
from typing import Any, Dict, List

import httpx
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class NodeStatus(Enum):
    """Enum for node status"""

    INIT = "init"
    READY = "ready"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"

class Node(ABC):
    """Defines an node in the processing graph"""

    def __init__(
            self,
            name: str,
            status: NodeStatus = NodeStatus.INIT,
            data_in: Dict[str, Any] | None = None,
            data_out: Dict[str, Any] | None = None,
        ):
        """Base class of a processing node.
        
        Args:
            name: Name of the node (unique identifier)
            status: Status for orchestrating node execution.
            data_in: Input data from preceding nodes.
            data_out: Output data for succeeding nodes.
        """
        self.name = name
        self.status = status
        self._data_in = data_in if data_in is not None else {}
        self._data_out = data_out if data_out is not None else {}
    
    def add_data_in(self, key: str, val: Any):
        """Add input data from other nodes."""
        if key in self._data_in:
            raise ValueError(f'key {key} allready present in data_in of node {self.name}')
        self._data_in[key] = val

    def get_data_out(self) -> Dict[str, Any]:
        """Returns the output data of the Node"""
        if not self.status == NodeStatus.SUCCESS:
            raise ValueError(f'Node {self.name} has status {self.status} - not ready to provide data_out')
        return self._data_out

    @abstractmethod
    async def _run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        pass

    async def run(self):
        if not self.status == NodeStatus.READY:
            raise ValueError(f'Node {self.name} is supposed to run but has status="{self.status}"')

        logger.debug(f'Start running node {self.name}')
        self.status = NodeStatus.RUNNING

        try:
            data_out = await self._run(data=self._data_in)
        except Exception as e:
            msg = f'Error occured while running node {self.name}: {e}'
            logger.error(msg=msg)
            raise type(e)(msg) from e

        self._data_out = data_out
        logger.info(f'Run of node {self.name} finished successfully.')
        self.status = NodeStatus.SUCCESS
        

class EdgeCategory(Enum):
    """Enum for edge categories"""
    
    MAIN = "main"
    TOOL = "tool"


class Edge(BaseModel):
    """Defines an edge in the processing graph"""
    source: str
    target: str
    relation: EdgeCategory


class Enpoint(Enum):
    INPUT = "input"
    OUTPUT = "ouput"


class ProcessingGraph(BaseModel):
    nodes: List[Node]
    edges: List[Edge]

    def _get_node_by_name(self, name: str) -> Node:
        """Helper to look up nodes by name.

        Args:
            name: The name of the node to find.
        """
        for node in self.nodes:
            if node.name == name:
                return node
        raise ValueError(f'No node found with name: {name}')

    def _get_starting_nodes(self) -> List[str]:
        """Find nodes with source='input' edges (no dependencies)."""
        starting_nodes = []
        for edge in self.edges:
            if edge.source == Enpoint.INPUT.value:
                starting_nodes.append(edge.target)
        return starting_nodes

    def _build_dependency_graph(self) -> Dict[str, List[str]]:
        """Build adjacency list mapping each node to its dependents."""
        dependencies = defaultdict(list)
        for edge in self.edges:
            if edge.source == Enpoint.INPUT.value or edge.target == Enpoint.OUTPUT.value:
                continue
            dependencies[edge.source].append(edge.target)
        return dependencies

    async def run(self):
        """Execute all nodes in topological order respecting dependencies.

        Uses Kahn's algorithm for topological sort to ensure nodes are
        executed only after all their dependencies have completed.
        """
        # Build dependency structures
        dependencies = self._build_dependency_graph()
        starting_nodes = self._get_starting_nodes()

        # TODO: continue from here

        # logger.info(f'Starting graph execution with {len(self.nodes)} nodes')
        # logger.debug(f'Starting nodes: {starting_nodes}')

        # # Initialize queue with starting nodes (nodes with no dependencies)
        # queue = []
        # for node_name in starting_nodes:
        #     node = self._get_node_by_name(node_name)
        #     node.status = NodeStatus.READY
        #     queue.append(node_name)

        # executed_count = 0

        # # Process nodes in topological order
        # while queue:
        #     # Dequeue next node to execute
        #     current_node_name = queue.pop(0)
        #     current_node = self._get_node_by_name(current_node_name)

        #     # Execute the node
        #     logger.info(f'Executing node: {current_node_name}')
        #     await current_node.run()
        #     executed_count += 1

        #     # Get output data from completed node
        #     output_data = current_node.get_data_out()

        #     # Transfer output to dependent nodes
        #     for dependent_node_name in dependencies[current_node_name]:
        #         dependent_node = self._get_node_by_name(dependent_node_name)

        #         # Add output data to dependent node's input
        #         for key, value in output_data.items():
        #             dependent_node.add_data_in(key, value)

        #         # Decrease in-degree (one less dependency to wait for)
        #         count[dependent_node_name] -= 1

        #         # If all dependencies satisfied, mark as ready and enqueue
        #         if count[dependent_node_name] == 0:
        #             dependent_node.status = NodeStatus.READY
        #             queue.append(dependent_node_name)
        #             logger.debug(f'Node {dependent_node_name} is now ready to execute')

        # # Check if all nodes were executed (cycle detection)
        # if executed_count != len(self.nodes):
        #     unexecuted = [node.name for node in self.nodes if node.status != NodeStatus.SUCCESS]
        #     raise ValueError(
        #         f'Graph execution incomplete. Executed {executed_count}/{len(self.nodes)} nodes. '
        #         f'Unexecuted nodes: {unexecuted}. This may indicate a cycle or missing "input" edges.'
        #     )

        # logger.info(f'Graph execution completed successfully. Executed {executed_count} nodes.')


class OpenAIChat(Node):
    """Open AI Chat model."""

    def __init__(
            self,
            http_client: httpx.AsyncClient,
            name: str,
            api_key: str,
            model: str,
            status: NodeStatus = NodeStatus.INIT,
            data_in: Dict[str, Any] | None = None,
            data_out: Dict[str, Any] | None = None,

        ):
        """Open AI Chat node.

        Args:
            name: Name of the node (unique identifier)
            status: Status for orchestrating node execution.
            http_client: An httpx.AsyncClient to use for the async requests.
            api_key: The OpenAI API key.
            model: The OpenAI model to use.
            data_in: Input data from preceding nodes.
            data_out: Output data for succeeding nodes.
        """
        super().__init__(
            name=name,
            status=status,
            data_in=data_in,
            data_out=data_out,
        )
        self._client = AsyncOpenAI(http_client=http_client, api_key=api_key)
        self._model = model
