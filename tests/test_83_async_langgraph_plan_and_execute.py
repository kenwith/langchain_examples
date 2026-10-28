"""Test for example 83: async langgraph plan and execute workflow graph structure.

| Example | Description |
|---------|-------------|
| 83      | Async LangGraph Plan and Execute |
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from langgraph.graph import END, START

from examples import example_83_async_langgraph_plan_and_execute as example


@pytest.mark.asyncio
async def test_workflow_graph_structure():
    """Verify the workflow graph has the expected nodes and edges."""
    with patch.object(example, "init_chat_model", return_value=MagicMock()) as mock_model:
        workflow = example.build_workflow()
        graph = workflow.get_graph()

        # Collect node IDs, excluding the special START and END nodes.
        node_ids = {node.id for node in graph.nodes if node.id not in (START, END)}
        assert node_ids == {"planner", "executor", "replanner"}

        # Check non-conditional edges.
        edges = {(edge.source, edge.target) for edge in graph.edges}
        assert ("planner", "executor") in edges
        assert ("executor", "replanner") in edges

        # Check conditional edges from replanner.
        replanner_out = [edge for edge in graph.edges if edge.source == "replanner"]
        assert any(edge.target == "executor" for edge in replanner_out)
        assert any(edge.target == END for edge in replanner_out)


if __name__ == "__main__":
    asyncio.run(test_workflow_graph_structure())
    print("Test passed.")
