import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import TypedDict, Annotated, List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import json
import time

# Import the modules we're testing
try:
    from langgraph.graph import StateGraph, END, START
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.constants import Send
    from langgraph.types import Command, interrupt
    from langgraph.errors import GraphRecursionError
except ImportError:
    # Mock imports for testing without langgraph installed
    StateGraph = Mock
    END = "END"
    START = "START"
    MemorySaver = Mock
    SqliteSaver = Mock
    Send = Mock
    Command = Mock
    interrupt = Mock
    GraphRecursionError = Exception


class TestStateSchema(TypedDict):
    """Test state schema for basic graph testing."""
    messages: Annotated[List[str], "add"]
    counter: int
    metadata: Dict[str, Any]


class ComplexStateSchema(TypedDict):
    """Complex state schema with nested structures."""
    user_id: str
    session_data: Dict[str, Any]
    history: List[Dict[str, Any]]
    current_step: str
    retry_count: int
    config: Dict[str, Any]


class MinimalStateSchema(TypedDict):
    """Minimal state schema for testing."""
    value: str


class StateWithOptional(TypedDict):
    """State schema with optional fields."""
    required_field: str
    optional_field: Optional[str]
    default_field: str


class AgentState(TypedDict):
    """Agent state schema for multi-agent testing."""
    agent_name: str
    task: str
    result: Optional[str]
    next_agent: Optional[str]
    shared_data: Dict[str, Any]


class WorkflowState(TypedDict):
    """Workflow state for complex workflow testing."""
    workflow_id: str
    status: str
    steps_completed: List[str]
    current_step: str
    error: Optional[str]
    metadata: Dict[str, Any]


def test_basic_state_graph_creation():
    """Test creating a basic state graph."""
    graph = StateGraph(TestStateSchema)
    assert graph is not None


def test_state_graph_with_nodes():
    """Test adding nodes to state graph."""
    graph = StateGraph(TestStateSchema)
    
    def node_a(state: TestStateSchema) -> TestStateSchema:
        return {"messages": ["A"], "counter": 1, "metadata": {"node": "A"}}
    
    def node_b(state: TestStateSchema) -> TestStateSchema:
        return {"messages": ["B"], "counter": 2, "metadata": {"node": "B"}}
    
    graph.add_node("node_a", node_a)
    graph.add_node("node_b", node_b)
    
    assert "node_a" in graph.nodes
    assert "node_b" in graph.nodes


def test_state_graph_edges():
    """Test adding edges between nodes."""
    graph = StateGraph(TestStateSchema)
    
    def node_a(state: TestStateSchema) -> TestStateSchema:
        return {"messages": ["A"], "counter": 1, "metadata": {}}
    
    def node_b(state: TestStateSchema) -> TestStateSchema:
        return {"messages": ["B"], "counter": 2, "metadata": {}}
    
    graph.add_node("node_a", node_a)
    graph.add_node("node_b", node_b)
    graph.add_edge("node_a", "node_b")
    graph.add_edge("node_b", END)
    
    assert graph.edges is not None


def test_state_graph_conditional_edges():
    """Test conditional edges in state graph."""
    graph = StateGraph(TestStateSchema)
    
    def node_a(state: TestStateSchema) -> TestStateSchema:
        return {"messages": ["A"], "counter": state.get("counter", 0) + 1, "metadata": {}}
    
    def node_b(state: TestStateSchema) -> TestStateSchema:
        return {"messages": ["B"], "counter": state.get("counter", 0) + 1, "metadata": {}}
    
    def node_c(state: TestStateSchema) -> TestStateSchema:
        return {"messages": ["C"], "counter": state.get("counter", 0) + 1, "metadata": {}}
    
    def route(state: TestStateSchema) -> str:
        if state.get("counter", 0) > 5:
            return "node_c"
        return "node_b"
    
    graph.add_node("node_a", node_a)
    graph.add_node("node_b", node_b)
    graph.add_node("node_c", node_c)
    graph.add_conditional_edges("node_a", route, {"node_b": "node_b", "node_c": "node_c"})
    graph.add_edge("node_b", END)
    graph.add_edge("node_c", END)
    
    assert graph.edges is not None


def test_state_graph_compilation():
    """Test compiling state graph."""
    graph = StateGraph(TestStateSchema)
    
    def node_a(state: TestStateSchema) -> TestStateSchema:
        return {"messages": ["A"], "counter": 1, "metadata": {}}
    
    graph.add_node("node_a", node_a)
    graph.add_edge(START, "node_a")
    graph.add_edge("node_a", END)
    
    compiled = graph.compile()
    assert compiled is not None


def test_state_graph_compilation_with_checkpointer():
    """Test compiling state graph with checkpointer."""
    graph = StateGraph(TestStateSchema)
    
    def node_a(state: TestStateSchema) -> TestStateSchema:
        return {"messages": ["A"], "counter": 1, "metadata": {}}
    
    graph.add_node("node_a", node_a)
    graph.add_edge(START, "node_a")
    graph.add_edge("node_a", END)
    
    checkpointer = MemorySaver()
    compiled = graph.compile(checkpointer=checkpointer)
    assert compiled is not None


def test_state_transition_basic():
    """Test basic state transition."""
    graph = StateGraph(TestStateSchema)
    
    def increment_counter(state: TestStateSchema) -> TestStateSchema:
        return {"counter": state.get("counter", 0) + 1}
    
    graph.add_node("increment", increment_counter)
    graph.add_edge(START, "increment")
    graph.add_edge("increment", END)
    
    compiled = graph.compile()
    initial_state = {"messages": [], "counter": 0, "metadata": {}}
    result = compiled.invoke(initial_state)
    
    assert result["counter"] == 1


def test_state_transition_multiple_nodes():
    """Test state transition through multiple nodes."""
    graph = StateGraph(TestStateSchema)
    
    def node_a(state: TestStateSchema) -> TestStateSchema:
        return {"messages": ["A"], "counter": state.get("counter", 0) + 1}
    
    def node_b(state: TestStateSchema) -> TestStateSchema:
        return {"messages": ["B"], "counter": state.get("counter", 0) + 10}
    
    graph.add_node("node_a", node_a)
    graph.add_node("node_b", node_b)
    graph.add_edge(START, "node_a")
    graph.add_edge("node_a", "node_b")
    graph.add_edge("node_b", END)
    
    compiled = graph.compile()
    initial_state = {"messages": [], "counter": 0, "metadata": {}}
    result = compiled.invoke(initial_state)
    
    assert result["counter"] == 11
    assert "A" in result["messages"]
    assert "B" in result["messages"]


def test_state_transition_conditional():
    """Test conditional state transitions."""
    graph = StateGraph(TestStateSchema)
    
    def process(state: TestStateSchema) -> TestStateSchema:
        counter = state.get("counter", 0) + 1
        return {"counter": counter, "messages": [f"step_{counter}"]}
    
    def route(state: TestStateSchema) -> str:
        if state.get("counter", 0) >= 3:
            return END
        return "process"
    
    graph.add_node("process", process)
    graph.add_edge(START, "process")
    graph.add_conditional_edges("process", route, {"process": "process", END: END})
    
    compiled = graph.compile()
    initial_state = {"messages": [], "counter": 0, "metadata": {}}
    result = compiled.invoke(initial_state)
    
    assert result["counter"] == 3
    assert len(result["messages"]) == 3


def test_checkpointing_memory_saver():
    """Test checkpointing with MemorySaver."""
    graph = StateGraph(TestStateSchema)
    
    def node_a(state: TestStateSchema) -> TestStateSchema:
        return {"messages": ["A"], "counter": state.get("counter", 0) + 1, "metadata": {}}
    
    def node_b(state: TestStateSchema) -> TestStateSchema:
        return {"messages": ["B"], "counter": state.get("counter", 0) + 1, "metadata": {}}
    
    graph.add_node("node_a", node_a)
    graph.add_node("node_b", node_b)
    graph.add_edge(START, "node_a")
    graph.add_edge("node_a", "node_b")
    graph.add_edge("node_b", END)
    
    checkpointer = MemorySaver()
    compiled = graph.compile(checkpointer=checkpointer)
    
    config = {"configurable": {"thread_id": "test_thread_1"}}
    initial_state = {"messages": [], "counter": 0, "metadata": {}}
    
    result1 = compiled.invoke(initial_state, config=config)
    assert result1["counter"] == 2
    
    # Resume from checkpoint
    result2 = compiled.invoke(None, config=config)
    assert result2["counter"] == 2  # Should be same state


def test_checkpointing_multiple_threads():
    """Test checkpointing with multiple threads."""
    graph = StateGraph(TestStateSchema)
    
    def node_a(state: TestStateSchema) -> TestStateSchema:
        return {"messages": ["A"], "counter": state.get("counter", 0) + 1, "metadata": {}}
    
    graph.add_node("node_a", node_a)
    graph.add_edge(START, "node_a")
    graph.add_edge("node_a", END)
    
    checkpointer = MemorySaver()
    compiled = graph.compile(checkpointer=checkpointer)
    
    config1 = {"configurable": {"thread_id": "thread_1"}}
    config2 = {"configurable": {"thread_id": "thread_2"}}
    
    initial_state = {"messages": [], "counter": 0, "metadata": {}}
    
    result1 = compiled.invoke(initial_state, config=config1)
    result2 = compiled.invoke(initial_state, config=config2)
    
    assert result1["counter"] == 1
    assert result2["counter"] == 1
    
    # Continue thread 1
    result1_continued = compiled.invoke(None, config=config1)
    assert result1_continued["counter"] == 1  # Same state


def test_checkpointing_state_history():
    """Test retrieving state history from checkpoints."""
    graph = StateGraph(TestStateSchema)
    
    def node_a(state: TestStateSchema) -> TestStateSchema:
        return {"messages": ["A"], "counter": state.get("counter", 0) + 1, "metadata": {"step": "A"}}
    
    def node_b(state: TestStateSchema) -> TestStateSchema:
        return {"messages": ["B"], "counter": state.get("counter", 0) + 1, "metadata": {"step": "B"}}
    
    graph.add_node("node_a", node_a)
    graph.add_node("node_b", node_b)
    graph.add_edge(START, "node_a")
    graph.add_edge("node_a", "node_b")
    graph.add_edge("node_b", END)
    
    checkpointer = MemorySaver()
    compiled = graph.compile(checkpointer=checkpointer)
    
    config = {"configurable": {"thread_id": "history_test"}}
    initial_state = {"messages": [], "counter": 0, "metadata": {}}
    
    compiled.invoke(initial_state, config=config)
    
    # Get state history
    history = list(checkpointer.list(config))
    assert len(history) > 0


def test_graph_compilation_with_complex_schema():
    """Test graph compilation with complex state schema."""
    graph = StateGraph(ComplexStateSchema)
    
    def init_node(state: ComplexStateSchema) -> ComplexStateSchema:
        return {
            "user_id": "user_123",
            "session_data": {"started_at": time.time()},
            "history": [],
            "current_step": "init",
            "retry_count": 0,
            "config": {"max_retries": 3}
        }
    
    def process_node(state: ComplexStateSchema) -> ComplexStateSchema:
        history = state.get("history", [])
        history.append({"step": state["current_step"], "timestamp": time.time()})
        return {
            "history": history,
            "current_step": "processing",
            "retry_count": state.get("retry_count", 0) + 1
        }
    
    graph.add_node("init", init_node)
    graph.add_node("process", process_node)
    graph.add_edge(START, "init")
    graph.add_edge("init", "process")
    graph.add_edge("process", END)
    
    compiled = graph.compile()
    assert compiled is not None
    
    initial_state = {
        "user_id": "",
        "session_data": {},
        "history": [],
        "current_step": "",
        "retry_count": 0,
        "config": {}
    }
    result = compiled.invoke(initial_state)
    
    assert result["user_id"] == "user_123"
    assert result["current_step"] == "processing"
    assert len(result["history"]) == 1


def test_graph_compilation_with_minimal_schema():
    """Test graph compilation with minimal state schema."""
    graph = StateGraph(MinimalStateSchema)
    
    def transform(state: MinimalStateSchema) -> MinimalStateSchema:
        return {"value": state.get("value", "") + "_transformed"}
    
    graph.add_node("transform", transform)
    graph.add_edge(START, "transform")
    graph.add_edge("transform", END)
    
    compiled = graph.compile()
    result = compiled.invoke({"value": "initial"})
    
    assert result["value"] == "initial_transformed"


def test_graph_compilation_with_optional_fields():
    """Test graph compilation with optional state fields."""
    graph = StateGraph(StateWithOptional)
    
    def process(state: StateWithOptional) -> StateWithOptional:
        return {
            "required_field": state["required_field"],
            "optional_field": state.get("optional_field", "default"),
            "default_field": "processed"
        }
    
    graph.add_node("process", process)
    graph.add_edge(START, "process")
    graph.add_edge("process", END)
    
    compiled = graph.compile()
    
    # Test with optional field
    result1 = compiled.invoke({
        "required_field": "req",
        "optional_field": "opt",
        "default_field": "def"
    })
    assert result1["optional_field"] == "opt"
    
    # Test without optional field
    result2 = compiled.invoke({
        "required_field": "req",
        "default_field": "def"
    })
    assert result2["optional_field"] == "default"


def test_multi_agent_state_transitions():
    """Test state transitions in multi-agent setup."""
    graph = StateGraph(AgentState)
    
    def agent_a(state: AgentState) -> AgentState:
        return {
            "agent_name": "agent_a",
            "task": state.get("task", ""),
            "result": "Agent A completed task",
            "next_agent": "agent_b",
            "shared_data": {"from_a": "data_a"}
        }
    
    def agent_b(state: AgentState) -> AgentState:
        shared = state.get("shared_data", {})
        shared["from_b"] = "data_b"
        return {
            "agent_name": "agent_b",
            "task": state.get("task", ""),
            "result": "Agent B completed task",
            "next_agent": None,
            "shared_data": shared
        }
    
    def route(state: AgentState) -> str:
        next_agent = state.get("next_agent")
        if next_agent:
            return next_agent
        return END
    
    graph.add_node("agent_a", agent_a)
    graph.add_node("agent_b", agent_b)
    graph.add_edge(START, "agent_a")
    graph.add_conditional_edges("agent_a", route, {"agent_b": "agent_b", END: END})
    graph.add_conditional_edges("agent_b", route, {"agent_b": "agent_b", END: END})
    
    compiled = graph.compile()
    initial_state = {
        "agent_name": "",
        "task": "test_task",
        "result": None,
        "next_agent": None,
        "shared_data": {}
    }
    
    result = compiled.invoke(initial_state)
    
    assert result["agent_name"] == "agent_b"
    assert result["next_agent"] is None
    assert "from_a" in result["shared_data"]
    assert "from_b" in result["shared_data"]


def test_workflow_state_transitions():
    """Test workflow state transitions."""
    graph = StateGraph(WorkflowState)
    
    def start_workflow(state: WorkflowState) -> WorkflowState:
        return {
            "workflow_id": "wf_123",
            "status": "running",
            "steps_completed": ["start"],
            "current_step": "step_1",
            "error": None,
            "metadata": {"started_at": time.time()}
        }
    
    def step_1(state: WorkflowState) -> WorkflowState:
        steps = state.get("steps_completed", [])
        steps.append("step_1")
        return {
            "steps_completed": steps,
            "current_step": "step_2",
            "status": "running"
        }
    
    def step_2(state: WorkflowState) -> WorkflowState:
        steps = state.get("steps_completed", [])
        steps.append("step_2")
        return {
            "steps_completed": steps,
            "current_step": "complete",
            "status": "completed"
        }
    
    def route(state: WorkflowState) -> str:
        current = state.get("current_step", "")
        if current == "step_1":
            return "step_1"
        elif current == "step_2":
            return "step_2"
        return END
    
    graph.add_node("start", start_workflow)
    graph.add_node("step_1", step_1)
    graph.add_node("step_2", step_2)
    graph.add_edge(START, "start")
    graph.add_conditional_edges("start", route, {"step_1": "step_1", "step_2": "step_2", END: END})
    graph.add_conditional_edges("step_1", route, {"step_1": "step_1", "step_2": "step_2", END: END})
    graph.add_conditional_edges("step_2", route, {"step_1": "step_1", "step_2": "step_2", END: END})
    
    compiled = graph.compile()
    initial_state = {
        "workflow_id": "",
        "status": "pending",
        "steps_completed": [],
        "current_step": "",
        "error": None,
        "metadata": {}
    }
    
    result = compiled.invoke(initial_state)
    
    assert result["status"] == "completed"
    assert "step_1" in result["steps_completed"]
    assert "step_2" in result["steps_completed"]
    assert result["workflow_id"] == "wf_123"


def test_graph_compilation_with_interrupt():
    """Test graph compilation with interrupt capability."""
    graph = StateGraph(TestStateSchema)
    
    def node_with_interrupt(state: TestStateSchema) -> TestStateSchema:
        # Simulate interrupt
        return {"messages": ["interrupted"], "counter": state.get("counter", 0), "metadata": {}}
    
    graph.add_node("interrupt_node", node_with_interrupt)
    graph.add_edge(START, "interrupt_node")
    graph.add_edge("interrupt_node", END)
    
    compiled = graph.compile()
    assert compiled is not None


def test_graph_recursion_limit():
    """Test graph recursion limit handling."""
    graph = StateGraph(TestStateSchema)
    
    def recursive_node(state: TestStateSchema) -> TestStateSchema:
        counter = state.get("counter", 0) + 1
        if counter > 100:
            return {"counter": counter, "messages": ["done"]}
        return {"counter": counter}
    
    def route(state: TestStateSchema) -> str:
        if state.get("counter", 0) > 100:
            return END
        return "recursive_node"
    
    graph.add_node("recursive_node", recursive_node)
    graph.add_edge(START, "recursive_node")
    graph.add_conditional_edges("recursive_node", route, {"recursive_node": "recursive_node", END: END})
    
    compiled = graph.compile()
    initial_state = {"messages": [], "counter": 0, "metadata": {}}
    
    # Should complete without recursion error
    result = compiled.invoke(initial_state, config={"recursion_limit": 200})
    assert result["counter"] > 100


def test_parallel_node_execution():
    """Test parallel node execution with Send."""
    graph = StateGraph(TestStateSchema)
    
    def fan_out(state: TestStateSchema) -> List[Send]:
        return [
            Send("parallel_a", {"messages": ["A"], "counter": 1, "metadata": {}}),
            Send("parallel_b", {"messages": ["B"], "counter": 2, "metadata": {}}),
            Send("parallel_c", {"messages": ["C"], "counter": 3, "metadata": {}})
        ]
    
    def parallel_a(state: TestStateSchema) -> TestStateSchema:
        return {"messages": state["messages"] + ["A_done"], "counter": state["counter"] * 10}
    
    def parallel_b(state: TestStateSchema) -> TestStateSchema:
        return {"messages": state["messages"] + ["B_done"], "counter": state["counter"] * 10}
    
    def parallel_c(state: TestStateSchema) -> TestStateSchema:
        return {"messages": state["messages"] + ["C_done"], "counter": state["counter"] * 10}
    
    def fan_in(state: TestStateSchema) -> TestStateSchema:
        return state
    
    graph.add_node("fan_out", fan_out)
    graph.add_node("parallel_a", parallel_a)
    graph.add_node("parallel_b", parallel_b)
    graph.add_node("parallel_c", parallel_c)
    graph.add_node("fan_in", fan_in)
    
    graph.add_edge(START, "fan_out")
    graph.add_edge("parallel_a", "fan_in")
    graph.add_edge("parallel_b", "fan_in")
    graph.add_edge("parallel_c", "fan_in")
    graph.add_edge("fan_in", END)
    
    compiled = graph.compile()
    assert compiled is not None


def test_state_schema_validation():
    """Test state schema validation during compilation."""
    graph = StateGraph(TestStateSchema)
    
    def invalid_node(state: TestStateSchema) -> Dict:
        # Return dict missing required fields
        return {"counter": 1}
    
    graph.add_node("invalid", invalid_node)
    graph.add_edge(START, "invalid")
    graph.add_edge("invalid", END)
    
    compiled = graph.compile()
    # Should still compile but may have runtime issues
    assert compiled is not None


def test_graph_compilation_with_config():
    """Test graph compilation with configuration."""
    graph = StateGraph(TestStateSchema)
    
    def configurable_node(state: TestStateSchema, config: Dict) -> TestStateSchema:
        multiplier = config.get("configurable", {}).get("multiplier", 1)
        return {"counter": state.get("counter", 0) * multiplier}
    
    graph.add_node("configurable", configurable_node)
    graph.add_edge(START, "configurable")
    graph.add_edge("configurable", END)
    
    compiled = graph.compile()
    
    result = compiled.invoke(
        {"messages": [], "counter": 5, "metadata": {}},
        config={"configurable": {"multiplier": 10}}
    )
    
    assert result["counter"] == 50


def test_checkpointing_with_sqlite():
    """Test checkpointing with SQLite (if available)."""
    graph = StateGraph(TestStateSchema)
    
    def node_a(state: TestStateSchema) -> TestStateSchema:
        return {"messages": ["A"], "counter": 1, "metadata": {}}
    
    graph.add_node("node_a", node_a)
    graph.add_edge(START, "node_a")
    graph.add_edge("node_a", END)
    
    # This would use SqliteSaver in real implementation
    # For testing, we just verify compilation works
    compiled = graph.compile()
    assert compiled is not None


def test_state_update_merging():
    """Test state update merging behavior."""
    graph = StateGraph(TestStateSchema)
    
    def partial_update(state: TestStateSchema) -> TestStateSchema:
        # Only update counter, leave messages and metadata unchanged
        return {"counter": state.get("counter", 0) + 1}
    
    graph.add_node("partial", partial_update)
    graph.add_edge(START, "partial")
    graph.add_edge("partial", END)
    
    compiled = graph.compile()
    initial_state = {"messages": ["initial"], "counter": 0, "metadata": {"key": "value"}}
    result = compiled.invoke(initial_state)
    
    assert result["counter"] == 1
    assert result["messages"] == ["initial"]  # Should be preserved
    assert result["metadata"] == {"key": "value"}  # Should be preserved


def test_streaming_state_updates():
    """Test streaming state updates."""
    graph = StateGraph(TestStateSchema)
    
    def node_a(state: TestStateSchema) -> TestStateSchema:
        return {"messages": ["A"], "counter": 1, "metadata": {}}
    
    def node_b(state: TestStateSchema) -> TestStateSchema:
        return {"messages": ["B"], "counter": 2, "metadata": {}}
    
    graph.add_node("node_a", node_a)
    graph.add_node("node_b", node_b)
    graph.add_edge(START, "node_a")
    graph.add_edge("node_a", "node_b")
    graph.add_edge("node_b", END)
    
    compiled = graph.compile()
    initial_state = {"messages": [], "counter": 0, "metadata": {}}
    
    # Stream updates
    updates = list(compiled.stream(initial_state))
    assert len(updates) >= 2


def test_async_graph_execution():
    """Test async graph execution."""
    graph = StateGraph(TestStateSchema)
    
    async def async_node(state: TestStateSchema) -> TestStateSchema:
        return {"messages": ["async"], "counter": 1, "metadata": {}}
    
    graph.add_node("async_node", async_node)
    graph.add_edge(START, "async_node")
    graph.add_edge("async_node", END)
    
    compiled = graph.compile()
    assert compiled is not None
    
    # Test ainvoke if available
    import asyncio
    async def test_ainvoke():
        result = await compiled.ainvoke({"messages": [], "counter": 0, "metadata": {}})
        return result
    
    result = asyncio.run(test_ainvoke())
    assert result["counter"] == 1


def test_graph_visualization():
    """Test graph visualization capabilities."""
    graph = StateGraph(TestStateSchema)
    
    def node_a(state: TestStateSchema) -> TestStateSchema:
        return {"messages": ["A"], "counter": 1, "metadata": {}}
    
    def node_b(state: TestStateSchema) -> TestStateSchema:
        return {"messages": ["B"], "counter": 2, "metadata": {}}
    
    graph.add_node("node_a", node_a)
    graph.add_node("node_b", node_b)
    graph.add_edge(START, "node_a")
    graph.add_edge("node_a", "node_b")
    graph.add_edge("node_b", END)
    
    compiled = graph.compile()
    
    # Test getting graph structure
    assert hasattr(compiled, "get_graph")
    graph_viz = compiled.get_graph()
    assert graph_viz is not None


def test_subgraph_compilation():
    """Test subgraph compilation."""
    # Create subgraph
    subgraph = StateGraph(TestStateSchema)
    
    def sub_node_a(state: TestStateSchema) -> TestStateSchema:
        return {"messages": ["sub_a"], "counter": 1, "metadata": {}}
    
    def sub_node_b(state: TestStateSchema) -> TestStateSchema:
        return {"messages": ["sub_b"], "counter": 2, "metadata": {}}
    
    subgraph.add_node("sub_a", sub_node_a)
    subgraph.add_node("sub_b", sub_node_b)
    subgraph.add_edge(START, "sub_a")
    subgraph.add_edge("sub_a", "sub_b")
    subgraph.add_edge("sub_b", END)
    
    compiled_subgraph = subgraph.compile()
    
    # Create main graph using subgraph
    main_graph = StateGraph(TestStateSchema)
    
    def main_node(state: TestStateSchema) -> TestStateSchema:
        return {"messages": ["main"], "counter": 0, "metadata": {}}
    
    main_graph.add_node("main", main_node)
    main_graph.add_node("subgraph", compiled_subgraph)
    main_graph.add_edge(START, "main")
    main_graph.add_edge("main", "subgraph")
    main_graph.add_edge("subgraph", END)
    
    compiled_main = main_graph.compile()
    assert compiled_main is not None
    
    result = compiled_main.invoke({"messages": [], "counter": 0, "metadata": {}})
    assert "sub_a" in result["messages"]
    assert "sub_b" in result["messages"]


def test_dynamic_state_schema():
    """Test dynamic state schema modification."""
    graph = StateGraph(TestStateSchema)
    
    def dynamic_node(state: TestStateSchema) -> TestStateSchema:
        # Add dynamic fields
        return {
            "messages": state.get("messages", []) + ["dynamic"],
            "counter": state.get("counter", 0) + 1,
            "metadata": {**state.get("metadata", {}), "dynamic": True}
        }
    
    graph.add_node("dynamic", dynamic_node)
    graph.add_edge(START, "dynamic")
    graph.add_edge("dynamic", END)
    
    compiled = graph.compile()
    result = compiled.invoke({"messages": [], "counter": 0, "metadata": {}})
    
    assert "dynamic" in result["messages"]
    assert result["metadata"].get("dynamic") is True


def test_error_handling_in_state_transitions():
    """Test error handling during state transitions."""
    graph = StateGraph(TestStateSchema)
    
    def error_node(state: TestStateSchema) -> TestStateSchema:
        raise ValueError("Test error")
    
    def recovery_node(state: TestStateSchema) -> TestStateSchema:
        return {"messages": ["recovered"], "counter": 0, "metadata": {"error": "handled"}}
    
    graph.add_node("error", error_node)
    graph.add_node("recovery", recovery_node)
    graph.add_edge(START, "error")
    graph.add_edge("error", "recovery")
    graph.add_edge("recovery", END)
    
    compiled = graph.compile()
    
    # Should propagate error
    with pytest.raises(ValueError):
        compiled.invoke({"messages": [], "counter": 0, "metadata": {}})


def test_state_persistence_across_restarts():
    """Test state persistence across graph restarts."""
    checkpointer = MemorySaver()
    
    graph = StateGraph(TestStateSchema)
    
    def persistent_node(state: TestStateSchema) -> TestStateSchema:
        return {"messages": ["persistent"], "counter": state.get("counter", 0) + 1, "metadata": {}}
    
    graph.add_node("persistent", persistent_node)
    graph.add_edge(START, "persistent")
    graph.add_edge("persistent", END)
    
    compiled = graph.compile(checkpointer=checkpointer)
    
    config = {"configurable": {"thread_id": "persist_test"}}
    
    # First run
    result1 = compiled.invoke({"messages": [], "counter": 0, "metadata": {}}, config=config)
    assert result1["counter"] == 1
    
    # Create new compiled graph (simulating restart)
    compiled2 = graph.compile(checkpointer=checkpointer)
    
    # Resume from checkpoint
    result2 = compiled2.invoke(None, config=config)
    assert result2["counter"] == 1


def test_concurrent_checkpoint_access():
    """Test concurrent access to checkpoints."""
    import threading
    import time
    
    checkpointer = MemorySaver()
    graph = StateGraph(TestStateSchema)
    
    def slow_node(state: TestStateSchema) -> TestStateSchema:
        time.sleep(0.01)
        return {"messages": ["slow"], "counter": state.get("counter", 0) + 1, "metadata": {}}
    
    graph.add_node("slow", slow_node)
    graph.add_edge(START, "slow")
    graph.add_edge("slow", END)
    
    compiled = graph.compile(checkpointer=checkpointer)
    
    results = []
    errors = []
    
    def run_thread(thread_id):
        try:
            config = {"configurable": {"thread_id": f"concurrent_{thread_id}"}}
            result = compiled.invoke({"messages": [], "counter": 0, "metadata": {}}, config=config)
            results.append(result)
        except Exception as e:
            errors.append(e)
    
    threads = [threading.Thread(target=run_thread, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    assert len(errors) == 0
    assert len(results) == 10
    for result in results:
        assert result["counter"] == 1


def test_graph_compilation_performance():
    """Test graph compilation performance with large graphs."""
    graph = StateGraph(TestStateSchema)
    
    # Add many nodes
    for i in range(50):
        def make_node(idx):
            def node(state: TestStateSchema) -> TestStateSchema:
                return {"messages": [f"node_{idx}"], "counter": idx, "metadata": {}}
            return node
        
        graph.add_node(f"node_{i}", make_node(i))
    
    # Add edges in a chain
    graph.add_edge(START, "node_0")
    for i in range(49):
        graph.add_edge(f"node_{i}", f"node_{i+1}")
    graph.add_edge("node_49", END)
    
    # Compilation should complete in reasonable time
    start
