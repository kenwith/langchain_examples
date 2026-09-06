"""
LangGraph Workflows Example

Demonstrates: Stateful graphs, cycles, branching, human-in-the-loop, persistence
Provider-agnostic using init_chat_model
"""
import os
from typing import Annotated, Literal
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

load_dotenv()


def get_model():
    model_name = os.getenv("LANGCHAIN_MODEL", "openai/gpt-4o-mini")
    return init_chat_model(model_name)


# =============================================================================
# Basic State Definition
# =============================================================================

class ChatState(BaseModel):
    messages: Annotated[list, add_messages] = Field(default_factory=list)
    user_name: str = "User"
    turn_count: int = 0


# =============================================================================
# Example 1: Simple Linear Workflow
# =============================================================================

def simple_workflow():
    """Simple linear workflow: classify -> respond"""
    print("=== Simple Linear Workflow ===")

    model = get_model()

    def classify_node(state: ChatState):
        """Classify the user's intent"""
        last_msg = state.messages[-1].content if state.messages else ""
        prompt = f"Classify this message: '{last_msg}' as 'question', 'greeting', or 'other'. Reply with just the category."
        result = model.invoke([HumanMessage(content=prompt)])
        category = result.content.strip().lower()
        return {"messages": [AIMessage(content=f"[Classified as: {category}]")]}

    def respond_node(state: ChatState):
        """Generate response based on classification"""
        last_ai = [m for m in state.messages if isinstance(m, AIMessage)][-1]
        category = last_ai.content.split(": ")[1].rstrip("]")
        
        if category == "greeting":
            response = f"Hello {state.user_name}! How can I help?"
        elif category == "question":
            response = model.invoke(state.messages).content
        else:
            response = "I'm not sure how to respond to that."
        
        return {"messages": [AIMessage(content=response)], "turn_count": state.turn_count + 1}

    # Build graph
    workflow = StateGraph(ChatState)
    workflow.add_node("classify", classify_node)
    workflow.add_node("respond", respond_node)
    workflow.add_edge(START, "classify")
    workflow.add_edge("classify", "respond")
    workflow.add_edge("respond", END)

    app = workflow.compile()

    # Test
    for msg in ["Hi there!", "What is LangGraph?", "Random statement"]:
        print(f"\nUser: {msg}")
        result = app.invoke({"messages": [HumanMessage(content=msg)], "user_name": "Alice"})
        for m in result["messages"]:
            if isinstance(m, AIMessage):
                print(f"Bot: {m.content}")


# =============================================================================
# Example 2: Branching Workflow
# =============================================================================

def branching_workflow():
    """Workflow with conditional branching"""
    print("\n=== Branching Workflow ===")

    model = get_model()

    class BranchState(BaseModel):
        messages: Annotated[list, add_messages] = Field(default_factory=list)
        route: Literal["technical", "creative", "general"] = "general"

    def router(state: BranchState):
        """Route based on message content"""
        last_msg = state.messages[-1].content.lower()
        if any(w in last_msg for w in ["code", "program", "debug", "api", "function"]):
            return "technical"
        elif any(w in last_msg for w in ["write", "story", "poem", "creative", "imagine"]):
            return "creative"
        return "general"

    def technical_node(state: BranchState):
        prompt = SystemMessage(content="You are a technical expert. Give concise, accurate answers.")
        result = model.invoke([prompt] + state.messages)
        return {"messages": [result]}

    def creative_node(state: BranchState):
        prompt = SystemMessage(content="You are a creative writer. Be imaginative and engaging.")
        result = model.invoke([prompt] + state.messages)
        return {"messages": [result]}

    def general_node(state: BranchState):
        result = model.invoke(state.messages)
        return {"messages": [result]}

    workflow = StateGraph(BranchState)
    workflow.add_node("technical", technical_node)
    workflow.add_node("creative", creative_node)
    workflow.add_node("general", general_node)
    workflow.add_conditional_edges(START, router, {
        "technical": "technical",
        "creative": "creative",
        "general": "general",
    })
    workflow.add_edge("technical", END)
    workflow.add_edge("creative", END)
    workflow.add_edge("general", END)

    app = workflow.compile()

    test_inputs = [
        "How do I debug a Python function?",
        "Write a short poem about coding",
        "What's the weather like?",
    ]

    for msg in test_inputs:
        print(f"\nUser: {msg}")
        result = app.invoke({"messages": [HumanMessage(content=msg)]})
        for m in result["messages"]:
            if isinstance(m, AIMessage):
                print(f"Bot: {m.content[:100]}...")


# =============================================================================
# Example 3: Human-in-the-Loop
# =============================================================================

def human_in_the_loop():
    """Workflow that pauses for human input"""
    print("\n=== Human-in-the-Loop ===")

    model = get_model()

    class HITLState(BaseModel):
        messages: Annotated[list, add_messages] = Field(default_factory=list)
        requires_approval: bool = False
        approved: bool = False

    def analyze_node(state: HITLState):
        """Analyze if action needs approval"""
        last_msg = state.messages[-1].content.lower()
        needs_approval = any(w in last_msg for w in ["delete", "remove", "send email", "purchase"])
        return {"requires_approval": needs_approval}

    def action_node(state: HITLState):
        """Execute the action"""
        action = "Action executed!" if state.approved else "Action cancelled."
        return {"messages": [AIMessage(content=action)]}

    def human_review(state: HITLState):
        """This node represents the human review pause"""
        # In real use, this would pause and wait for external input
        # For demo, we simulate approval
        return {"approved": True, "messages": [AIMessage(content="[Human approved]")]}

    workflow = StateGraph(HITLState)
    workflow.add_node("analyze", analyze_node)
    workflow.add_node("human_review", human_review)
    workflow.add_node("action", action_node)

    workflow.add_edge(START, "analyze")
    workflow.add_conditional_edges(
        "analyze",
        lambda s: "human_review" if s.requires_approval else "action",
        {"human_review": "human_review", "action": "action"}
    )
    workflow.add_edge("human_review", "action")
    workflow.add_edge("action", END)

    app = workflow.compile()

    test_inputs = [
        "Please summarize this text",
        "Delete all my files",
    ]

    for msg in test_inputs:
        print(f"\nUser: {msg}")
        result = app.invoke({"messages": [HumanMessage(content=msg)]})
        for m in result["messages"]:
            if isinstance(m, AIMessage):
                print(f"Bot: {m.content}")


# =============================================================================
# Example 4: Cyclic Workflow (Refinement Loop)
# =============================================================================

def refinement_loop():
    """Iterative refinement with a loop"""
    print("\n=== Refinement Loop ===")

    model = get_model()

    class RefinementState(BaseModel):
        messages: Annotated[list, add_messages] = Field(default_factory=list)
        draft: str = ""
        iteration: int = 0
        max_iterations: int = 3
        approved: bool = False

    def generate_draft(state: RefinementState):
        """Generate initial draft or refine"""
        if state.iteration == 0:
            prompt = f"Write a short paragraph about: {state.messages[-1].content}"
        else:
            prompt = f"Refine this draft (iteration {state.iteration}):\n{state.draft}\n\nMake it more concise and engaging."
        result = model.invoke([HumanMessage(content=prompt)])
        return {"draft": result.content, "iteration": state.iteration + 1}

    def review_draft(state: RefinementState):
        """Review the draft"""
        if state.iteration >= state.max_iterations:
            return {"approved": True, "messages": [AIMessage(content=f"Final draft (after {state.iteration} iterations):\n{state.draft}")]}

        # Simulate review - auto-approve after 2 iterations for demo
        approved = state.iteration >= 2
        return {"approved": approved}

    workflow = StateGraph(RefinementState)
    workflow.add_node("generate", generate_draft)
    workflow.add_node("review", review_draft)

    workflow.add_edge(START, "generate")
    workflow.add_edge("generate", "review")
    workflow.add_conditional_edges(
        "review",
        lambda s: END if s.approved else "generate",
        {True: END, False: "generate"}
    )

    app = workflow.compile()

    result = app.invoke({"messages": [HumanMessage(content="LangGraph workflows")]})
    for m in result["messages"]:
        if isinstance(m, AIMessage):
            print(m.content)


# =============================================================================
# Example 5: Persistent Checkpointing
# =============================================================================

def persistent_workflow():
    """Workflow with memory persistence"""
    print("\n=== Persistent Workflow (MemorySaver) ===")

    model = get_model()
    memory = MemorySaver()

    class PersistentState(BaseModel):
        messages: Annotated[list, add_messages] = Field(default_factory=list)
        session_id: str = "default"

    def chat_node(state: PersistentState):
        result = model.invoke(state.messages)
        return {"messages": [result]}

    workflow = StateGraph(PersistentState)
    workflow.add_node("chat", chat_node)
    workflow.add_edge(START, "chat")
    workflow.add_edge("chat", END)

    app = workflow.compile(checkpointer=memory)

    config = {"configurable": {"thread_id": "session-123"}}

    # First interaction
    result = app.invoke(
        {"messages": [HumanMessage(content="My name is Bob")]},
        config=config
    )
    print(f"Bot: {result['messages'][-1].content}")

    # Second interaction - should remember name
    result = app.invoke(
        {"messages": [HumanMessage(content="What's my name?")]},
        config=config
    )
    print(f"Bot: {result['messages'][-1].content}")

    # Different session - should not remember
    result = app.invoke(
        {"messages": [HumanMessage(content="What's my name?")]},
        config={"configurable": {"thread_id": "session-456"}}
    )
    print(f"Bot (new session): {result['messages'][-1].content}")


if __name__ == "__main__":
    simple_workflow()
    branching_workflow()
    human_in_the_loop()
    refinement_loop()
    persistent_workflow()
    print("\nAll LangGraph workflow examples completed!")