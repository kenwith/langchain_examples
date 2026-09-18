import json
from typing import Literal, TypedDict

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

# Define the state schema
class AgentState(TypedDict):
    input: str
    decision: str
    output: str

# Initialize the model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Define the nodes
def generate_decision(state: AgentState) -> AgentState:
    """Generate a decision based on the input using the LLM."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a router. Respond with exactly 'YES' or 'NO' based on whether the input mentions a weather-related topic."),
        ("human", "{input}")
    ])
    chain = prompt | llm | StrOutputParser()
    decision = chain.invoke({"input": state["input"]}).strip().upper()
    return {"decision": decision}

def process_yes(state: AgentState) -> AgentState:
    """Branch for YES decision."""
    return {"output": f"Handled as weather: {state['input']}"}

def process_no(state: AgentState) -> AgentState:
    """Branch for NO decision."""
    return {"output": f"Handled as non-weather: {state['input']}"}

# Define the conditional edge function
def route_based_on_decision(state: AgentState) -> Literal["process_yes", "process_no"]:
    """Route to the appropriate branch based on the decision."""
    if state["decision"] == "YES":
        return "process_yes"
    else:
        return "process_no"

# Build the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("generate_decision", generate_decision)
workflow.add_node("process_yes", process_yes)
workflow.add_node("process_no", process_no)

# Add edges
workflow.add_edge(START, "generate_decision")
workflow.add_conditional_edges(
    "generate_decision",
    route_based_on_decision,
    {
        "process_yes": "process_yes",
        "process_no": "process_no",
    }
)
workflow.add_edge("process_yes", END)
workflow.add_edge("process_no", END)

# Compile the graph
app = workflow.compile()

# Example usage
if __name__ == "__main__":
    test_inputs = [
        "What's the weather like today?",
        "Tell me a joke.",
    ]
    for user_input in test_inputs:
        result = app.invoke({"input": user_input})
        print(f"Input: {user_input}")
        print(f"Decision: {result['decision']}")
        print(f"Output: {result['output']}\n")
