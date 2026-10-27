"""
Ollama LangGraph workflow example.

This module assembles a three-stage LangGraph workflow using Ollama's
ChatOllama as the language model. The workflow writes a short story,
critiques it, and rewrites it based on the critique.
"""

from typing import TypedDict

from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph


class WorkflowState(TypedDict):
    topic: str
    story: str
    critique: str
    final_story: str


llm = ChatOllama(model="llama3", temperature=0.7)


def write_story(state: WorkflowState) -> dict:
    prompt = f"Write a short story about {state['topic']}."
    story = llm.invoke(prompt).content
    return {"story": story}


def critique_story(state: WorkflowState) -> dict:
    prompt = f"Critique the following story and suggest improvements:\n\n{state['story']}"
    critique = llm.invoke(prompt).content
    return {"critique": critique}


def improve_story(state: WorkflowState) -> dict:
    prompt = (
        f"Rewrite the story based on this critique:\n\n"
        f"Story:\n{state['story']}\n\n"
        f"Critique:\n{state['critique']}"
    )
    final_story = llm.invoke(prompt).content
    return {"final_story": final_story}


def build_workflow():
    """Build and compile the LangGraph workflow.

    The workflow graph is assembled with three nodes:
    - write_story: generates an initial story from a topic.
    - critique_story: critiques the story and suggests improvements.
    - improve_story: rewrites the story based on the critique.

    Edges connect write_story -> critique_story -> improve_story -> END.
    """
    workflow = StateGraph(WorkflowState)

    workflow.add_node("write_story", write_story)
    workflow.add_node("critique_story", critique_story)
    workflow.add_node("improve_story", improve_story)

    workflow.set_entry_point("write_story")

    workflow.add_edge("write_story", "critique_story")
    workflow.add_edge("critique_story", "improve_story")
    workflow.add_edge("improve_story", END)

    return workflow.compile()


if __name__ == "__main__":
    app = build_workflow()
    result = app.invoke({"topic": "a lost key"})

    print("Topic:", result["topic"])
    print("\nStory:\n", result["story"])
    print("\nCritique:\n", result["critique"])
    print("\nFinal story:\n", result["final_story"])
