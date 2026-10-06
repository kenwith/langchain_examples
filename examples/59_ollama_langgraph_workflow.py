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


workflow = StateGraph(WorkflowState)

workflow.add_node("write_story", write_story)
workflow.add_node("critique_story", critique_story)
workflow.add_node("improve_story", improve_story)

workflow.set_entry_point("write_story")

workflow.add_edge("write_story", "critique_story")
workflow.add_edge("critique_story", "improve_story")
workflow.add_edge("improve_story", END)

app = workflow.compile()

if __name__ == "__main__":
    result = app.invoke({"topic": "a lost key"})

    print("Topic:", result["topic"])
    print("\nStory:\n", result["story"])
    print("\nCritique:\n", result["critique"])
    print("\nFinal story:\n", result["final_story"])
