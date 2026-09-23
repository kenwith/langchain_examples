"""Basic chain example."""

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def run_chain(topic: str) -> None:
    """Run a basic chain for the given topic."""
    llm = OpenAI(temperature=0.7)
    prompt = PromptTemplate(
        input_variables=["topic"],
        template="Tell me a fun fact about {topic}.",
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(topic)
    print(response)

if __name__ == "__main__":
    run_chain("space")
    run_chain("oceans")
