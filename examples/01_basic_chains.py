"""Basic chain example.

Provider configuration:
- Requires an OpenAI API key. Set the OPENAI_API_KEY environment variable,
  or create a .env file in the project root with:
  OPENAI_API_KEY=your-api-key
- The default model is "gpt-4o-mini". Override with OPENAI_MODEL if needed.

Expected output:
A generated response from the model, e.g.:

```
Why did the chicken cross the road?
To get to the other side.
```
"""

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser


def run_example():
    """Run the basic chain example."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    prompt = PromptTemplate.from_template("Tell me a {adjective} joke.")
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"adjective": "funny"})
    print(response)


if __name__ == "__main__":
    run_example()
