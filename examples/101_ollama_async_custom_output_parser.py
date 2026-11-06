"""101: Async custom output parser with Ollama.

This example demonstrates an asynchronous custom output parser with Ollama as
the model provider. It uses the provider-agnostic ``init_chat_model`` helper so
the same code can be pointed at a different chat model backend without changing
the example logic.
"""

import asyncio
import os
from typing import Dict, List

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.outputs import Generation


# ---------------------------------------------------------------------------
# Custom async output parser
# ---------------------------------------------------------------------------
class KeyValueParser(BaseOutputParser[Dict[str, str]]):
    """Parse a ``label: value`` response into a dictionary.

    The parser expects one ``label: value`` pair per line and overrides
    :meth:`aparse_result` to demonstrate an async post-processing hook.
    """

    def get_format_instructions(self) -> str:
        """Return instructions describing the expected response format."""
        return (
            "Return each answer as 'label: value' on its own line.\n"
            "Example:\n"
            "title: The Matrix\n"
            "year: 1999\n"
        )

    def parse(self, text: str) -> Dict[str, str]:
        """Parse colon-separated lines into a dictionary."""
        result = {}
        for line in text.strip().splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                result[key.strip()] = value.strip()
        return result

    async def aparse_result(
        self, result: List[Generation], *, partial: bool = False
    ) -> Dict[str, str]:
        """Asynchronously parse a generation list and normalize keys.

        The synchronous :meth:`parse` is called first, followed by an async
        placeholder step where real validation or enrichment could be added.
        """
        if partial:
            raise ValueError("Partial parsing is not supported by this parser")
        parsed = self.parse(result[0].text)
        # Placeholder for real async post-processing, such as database
        # validation or enrichment through an external service.
        await asyncio.sleep(0)
        return {key.lower(): value for key, value in parsed.items()}


# ---------------------------------------------------------------------------
# Example runner
# ---------------------------------------------------------------------------
async def run_example() -> None:
    """Run the async custom parser example with an Ollama-hosted model."""
    model = init_chat_model(
        os.getenv("OLLAMA_MODEL", "llama3.1"),
        provider="ollama",
        temperature=0,
    )
    parser = KeyValueParser()
    prompt = (
        "Provide details for the movie 'Inception'.\n"
        f"{parser.get_format_instructions()}"
    )
    message = await model.ainvoke(prompt)
    parsed = await parser.ainvoke(message)
    print(parsed)


if __name__ == "__main__":
    asyncio.run(run_example())
