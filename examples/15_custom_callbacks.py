"""
Example 15: Custom Callbacks

This example demonstrates how to implement a custom callback handler in LangChain.

The handler:
- Accumulates token usage metadata across all LLM calls in a chain.
- Logs the order of callback lifecycle events as they occur.

Typical callback lifecycle for an LLMChain:
1. on_chain_start
2. on_llm_start
3. on_llm_end
4. on_chain_end
If an error occurs, on_llm_error and/or on_chain_error will be triggered.
"""

import os
from typing import Any, Dict, List

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_openai import ChatOpenAI


class TokenUsageCallbackHandler(BaseCallbackHandler):
    """Custom callback that accumulates token usage and logs lifecycle events."""

    def __init__(self) -> None:
        self.token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self.event_order: List[str] = []

    def _record_event(self, event_name: str) -> None:
        self.event_order.append(event_name)
        print(f"[callback] {event_name}")

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        self._record_event("on_chain_start")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        self._record_event("on_chain_end")

    def on_chain_error(self, error: BaseException, **kwargs: Any) -> None:
        self._record_event("on_chain_error")

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        self._record_event("on_llm_start")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        self._record_event("on_llm_end")

        # Accumulate token usage from the LLM result
        if response.llm_output and "token_usage" in response.llm_output:
            usage = response.llm_output["token_usage"]
            self.token_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
            self.token_usage["completion_tokens"] += usage.get("completion_tokens", 0)
            self.token_usage["total_tokens"] += usage.get("total_tokens", 0)
        else:
            print("[callback] No token usage found in LLM output")

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        self._record_event("on_llm_error")

    def print_usage_summary(self) -> None:
        """Print a formatted summary of accumulated token usage."""
        print("\n--- Token Usage Summary ---")
        print(f"Prompt tokens: {self.token_usage['prompt_tokens']}")
        print(f"Completion tokens: {self.token_usage['completion_tokens']}")
        print(f"Total tokens: {self.token_usage['total_tokens']}")


def main() -> None:
    # Ensure the API key is available
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set the OPENAI_API_KEY environment variable.")
        return

    callback = TokenUsageCallbackHandler()

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    prompt = PromptTemplate.from_template("Tell me a short joke about {topic}.")
    chain = LLMChain(llm=llm, prompt=prompt)

    result = chain.run({"topic": "programming"}, callbacks=[callback])

    print("\n--- Event Order ---")
    for i, event in enumerate(callback.event_order, 1):
        print(f"{i}. {event}")

    callback.print_usage_summary()

    print("\n--- Result ---")
    print(result)


if __name__ == "__main__":
    main()
