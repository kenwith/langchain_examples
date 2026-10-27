"""
Example 15: Custom Callbacks

This example demonstrates how to implement a custom callback handler in LangChain.

The handler:
- Accumulates token usage metadata across all LLM calls in a chain.
- Records per-call latency for each LLM request.
- Logs the order of callback lifecycle events as they occur.
- Prints token usage and latency for each LLM response in on_llm_end.

Typical callback lifecycle for an LLMChain:
1. on_chain_start
2. on_llm_start
3. on_llm_end
4. on_chain_end
If an error occurs, on_llm_error and/or on_chain_error will be triggered.
"""

import os
import time
from typing import Any, Dict, List

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_openai import ChatOpenAI


class TokenUsageCallbackHandler(BaseCallbackHandler):
    """Custom callback that accumulates token usage, measures latency, and logs lifecycle events."""

    def __init__(self) -> None:
        self.token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self.llm_latencies: List[float] = []
        self._llm_start_times: Dict[Any, float] = {}
        self.event_order: List[str] = []

    def _record_event(self, event_name: str) -> None:
        """Append an event name to the order log and print it."""
        self.event_order.append(event_name)
        print(f"[callback] {event_name}")

    def _record_llm_latency(self, run_id: Any) -> float | None:
        """Record and return the latency for an LLM call using its run_id.

        Args:
            run_id: The run_id of the LLM call, used to find the start time.

        Returns:
            The latency in seconds, or None if a matching start time is unavailable.
        """
        if run_id is None:
            return None
        start_time = self._llm_start_times.pop(run_id, None)
        if start_time is None:
            return None
        latency = time.time() - start_time
        self.llm_latencies.append(latency)
        return latency

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Log the start of a chain."""
        self._record_event("on_chain_start")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Log the end of a chain."""
        self._record_event("on_chain_end")

    def on_chain_error(self, error: BaseException, **kwargs: Any) -> None:
        """Log an error raised during a chain."""
        self._record_event("on_chain_error")

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Log the start of an LLM call and record its start time."""
        self._record_event("on_llm_start")
        run_id = kwargs.get("run_id")
        if run_id is not None:
            self._llm_start_times[run_id] = time.time()

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Log the end of an LLM call, print token usage, and record latency."""
        self._record_event("on_llm_end")

        # Print token usage from the LLM response for this call
        if response.llm_output and "token_usage" in response.llm_output:
            usage = response.llm_output["token_usage"]
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)
            print(
                f"[callback] Token usage - prompt: {prompt_tokens}, "
                f"completion: {completion_tokens}, total: {total_tokens}"
            )

            # Accumulate token usage across all LLM calls
            self.token_usage["prompt_tokens"] += prompt_tokens
            self.token_usage["completion_tokens"] += completion_tokens
            self.token_usage["total_tokens"] += total_tokens
        else:
            print("[callback] No token usage found in LLM output")

        latency = self._record_llm_latency(kwargs.get("run_id"))
        if latency is not None:
            print(f"[callback] Latency: {latency:.2f}s")

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        """Log an error raised during an LLM call and record its latency."""
        self._record_event("on_llm_error")
        latency = self._record_llm_latency(kwargs.get("run_id"))
        if latency is not None:
            print(f"[callback] Error latency: {latency:.2f}s")

    def print_usage_summary(self) -> None:
        """Print a formatted summary of accumulated token usage."""
        print("\n--- Token Usage Summary ---")
        print(f"Prompt tokens: {self.token_usage['prompt_tokens']}")
        print(f"Completion tokens: {self.token_usage['completion_tokens']}")
        print(f"Total tokens: {self.token_usage['total_tokens']}")

    def print_latency_summary(self) -> None:
        """Print a formatted summary of recorded LLM call latencies."""
        if not self.llm_latencies:
            print("\n--- Latency Summary ---")
            print("No LLM call latencies recorded.")
            return
        total = sum(self.llm_latencies)
        average = total / len(self.llm_latencies)
        print("\n--- Latency Summary ---")
        print(f"LLM calls: {len(self.llm_latencies)}")
        print(f"Total latency: {total:.2f}s")
        print(f"Average latency: {average:.2f}s")

    def print_callback_events(self) -> None:
        """Print a formatted summary of callback lifecycle events."""
        print("\n--- Callback Events ---")
        for i, event in enumerate(self.event_order, 1):
            print(f"{i}. {event}")


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

    callback.print_callback_events()
    callback.print_usage_summary()
    callback.print_latency_summary()

    print("\n--- Result ---")
    print(result)


if __name__ == "__main__":
    main()
