"""
Batch Inference Example

Demonstrates batch processing with semaphore-limited concurrency,
progress tracking, and error handling using LangChain's init_chat_model.

Key features:
- Semaphore-based concurrency control
- Progress tracking with callbacks
- Graceful error handling with retry logic
- Result aggregation and reporting
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Callable, Awaitable

from langchain.chat_models import init_chat_model


# ============================================================
# Configuration
# ============================================================

@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    max_concurrent: int = 3
    max_retries: int = 2
    retry_delay: float = 1.0
    timeout: float = 30.0


@dataclass
class BatchResult:
    """Result of a single batch item processing."""
    index: int
    input_data: Any
    output: Any = None
    error: Exception | None = None
    retries: int = 0
    duration: float = 0.0


# ============================================================
# Core Batch Processing
# ============================================================

async def process_with_semaphore(
    semaphore: asyncio.Semaphore,
    func: Callable[..., Awaitable[Any]],
    *args,
    **kwargs
) -> Any:
    """Execute a coroutine with semaphore-based concurrency limit."""
    async with semaphore:
        return await func(*args, **kwargs)


async def process_with_retry(
    func: Callable[..., Awaitable[Any]],
    *args,
    config: BatchConfig,
    **kwargs
) -> Any:
    """Execute a coroutine with retry logic."""
    last_exception = None
    
    for attempt in range(config.max_retries + 1):
        try:
            return await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=config.timeout
            )
        except Exception as e:
            last_exception = e
            if attempt < config.max_retries:
                await asyncio.sleep(config.retry_delay * (attempt + 1))
    
    raise last_exception


async def process_batch_item(
    index: int,
    input_data: Any,
    processor: Callable[..., Awaitable[Any]],
    semaphore: asyncio.Semaphore,
    config: BatchConfig,
    progress_callback: Callable[[int, int], None] | None = None
) -> BatchResult:
    """Process a single batch item with concurrency control and error handling."""
    start_time = time.perf_counter()
    
    try:
        result = await process_with_semaphore(
            semaphore,
            process_with_retry,
            processor,
            input_data,
            config=config
        )
        return BatchResult(
            index=index,
            input_data=input_data,
            output=result,
            duration=time.perf_counter() - start_time
        )
    except Exception as e:
        return BatchResult(
            index=index,
            input_data=input_data,
            error=e,
            duration=time.perf_counter() - start_time
        )
    finally:
        if progress_callback:
            progress_callback(index, 1)


async def run_batch(
    items: list[Any],
    processor: Callable[..., Awaitable[Any]],
    config: BatchConfig | None = None,
    progress_callback: Callable[[int, int], None] | None = None
) -> list[BatchResult]:
    """
    Run batch processing with controlled concurrency.
    
    Args:
        items: List of input items to process
        processor: Async function to process each item
        config: BatchConfig for concurrency and retry settings
        progress_callback: Optional callback(completed, total) for progress tracking
    
    Returns:
        List of BatchResult objects in order of input items
    """
    config = config or BatchConfig()
    semaphore = asyncio.Semaphore(config.max_concurrent)
    completed = 0
    total = len(items)
    
    def wrapped_progress(idx: int, count: int):
        nonlocal completed
        completed += count
        if progress_callback:
            progress_callback(completed, total)
    
    tasks = [
        process_batch_item(i, item, processor, semaphore, config, wrapped_progress)
        for i, item in enumerate(items)
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=False)
    
    # Sort by original index to maintain order
    results.sort(key=lambda r: r.index)
    return results


# ============================================================
# Example Processors
# ============================================================

async def llm_processor(prompt: str, model: str = "gpt-4o-mini") -> str:
    """Process a prompt using an LLM."""
    chat_model = init_chat_model(model, temperature=0)
    response = await chat_model.ainvoke(prompt)
    return response.content


async def mock_processor(item: dict) -> dict:
    """Mock processor for testing without API calls."""
    await asyncio.sleep(0.1)  # Simulate work
    return {"processed": True, "input": item}


# ============================================================
# Progress Tracking Helpers
# ============================================================

def create_progress_tracker(total: int, prefix: str = "Progress") -> Callable[[int, int], None]:
    """Create a simple progress tracking callback."""
    def track(completed: int, total_items: int):
        pct = (completed / total_items) * 100
        bar_len = 30
        filled = int(bar_len * completed / total_items)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"\r{prefix}: [{bar}] {completed}/{total_items} ({pct:.1f}%)", end="", flush=True)
        if completed == total_items:
            print()
    return track


# ============================================================
# Result Analysis
# ============================================================

def analyze_results(results: list[BatchResult]) -> dict:
    """Analyze batch results and return summary statistics."""
    successful = [r for r in results if r.error is None]
    failed = [r for r in results if r.error is not None]
    
    durations = [r.duration for r in successful]
    
    return {
        "total": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "success_rate": len(successful) / len(results) * 100 if results else 0,
        "avg_duration": sum(durations) / len(durations) if durations else 0,
        "min_duration": min(durations) if durations else 0,
        "max_duration": max(durations) if durations else 0,
        "total_duration": sum(r.duration for r in results),
        "errors": [{"index": r.index, "error": str(r.error)} for r in failed]
    }


def print_results_table(results: list[BatchResult]) -> None:
    """Print results in a readable table format."""
    print("\n" + "=" * 80)
    print(f"{'Index':<6} | {'Status':<10} | {'Duration':>8} | {'Details'}")
    print("-" * 80)
    
    for r in results:
        status = "SUCCESS" if r.error is None else "FAILED"
        duration = f"{r.duration:.3f}s"
        
        if r.error:
            details = f"Error: {type(r.error).__name__}: {r.error}"
        else:
            output_preview = str(r.output)[:50] + "..." if len(str(r.output)) > 50 else str(r.output)
            details = f"Output: {output_preview}"
        
        print(f"{r.index:<6} | {status:<10} | {duration:>8} | {details}")
    
    print("=" * 80)


# ============================================================
# Demo
# ============================================================

async def main():
    """Run batch inference demonstration."""
    print("=" * 60)
    print("Batch Inference Demo")
    print("=" * 60)
    
    # Sample prompts for processing
    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in one sentence.",
        "Write a haiku about programming.",
        "What is 2 + 2?",
        "Name three programming languages.",
        "What is the speed of light?",
        "Define machine learning.",
        "What is Python?",
        "Explain recursion briefly.",
        "What is an API?",
    ]
    
    # Configuration
    config = BatchConfig(
        max_concurrent=3,
        max_retries=1,
        retry_delay=0.5,
        timeout=15.0
    )
    
    print(f"\nProcessing {len(prompts)} items with max_concurrent={config.max_concurrent}")
    print("-" * 60)
    
    # Create progress tracker
    progress = create_progress_tracker(len(prompts), "Batch")
    
    # Run batch with mock processor (no API key needed)
    print("\nUsing mock processor (no API calls)...")
    results = await run_batch(
        items=prompts,
        processor=mock_processor,
        config=config,
        progress_callback=progress
    )
    
    # Analyze and display results
    summary = analyze_results(results)
    print_results_table(results)
    
    print("\nSummary:")
    print(f"  Total:        {summary['total']}")
    print(f"  Successful:   {summary['successful']}")
    print(f"  Failed:       {summary['failed']}")
    print(f"  Success Rate: {summary['success_rate']:.1f}%")
    print(f"  Avg Duration: {summary['avg_duration']:.3f}s")
    print(f"  Total Time:   {summary['total_duration']:.3f}s")
    
    # Demonstrate error handling with a failing processor
    print("\n" + "=" * 60)
    print("Error Handling Demo")
    print("=" * 60)
    
    async def failing_processor(item: str) -> str:
        if "error" in item.lower():
            raise ValueError(f"Simulated error for: {item}")
        await asyncio.sleep(0.05)
        return f"Processed: {item}"
    
    test_items = ["item1", "item2", "trigger error", "item4", "another error"]
    error_config = BatchConfig(max_concurrent=2, max_retries=2, retry_delay=0.1)
    
    print(f"\nProcessing {len(test_items)} items (some will fail)...")
    error_progress = create_progress_tracker(len(test_items), "Errors")
    
    error_results = await run_batch(
        items=test_items,
        processor=failing_processor,
        config=error_config,
        progress_callback=error_progress
    )
    
    print_results_table(error_results)
    
    error_summary = analyze_results(error_results)
    print(f"\nErrors caught: {error_summary['failed']}")
    for err in error_summary['errors']:
        print(f"  Index {err['index']}: {err['error']}")


if __name__ == "__main__":
    asyncio.run(main())
