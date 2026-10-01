# LangChain Examples

A collection of practical, runnable examples for building applications with [LangChain](https://github.com/langchain-ai/langchain). Each example focuses on a specific LangChain concept and is designed to be easy to read, modify, and reuse.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Examples](#examples)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## Overview

This repository contains self-contained examples that show how to use LangChain for common LLM tasks. The examples are written in Python and can be run from the command line, used as references, or adapted into your own projects.

All example scripts live in the `examples/` directory and are numbered in a suggested learning order. The list below reflects the current set of files; new examples are added regularly. These examples are continuously updated to work with the latest LangChain release and best practices.

## Quick Start

1. Clone the repository:

   ```bash
   git clone https://github.com/kenwith/langchain_examples.git
   cd langchain_examples
   ```

2. Create and activate a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run an example:

   ```bash
   python examples/01_basic_llm.py
   ```

## Examples

All example scripts live in the `examples/` directory. The list below reflects the current set of files; new examples are added regularly. These examples are continuously updated to work with the latest LangChain release and best practices.

- [01_basic_llm.py](examples/01_basic_llm.py) - Make a simple LLM call.
- [02_prompt_templates.py](examples/02_prompt_templates.py) - Create reusable prompt templates.
- [03_chains.py](examples/03_chains.py) - Combine components into a single chain.
- [04_agents.py](examples/04_agents.py) - Use agents to dynamically choose actions.
- [05_memory.py](examples/05_memory.py) - Add conversation memory to chains and agents.
- [06_document_loaders.py](examples/06_document_loaders.py) - Load data from various sources.
- [07_embeddings.py](examples/07_embeddings.py) - Generate text embeddings.
- [08_vector_stores.py](examples/08_vector_stores.py) - Store and query vector embeddings.
- [09_question_answering.py](examples/09_question_answering.py) - Answer questions over documents.
- [10_summarization.py](examples/10_summarization.py) - Summarize long documents.

## Testing

The examples are meant to be run directly and do not have a formal test suite. You can verify an example works by running it with Python and checking the output. If you find a bug, please open an issue or submit a pull request.

## Contributing

Contributions are welcome! If you have an idea for a new example, an improvement to an existing one, or a fix for a bug, please open an issue or submit a pull request. Make sure your code follows the existing style and includes a clear description.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
